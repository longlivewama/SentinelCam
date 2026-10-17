"""
Multi-person tracking (`SceneTracker`): ids that persist, survive brief
loss, end cleanly, carry each person's own pose, and never leak between
sessions.
"""
from collections import namedtuple

from app.services.detection.person_tracker import PersonTracker
from app.services.detection.scene_tracking import (
    STATE_CONFIRMED,
    STATE_LOST,
    STATE_TENTATIVE,
    SceneTracker,
)

Person = namedtuple("Person", ["bbox", "keypoints", "confidence"])

STEP = 0.2  # seconds between processed frames (stride 5 at 25 fps)


def person(x, y=50.0, w=40.0, h=120.0, kp_value=1.0, confidence=0.9, keypoints=True):
    kps = [(x + i, y + i, kp_value) for i in range(17)] if keypoints else []
    return Person(bbox=(x, y, x + w, y + h), keypoints=kps, confidence=confidence)


def run(tracker, frames, start=0):
    out = []
    for offset, people in enumerate(frames):
        index = start + offset
        out.append(tracker.update(people, now=index * STEP, frame_index=index))
    return out


def ids(tracked, state=None):
    return sorted(p.track_id for p in tracked if state is None or p.state == state)


def test_people_walking_keep_their_ids():
    tracker = SceneTracker()
    frames = [[person(10 + 4 * i), person(200 - 4 * i)] for i in range(20)]
    results = run(tracker, frames)
    assert all(ids(r) == [1, 2] for r in results)


def test_a_new_track_is_tentative_until_seen_again():
    tracker = SceneTracker(min_hits=2)
    first, second = run(tracker, [[person(10)], [person(12)]])
    assert first[0].state == STATE_TENTATIVE
    assert second[0].state == STATE_CONFIRMED


def test_brief_detection_loss_keeps_the_id_and_reports_lost():
    tracker = SceneTracker(max_age_seconds=1.5)
    run(tracker, [[person(10)], [person(11)], [person(12)]])
    gap = run(tracker, [[], []], start=3)               # 0.4 s unseen
    assert [(p.track_id, p.state) for p in gap[-1]] == [(1, STATE_LOST)]
    back = run(tracker, [[person(14)]], start=5)[0]
    assert [(p.track_id, p.state) for p in back] == [(1, STATE_CONFIRMED)]


def test_a_person_gone_longer_than_max_age_gets_a_new_id():
    tracker = SceneTracker(tracker=PersonTracker(max_age_seconds=0.5), max_age_seconds=0.5)
    run(tracker, [[person(10)], [person(11)]])
    run(tracker, [[]] * 5, start=2)                      # 1.0 s unseen
    back = run(tracker, [[person(12)]], start=7)[0]
    assert ids(back) == [2]


def test_people_entering_and_leaving():
    tracker = SceneTracker(tracker=PersonTracker(max_age_seconds=0.3), max_age_seconds=0.3)
    results = run(tracker, [
        [person(10)],
        [person(12), person(300)],    # second person enters
        [person(14), person(302)],
        [person(304)],                # first leaves
        [person(306)],
        [person(308)],
    ])
    assert ids(results[1]) == [1, 2]
    assert ids(results[-1]) == [2]
    summaries = {s.track_id: s for s in tracker.summaries()}
    assert summaries[1].frames_observed == 3
    assert summaries[2].first_seen == STEP


def test_each_track_carries_its_own_keypoints_and_confidence():
    tracker = SceneTracker()
    tracked = run(tracker, [[person(10, kp_value=0.8, confidence=0.71), person(200, kp_value=0.4, confidence=0.55)]])[0]
    by_id = {p.track_id: p for p in tracked}
    assert by_id[1].keypoints[0] == (10.0, 50.0, 0.8)
    assert by_id[1].confidence == 0.71
    assert by_id[2].keypoints[0] == (200.0, 50.0, 0.4)
    assert by_id[2].confidence == 0.55


def test_a_person_without_a_usable_pose_is_still_tracked():
    tracker = SceneTracker()
    tracked = run(tracker, [[person(10, keypoints=False)], [person(11, keypoints=False)]])[-1]
    assert ids(tracked) == [1]
    assert tracked[0].keypoints == []
    assert tracked[0].torso_angle is None


def test_an_empty_scene_is_fine():
    tracker = SceneTracker()
    assert run(tracker, [[], [], []]) == [[], [], []]
    assert tracker.summaries() == []


def test_sessions_are_isolated():
    """Two cameras (or two videos) never share ids or state. Ultralytics'
    trackers number tracks from one class-level counter shared by every
    instance; this must not."""
    camera_a, camera_b = SceneTracker(), SceneTracker()
    run(camera_a, [[person(10), person(200)]] * 3)
    b = run(camera_b, [[person(500)]])[0]
    assert ids(b) == [1]
    assert ids(camera_a.current) == [1, 2]
    assert camera_b.get(2) is None


def test_a_new_session_starts_from_scratch():
    """Per-video reset: a fresh tracker for the next upload."""
    first = SceneTracker()
    run(first, [[person(10)], [person(11)], [person(12)]])
    second = SceneTracker()
    assert ids(run(second, [[person(10)]])[0]) == [1]
    assert second.summaries()[0].frames_observed == 1


def test_ended_track_summaries_are_bounded():
    tracker = SceneTracker(tracker=PersonTracker(max_age_seconds=0.1), max_age_seconds=0.1, max_finished_summaries=3)
    for i in range(10):
        run(tracker, [[person(10 + 500 * (i % 2))], []], start=2 * i)
    assert len(tracker.summaries()) <= 4  # 3 finished + at most 1 live


def test_torso_history_skips_frames_without_a_torso():
    tracker = SceneTracker()
    upright = person(10)
    no_pose = person(11, keypoints=False)
    run(tracker, [[upright], [no_pose], [person(12)]])
    angles = tracker.recent_torso_angles(1, window_seconds=10)
    assert len(angles) == 2  # the no-pose frame is not counted either way
