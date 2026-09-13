"""
The short-term person tracker behind the fall-clip annotation.

Everything here is pure geometry over plain tuples - no model, no torch,
no video - because the tracker's whole job is frame-to-frame association
and that is exactly what has to be pinned down: the same person keeps one
id, two people keep two, and an id is never quietly handed to somebody
else.
"""
import pytest

from app.services.detection.person_tracker import (
    DUPLICATE_CONTAINMENT,
    MAX_AGE_SECONDS,
    PersonTracker,
    containment,
    iou,
    suppress_duplicate_boxes,
)


def box(x, y, w=60, h=140):
    return (float(x), float(y), float(x + w), float(y + h))


def ids(tracks):
    return [t.track_id for t in tracks]


# --- geometry -------------------------------------------------------------

def test_iou_and_containment_disagree_when_one_box_is_inside_another():
    """The case the association logic is built around: a fall box that is
    a sub-region of a person box scores modestly on IoU and almost
    perfectly on containment."""
    person = (100.0, 100.0, 300.0, 500.0)
    lower_body = (100.0, 300.0, 300.0, 500.0)

    assert iou(lower_body, person) == pytest.approx(0.5, abs=0.01)
    assert containment(lower_body, person) == pytest.approx(1.0, abs=0.01)


def test_disjoint_boxes_score_zero_on_both_measures():
    assert iou(box(0, 0), box(500, 500)) == 0.0
    assert containment(box(0, 0), box(500, 500)) == 0.0


# --- identity over time ---------------------------------------------------

def test_one_person_walking_keeps_one_id():
    tracker = PersonTracker()
    seen = []
    for step in range(12):
        tracks = tracker.update([box(100 + step * 8, 200)], now=step * 0.2, frame_index=step * 5)
        seen.extend(ids(tracks))

    assert set(seen) == {1}, f"the same person was split across ids {sorted(set(seen))}"


def test_three_people_get_three_stable_ids():
    tracker = PersonTracker()
    lanes = [100, 300, 500]

    first = ids(tracker.update([box(x, 200) for x in lanes], now=0.0, frame_index=0))
    assert sorted(first) == [1, 2, 3]

    for step in range(1, 10):
        tracks = tracker.update(
            [box(x + step * 5, 200) for x in lanes], now=step * 0.2, frame_index=step * 5,
        )
        assert ids(tracks) == first, "a lane changed identity between frames"


def test_ids_are_not_reused_after_a_track_is_dropped():
    """A recycled id would silently relabel a new person as the one a
    clip already named."""
    tracker = PersonTracker(max_age_seconds=0.5)
    tracker.update([box(100, 200)], now=0.0, frame_index=0)

    # Long enough with nothing in frame that the first track ages out.
    tracker.update([], now=5.0, frame_index=150)
    later = tracker.update([box(100, 200)], now=5.2, frame_index=156)

    assert ids(later) == [2]
    assert tracker.get(1) is None


def test_a_track_survives_a_short_detection_gap():
    """The pose model loses subjects for a frame or two in dark footage;
    that must not restart the id."""
    tracker = PersonTracker()
    tracker.update([box(100, 200)], now=0.0, frame_index=0)

    for step in range(1, 5):  # ~0.8s of nothing detected
        tracker.update([], now=step * 0.2, frame_index=step * 5)

    resumed = tracker.update([box(120, 205)], now=1.0, frame_index=25)
    assert ids(resumed) == [1]


def test_a_track_is_dropped_once_it_is_older_than_the_age_limit():
    tracker = PersonTracker()
    tracker.update([box(100, 200)], now=0.0, frame_index=0)
    tracker.update([], now=MAX_AGE_SECONDS + 0.1, frame_index=60)

    assert tracker.get(1) is None
    assert tracker.tracks == []


def test_two_people_crossing_do_not_swap_ids_while_they_stay_apart():
    tracker = PersonTracker()
    left, right = 100, 500
    tracker.update([box(left, 200), box(right, 200)], now=0.0, frame_index=0)

    for step in range(1, 8):
        # They approach each other but never overlap.
        a, b = left + step * 20, right - step * 20
        tracks = tracker.update([box(a, 200), box(b, 200)], now=step * 0.2, frame_index=step * 5)
        assert ids(tracks) == [1, 2]


def test_history_records_only_the_frames_a_subject_was_seen_on():
    """Gaps must stay gaps - the renderer relies on this to avoid drawing
    a box where nothing was observed."""
    tracker = PersonTracker()
    tracker.update([box(100, 200)], now=0.0, frame_index=0)
    tracker.update([], now=0.2, frame_index=5)
    tracker.update([box(110, 200)], now=0.4, frame_index=10)

    assert [frame for frame, _ in tracker.get(1).history] == [0, 10]


def test_the_tracker_is_deterministic_regardless_of_detection_order():
    boxes = [box(100, 200), box(400, 210), box(250, 600)]

    forward = PersonTracker()
    forward.update(boxes, now=0.0, frame_index=0)
    reverse = PersonTracker()
    reverse.update(list(reversed(boxes)), now=0.0, frame_index=0)

    # Same boxes on the next frame: both trackers must agree about which
    # box belongs to which of their own tracks.
    moved = [box(105, 200), box(405, 210), box(255, 600)]
    forward_map = {t.track_id: t.bbox for t in forward.update(moved, now=0.2, frame_index=5)}
    reverse_map = {t.track_id: t.bbox for t in reverse.update(moved, now=0.2, frame_index=5)}

    assert sorted(forward_map.values()) == sorted(reverse_map.values())
    assert len(forward_map) == len(reverse_map) == 3


# --- duplicate suppression ------------------------------------------------

def test_a_near_duplicate_detection_is_collapsed_into_one_subject():
    """One person reported twice - a full-body box and the same body plus
    an outstretched arm - is the artifact that used to open a second track
    and steal the subject on the next frame."""
    full = (175.0, 239.0, 497.0, 787.0)
    arm_clipped = (178.0, 242.0, 381.0, 793.0)

    kept = suppress_duplicate_boxes([arm_clipped, full])

    assert kept == [full], "the larger of the two duplicates should survive"
    assert containment(arm_clipped, full) >= DUPLICATE_CONTAINMENT


def test_two_genuinely_separate_people_both_survive_suppression():
    people = [box(100, 200), box(400, 200)]
    assert sorted(suppress_duplicate_boxes(people)) == sorted(people)


def test_partially_overlapping_people_both_survive_suppression():
    """Standing side by side with some overlap is not a duplicate."""
    a = (100.0, 200.0, 300.0, 600.0)
    b = (250.0, 200.0, 450.0, 600.0)
    assert len(suppress_duplicate_boxes([a, b])) == 2


def test_suppression_keeps_the_same_subject_on_one_id_across_frames():
    tracker = PersonTracker()
    for step in range(6):
        raw = [box(100 + step * 6, 200)]
        if step == 3:
            # The duplicate the detector emits on one frame.
            raw.append(box(102 + step * 6, 203, w=50, h=130))
        tracks = tracker.update(suppress_duplicate_boxes(raw), now=step * 0.2, frame_index=step * 5)
        assert ids(tracks) == [1]
