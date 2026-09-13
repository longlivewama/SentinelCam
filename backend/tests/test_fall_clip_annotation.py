"""
The annotation coordinator: tracking + association + the plan a clip is
drawn from.

These tests drive `FallClipAnnotator` exactly the way
`services/video_analysis.py` does - `observe()` on every processed frame,
`plan_for_event()` when the fall gate fires - with synthetic boxes
standing in for the two models, so the scenarios that matter (three
people and one faller, two people falling one after the other, tracking
dropping out mid-clip) are exact rather than approximate.

The invariant underneath all of them: the annotation layer is downstream
of the fall decision. It never creates, suppresses, retimes or re-scores
a fall - it only decides where to draw, and is allowed to decline.
"""
from collections import namedtuple

import pytest

from app.services.detection.fall_annotation import FallClipAnnotator

Detected = namedtuple("Detected", ["bbox"])

FPS = 30.0
STRIDE = 5


def annotator(**kwargs):
    params = dict(fps=FPS, stride=STRIDE, pre_event_seconds=3, post_event_seconds=3)
    params.update(kwargs)
    return FallClipAnnotator(**params)


def person(x, y, w=200, h=400):
    return Detected(bbox=(float(x), float(y), float(x + w), float(y + h)))


def fall_event(bbox, confidence=0.82, sustained=0.6):
    """The dict shape `fall_detection.py`'s gates return."""
    return {
        "track_id": 0,
        "confidence": confidence,
        "sustained_seconds": sustained,
        "detector": "model",
        "bbox": tuple(float(v) for v in bbox),
    }


def run(ann, script):
    """`script` is [(frame_index, [people], [fall boxes]), ...]."""
    for frame_index, people, falls in script:
        ann.observe(frame_index, frame_index / FPS, people, [Detected(bbox=b) for b in falls])


# --- one person, one fall -------------------------------------------------

def test_one_person_one_fall_is_annotated_with_that_persons_track():
    ann = annotator()
    subject = lambda step: person(100 + step * 4, 100)  # noqa: E731
    falling = lambda step: (110.0 + step * 4, 300.0, 290.0 + step * 4, 490.0)  # noqa: E731

    script = [(f, [subject(i)], [falling(i)]) for i, f in enumerate(range(0, 60, STRIDE))]
    run(ann, script)

    plan = ann.plan_for_event(fall_event(falling(11)), 55, 55 / FPS)

    assert plan is not None
    assert plan.matched is True
    assert plan.track_id == 1
    assert plan.label.detail == "ID 1 | 0.82"
    assert plan.confidence == pytest.approx(0.82)


def test_the_plans_boxes_follow_the_person_rather_than_standing_still():
    ann = annotator()
    for step, frame_index in enumerate(range(0, 60, STRIDE)):
        x = 100 + step * 20
        ann.observe(frame_index, frame_index / FPS, [person(x, 100)],
                    [Detected(bbox=(float(x + 10), 300.0, float(x + 190), 490.0))])

    plan = ann.plan_for_event(fall_event((330.0, 300.0, 510.0, 490.0)), 55, 55 / FPS)

    xs = [bbox[0] for _frame, bbox in plan.observations]
    assert len(set(xs)) > 1, "every frame got the same box"
    assert xs[-1] - xs[0] >= 200


# --- three people, one falls ----------------------------------------------

def build_three_person_scene(ann, faller_index, frames=range(0, 60, STRIDE)):
    """Three people in their own lanes; one of them has the fall box on
    them for the whole window."""
    lanes = [40, 300, 560]
    fall_boxes = []
    for step, frame_index in enumerate(frames):
        people = [person(x, 100) for x in lanes]
        x = lanes[faller_index]
        fall_box = (float(x + 10), 300.0, float(x + 190), 490.0)
        fall_boxes.append(fall_box)
        ann.observe(frame_index, frame_index / FPS, people, [Detected(bbox=fall_box)])
    return list(frames), fall_boxes


def test_three_people_one_falls_annotates_only_the_one_who_fell():
    ann = annotator()
    frames, fall_boxes = build_three_person_scene(ann, faller_index=2)

    plan = ann.plan_for_event(fall_event(fall_boxes[-1]), frames[-1], frames[-1] / FPS)

    assert plan is not None
    assert plan.matched is True
    # Lanes are created left-to-right on the first frame, so the third
    # lane is track 3.
    assert plan.track_id == 3

    # Every box the clip will draw sits on the third lane, never on the
    # other two.
    for _frame, (x1, _y1, x2, _y2) in plan.observations:
        assert x1 >= 500, "the annotation drifted onto another person"
        assert x2 <= 780


@pytest.mark.parametrize("faller_index, expected_track", [(0, 1), (1, 2), (2, 3)])
def test_whichever_of_the_three_falls_is_the_one_named(faller_index, expected_track):
    ann = annotator()
    frames, fall_boxes = build_three_person_scene(ann, faller_index=faller_index)

    plan = ann.plan_for_event(fall_event(fall_boxes[-1]), frames[-1], frames[-1] / FPS)

    assert plan is not None
    assert plan.track_id == expected_track


def test_the_first_person_in_the_list_is_not_the_default_answer():
    """The cheap wrong implementation - annotate people[0] - would pass
    the faller_index=0 case above and fail here."""
    ann = annotator()
    frames, fall_boxes = build_three_person_scene(ann, faller_index=1)

    plan = ann.plan_for_event(fall_event(fall_boxes[-1]), frames[-1], frames[-1] / FPS)
    assert plan.track_id != 1


# --- two falls ------------------------------------------------------------

def test_two_people_falling_in_turn_get_one_plan_each_on_their_own_track():
    ann = annotator()
    left, right = 40, 560

    # Phase 1: the left-hand person goes down.
    left_fall = (50.0, 300.0, 230.0, 490.0)
    for frame_index in range(0, 60, STRIDE):
        ann.observe(
            frame_index, frame_index / FPS,
            [person(left, 100), person(right, 100)],
            [Detected(bbox=left_fall)],
        )
    first = ann.plan_for_event(fall_event(left_fall), 55, 55 / FPS)

    # Phase 2, later in the same video: the right-hand person goes down.
    right_fall = (570.0, 300.0, 750.0, 490.0)
    for frame_index in range(60, 130, STRIDE):
        ann.observe(
            frame_index, frame_index / FPS,
            [person(left, 100), person(right, 100)],
            [Detected(bbox=right_fall)],
        )
    second = ann.plan_for_event(fall_event(right_fall, confidence=0.66), 125, 125 / FPS)

    assert first is not None and second is not None
    assert first.track_id != second.track_id
    assert first.label.detail == "ID 1 | 0.82"
    assert second.label.detail == f"ID {second.track_id} | 0.66"
    assert second.observations[-1][1][0] >= 500


def test_two_simultaneous_falls_are_each_matched_to_their_own_evidence():
    ann = annotator()
    left_fall = (50.0, 300.0, 230.0, 490.0)
    right_fall = (570.0, 300.0, 750.0, 490.0)

    for frame_index in range(0, 60, STRIDE):
        ann.observe(
            frame_index, frame_index / FPS,
            [person(40, 100), person(560, 100)],
            [Detected(bbox=left_fall), Detected(bbox=right_fall)],
        )

    plan_left = ann.plan_for_event(fall_event(left_fall), 55, 55 / FPS)
    plan_right = ann.plan_for_event(fall_event(right_fall), 55, 55 / FPS)

    assert plan_left is not None and plan_right is not None
    assert plan_left.track_id != plan_right.track_id
    assert plan_left.observations[-1][1][0] < 300
    assert plan_right.observations[-1][1][0] >= 500


# --- no fall, and falls that cannot be attributed -------------------------

def test_a_video_with_no_fall_produces_no_plans():
    ann = annotator()
    for frame_index in range(0, 90, STRIDE):
        ann.observe(frame_index, frame_index / FPS, [person(100, 100)], [])

    assert ann._open_plans == []


def test_a_fall_on_no_person_is_annotated_as_the_detectors_own_region():
    """The real-video case: the model fires on a dark car door with a
    bystander standing beside it. Naming the bystander would be a
    confident false statement, so the clip gets the detector's own box
    and no track id."""
    ann = annotator()
    bystander = person(345, 337, w=97, h=142)
    car_door = (251.0, 350.0, 371.0, 507.0)

    for frame_index in range(0, 60, STRIDE):
        ann.observe(frame_index, frame_index / FPS, [bystander], [Detected(bbox=car_door)])

    plan = ann.plan_for_event(fall_event(car_door, confidence=0.7), 55, 55 / FPS)

    assert plan is not None
    assert plan.matched is False
    assert plan.track_id is None
    assert plan.label.detail == "UNMATCHED | 0.70"
    assert plan.observations, "the detector's own region should still be drawn"


def test_an_ambiguous_fall_is_not_pinned_on_either_candidate():
    ann = annotator()
    left = person(100, 100)
    right = person(300, 100)
    straddling = (250.0, 100.0, 350.0, 500.0)

    for frame_index in range(0, 60, STRIDE):
        ann.observe(frame_index, frame_index / FPS, [left, right], [Detected(bbox=straddling)])

    plan = ann.plan_for_event(fall_event(straddling), 55, 55 / FPS)

    assert plan is not None
    assert plan.track_id is None, "an even split between two people must not name one of them"


def test_an_event_with_no_evidence_box_yields_no_plan_at_all():
    """Nothing to anchor an annotation to - the clip is written exactly as
    it was before this feature existed."""
    ann = annotator()
    ann.observe(0, 0.0, [person(100, 100)], [])

    assert ann.plan_for_event({"confidence": 0.9}, 0, 0.0) is None


def test_a_fall_before_anything_has_been_observed_yields_no_plan():
    assert annotator().plan_for_event(fall_event((0, 0, 10, 10)), 0, 0.0) is None


# --- the clip window ------------------------------------------------------

def test_a_plan_carries_the_pre_event_boxes_it_was_seeded_with():
    """A clip starts three seconds BEFORE the event, so the plan needs
    boxes from before it existed."""
    ann = annotator()
    for step, frame_index in enumerate(range(0, 200, STRIDE)):
        subject = person(100 + step, 100)
        ann.observe(frame_index, frame_index / FPS, [subject], [Detected(bbox=(110.0 + step, 300.0, 290.0 + step, 490.0))])

    plan = ann.plan_for_event(fall_event((110.0 + 39, 300.0, 290.0 + 39, 490.0)), 195, 195 / FPS)

    first_frame = plan.observations[0][0]
    assert first_frame <= 195 - 3 * FPS, "the pre-event window is missing from the plan"


def test_a_plan_keeps_collecting_boxes_until_it_is_released():
    """The clip is written seconds after the event, and the tracker prunes
    aggressively in between; without the subscription the second half of
    every clip would lose its box."""
    ann = annotator()
    for step, frame_index in enumerate(range(0, 60, STRIDE)):
        ann.observe(frame_index, frame_index / FPS, [person(100 + step * 4, 100)],
                    [Detected(bbox=(110.0 + step * 4, 300.0, 290.0 + step * 4, 490.0))])

    plan = ann.plan_for_event(fall_event((154.0, 300.0, 334.0, 490.0)), 55, 55 / FPS)
    observations_at_firing = len(plan.observations)

    for step, frame_index in enumerate(range(60, 150, STRIDE), start=12):
        ann.observe(frame_index, frame_index / FPS, [person(100 + step * 4, 100)], [])

    assert len(plan.observations) > observations_at_firing
    assert plan.observations[-1][0] >= 145


def test_a_released_plan_stops_collecting():
    ann = annotator()
    for step, frame_index in enumerate(range(0, 60, STRIDE)):
        ann.observe(frame_index, frame_index / FPS, [person(100 + step * 4, 100)],
                    [Detected(bbox=(110.0 + step * 4, 300.0, 290.0 + step * 4, 490.0))])
    plan = ann.plan_for_event(fall_event((154.0, 300.0, 334.0, 490.0)), 55, 55 / FPS)

    ann.release(plan)
    frozen = len(plan.observations)

    for frame_index in range(60, 120, STRIDE):
        ann.observe(frame_index, frame_index / FPS, [person(300, 100)], [])

    assert plan.closed is True
    assert len(plan.observations) == frozen


def test_releasing_twice_and_releasing_nothing_are_both_harmless():
    ann = annotator()
    ann.release(None)
    ann.observe(0, 0.0, [person(100, 100)], [Detected(bbox=(110.0, 300.0, 290.0, 490.0))])
    plan = ann.plan_for_event(fall_event((110.0, 300.0, 290.0, 490.0)), 0, 0.0)
    ann.release(plan)
    ann.release(plan)


# --- tracking loss --------------------------------------------------------

def test_a_subject_lost_mid_clip_leaves_a_gap_rather_than_a_frozen_box():
    ann = annotator()
    for step, frame_index in enumerate(range(0, 40, STRIDE)):
        ann.observe(frame_index, frame_index / FPS, [person(100 + step * 4, 100)],
                    [Detected(bbox=(110.0 + step * 4, 300.0, 290.0 + step * 4, 490.0))])
    plan = ann.plan_for_event(fall_event((138.0, 300.0, 318.0, 490.0)), 35, 35 / FPS)

    # The pose model loses the subject entirely for two seconds.
    for frame_index in range(40, 100, STRIDE):
        ann.observe(frame_index, frame_index / FPS, [], [])

    timeline = plan.timeline()
    assert timeline.box_at(35) is not None
    assert timeline.box_at(70) is None, "a box was drawn for a frame nobody was seen on"
    assert timeline.box_at(95) is None


def test_tracking_loss_does_not_raise_or_discard_the_plan():
    ann = annotator()
    ann.observe(0, 0.0, [person(100, 100)], [Detected(bbox=(110.0, 300.0, 290.0, 490.0))])
    plan = ann.plan_for_event(fall_event((110.0, 300.0, 290.0, 490.0)), 0, 0.0)

    for frame_index in range(5, 300, STRIDE):
        ann.observe(frame_index, frame_index / FPS, [], [])

    assert plan is not None
    assert plan.observations, "the boxes already collected must survive the loss"


# --- passing detections through -------------------------------------------

def test_duplicate_person_detections_do_not_split_the_subject_mid_clip():
    """One person reported twice on a single frame used to open a second
    track that then took over the clip, changing the id half way
    through."""
    ann = annotator()
    for step, frame_index in enumerate(range(0, 100, STRIDE)):
        x = 100 + step * 4
        people = [person(x, 100)]
        if step == 10:
            people.append(person(x + 3, 103, w=194, h=394))
        ann.observe(frame_index, frame_index / FPS, people,
                    [Detected(bbox=(float(x + 10), 300.0, float(x + 190), 490.0))])

    plan = ann.plan_for_event(fall_event((float(100 + 19 * 4 + 10), 300.0, float(100 + 19 * 4 + 190), 490.0)), 95, 95 / FPS)

    assert plan is not None
    assert plan.track_id == 1


def test_detections_without_a_box_are_skipped_rather_than_crashing():
    ann = annotator()
    Broken = namedtuple("Broken", ["bbox"])
    ann.observe(0, 0.0, [Broken(bbox=None), person(100, 100)], [])
    assert len(ann._person_boxes_now) == 1
