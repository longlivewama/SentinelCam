"""
Turning sparse observations into a box on every clip frame, and drawing it.

Two properties matter more than anything cosmetic:

  * the box MOVES. Inference runs on every stride-th frame, so a clip has
    a real observation roughly six times a second; the frames in between
    are interpolated between two real sightings. A single fixed box for
    the whole clip would be the wrong picture of a fall.
  * the box is never INVENTED. Outside the observed span, or across a gap
    too long to bridge honestly, the renderer draws nothing at all.
"""
import numpy as np
import pytest

from app.services.detection.annotated_clip_renderer import (
    COLOR_MATCHED,
    COLOR_UNMATCHED,
    BoxTimeline,
    draw_annotation,
    is_renderable,
    iter_annotated_frames,
    label_for,
    max_gap_frames_for,
)

WIDTH, HEIGHT = 320, 240


def blank(value=30):
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def colored_pixels(frame, color, tolerance=40):
    diff = np.abs(frame.astype(int) - np.array(color, dtype=int))
    return int((diff.max(axis=2) <= tolerance).sum())


# --- the timeline ---------------------------------------------------------

def test_an_observed_frame_returns_exactly_what_was_observed():
    timeline = BoxTimeline([(0, (10, 20, 60, 140)), (5, (30, 20, 80, 140))], max_gap_frames=22)
    assert timeline.box_at(5) == (30.0, 20.0, 80.0, 140.0)


def test_frames_between_observations_are_interpolated():
    timeline = BoxTimeline([(0, (0, 0, 100, 100)), (10, (100, 0, 200, 100))], max_gap_frames=22)

    midpoint = timeline.box_at(5)
    assert midpoint == pytest.approx((50.0, 0.0, 150.0, 100.0))

    quarter = timeline.box_at(2)
    assert quarter[0] == pytest.approx(20.0)


def test_the_box_moves_across_the_clip_rather_than_sitting_still():
    observations = [(frame, (frame, 0, frame + 100, 100)) for frame in range(0, 60, 5)]
    timeline = BoxTimeline(observations, max_gap_frames=22)

    positions = [timeline.box_at(f)[0] for f in range(0, 56)]
    assert positions == sorted(positions)
    assert positions[-1] - positions[0] > 40, "the box barely moved"
    assert len(set(positions)) > 20, "the box only updated on observed frames"


def test_nothing_is_drawn_before_the_first_or_after_the_last_observation():
    timeline = BoxTimeline([(10, (0, 0, 50, 50)), (20, (0, 0, 50, 50))], max_gap_frames=22)
    assert timeline.box_at(9) is None
    assert timeline.box_at(21) is None
    assert timeline.box_at(10) is not None


def test_a_gap_too_long_to_bridge_is_left_empty():
    """Tracking lost the subject for a second; drawing a straight line
    through wherever they actually went would be fiction."""
    timeline = BoxTimeline([(0, (0, 0, 50, 50)), (100, (300, 300, 350, 350))], max_gap_frames=22)

    assert timeline.box_at(50) is None
    assert timeline.box_at(0) is not None
    assert timeline.box_at(100) is not None


def test_an_empty_timeline_yields_no_boxes_at_all():
    timeline = BoxTimeline([], max_gap_frames=22)
    assert len(timeline) == 0
    assert timeline.span is None
    assert timeline.box_at(0) is None


def test_observations_are_accepted_out_of_order():
    timeline = BoxTimeline([(10, (5, 5, 15, 15)), (0, (0, 0, 10, 10))], max_gap_frames=22)
    assert timeline.span == (0, 10)
    assert timeline.box_at(5) == pytest.approx((2.5, 2.5, 12.5, 12.5))


def test_the_interpolation_budget_covers_at_least_one_stride():
    assert max_gap_frames_for(fps=30, stride=5) >= 5
    assert max_gap_frames_for(fps=30, stride=60) >= 60
    assert max_gap_frames_for(fps=0, stride=1) > 0


# --- drawing --------------------------------------------------------------

def test_a_matched_fall_draws_a_red_box_with_the_track_id_and_confidence():
    frame = blank()
    assert draw_annotation(frame, (80, 60, 220, 180), label_for(3, 0.74)) is True
    assert colored_pixels(frame, COLOR_MATCHED) > 300


def test_an_unmatched_fall_is_drawn_in_a_visibly_different_style():
    """Amber and dashed, so an operator can tell "this person fell" from
    "the detector fired here and no person could be named"."""
    matched, unmatched = blank(), blank()
    draw_annotation(matched, (80, 60, 220, 180), label_for(3, 0.74))
    draw_annotation(unmatched, (80, 60, 220, 180), label_for(None, 0.74))

    assert colored_pixels(unmatched, COLOR_UNMATCHED) > 300
    assert colored_pixels(unmatched, COLOR_MATCHED) == 0
    assert colored_pixels(matched, COLOR_UNMATCHED) == 0


def test_the_label_says_the_track_id_and_the_events_own_confidence():
    label = label_for(3, 0.742)
    assert label.headline == "FALL DETECTED"
    assert label.detail == "ID 3 | 0.74"
    assert label.matched is True


def test_an_unnamed_fall_never_shows_a_fabricated_id():
    label = label_for(None, 0.7)
    assert "ID" not in label.detail
    assert label.detail == "UNMATCHED | 0.70"
    assert label.matched is False


def test_a_box_at_the_very_top_still_gets_its_headline_inside_the_frame():
    frame = blank()
    assert draw_annotation(frame, (10, 0, 200, 120), label_for(1, 0.9)) is True
    # The headline band is tucked inside the box rather than off-frame.
    assert colored_pixels(frame[:40], COLOR_MATCHED) > 100


def test_a_box_at_the_very_bottom_still_gets_its_detail_inside_the_frame():
    frame = blank()
    assert draw_annotation(frame, (10, HEIGHT - 80, 200, HEIGHT - 1), label_for(1, 0.9)) is True
    assert colored_pixels(frame[HEIGHT - 90:], COLOR_MATCHED) > 100


def test_a_box_outside_the_frame_is_declined_rather_than_clipped_into_nonsense():
    frame = blank()
    assert draw_annotation(frame, (WIDTH + 50, HEIGHT + 50, WIDTH + 90, HEIGHT + 90), label_for(1, 0.5)) is False
    assert colored_pixels(frame, COLOR_MATCHED) == 0


def test_a_degenerate_box_is_declined():
    frame = blank()
    assert draw_annotation(frame, (100, 100, 100, 100), label_for(1, 0.5)) is False


# --- corrupt input --------------------------------------------------------

def test_corrupt_frames_are_recognised_and_left_alone():
    assert is_renderable(blank()) is True
    assert is_renderable(None) is False
    assert is_renderable(np.zeros((0, 0, 3), dtype=np.uint8)) is False
    assert is_renderable(np.zeros((10, 10), dtype=np.uint8)) is False       # no channels
    assert is_renderable(np.zeros((10, 10, 4), dtype=np.uint8)) is False    # not BGR


def test_drawing_on_a_corrupt_frame_reports_failure_instead_of_raising():
    assert draw_annotation(None, (0, 0, 10, 10), label_for(1, 0.5)) is False
    assert draw_annotation(np.zeros((4, 4), dtype=np.uint8), (0, 0, 3, 3), label_for(1, 0.5)) is False


def test_one_corrupt_frame_does_not_cost_the_clip_its_other_frames():
    frames = [blank(), None, blank()]
    timeline = BoxTimeline([(0, (40, 40, 200, 180)), (2, (40, 40, 200, 180))], max_gap_frames=22)

    rendered = list(iter_annotated_frames(frames, 0, timeline, label_for(2, 0.8)))

    assert rendered[1] is None
    assert colored_pixels(rendered[0], COLOR_MATCHED) > 300
    assert colored_pixels(rendered[2], COLOR_MATCHED) > 300


# --- the clip pass --------------------------------------------------------

def test_only_frames_inside_the_tracked_span_are_annotated():
    frames = [blank() for _ in range(10)]
    timeline = BoxTimeline([(3, (40, 40, 200, 180)), (6, (60, 40, 220, 180))], max_gap_frames=22)

    rendered = list(iter_annotated_frames(frames, 0, timeline, label_for(1, 0.8)))

    annotated = [i for i, f in enumerate(rendered) if colored_pixels(f, COLOR_MATCHED) > 300]
    assert annotated == [3, 4, 5, 6]


def test_the_callers_frames_are_never_modified_in_place():
    """Clip frames are shared with the analyser's ring buffer and with any
    other clip still collecting, so drawing in place would corrupt them."""
    frames = [blank() for _ in range(4)]
    originals = [f.copy() for f in frames]
    timeline = BoxTimeline([(0, (40, 40, 200, 180)), (3, (40, 40, 200, 180))], max_gap_frames=22)

    list(iter_annotated_frames(frames, 0, timeline, label_for(1, 0.8)))

    for frame, original in zip(frames, originals):
        assert np.array_equal(frame, original)


def test_frame_indexes_are_offset_by_the_clips_position_in_the_video():
    """A clip starts part-way into the video; its first frame is not
    frame 0 of the timeline."""
    frames = [blank() for _ in range(6)]
    timeline = BoxTimeline([(511, (40, 40, 200, 180)), (514, (40, 40, 200, 180))], max_gap_frames=22)

    rendered = list(iter_annotated_frames(frames, 511, timeline, label_for(1, 0.8)))
    annotated = [i for i, f in enumerate(rendered) if colored_pixels(f, COLOR_MATCHED) > 300]

    assert annotated == [0, 1, 2, 3]
