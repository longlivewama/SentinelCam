"""
Tests for the trained fall detector's temporal gate
(fall_detection.ModelFallDetector) and the strategy selection in
fall_pipeline.py.

Deliberately model-free: these drive the gate with synthetic boxes rather
than real inference, because what they verify is the logic that decides
whether an alert is raised - the sustained-duration requirement, the
debounce, and per-subject tracking. Whether the checkpoint itself detects
falls is measured in ml/ (see ml/MODEL_CARD.md); duplicating that here
would make the unit suite depend on torch and a 5MB artifact.
"""
from collections import namedtuple

import pytest

from app.config import settings
from app.services.detection.fall_detection import ModelFallDetector
from app.services.detection import fall_pipeline as fp

Box = namedtuple("Box", ["bbox", "confidence"])

# A wide, low box - the shape the detector produces for someone on the floor.
FALLEN = Box(bbox=(100.0, 300.0, 300.0, 380.0), confidence=0.8)


def _feed(detector, detection, times):
    """Runs the gate over a list of timestamps, returning every event."""
    events = []
    for t in times:
        events.extend(detector.update([detection] if detection else [], now=t))
    return events


def test_a_single_frame_detection_does_not_fire():
    """The model is ~81% precise per image; one frame's opinion must not
    raise an alert on its own."""
    detector = ModelFallDetector(min_sustained_seconds=0.6)
    assert detector.update([FALLEN], now=0.0) == []


def test_detection_sustained_past_the_gate_fires_once():
    detector = ModelFallDetector(min_sustained_seconds=0.6)
    events = _feed(detector, FALLEN, [0.0, 0.2, 0.4, 0.6, 0.8])
    assert len(events) == 1
    assert events[0]["detector"] == "model"
    assert events[0]["sustained_seconds"] >= 0.6


def test_reported_confidence_is_the_peak_seen_not_the_last_frame():
    detector = ModelFallDetector(min_sustained_seconds=0.4)
    detector.update([Box(bbox=FALLEN.bbox, confidence=0.55)], now=0.0)
    detector.update([Box(bbox=FALLEN.bbox, confidence=0.95)], now=0.2)
    events = detector.update([Box(bbox=FALLEN.bbox, confidence=0.60)], now=0.5)
    assert events[0]["confidence"] == 0.95


def test_a_flickering_detection_never_accumulates_enough_time():
    """Present, gone, present, gone - each reappearance restarts the
    sustain timer, so intermittent noise cannot add up to an alert."""
    detector = ModelFallDetector(min_sustained_seconds=0.6)
    events = []
    for i in range(12):
        detection = [FALLEN] if i % 2 == 0 else []
        events.extend(detector.update(detection, now=i * 0.2))
    assert events == []


def test_debounce_prevents_re_firing_while_the_subject_stays_down():
    detector = ModelFallDetector(min_sustained_seconds=0.6, debounce_seconds=10.0)
    events = _feed(detector, FALLEN, [t * 0.2 for t in range(40)])  # 0 .. 7.8s
    assert len(events) == 1


def test_it_fires_again_after_the_debounce_window_expires():
    detector = ModelFallDetector(min_sustained_seconds=0.6, debounce_seconds=2.0)
    events = _feed(detector, FALLEN, [t * 0.2 for t in range(40)])
    assert len(events) > 1


def test_two_separate_subjects_are_tracked_independently():
    far_away = Box(bbox=(900.0, 300.0, 1100.0, 380.0), confidence=0.7)
    detector = ModelFallDetector(min_sustained_seconds=0.6)
    events = []
    for t in (0.0, 0.2, 0.4, 0.6, 0.8):
        events.extend(detector.update([FALLEN, far_away], now=t))
    assert len({e["track_id"] for e in events}) == 2


def test_a_subject_that_drifts_slightly_stays_one_track():
    """Small frame-to-frame movement must not split into new tracks, which
    would reset the sustain timer forever and suppress every alert."""
    detector = ModelFallDetector(min_sustained_seconds=0.6)
    events = []
    for i, t in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
        drift = i * 6.0
        moved = Box(bbox=(100 + drift, 300 + drift, 300 + drift, 380 + drift), confidence=0.8)
        events.extend(detector.update([moved], now=t))
    assert len(events) == 1


# --- strategy selection ---------------------------------------------------

def test_auto_uses_the_trained_model_when_it_loads(monkeypatch):
    monkeypatch.setattr(settings, "FALL_DETECTION_MODE", "auto")
    monkeypatch.setattr(fp.fall_object_detector.__class__, "is_available", property(lambda self: True))
    assert fp.resolve_mode() == fp.MODE_MODEL


def test_auto_falls_back_to_the_heuristic_when_the_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "FALL_DETECTION_MODE", "auto")
    monkeypatch.setattr(fp.fall_object_detector.__class__, "is_available", property(lambda self: False))
    assert fp.resolve_mode() == fp.MODE_HEURISTIC


def test_explicit_modes_are_honoured_regardless_of_model_availability(monkeypatch):
    monkeypatch.setattr(fp.fall_object_detector.__class__, "is_available", property(lambda self: True))
    monkeypatch.setattr(settings, "FALL_DETECTION_MODE", "heuristic")
    assert fp.resolve_mode() == fp.MODE_HEURISTIC


def test_an_unrecognised_mode_falls_back_to_auto_instead_of_crashing(monkeypatch):
    monkeypatch.setattr(settings, "FALL_DETECTION_MODE", "nonsense")
    monkeypatch.setattr(fp.fall_object_detector.__class__, "is_available", property(lambda self: False))
    assert fp.resolve_mode() == fp.MODE_HEURISTIC


@pytest.mark.parametrize("mode,expect_model,expect_heuristic", [
    (fp.MODE_MODEL, True, False),
    (fp.MODE_HEURISTIC, False, True),
    (fp.MODE_HYBRID, True, True),
])
def test_pipeline_wires_up_the_right_gates(mode, expect_model, expect_heuristic):
    pipeline = fp.FallPipeline(mode=mode)
    assert (pipeline._model_gate is not None) is expect_model
    assert (pipeline._heuristic is not None) is expect_heuristic


def test_pipeline_tags_events_with_the_detector_that_fired_them(monkeypatch):
    """The Event row records which strategy made the call, because the two
    compute confidence differently and are not comparable."""
    monkeypatch.setattr(fp.fall_object_detector, "detect", lambda frame: [FALLEN])

    pipeline = fp.FallPipeline(mode=fp.MODE_MODEL)
    events = []
    for t in (0.0, 0.2, 0.4, 0.6, 0.8):
        events.extend(pipeline.update(frame=None, people=[], now=t))

    assert len(events) == 1
    assert events[0]["detector"] == "model"


def test_pipeline_in_model_mode_never_touches_the_pose_heuristic(monkeypatch):
    """A fall reported in model mode must come from the model, not from
    the geometric proxy running underneath it."""
    monkeypatch.setattr(fp.fall_object_detector, "detect", lambda frame: [])

    pipeline = fp.FallPipeline(mode=fp.MODE_MODEL)
    assert pipeline._heuristic is None
    assert pipeline.update(frame=None, people=[], now=0.0) == []
