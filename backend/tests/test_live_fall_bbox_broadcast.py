"""
The live-view bounding box (see detection/engine.py's `_broadcast_fall_bbox`)
must be exactly the box the fall gate itself fired on - never a fabricated
or re-derived one - and must never fire for an event that carries no box.

These drive `DetectionEngine._process_frame` directly with fake pipeline
collaborators so no YOLO model, camera, or database is needed: the method
only touches `self.extract_people`/`self.extract_objects` (stubbed here)
and the module-level `recording_engine`/`realtime_broadcaster` (also
stubbed), so this is a pure unit test of the wiring between a fall event
and what gets published.
"""
import numpy as np

from app.services.detection.engine import DetectionEngine


class FakeFallPipeline:
    def __init__(self, events):
        self._events = events

    def update(self, frame, people):
        return self._events


class FakeViolenceDetector:
    def update(self, people):
        return []


class FakeCrowdDetector:
    def update(self, person_count, threshold):
        return None


def _run(monkeypatch, fall_events, frame_shape=(480, 640, 3)):
    engine = DetectionEngine()
    monkeypatch.setattr(engine, "extract_people", lambda frame: [])
    monkeypatch.setattr(engine, "extract_objects", lambda frame: ([], []))
    monkeypatch.setattr("app.services.detection.engine.recording_engine.trigger_event", lambda *a, **k: None)

    published = []
    monkeypatch.setattr(
        "app.services.detection.engine.realtime_broadcaster.publish",
        lambda event_type, data, owner_user_id=None: published.append((event_type, data)),
    )

    frame = np.zeros(frame_shape, dtype="uint8")
    engine._process_frame(
        camera_id=1,
        frame=frame,
        fall_pipeline=FakeFallPipeline(fall_events),
        violence_detector=FakeViolenceDetector(),
        crowd_detector=FakeCrowdDetector(),
        crowd_threshold=20,
    )
    return published


def test_the_broadcast_bbox_is_exactly_the_events_own_box(monkeypatch):
    fall_event = {
        "confidence": 0.87,
        "detector": "model",
        "track_id": 3,
        "bbox": (10.0, 20.0, 110.0, 220.0),
    }
    published = _run(monkeypatch, [fall_event])

    bbox_events = [data for event_type, data in published if event_type == "fall.bbox"]
    assert len(bbox_events) == 1
    payload = bbox_events[0]
    assert payload["camera_id"] == 1
    assert payload["bbox"] == [10.0, 20.0, 110.0, 220.0]
    assert payload["frame_width"] == 640
    assert payload["frame_height"] == 480
    assert payload["confidence"] == 0.87
    assert payload["track_id"] == 3
    assert payload["detector"] == "model"


def test_no_bbox_is_invented_when_the_event_carries_none(monkeypatch):
    # A fall event missing its own bbox key (should not happen for either
    # gate today, but must fail closed rather than fabricate a box).
    published = _run(monkeypatch, [{"confidence": 0.5, "detector": "heuristic"}])
    assert "fall.bbox" not in [event_type for event_type, _ in published]


def test_no_broadcast_at_all_when_no_fall_fires(monkeypatch):
    published = _run(monkeypatch, [])
    assert published == []
