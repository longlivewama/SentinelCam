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
from types import SimpleNamespace

import numpy as np

from app.services.detection.engine import DetectionEngine


class FakeFallPipeline:
    # Mirrors FallPipeline's interface: no tracker, pose not required.
    scene = None
    needs_pose = False

    def __init__(self, events):
        self._events = events

    def update(self, frame, people, now=None, frame_index=None):
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


def test_a_held_event_is_drawn_on_its_persons_current_box(monkeypatch):
    """Released up to 2 s after it fired, a held event's fire-time box would
    hang where the person used to be on the live view. While its attributed
    person is still tracked, their current box is drawn instead; anything
    else keeps the event's own box."""
    engine = DetectionEngine()
    published = []
    monkeypatch.setattr(
        "app.services.detection.engine.realtime_broadcaster.publish",
        lambda event_type, data, owner_user_id=None: published.append(data),
    )
    scene = SimpleNamespace(current=[
        SimpleNamespace(track_id=4, bbox=(300.0, 200.0, 420.0, 470.0), state="confirmed"),
    ])
    frame = np.zeros((480, 640, 3), dtype="uint8")
    fired_box = (100.0, 50.0, 200.0, 210.0)

    held = {"bbox": fired_box, "person_track_id": 4, "fired_at": 10.0, "confidence": 0.7}
    engine._broadcast_fall_bbox(1, frame, held, scene)
    assert published[-1]["bbox"] == [300.0, 200.0, 420.0, 470.0]

    immediate = {"bbox": fired_box, "person_track_id": 4, "confidence": 0.7}
    engine._broadcast_fall_bbox(1, frame, immediate, scene)
    assert published[-1]["bbox"] == list(fired_box)

    held_person_gone = dict(held, person_track_id=9)
    engine._broadcast_fall_bbox(1, frame, held_person_gone, scene)
    assert published[-1]["bbox"] == list(fired_box)
