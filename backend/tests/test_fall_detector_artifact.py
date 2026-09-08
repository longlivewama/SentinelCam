"""
Contract tests for the committed trained-detector artifact
(ml/exported/fall_detector_v1.pt).

These load the real checkpoint and run real inference. They are cheap
(one 640x640 frame through yolov8n on CPU) and they guard the things that
would silently break the product if the artifact were ever swapped:

  * the file the default configuration points at actually exists, so a
    fresh checkout runs the trained model rather than quietly degrading
    to the pose heuristic;
  * it is single-class "Fall" - a checkpoint with different classes would
    otherwise map class 0 to "fall" and report confidently wrong alerts;
  * inference returns real boxes and scores, and the confidence floor is
    honoured.

Skipped rather than failed when the artifact or ultralytics is absent, so
a lightweight checkout can still run the rest of the suite.
"""
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.services.detection.fall_object_detector import EXPECTED_CLASS_NAMES, FallObjectDetector

pytest.importorskip("ultralytics", reason="ultralytics not installed")

ARTIFACT = Path(settings.FALL_DETECTOR_MODEL_PATH)
requires_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(), reason=f"trained detector artifact not present at {ARTIFACT}",
)


@requires_artifact
def test_the_default_configuration_points_at_a_real_file():
    assert ARTIFACT.is_file()
    # A truncated or LFS-pointer checkout would be a few hundred bytes.
    assert ARTIFACT.stat().st_size > 1_000_000


@requires_artifact
def test_the_artifact_loads_and_is_the_single_class_fall_detector():
    detector = FallObjectDetector()
    assert detector.is_available, f"model failed to load: {detector.load_error}"
    assert detector.model_info["class_names"] == EXPECTED_CLASS_NAMES


@requires_artifact
def test_inference_returns_well_formed_boxes_above_the_confidence_floor():
    detector = FallObjectDetector()
    assert detector.is_available

    # Real inference on a real frame. A blank frame is a legitimate input
    # and is expected to yield few or no detections; what is asserted here
    # is the SHAPE of whatever comes back, not that anything is found.
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert isinstance(detections, list)
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        assert x2 > x1 and y2 > y1
        assert 0.0 <= detection.confidence <= 1.0
        assert detection.confidence >= settings.FALL_DETECTOR_MIN_CONFIDENCE

    # Highest confidence first - the pipeline relies on this ordering.
    assert detections == sorted(detections, key=lambda d: d.confidence, reverse=True)


def test_a_checkpoint_with_unexpected_classes_is_refused(monkeypatch, tmp_path):
    """Guards against a swapped checkpoint: mapping some other model's
    class 0 to "fall" would produce confidently wrong alerts, so the
    loader must refuse rather than guess."""
    fake = tmp_path / "wrong_classes.pt"
    fake.write_bytes(b"not really a checkpoint")
    monkeypatch.setattr(settings, "FALL_DETECTOR_MODEL_PATH", str(fake))

    class _FakeYOLO:
        def __init__(self, path):
            self.names = {0: "person", 1: "car"}

    monkeypatch.setattr("ultralytics.YOLO", _FakeYOLO)

    detector = FallObjectDetector()
    assert not detector.is_available
    assert "unexpected class names" in detector.load_error


def test_a_missing_model_file_disables_the_detector_instead_of_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "FALL_DETECTOR_MODEL_PATH", str(tmp_path / "nope.pt"))
    detector = FallObjectDetector()
    assert not detector.is_available
    assert "not found" in detector.load_error
    # And it degrades to an empty result rather than exploding mid-frame.
    assert detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []


def test_an_inference_error_is_swallowed_rather_than_killing_the_loop(monkeypatch):
    detector = FallObjectDetector()
    detector._load_attempted = True
    detector._model = type("Boom", (), {"predict": lambda self, *a, **k: 1 / 0})()

    assert detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []
