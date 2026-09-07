from pathlib import Path

import pytest

from app.services.detection.engine import PersonDetection
from app.services.detection.fall_classifier import FallClassifier

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "ml" / "exported" / "fall_classifier_v1.onnx"


def _standing_person() -> PersonDetection:
    kps = [(0, 0, 0.0)] * 17
    kps[5] = (300, 150, 0.9)
    kps[6] = (340, 150, 0.9)
    kps[11] = (300, 300, 0.9)
    kps[12] = (340, 300, 0.9)
    kps[13] = (300, 380, 0.9)
    kps[14] = (340, 380, 0.9)
    kps[15] = (300, 450, 0.9)
    kps[16] = (340, 450, 0.9)
    return PersonDetection(bbox=(280, 100, 360, 460), keypoints=kps)


def _fallen_person() -> PersonDetection:
    kps = [(0, 0, 0.0)] * 17
    kps[5] = (200, 300, 0.9)
    kps[6] = (240, 300, 0.9)
    kps[11] = (300, 305, 0.9)
    kps[12] = (340, 305, 0.9)
    kps[13] = (400, 300, 0.9)
    kps[14] = (440, 300, 0.9)
    kps[15] = (500, 300, 0.9)
    kps[16] = (540, 300, 0.9)
    return PersonDetection(bbox=(180, 280, 560, 330), keypoints=kps)


def test_classifier_disabled_by_default_returns_zero_scores():
    classifier = FallClassifier()  # settings.FALL_CLASSIFIER_MODEL_PATH is "" in the test env
    assert classifier.is_available is False
    scores = classifier.score_people([_standing_person(), _fallen_person()], 640, 480)
    assert scores == [0.0, 0.0]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="ml/exported/fall_classifier_v1.onnx not built")
def test_classifier_discriminates_standing_vs_fallen(monkeypatch):
    import app.config as config_module

    monkeypatch.setattr(config_module.settings, "FALL_CLASSIFIER_MODEL_PATH", str(MODEL_PATH))
    classifier = FallClassifier()
    assert classifier.is_available is True

    scores = classifier.score_people([_standing_person(), _fallen_person()], 640, 480)
    standing_score, fallen_score = scores
    assert standing_score < 0.1
    assert fallen_score > 0.9


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="ml/exported/fall_classifier_v1.onnx not built")
def test_classifier_missing_file_fails_safe(monkeypatch):
    import app.config as config_module

    monkeypatch.setattr(config_module.settings, "FALL_CLASSIFIER_MODEL_PATH", "/nonexistent/model.onnx")
    classifier = FallClassifier()
    assert classifier.is_available is False
    assert classifier.score_people([_standing_person()], 640, 480) == [0.0]
