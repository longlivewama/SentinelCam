"""
Tests for the validation runner's safety guards and its replay mechanism.

The guards matter more than they look: a validation run that quietly
measured the pose heuristic, or a stock COCO checkpoint, while labelling
its output "SentinelCam fall detector" would be worse than no validation
at all - it would produce confident numbers about the wrong thing. So
`verify_pipeline` must refuse rather than degrade, and that refusal is
tested for each way it can go wrong.

The replay equivalence test underpins the entire threshold sweep: the
sweep only avoids re-running inference per grid point because filtering
cached boxes is exactly equivalent to re-running at a higher confidence.
If that stopped being true, every swept row would be wrong.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import settings  # noqa: E402
from ml.validation import runner  # noqa: E402
from ml.validation.runner import (  # noqa: E402
    PipelineUnavailable,
    RawFrameDetections,
    ScannedVideo,
    replay,
    verify_pipeline,
)


def scanned(frames):
    """A ScannedVideo from [(time, [(confidence), ...]), ...]."""
    return ScannedVideo(
        filename="synthetic.mp4",
        duration_seconds=frames[-1][0] + 1 if frames else 0.0,
        fps=25.0,
        frame_count=len(frames) * 5,
        processed_frames=len(frames),
        frames=[
            RawFrameDetections(
                video_time_seconds=t,
                boxes=[{"bbox": [100.0, 300.0, 300.0, 380.0], "confidence": c} for c in confs],
            )
            for t, confs in frames
        ],
        inference_seconds=0.0,
    )


# --- replay: the basis of the sweep --------------------------------------

def test_replay_fires_once_a_detection_persists_past_the_gate():
    video = scanned([(t * 0.2, [0.8]) for t in range(10)])
    detections = replay(video, min_confidence=0.4, min_sustained_seconds=0.6)

    assert len(detections) == 1
    assert detections[0].detector == "model"
    assert detections[0].video_time_seconds == pytest.approx(0.6, abs=0.01)


def test_replay_does_not_fire_when_the_detection_is_too_brief():
    video = scanned([(0.0, [0.8]), (0.2, [0.8]), (0.4, []), (0.6, []), (0.8, [])])
    assert replay(video, min_confidence=0.4, min_sustained_seconds=0.6) == []


def test_raising_the_confidence_threshold_filters_boxes_out():
    """The mechanism the sweep relies on: the same cached scan, read at a
    higher threshold, yields the detections a stricter run would."""
    video = scanned([(t * 0.2, [0.5]) for t in range(10)])

    assert len(replay(video, min_confidence=0.4, min_sustained_seconds=0.6)) == 1
    # Every box is below 0.6, so a stricter run sees nothing at all.
    assert replay(video, min_confidence=0.6, min_sustained_seconds=0.6) == []


def test_raising_the_sustained_duration_suppresses_short_detections():
    video = scanned([(t * 0.2, [0.8]) for t in range(5)])  # 0.0 .. 0.8s

    assert len(replay(video, min_confidence=0.4, min_sustained_seconds=0.6)) == 1
    # A 0.8s run cannot satisfy a 1.5s gate.
    assert replay(video, min_confidence=0.4, min_sustained_seconds=1.5) == []


def test_replay_reports_the_peak_confidence_of_the_run():
    video = scanned([(0.0, [0.55]), (0.2, [0.95]), (0.4, [0.60]), (0.6, [0.62]), (0.8, [0.58])])
    detections = replay(video, min_confidence=0.4, min_sustained_seconds=0.6)
    assert detections[0].confidence == pytest.approx(0.95)


def test_replay_is_equivalent_to_having_scanned_at_the_higher_threshold():
    """The exactness claim behind the sweep, stated as a test: filtering a
    permissive scan to >= C produces the same detections as a scan that
    had been run at C in the first place."""
    frames = [(t * 0.2, [0.30, 0.75]) for t in range(10)]
    permissive = scanned(frames)

    # What a scan run at conf=0.5 would have cached: only boxes >= 0.5.
    strict = scanned([(t, [c for c in confs if c >= 0.5]) for t, confs in frames])

    from_permissive = replay(permissive, min_confidence=0.5, min_sustained_seconds=0.6)
    from_strict = replay(strict, min_confidence=0.5, min_sustained_seconds=0.6)

    assert [(d.video_time_seconds, d.confidence) for d in from_permissive] == \
           [(d.video_time_seconds, d.confidence) for d in from_strict]


def test_replay_of_an_empty_scan_produces_nothing():
    assert replay(scanned([]), min_confidence=0.4, min_sustained_seconds=0.6) == []


def test_replay_debounces_a_subject_who_stays_on_the_ground():
    """One long fall must not become dozens of alerts."""
    video = scanned([(t * 0.2, [0.8]) for t in range(150)])  # 30 seconds
    detections = replay(video, min_confidence=0.4, min_sustained_seconds=0.6)

    # 30s of continuous detection at a 10s debounce is a handful of
    # alerts, not one per processed frame.
    assert 1 <= len(detections) <= 4


# --- guards: refuse rather than measure the wrong thing ------------------

def test_verify_pipeline_refuses_when_no_model_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "FALL_DETECTOR_MODEL_PATH", "")
    with pytest.raises(PipelineUnavailable, match="FALL_DETECTOR_MODEL_PATH is empty"):
        verify_pipeline()


def test_verify_pipeline_refuses_when_the_checkpoint_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "FALL_DETECTOR_MODEL_PATH", str(tmp_path / "absent.pt"))
    with pytest.raises(PipelineUnavailable, match="not found"):
        verify_pipeline()


def test_verify_pipeline_refuses_a_stock_ultralytics_checkpoint(monkeypatch, tmp_path):
    """Guards the specific mistake of evaluating an untrained yolov8n that
    happens to sit at the configured path and reporting it as ours."""
    stock = tmp_path / "yolov8n.pt"
    stock.write_bytes(b"not really a checkpoint")
    monkeypatch.setattr(settings, "FALL_DETECTOR_MODEL_PATH", str(stock))

    class _Loaded:
        is_available = True
        model_info = {"class_names": {0: "Fall"}}

    monkeypatch.setattr(runner, "_sha256", lambda p: "0" * 64)
    monkeypatch.setattr(
        "app.services.detection.fall_object_detector.fall_object_detector", _Loaded(),
    )
    with pytest.raises(PipelineUnavailable, match="stock Ultralytics checkpoint"):
        verify_pipeline()


def test_verify_pipeline_refuses_a_checkpoint_with_the_wrong_classes(monkeypatch, tmp_path):
    other = tmp_path / "some_other_detector.pt"
    other.write_bytes(b"x")
    monkeypatch.setattr(settings, "FALL_DETECTOR_MODEL_PATH", str(other))

    class _WrongClasses:
        is_available = True
        model_info = {"class_names": {0: "person", 1: "car"}}

    monkeypatch.setattr(
        "app.services.detection.fall_object_detector.fall_object_detector", _WrongClasses(),
    )
    with pytest.raises(PipelineUnavailable, match="expected"):
        verify_pipeline()


def test_verify_pipeline_refuses_when_the_detector_failed_to_load(monkeypatch, tmp_path):
    present = tmp_path / "fall_detector_v1.pt"
    present.write_bytes(b"x")
    monkeypatch.setattr(settings, "FALL_DETECTOR_MODEL_PATH", str(present))

    class _Broken:
        is_available = False
        load_error = "failed to load checkpoint: boom"

    monkeypatch.setattr(
        "app.services.detection.fall_object_detector.fall_object_detector", _Broken(),
    )
    with pytest.raises(PipelineUnavailable, match="failed to load"):
        verify_pipeline()


# --- the real artifact ----------------------------------------------------

ARTIFACT = Path(settings.FALL_DETECTOR_MODEL_PATH) if settings.FALL_DETECTOR_MODEL_PATH else None
requires_artifact = pytest.mark.skipif(
    ARTIFACT is None or not ARTIFACT.exists(),
    reason="trained detector artifact not present",
)


@requires_artifact
def test_verify_pipeline_accepts_the_real_committed_detector():
    """The positive case: with the shipped checkpoint, verification passes
    and reports the identity that will be printed in the report."""
    pytest.importorskip("ultralytics")
    identity = verify_pipeline()

    assert identity.class_names == {"0": "Fall"}
    assert identity.mode == "model"
    assert identity.model_sha256
    assert identity.stride >= 1
    assert "fall_detector" in Path(identity.model_path).stem


@requires_artifact
def test_the_reported_sha256_matches_the_committed_metadata():
    """Ties a validation report to a specific, documented artifact."""
    import json
    pytest.importorskip("ultralytics")

    meta = json.loads((REPO_ROOT / "ml" / "exported" / "fall_detector_v1.metadata.json").read_text())
    identity = verify_pipeline()
    assert identity.model_sha256 == meta["file"]["sha256"]
