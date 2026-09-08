"""
Drives the REAL SentinelCam inference pipeline over a labelled corpus.

There is no evaluation-only detector here. The runner imports
`app.services.detection.*` from the backend and calls the same
`video_scan.scan_video` loop the upload analyser runs, with the same
`FallPipeline`, the same `ModelFallDetector` gate and the same trained
checkpoint. Anything less would measure a re-implementation, and the
resulting numbers would be about the evaluator rather than the product.

THE THRESHOLD SWEEP IS EXACT, NOT APPROXIMATE
---------------------------------------------
Re-running YOLO for every (confidence, sustained-seconds) pair would cost
`grid_size x corpus_hours` of inference. It is avoided without
approximating anything, because of two properties of the pipeline:

  1. `FALL_DETECTOR_MIN_SUSTAINED_SECONDS` never reaches the model. It is
     consumed entirely by `ModelFallDetector`, downstream of inference.

  2. `FALL_DETECTOR_MIN_CONFIDENCE` reaches the model only as YOLO's
     `conf=` argument, which drops boxes below the threshold BEFORE NMS.
     NMS keeps the highest-scoring box of an overlapping group and
     suppresses lower-scoring ones, so a box below C can never suppress a
     box above C. The set of surviving boxes with score >= C is therefore
     identical whether inference ran at `conf=C` or at any `conf=C_min <= C`
     followed by filtering to `>= C`.

So: run inference ONCE per video at the lowest confidence in the grid,
cache every raw box, then replay the cache through a fresh
`ModelFallDetector` for each grid point. Each replay is arithmetic on
cached numbers, and the result is bit-identical to a full re-run.

The cache is keyed on the video's content hash plus the model's own
sha256, so editing a clip or swapping the checkpoint invalidates it
automatically rather than silently reporting stale numbers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"

# The validation package is a development tool that measures the running
# product, so it imports the product rather than duplicating it. (The
# reverse never happens: nothing under backend/ imports ml/.)
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from .annotations import AnnotatedVideo, Corpus  # noqa: E402
from .metrics import Detection  # noqa: E402

logger = logging.getLogger(__name__)

CACHE_VERSION = 1


@dataclass(frozen=True)
class PipelineIdentity:
    """Exactly what produced a set of numbers. Recorded in the cache and
    printed in the report, so a result can never be read without knowing
    which checkpoint and which classes it came from."""

    model_path: str
    model_sha256: str
    class_names: Dict[str, str]
    stride: int
    mode: str
    ultralytics_version: str

    def as_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "class_names": self.class_names,
            "frame_stride": self.stride,
            "fall_detection_mode": self.mode,
            "ultralytics_version": self.ultralytics_version,
        }


class PipelineUnavailable(RuntimeError):
    """The real pipeline could not be loaded. Never fall back to anything."""


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pipeline() -> PipelineIdentity:
    """Loads the production detector and asserts it is the real thing.

    Raises rather than degrading: an evaluation run that quietly measured
    the pose heuristic, or an untrained yolov8n, while reporting itself as
    'the fall detector' would be worse than no evaluation at all.
    """
    from app.config import settings
    from app.services.detection.fall_object_detector import fall_object_detector

    model_path = settings.FALL_DETECTOR_MODEL_PATH
    if not model_path:
        raise PipelineUnavailable(
            "FALL_DETECTOR_MODEL_PATH is empty. The validation runner measures the "
            "trained detector; point it at ml/exported/fall_detector_v1.pt."
        )

    path = Path(model_path)
    if not path.exists():
        raise PipelineUnavailable(
            f"Trained model not found at {path}. The checkpoint is committed at "
            "ml/exported/fall_detector_v1.pt; set FALL_DETECTOR_MODEL_PATH if you keep it elsewhere."
        )

    if not fall_object_detector.is_available:
        raise PipelineUnavailable(
            f"The trained detector failed to load: {fall_object_detector.load_error}"
        )

    info = fall_object_detector.model_info
    class_names = {str(k): str(v) for k, v in (info.get("class_names") or {}).items()}
    if class_names != {"0": "Fall"}:
        raise PipelineUnavailable(
            f"Loaded model reports classes {class_names}, expected {{'0': 'Fall'}}. "
            "Refusing to evaluate a checkpoint that is not the single-class fall detector."
        )

    # Guard against the specific mistake of evaluating a stock COCO model
    # that happens to sit at the configured path.
    stem = path.stem.lower()
    if stem in ("yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolov8n-pose"):
        raise PipelineUnavailable(
            f"{path} looks like a stock Ultralytics checkpoint, not the fine-tuned "
            "fall detector. Refusing to report its results as SentinelCam's."
        )

    from app.services.detection.fall_pipeline import resolve_mode

    try:
        import ultralytics
        version = ultralytics.__version__
    except Exception:  # pragma: no cover - only if ultralytics changes shape
        version = "unknown"

    return PipelineIdentity(
        model_path=str(path),
        model_sha256=_sha256(path),
        class_names=class_names,
        stride=max(int(settings.VIDEO_ANALYSIS_FRAME_STRIDE), 1),
        mode=resolve_mode(),
        ultralytics_version=version,
    )


# --- raw per-frame detections -------------------------------------------

@dataclass
class RawFrameDetections:
    """Every `Fall` box the model produced on one processed frame."""

    video_time_seconds: float
    boxes: List[dict]  # [{"bbox": [x1,y1,x2,y2], "confidence": float}, ...]


@dataclass
class ScannedVideo:
    filename: str
    duration_seconds: float
    fps: float
    frame_count: int
    processed_frames: int
    frames: List[RawFrameDetections]
    inference_seconds: float


class _NullPipeline:
    """Passed to `scan_video` when the runner wants raw boxes rather than
    gated events: it collects the model's output per frame and emits no
    events, so the sustain gate can be applied afterwards at many
    thresholds. The inference call itself is still the production one."""

    def __init__(self, detector, min_confidence: float):
        self._detector = detector
        self._min_confidence = min_confidence
        self.frames: List[RawFrameDetections] = []

    def update(self, frame, people, now=None):
        boxes = [
            {"bbox": [float(v) for v in d.bbox], "confidence": float(d.confidence)}
            for d in self._detector.detect(frame)
            if d.confidence >= self._min_confidence
        ]
        self.frames.append(RawFrameDetections(video_time_seconds=float(now or 0.0), boxes=boxes))
        return []


def scan_corpus_video(
    video_path: Path,
    min_confidence: float,
    stride: Optional[int] = None,
    run_pose: bool = False,
) -> ScannedVideo:
    """Runs the production frame loop over one video and returns every raw
    box, for later replay through the sustain gate.

    `run_pose` drives the pose model as production does. It is off by
    default because in `model` mode `FallPipeline` ignores the pose output
    entirely (it feeds only the heuristic), so running it changes no fall
    event - it is pure cost. The report records which was used.
    """
    from app.services.detection.fall_object_detector import fall_object_detector
    from app.services.detection.video_scan import open_video, scan_video

    extract_people = None
    if run_pose:
        from app.services.detection.engine import detection_engine
        detection_engine.ensure_models()
        extract_people = detection_engine.extract_people

    cap, properties = open_video(str(video_path))
    collector = _NullPipeline(fall_object_detector, min_confidence)

    started = time.monotonic()
    processed = 0
    try:
        for scanned in scan_video(cap, properties, collector, extract_people, stride=stride):
            if scanned.processed:
                processed += 1
    finally:
        cap.release()
    inference_seconds = time.monotonic() - started

    duration = properties.duration_seconds
    if not duration:
        # A container with no frame count still has a real timeline; the
        # last processed frame's timestamp is the best available bound.
        duration = collector.frames[-1].video_time_seconds if collector.frames else 0.0

    return ScannedVideo(
        filename=video_path.name,
        duration_seconds=float(duration),
        fps=properties.fps,
        frame_count=properties.frame_count,
        processed_frames=processed,
        frames=collector.frames,
        inference_seconds=inference_seconds,
    )


# --- replay: raw boxes -> gated incidents --------------------------------

class _CachedBox:
    __slots__ = ("bbox", "confidence")

    def __init__(self, bbox, confidence):
        self.bbox = tuple(bbox)
        self.confidence = confidence


def replay(
    scanned: ScannedVideo,
    min_confidence: float,
    min_sustained_seconds: float,
    debounce_seconds: Optional[float] = None,
) -> List[Detection]:
    """Feeds cached raw boxes through a fresh production
    `ModelFallDetector` at the given thresholds. Equivalent to a full
    re-run - see the module docstring for why."""
    from app.services.detection.fall_detection import FALL_DEBOUNCE_SECONDS, ModelFallDetector

    gate = ModelFallDetector(
        min_sustained_seconds=min_sustained_seconds,
        debounce_seconds=FALL_DEBOUNCE_SECONDS if debounce_seconds is None else debounce_seconds,
    )

    detections: List[Detection] = []
    for frame in scanned.frames:
        boxes = [
            _CachedBox(b["bbox"], b["confidence"])
            for b in frame.boxes
            if b["confidence"] >= min_confidence
        ]
        for event in gate.update(boxes, now=frame.video_time_seconds):
            detections.append(Detection(
                video_time_seconds=frame.video_time_seconds,
                confidence=float(event.get("confidence", 0.0)),
                detector=str(event.get("detector", "model")),
                track_id=event.get("track_id"),
                sustained_seconds=event.get("sustained_seconds"),
            ))
    return detections


# --- caching --------------------------------------------------------------

def _cache_key(video_path: Path, identity: PipelineIdentity, min_confidence: float, run_pose: bool) -> str:
    stat = video_path.stat()
    material = "|".join([
        str(CACHE_VERSION),
        video_path.name,
        str(stat.st_size),
        str(int(stat.st_mtime)),
        identity.model_sha256,
        str(identity.stride),
        f"{min_confidence:.4f}",
        str(run_pose),
    ])
    return hashlib.sha256(material.encode()).hexdigest()[:24]


def scan_or_load(
    video_path: Path,
    identity: PipelineIdentity,
    min_confidence: float,
    cache_dir: Optional[Path],
    run_pose: bool = False,
    stride: Optional[int] = None,
) -> ScannedVideo:
    """Raw detections for one video, from cache when the video, the model
    and the settings are all unchanged."""
    if cache_dir is None:
        return scan_corpus_video(video_path, min_confidence, stride=stride, run_pose=run_pose)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{video_path.stem}.{_cache_key(video_path, identity, min_confidence, run_pose)}.json"

    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text())
            logger.info("Using cached detections for %s", video_path.name)
            return ScannedVideo(
                filename=payload["filename"],
                duration_seconds=payload["duration_seconds"],
                fps=payload["fps"],
                frame_count=payload["frame_count"],
                processed_frames=payload["processed_frames"],
                frames=[
                    RawFrameDetections(video_time_seconds=f["t"], boxes=f["boxes"])
                    for f in payload["frames"]
                ],
                inference_seconds=payload.get("inference_seconds", 0.0),
            )
        except (KeyError, ValueError, TypeError):
            logger.warning("Cache file %s is unreadable; re-running inference", cache_file)

    scanned = scan_corpus_video(video_path, min_confidence, stride=stride, run_pose=run_pose)
    cache_file.write_text(json.dumps({
        "filename": scanned.filename,
        "duration_seconds": scanned.duration_seconds,
        "fps": scanned.fps,
        "frame_count": scanned.frame_count,
        "processed_frames": scanned.processed_frames,
        "inference_seconds": scanned.inference_seconds,
        "pipeline": identity.as_dict(),
        "frames": [{"t": f.video_time_seconds, "boxes": f.boxes} for f in scanned.frames],
    }))
    return scanned


def scan_corpus(
    corpus: Corpus,
    videos_dir: Path,
    identity: PipelineIdentity,
    min_confidence: float,
    cache_dir: Optional[Path] = None,
    run_pose: bool = False,
    stride: Optional[int] = None,
) -> "tuple[Dict[str, ScannedVideo], List[str]]":
    """Runs (or loads) raw detections for every annotated video. Returns
    the results plus the filenames that were annotated but missing on
    disk - reported rather than skipped silently, since a missing clip
    would otherwise look like a video that produced no detections."""
    scanned: Dict[str, ScannedVideo] = {}
    missing: List[str] = []

    for video in corpus.videos:
        path = videos_dir / video.filename
        if not path.exists():
            missing.append(video.filename)
            continue
        logger.info("Scanning %s", video.filename)
        scanned[video.filename] = scan_or_load(
            path, identity, min_confidence, cache_dir, run_pose=run_pose, stride=stride,
        )

    return scanned, missing


def annotated_durations_match(
    corpus: Corpus, scanned: Dict[str, ScannedVideo], tolerance: float = 1.0
) -> List[str]:
    """Filenames whose annotated duration disagrees with the decoded file
    by more than `tolerance` seconds. A mismatch usually means the
    annotation was written against a different cut, which would put every
    incident time in the wrong place."""
    mismatched = []
    for video in corpus.videos:
        found = scanned.get(video.filename)
        if found is None or found.duration_seconds <= 0:
            continue
        if abs(found.duration_seconds - video.duration_seconds) > tolerance:
            mismatched.append(
                f"{video.filename}: annotated {video.duration_seconds:.1f}s, "
                f"decoded {found.duration_seconds:.1f}s"
            )
    return mismatched


def detections_for_video(video: AnnotatedVideo, scanned: Dict[str, ScannedVideo],
                         min_confidence: float, min_sustained_seconds: float) -> Sequence[Detection]:
    found = scanned.get(video.filename)
    if found is None:
        return ()
    return replay(found, min_confidence, min_sustained_seconds)
