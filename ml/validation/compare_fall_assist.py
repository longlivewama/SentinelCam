"""
Does pose estimation / multi-person tracking improve fall alerting?

Compares the production fall pipeline with and without the two optional
assistance steps in `backend/app/services/detection/fall_pipeline.py`:

  pose verification   veto an event when the person it was attributed to
                      was measured upright (FALL_POSE_VERIFICATION)
  track dedupe        one event per tracked person per window
                      (FALL_TRACK_DEDUPE)

on the same labelled corpus `ml.validation.evaluate` uses, scored by the
same incident matcher (`metrics.evaluate`). Nothing here is a
re-implementation: inference runs through the production
`detection_engine.extract_people` and `fall_object_detector.detect`, and
every variant is replayed through the production `FallPipeline` class.

Inference runs ONCE per video. Each processed frame's raw `Fall` boxes (at
the production confidence floor) and the pose model's people are cached,
then replayed through a fresh `FallPipeline` per variant - neither assist
step can influence inference, so the replay is exact.

Usage (corpus layout as for ml.validation.evaluate):

    cd backend && source venv/bin/activate && cd ..
    python -m ml.validation.compare_fall_assist [--json out.json]

The corpus is footage of real people and is never committed; nor is the
cache, which lives in the corpus directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .annotations import AnnotationError, load_corpus
from .metrics import Detection, evaluate
from .runner import PipelineUnavailable, verify_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = REPO_ROOT / "ml" / "data" / "validation"
CACHE_VERSION = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variant:
    name: str
    tracking: bool
    pose_verification: bool
    track_dedupe: bool
    upright_max_degrees: Optional[float] = None
    # FALL_POSE_CONFIRM_SECONDS for this variant; 0 = immediate veto.
    confirm_seconds: Optional[float] = None


def default_variants(upright_grid: List[float], confirm_grid: List[float]) -> List[Variant]:
    variants = [
        Variant("baseline (no tracking)", tracking=False, pose_verification=False, track_dedupe=False),
        Variant("tracking only", tracking=True, pose_verification=False, track_dedupe=False),
        Variant("track dedupe", tracking=True, pose_verification=False, track_dedupe=True),
    ]
    for degrees in upright_grid:
        for confirm in confirm_grid:
            label = "immediate veto" if confirm <= 0 else f"confirm {confirm:g}s"
            variants.append(Variant(
                f"pose verification, {label} (<{degrees:g} deg)", tracking=True,
                pose_verification=True, track_dedupe=False, upright_max_degrees=degrees,
                confirm_seconds=confirm,
            ))
            variants.append(Variant(
                f"pose verification, {label} (<{degrees:g} deg) + dedupe", tracking=True,
                pose_verification=True, track_dedupe=True, upright_max_degrees=degrees,
                confirm_seconds=confirm,
            ))
    return variants


# --- inference, once per video ------------------------------------------

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_video_once(video_path: Path, identity, cache_dir: Optional[Path]) -> dict:
    from app.config import settings
    from app.services.detection.engine import detection_engine
    from app.services.detection.fall_object_detector import fall_object_detector
    from app.services.detection.video_scan import open_video, scan_video

    stride = max(int(settings.VIDEO_ANALYSIS_FRAME_STRIDE), 1)
    key = hashlib.sha256("|".join([
        str(CACHE_VERSION), _file_sha256(video_path), identity.model_sha256,
        str(settings.FALL_DETECTOR_MIN_CONFIDENCE), str(stride),
        settings.YOLO_POSE_MODEL, str(settings.POSE_INFERENCE_IMGSZ),
    ]).encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{video_path.stem}.{key}.json" if cache_dir else None
    if cache_file is not None and cache_file.exists():
        return json.loads(cache_file.read_text())

    # The model names are relative to the backend's working directory -
    # where the app runs and where the weights live. Resolved explicitly so
    # running this from the repo root uses those weights instead of having
    # Ultralytics download a second copy into the current directory.
    for attr in ("YOLO_POSE_MODEL", "YOLO_MODEL"):
        candidate = REPO_ROOT / "backend" / getattr(settings, attr)
        if not Path(getattr(settings, attr)).is_absolute() and candidate.exists():
            setattr(settings, attr, str(candidate))
    detection_engine.ensure_models()
    frames: List[dict] = []

    class _Recorder:
        """Captures this frame's raw inference for replay; emits nothing."""

        def update(self, frame, people, now=None, frame_index=None):
            boxes = [
                {"bbox": [float(v) for v in d.bbox], "confidence": float(d.confidence)}
                for d in fall_object_detector.detect(frame)
            ]
            frames.append({
                "i": frame_index, "t": float(now or 0.0), "fall_boxes": boxes,
                "people": [
                    {"bbox": [float(v) for v in p.bbox],
                     "keypoints": [list(map(float, k)) for k in p.keypoints],
                     "confidence": None if p.confidence is None else float(p.confidence)}
                    for p in people
                ],
            })
            return []

    cap, properties = open_video(str(video_path))
    started = time.monotonic()
    try:
        for _ in scan_video(cap, properties, _Recorder(), detection_engine.extract_people, stride=stride):
            pass
    finally:
        cap.release()
    scanned = {
        "filename": video_path.name,
        "duration_seconds": properties.duration_seconds,
        "inference_seconds": time.monotonic() - started,
        "frames": frames,
    }
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(scanned))
    return scanned


# --- replay through the production FallPipeline --------------------------

class _CachedBox:
    __slots__ = ("bbox", "confidence")

    def __init__(self, bbox, confidence):
        self.bbox = tuple(bbox)
        self.confidence = confidence


class _CachedDetector:
    """Stands in for fall_object_detector: returns the boxes the real
    model produced for the frame being replayed."""

    def __init__(self):
        self.boxes: List[_CachedBox] = []

    def detect(self, frame):
        return list(self.boxes)


def replay(scanned: dict, variant: Variant) -> Dict[str, object]:
    from app.config import settings
    from app.services.detection.engine import PersonDetection
    from app.services.detection.fall_pipeline import FallPipeline, build_scene_tracker

    previous_upright = settings.FALL_POSE_UPRIGHT_MAX_DEGREES
    previous_confirm = settings.FALL_POSE_CONFIRM_SECONDS
    if variant.upright_max_degrees is not None:
        settings.FALL_POSE_UPRIGHT_MAX_DEGREES = variant.upright_max_degrees
    if variant.confirm_seconds is not None:
        settings.FALL_POSE_CONFIRM_SECONDS = variant.confirm_seconds
    try:
        detector = _CachedDetector()
        pipeline = FallPipeline(
            mode="model",
            scene=build_scene_tracker() if variant.tracking else None,
            pose_verification=variant.pose_verification,
            track_dedupe=variant.track_dedupe,
            fall_detector=detector,
        )
        detections: List[Detection] = []
        suppressed: List[dict] = []
        for frame in scanned["frames"]:
            detector.boxes = [_CachedBox(b["bbox"], b["confidence"]) for b in frame["fall_boxes"]]
            people = [
                PersonDetection(bbox=tuple(p["bbox"]), keypoints=[tuple(k) for k in p["keypoints"]], confidence=p["confidence"])
                for p in frame["people"]
            ]
            for event in pipeline.update(None, people, now=frame["t"], frame_index=frame["i"]):
                detections.append(Detection(
                    video_time_seconds=frame["t"],
                    confidence=float(event.get("confidence", 0.0)),
                    detector=str(event.get("detector", "model")),
                    track_id=event.get("person_track_id"),
                    sustained_seconds=event.get("sustained_seconds"),
                ))
            for event in pipeline.last_suppressed_events:
                suppressed.append({"t": frame["t"], "reason": event.get("suppressed_reason"), "pose": event.get("pose")})
        # The video ended: events still held by pose verification are
        # decided now, at the last processed frame's time - exactly what
        # the upload analyser does.
        last_t = scanned["frames"][-1]["t"] if scanned["frames"] else 0.0
        for event in pipeline.flush():
            detections.append(Detection(
                video_time_seconds=last_t,
                confidence=float(event.get("confidence", 0.0)),
                detector=str(event.get("detector", "model")),
                track_id=event.get("person_track_id"),
                sustained_seconds=event.get("sustained_seconds"),
            ))
        for event in pipeline.last_suppressed_events:
            suppressed.append({"t": last_t, "reason": event.get("suppressed_reason"), "pose": event.get("pose")})
        return {"detections": detections, "suppressed": suppressed}
    finally:
        settings.FALL_POSE_UPRIGHT_MAX_DEGREES = previous_upright
        settings.FALL_POSE_CONFIRM_SECONDS = previous_confirm


def _fmt(value, pct=False, digits=2):
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%" if pct else f"{value:.{digits}f}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--json", type=Path, default=None, help="Also write machine-readable results here")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--upright-grid", type=float, nargs="+", default=[20.0, 30.0, 40.0, 50.0])
    parser.add_argument(
        "--confirm-grid", type=float, nargs="+", default=[0.0, 2.0],
        help="FALL_POSE_CONFIRM_SECONDS values to compare; 0 = immediate veto",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        corpus = load_corpus(args.corpus / "annotations.json")
    except AnnotationError as exc:
        print(f"Cannot evaluate: {exc}", file=sys.stderr)
        return 2
    try:
        identity = verify_pipeline()
    except PipelineUnavailable as exc:
        print(f"Refusing to evaluate: {exc}", file=sys.stderr)
        return 3

    from app.config import settings

    cache_dir = None if args.no_cache else args.corpus / ".assist_cache"
    scanned: Dict[str, dict] = {}
    for video in corpus.videos:
        path = args.corpus / "videos" / video.filename
        if path.exists():
            scanned[video.filename] = scan_video_once(path, identity, cache_dir)
    evaluated = type(corpus)(
        videos=[v for v in corpus.videos if v.filename in scanned], source_path=corpus.source_path,
    )
    if not evaluated.videos:
        print("No corpus videos found on disk.", file=sys.stderr)
        return 2

    rows = []
    for variant in default_variants(args.upright_grid, args.confirm_grid):
        replayed = {name: replay(found, variant) for name, found in scanned.items()}
        summary = evaluate(
            evaluated, {n: r["detections"] for n, r in replayed.items()},
            float(settings.FALL_DETECTOR_MIN_CONFIDENCE), float(settings.FALL_DETECTOR_MIN_SUSTAINED_SECONDS),
        )
        rows.append({
            "variant": variant.name,
            "true_positives": summary.true_positives,
            "missed": summary.false_negatives,
            "false_alerts": summary.false_positives,
            "incident_recall": summary.incident_recall,
            "incident_precision": summary.incident_precision,
            "incident_f1": summary.incident_f1,
            "false_alerts_per_hour": summary.false_alerts_per_hour,
            "median_latency_seconds": summary.median_latency_seconds,
            "events_withheld": sum(len(r["suppressed"]) for r in replayed.values()),
        })

    hours = sum(v.duration_seconds for v in evaluated.videos) / 3600.0
    print(f"\nCorpus: {len(evaluated.videos)} videos, {evaluated.total_incidents} falls, "
          f"{hours * 60:.1f} min; mode=model, conf>={settings.FALL_DETECTOR_MIN_CONFIDENCE}, "
          f"sustain>={settings.FALL_DETECTOR_MIN_SUSTAINED_SECONDS}s, stride {settings.VIDEO_ANALYSIS_FRAME_STRIDE}\n")
    print("| Variant | TP | Missed | False alerts | Recall | Precision | F1 | False alerts/h | Median latency (s) | Withheld |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['variant']} | {row['true_positives']} | {row['missed']} | {row['false_alerts']} | "
            f"{_fmt(row['incident_recall'], pct=True)} | {_fmt(row['incident_precision'], pct=True)} | "
            f"{_fmt(row['incident_f1'])} | {_fmt(row['false_alerts_per_hour'], digits=0)} | "
            f"{_fmt(row['median_latency_seconds'])} | {row['events_withheld']} |"
        )
    if args.json:
        args.json.write_text(json.dumps({"corpus_minutes": hours * 60, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
