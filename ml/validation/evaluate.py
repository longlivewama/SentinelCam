"""
CLI entry point for video-level validation.

    python -m ml.validation.evaluate                     # evaluate at production defaults
    python -m ml.validation.evaluate --sweep             # plus the threshold grid
    python -m ml.validation.evaluate --corpus /path/dir  # a corpus somewhere else

Run from the repository root. The corpus lives outside git (see
ml/validation/README.md): real footage of people falling is personal data
and must not be committed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from .annotations import AnnotationError, load_corpus
from .metrics import evaluate
from .report import render_report
from .runner import (
    PipelineUnavailable,
    annotated_durations_match,
    replay,
    scan_corpus,
    verify_pipeline,
)

logger = logging.getLogger("ml.validation")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = REPO_ROOT / "ml" / "data" / "validation"
DEFAULT_REPORT_PATH = REPO_ROOT / "ml" / "reports" / "fall_detector_video_validation.md"
DEFAULT_JSON_PATH = REPO_ROOT / "ml" / "reports" / "fall_detector_video_validation.json"

# The sweep grid. Confidence brackets the 0.4 production default from
# clearly-permissive to clearly-strict; sustained duration brackets the
# 0.6s default from "one processed frame" upward. At the default stride of
# 5 on 25fps footage a processed frame is 0.2s apart, so 0.2/0.4 probe
# whether the gate is doing anything at all, and 1.2 matches the pose
# heuristic's gate for comparison.
DEFAULT_CONFIDENCE_GRID = (0.25, 0.40, 0.55, 0.70)
DEFAULT_SUSTAINED_GRID = (0.2, 0.4, 0.6, 1.0, 1.5)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m ml.validation.evaluate",
        description="Video-level operational validation of the SentinelCam fall detector.",
    )
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS_DIR,
        help=f"Corpus directory containing videos/ and annotations.json (default: {DEFAULT_CORPUS_DIR})",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Markdown report output path")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON_PATH, help="Machine-readable results output path")
    parser.add_argument("--sweep", action="store_true", help="Also evaluate the threshold grid")
    parser.add_argument(
        "--confidence-grid", type=float, nargs="+", default=list(DEFAULT_CONFIDENCE_GRID),
        help="Confidence values to sweep",
    )
    parser.add_argument(
        "--sustained-grid", type=float, nargs="+", default=list(DEFAULT_SUSTAINED_GRID),
        help="Sustained-duration values (seconds) to sweep",
    )
    parser.add_argument(
        "--run-pose", action="store_true",
        help="Also run the pose model, exactly as production does. Slower and, in 'model' mode, "
             "changes no fall event (FallPipeline feeds pose only to the heuristic).",
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignore and do not write the detection cache")
    parser.add_argument("--stride", type=int, default=None, help="Override VIDEO_ANALYSIS_FRAME_STRIDE")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    corpus_dir: Path = args.corpus
    annotations_path = corpus_dir / "annotations.json"
    videos_dir = corpus_dir / "videos"

    # --- corpus ----------------------------------------------------------
    try:
        corpus = load_corpus(annotations_path)
    except AnnotationError as exc:
        print(f"\nCannot evaluate: {exc}\n", file=sys.stderr)
        print(
            "The validation corpus is intentionally NOT in git (it is footage of real people).\n"
            "To run this, create the layout documented in ml/validation/README.md:\n"
            f"  {corpus_dir}/videos/*.mp4\n"
            f"  {corpus_dir}/annotations.json\n",
            file=sys.stderr,
        )
        return 2

    if not corpus.videos:
        print("The annotation file contains no videos.", file=sys.stderr)
        return 2

    # --- pipeline: verify BEFORE any measurement -------------------------
    try:
        identity = verify_pipeline()
    except PipelineUnavailable as exc:
        print(f"\nRefusing to evaluate: {exc}\n", file=sys.stderr)
        return 3

    logger.info(
        "Pipeline verified: %s (sha256 %s…), classes=%s, mode=%s",
        identity.model_path, identity.model_sha256[:12], identity.class_names, identity.mode,
    )
    if identity.mode != "model":
        print(
            f"\nRefusing to evaluate: fall detection mode resolved to {identity.mode!r}, not 'model'. "
            "This framework measures the trained detector; set FALL_DETECTION_MODE=model or 'auto' "
            "with a loadable checkpoint.\n",
            file=sys.stderr,
        )
        return 3

    from app.config import settings  # after runner.py has put backend/ on sys.path

    production_confidence = float(settings.FALL_DETECTOR_MIN_CONFIDENCE)
    production_sustained = float(settings.FALL_DETECTOR_MIN_SUSTAINED_SECONDS)

    confidence_grid = sorted(set(args.confidence_grid)) if args.sweep else [production_confidence]
    sustained_grid = sorted(set(args.sustained_grid)) if args.sweep else [production_sustained]
    # Inference runs once at the lowest confidence anyone will ask for.
    scan_confidence = min(confidence_grid + [production_confidence])

    cache_dir = None if args.no_cache else corpus_dir / ".detection_cache"

    # --- inference --------------------------------------------------------
    scanned, missing = scan_corpus(
        corpus, videos_dir, identity, scan_confidence,
        cache_dir=cache_dir, run_pose=args.run_pose, stride=args.stride,
    )
    if missing:
        logger.warning("%d annotated video(s) missing on disk: %s", len(missing), ", ".join(missing))
    if not scanned:
        print(
            f"\nNone of the {len(corpus.videos)} annotated videos were found in {videos_dir}.\n",
            file=sys.stderr,
        )
        return 2

    mismatches = annotated_durations_match(corpus, scanned)
    for line in mismatches:
        logger.warning("Duration mismatch: %s", line)

    evaluated_corpus = type(corpus)(
        videos=[v for v in corpus.videos if v.filename in scanned],
        source_path=corpus.source_path,
    )

    def evaluate_at(min_conf: float, min_sustained: float):
        detections = {
            name: replay(found, min_conf, min_sustained) for name, found in scanned.items()
        }
        return evaluate(evaluated_corpus, detections, min_conf, min_sustained)

    summary = evaluate_at(production_confidence, production_sustained)

    # --- sweep -------------------------------------------------------------
    sweep_rows = []
    if args.sweep:
        for min_conf in confidence_grid:
            for min_sustained in sustained_grid:
                point = evaluate_at(min_conf, min_sustained)
                sweep_rows.append({
                    "min_confidence": min_conf,
                    "min_sustained_seconds": min_sustained,
                    "incident_recall": point.incident_recall,
                    "incident_precision": point.incident_precision,
                    "incident_f1": point.incident_f1,
                    "false_alerts_per_hour": point.false_alerts_per_hour,
                    "mean_latency_seconds": point.mean_latency_seconds,
                    "median_latency_seconds": point.median_latency_seconds,
                    "true_positives": point.true_positives,
                    "missed": point.false_negatives,
                    "false_alerts": point.false_positives,
                    "is_current": (
                        abs(min_conf - production_confidence) < 1e-9
                        and abs(min_sustained - production_sustained) < 1e-9
                    ),
                })

    # --- output ------------------------------------------------------------
    report = render_report(
        evaluated_corpus, summary, identity,
        sweep_rows=sweep_rows or None,
        missing_videos=missing or None,
        duration_mismatches=mismatches or None,
        run_pose=args.run_pose,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)

    args.json.write_text(json.dumps({
        "pipeline": identity.as_dict(),
        "corpus": {
            "videos": len(evaluated_corpus.videos),
            "fall_clips": len(evaluated_corpus.fall_videos),
            "hard_negative_clips": len(evaluated_corpus.hard_negative_videos),
            "total_duration_seconds": round(evaluated_corpus.total_duration_seconds, 2),
            "total_incidents": evaluated_corpus.total_incidents,
            "missing_videos": missing,
            "duration_mismatches": mismatches,
            "hard_negative_activities": evaluated_corpus.composition(),
            "fall_taxonomy": evaluated_corpus.fall_taxonomy(),
            "condition_coverage": evaluated_corpus.condition_coverage(),
            # The readiness check. Empty means the corpus covers every
            # bucket ml/validation/README.md asks for; anything present is
            # the shopping list, and the reason `validation_status` below
            # says what it says.
            "missing_coverage": evaluated_corpus.missing_coverage(),
        },
        "configuration": {
            "min_confidence": production_confidence,
            "min_sustained_seconds": production_sustained,
            "frame_stride": identity.stride,
            "run_pose": args.run_pose,
        },
        "results": {
            "true_positives": summary.true_positives,
            "false_negatives": summary.false_negatives,
            "false_positives": summary.false_positives,
            "duplicate_detections": summary.duplicate_detections,
            "incident_recall": summary.incident_recall,
            "incident_precision": summary.incident_precision,
            "incident_f1": summary.incident_f1,
            "hard_negative_hours": round(summary.hard_negative_hours, 4),
            "false_alerts_per_hour": summary.false_alerts_per_hour,
            "mean_latency_seconds": summary.mean_latency_seconds,
            "median_latency_seconds": summary.median_latency_seconds,
            "max_latency_seconds": summary.max_latency_seconds,
            "false_alert_categories": summary.false_alert_categories(),
            "false_alerts": summary.false_alert_details(),
            "missed_falls": summary.missed_fall_details(),
            "confidence_distribution": summary.confidence_distribution(),
            "by_condition": {
                dimension: [g.as_dict() for g in groups]
                for dimension, groups in summary.by_all_conditions().items()
            },
            "by_fall_direction": [g.as_dict() for g in summary.by_fall_direction()],
            "by_fall_speed": [g.as_dict() for g in summary.by_fall_speed()],
            "by_hard_negative_activity": [
                g.as_dict() for g in summary.by_hard_negative_activity()
            ],
        },
        "sweep": sweep_rows,
        # Stated in the machine-readable output as well as the report, so a
        # dashboard or CI job cannot present these numbers as validated
        # performance without also carrying the caveat that they are not.
        "validation_status": (
            "Evaluation framework ready; independent real-world video validation pending "
            "labelled footage."
            if evaluated_corpus.missing_coverage()
            else "Corpus meets the coverage specification in ml/validation/README.md."
        ),
    }, indent=2) + "\n")

    # --- console summary ---------------------------------------------------
    print()
    print(f"Videos evaluated      {len(evaluated_corpus.videos)} "
          f"({len(evaluated_corpus.fall_videos)} fall, {len(evaluated_corpus.hard_negative_videos)} hard negative)")
    print(f"Footage               {evaluated_corpus.total_duration_seconds / 60:.1f} min")
    print(f"Fall incidents        {summary.total_incidents}")
    recall = summary.incident_recall
    print(f"Incident recall       {'n/a' if recall is None else f'{recall * 100:.1f}%'} "
          f"({summary.true_positives}/{summary.total_incidents})")
    f1 = summary.incident_f1
    print(f"Incident F1            {'n/a' if f1 is None else f'{f1:.3f}'}")
    far = summary.false_alerts_per_hour
    print(f"False alerts/hour     {'n/a' if far is None else f'{far:.2f}'} "
          f"(over {summary.hard_negative_hours:.2f} h of hard negatives)")
    latency = summary.mean_latency_seconds
    print(f"Mean latency          {'n/a' if latency is None else f'{latency:.2f}s'}")
    missing_coverage = evaluated_corpus.missing_coverage()
    if missing_coverage:
        print()
        print("Status                Evaluation framework ready; independent real-world video")
        print("                      validation pending labelled footage.")
        print(f"                      Unrepresented: {', '.join(sorted(missing_coverage))}")
    print()
    print(f"Report  {args.report}")
    print(f"JSON    {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
