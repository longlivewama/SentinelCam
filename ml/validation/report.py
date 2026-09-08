"""
Renders an evaluation into a human-readable Markdown report.

Formatting rules that exist to stop a reader over-trusting the numbers:

  * A metric that could not be computed prints "n/a (reason)", never 0.
    Zero recall and un-measurable recall look identical in a table and
    mean opposite things.
  * Every headline figure is followed by its denominator. "83% recall"
    is a very different claim at n=6 than at n=600, and the table should
    not let anyone quote it without the n.
  * The sample-size caveat is generated from the actual corpus size and
    sits at the TOP of the report, not in a footnote.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from .annotations import Corpus
from .metrics import POST_TOLERANCE_SECONDS, PRE_TOLERANCE_SECONDS, EvaluationSummary

# Below this many annotated falls, the report leads with an explicit
# "preliminary" banner. Chosen to match the honest reading of a corpus
# this size: with fewer than 30 incidents a single miss moves recall by
# more than 3 points, so differences between threshold settings are mostly
# noise.
PRELIMINARY_INCIDENT_THRESHOLD = 30


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _secs(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def _rate(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_report(
    corpus: Corpus,
    summary: EvaluationSummary,
    pipeline_identity,
    sweep_rows: Optional[List[dict]] = None,
    missing_videos: Optional[List[str]] = None,
    duration_mismatches: Optional[List[str]] = None,
    run_pose: bool = False,
) -> str:
    lines: List[str] = []
    add = lines.append

    add("# Fall detector — video-level validation report")
    add("")
    add(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`python -m ml.validation.evaluate` (see `ml/validation/README.md`).")
    add("")

    # --- honesty banner, first thing on the page --------------------------
    n_incidents = summary.total_incidents
    n_videos = len(corpus.videos)
    if n_incidents == 0:
        add("> **No fall incidents in this corpus.** Recall is not measured. The figures below "
            "describe false-alert behaviour on hard-negative footage only.")
    elif n_incidents < PRELIMINARY_INCIDENT_THRESHOLD:
        add(f"> **Preliminary result — {n_incidents} annotated fall incident(s) across {n_videos} "
            f"clip(s).** At this sample size a single missed fall moves recall by "
            f"{100 / n_incidents:.1f} percentage points, so differences between threshold settings "
            "are largely noise. These numbers are a sanity check on operational behaviour, "
            "**not** evidence of production or clinical safety.")
    else:
        add(f"> {n_incidents} annotated fall incidents across {n_videos} clips. Still a small "
            "corpus by any clinical standard — read the Limitations section before quoting "
            "anything here.")
    add("")

    # --- what produced these numbers -------------------------------------
    add("## Pipeline under test")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Model | `{pipeline_identity.model_path}` |")
    add(f"| Model sha256 | `{pipeline_identity.model_sha256[:16]}…` |")
    add(f"| Classes | `{pipeline_identity.class_names}` |")
    add(f"| Fall detection mode | `{pipeline_identity.mode}` |")
    add(f"| Frame stride | every {pipeline_identity.stride} frames |")
    add(f"| Pose model run | {'yes (as production)' if run_pose else 'no — ignored in model mode, see README'} |")
    add(f"| Ultralytics | {pipeline_identity.ultralytics_version} |")
    add(f"| `FALL_DETECTOR_MIN_CONFIDENCE` | {summary.min_confidence} |")
    add(f"| `FALL_DETECTOR_MIN_SUSTAINED_SECONDS` | {summary.min_sustained_seconds} |")
    add("")

    if missing_videos:
        add(f"> ⚠ **{len(missing_videos)} annotated video(s) were not found on disk** and are "
            "excluded from every figure below: " + ", ".join(f"`{n}`" for n in missing_videos))
        add("")
    if duration_mismatches:
        add("> ⚠ **Annotated duration disagrees with the decoded file** for the following clips. "
            "The annotation may have been written against a different cut, which would put every "
            "incident time in the wrong place:")
        for line in duration_mismatches:
            add(f"> - {line}")
        add("")

    # --- dataset ----------------------------------------------------------
    add("## 1. Dataset composition")
    add("")
    add("| | Count | Duration |")
    add("|---|---:|---:|")
    fall_secs = sum(v.duration_seconds for v in corpus.fall_videos)
    hn_secs = sum(v.duration_seconds for v in corpus.hard_negative_videos)
    add(f"| Fall clips | {len(corpus.fall_videos)} | {fall_secs / 60:.1f} min |")
    add(f"| Hard-negative clips | {len(corpus.hard_negative_videos)} | {hn_secs / 60:.1f} min |")
    add(f"| **Total** | **{n_videos}** | **{corpus.total_duration_seconds / 60:.1f} min "
        f"({corpus.total_duration_seconds / 3600:.2f} h)** |")
    add("")
    add(f"Annotated fall incidents: **{corpus.total_incidents}**")
    add("")

    composition = corpus.composition()
    if composition:
        add("Hard negatives by activity:")
        add("")
        add("| Activity | Clips |")
        add("|---|---:|")
        for activity, count in composition.items():
            add(f"| {activity} | {count} |")
        add("")

    # --- headline metrics -------------------------------------------------
    add("## 2. Incident-level results")
    add("")
    add("| Metric | Value | Basis |")
    add("|---|---:|---|")
    add(f"| Incident recall | {_pct(summary.incident_recall)} | "
        f"{summary.true_positives} of {summary.total_incidents} annotated falls |")
    add(f"| Incident precision | {_pct(summary.incident_precision)} | "
        f"{summary.true_positives} of {summary.true_positives + summary.false_positives} alerts |")
    add(f"| Missed falls | {summary.false_negatives} | |")
    add(f"| False alerts | {summary.false_positives} | |")
    add(f"| Duplicate alerts | {summary.duplicate_detections} | same fall re-firing past the debounce |")
    add("")

    add("## 3. Operational results")
    add("")
    add("| Metric | Value | Basis |")
    add("|---|---:|---|")
    add(f"| Hard-negative footage | {summary.hard_negative_hours:.2f} h | ordinary activity only |")
    add(f"| **False alerts / hour** | **{_rate(summary.false_alerts_per_hour)}** | "
        f"{summary.hard_negative_false_alerts} alerts over hard negatives |")
    add(f"| False alerts / hour (whole corpus) | {_rate(summary.all_false_alerts_per_hour)} | "
        f"{summary.false_positives} over {summary.total_hours:.2f} h |")
    add(f"| Mean detection latency | {_secs(summary.mean_latency_seconds)} | "
        f"n={len(summary.latencies)} |")
    add(f"| Median detection latency | {_secs(summary.median_latency_seconds)} | "
        f"{'n<3, not reported' if summary.median_latency_seconds is None and summary.latencies else f'n={len(summary.latencies)}'} |")
    add(f"| Worst detection latency | {_secs(summary.max_latency_seconds)} | |")
    add("")
    add("Latency is measured from the **start of the annotated fall** to the first matching alert. "
        "A late alert is still a successful detection — the product premise is that harm comes "
        "from lying unnoticed — so latency is reported alongside recall rather than folded into it.")
    add("")

    # --- per video --------------------------------------------------------
    add("## 4. Per-video results")
    add("")
    add("| Video | Category | Duration | GT falls | Detected | Missed | False alerts | Status |")
    add("|---|---|---:|---:|---:|---:|---:|---|")
    for result in summary.results:
        add(
            f"| `{result.video.filename}` | {result.video.category} | "
            f"{result.video.duration_seconds:.1f}s | {result.ground_truth_falls} | "
            f"{result.detected_falls} | {len(result.missed)} | {len(result.false_alerts)} | "
            f"{result.status} |"
        )
    add("")

    # --- false positives ---------------------------------------------------
    add("## 5. Hard-negative analysis (false positives)")
    add("")
    categories = summary.false_alert_categories()
    if not categories:
        add("No false alerts were produced on this corpus.")
        add("")
        if summary.hard_negative_hours < 1.0:
            add(f"> This is measured over only {summary.hard_negative_hours:.2f} h of hard-negative "
                "footage. Zero false alerts here is consistent with a true rate of up to roughly "
                f"{3 / max(summary.hard_negative_hours, 1e-9):.1f}/hour "
                "(rule-of-three upper bound at 95% confidence) — it is not evidence of a low rate.")
            add("")
    else:
        add("| Activity | False alerts |")
        add("|---|---:|")
        for activity, count in categories.items():
            add(f"| {activity} | {count} |")
        add("")
        add("### Every false alert")
        add("")
        add("| Video | At | Confidence | Activity | Moment |")
        add("|---|---:|---:|---|---|")
        for row in summary.false_alert_details():
            add(f"| `{row['video']}` | {row['video_time_seconds']:.1f}s | {row['confidence']:.2f} | "
                f"{row['activity'] or '—'} | {row['activity_at_moment'] or '—'} |")
        add("")

    # --- missed falls ------------------------------------------------------
    add("## 6. Missed falls")
    add("")
    missed = summary.missed_fall_details()
    if not missed:
        add("Every annotated fall was detected." if summary.total_incidents
            else "No falls were annotated in this corpus.")
        add("")
    else:
        add("| Video | Incident | Fall interval | Detections in clip | Notes |")
        add("|---|---|---|---:|---|")
        for row in missed:
            add(f"| `{row['video']}` | {row['incident_id']} | "
                f"{row['start_seconds']:.1f}s–{row['end_seconds']:.1f}s | "
                f"{row['detections_in_video']} | {row['notes'] or '—'} |")
        add("")
        add("A clip with detections but a missed fall means the alert landed outside the matching "
            "window — worth watching before assuming the detector failed to see anything.")
        add("")

    # --- sweep -------------------------------------------------------------
    add("## 7. Threshold sweep")
    add("")
    if not sweep_rows:
        add("Not run. Pass `--sweep` to evaluate the threshold grid.")
        add("")
    else:
        add("Each row is a full re-evaluation of the corpus. Inference runs once at the lowest "
            "confidence in the grid; higher thresholds filter the cached boxes and replay them "
            "through a fresh sustain gate, which is exactly equivalent to re-running the model "
            "(see `ml/validation/runner.py` for the argument).")
        add("")
        add("| Confidence | Sustained (s) | Incident recall | Precision | False alerts/h | "
            "Avg latency | Missed | False alerts |")
        add("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sweep_rows:
            marker = " ←current" if row.get("is_current") else ""
            add(
                f"| {row['min_confidence']:.2f}{marker} | {row['min_sustained_seconds']:.1f} | "
                f"{_pct(row['incident_recall'])} | {_pct(row['incident_precision'])} | "
                f"{_rate(row['false_alerts_per_hour'])} | {_secs(row['mean_latency_seconds'])} | "
                f"{row['missed']} | {row['false_alerts']} |"
            )
        add("")

    # --- limitations -------------------------------------------------------
    add("## 8. Limitations")
    add("")
    add(f"1. **Sample size.** {n_incidents} fall incident(s) over "
        f"{summary.hard_negative_hours:.2f} h of hard-negative footage. Confidence intervals on "
        "every figure above are wide; treat rankings between threshold settings as indicative, "
        "not decisive.")
    add("2. **Matching rule sensitivity.** Recall depends on the ±tolerance window "
        f"(−{PRE_TOLERANCE_SECONDS:.1f}s / +{POST_TOLERANCE_SECONDS:.0f}s). A different window gives "
        "different numbers; the rule is "
        "documented in `ml/validation/metrics.py` and should be quoted alongside any figure.")
    add("3. **Annotation subjectivity.** \"When did the fall start\" is fuzzy to within roughly a "
        "second, which directly perturbs latency.")
    add("4. **Corpus realism.** These clips are not a random sample of the target deployment's "
        "footage. Camera height, lens, lighting and occlusion in a real installation will differ.")
    add("5. **No per-subject ground truth.** Incidents are annotated as time intervals, so a clip "
        "where the detector alerts on the wrong person at the right moment scores as a success.")
    add("")

    add("## 9. Next data-collection priorities")
    add("")
    add("Ranked by how much each would tighten the numbers above:")
    add("")
    add(f"1. **More fall incidents.** {n_incidents} is not enough to separate threshold settings. "
        "~30 is the point where a single miss stops moving recall by several points.")
    add("2. **More hard-negative hours.** False alerts/hour is the figure an operator cares about "
        f"and it currently rests on {summary.hard_negative_hours:.2f} h. Hours, not minutes, are "
        "needed before a rate can be quoted.")
    if categories:
        worst = next(iter(categories))
        add(f"3. **More `{worst}` footage** — the activity producing the most false alerts here.")
    else:
        add("3. **Floor-level activity** (sit-ups, crouching, a child playing, lying on a sofa) — "
            "the failure modes the frame-level test split contains none of.")
    add("4. **Target-environment footage** at the real camera height and lighting.")
    add("")

    return "\n".join(lines) + "\n"
