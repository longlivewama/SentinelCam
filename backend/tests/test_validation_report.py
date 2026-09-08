"""
Tests for the validation report renderer (ml/validation/report.py).

The renderer is the only part of the framework a human actually reads, and
it is the part that runs for the first time on the day someone finally has
footage - the worst possible moment to discover a formatting crash or, far
worse, a caveat that quietly went missing.

So what is tested here is not "does it produce Markdown". It is the set of
statements the report makes about its own trustworthiness:

  * an un-measurable metric renders as "n/a", never as 0
  * the sample-size banner matches the corpus that was actually evaluated
  * a missing clip or a duration mismatch is surfaced, not swallowed
  * the documented matching window in the Limitations section is the
    window the matcher actually used

That last one is why the tolerances are asserted against the constants
rather than against literals: a report that misstates its own matching
rule is worse than one that omits it, because it reads as authoritative.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.validation.annotations import (  # noqa: E402
    ActivityInterval,
    AnnotatedVideo,
    Corpus,
    Incident,
)
from ml.validation.metrics import (  # noqa: E402
    POST_TOLERANCE_SECONDS,
    PRE_TOLERANCE_SECONDS,
    Detection,
    evaluate,
)
from ml.validation.report import PRELIMINARY_INCIDENT_THRESHOLD, render_report  # noqa: E402


class FakeIdentity:
    """Stands in for runner.PipelineIdentity - the renderer only reads
    attributes off it, and building a real one needs the checkpoint."""

    model_path = "/repo/ml/exported/fall_detector_v1.pt"
    model_sha256 = "da38c55f24bb6a63" + "0" * 48
    class_names = {"0": "Fall"}
    mode = "model"
    stride = 5
    ultralytics_version = "8.3.0"


def fall_video(filename="f1.mp4", duration=60.0, incidents=()):
    return AnnotatedVideo(
        filename=filename, duration_seconds=duration, category="fall",
        incidents=list(incidents),
    )


def hard_negative(filename="h1.mp4", duration=1800.0, activity="sitting", intervals=()):
    return AnnotatedVideo(
        filename=filename, duration_seconds=duration, category="hard_negative",
        incidents=[], activity=activity, activity_intervals=list(intervals),
    )


def incident(start, end, id="i1", notes=""):
    return Incident(id=id, start_seconds=start, end_seconds=end, notes=notes)


def render(videos, detections=None, **kwargs):
    corpus = Corpus(videos=list(videos))
    summary = evaluate(corpus, detections or {}, 0.4, 0.6)
    return render_report(corpus, summary, FakeIdentity(), **kwargs)


# --- the caveats that must never go missing ------------------------------

def test_a_corpus_with_no_falls_says_recall_is_not_measured():
    """Rather than printing 0% recall, which reads as total failure."""
    text = render([hard_negative()])

    assert "No fall incidents in this corpus" in text
    assert "Recall is not measured" in text
    assert "| Incident recall | n/a |" in text


def test_a_small_corpus_leads_with_the_preliminary_banner():
    videos = [fall_video(f"f{i}.mp4", 60.0, [incident(10.0, 12.0)]) for i in range(3)]
    text = render(videos)

    assert "**Preliminary result — 3 annotated fall incident(s)" in text
    assert "not** evidence of production or clinical safety" in text
    # The banner sits above the results, not in a footnote.
    assert text.index("Preliminary result") < text.index("## 2. Incident-level results")


def test_the_banner_states_how_far_one_miss_moves_recall():
    """The number is derived from the corpus, so it cannot drift out of
    date the way a hardcoded caveat would."""
    videos = [fall_video(f"f{i}.mp4", 60.0, [incident(10.0, 12.0)]) for i in range(4)]
    text = render(videos)

    assert "moves recall by 25.0 percentage points" in text


def test_a_larger_corpus_still_warns_before_anything_is_quoted():
    n = PRELIMINARY_INCIDENT_THRESHOLD + 1
    videos = [fall_video(f"f{i}.mp4", 60.0, [incident(10.0, 12.0)]) for i in range(n)]
    text = render(videos)

    assert "Preliminary result" not in text
    assert "Still a small corpus by any clinical standard" in text


def test_the_limitations_quote_the_matching_window_actually_used():
    """Guards a report that misstates its own rule: the tolerances are
    read from metrics.py, not written out as literals here."""
    text = render([fall_video(incidents=[incident(10.0, 12.0)])])

    assert f"−{PRE_TOLERANCE_SECONDS:.1f}s" in text
    assert f"+{POST_TOLERANCE_SECONDS:.0f}s" in text


def test_zero_false_alerts_over_thin_footage_is_not_sold_as_a_low_rate():
    """The rule-of-three bound is the difference between 'we measured a
    low rate' and 'we did not measure enough to know'."""
    text = render([hard_negative(duration=600.0)])  # 10 minutes

    assert "No false alerts were produced" in text
    assert "rule-of-three upper bound" in text
    assert "it is not evidence of a low rate" in text


def test_a_large_hard_negative_corpus_drops_the_rule_of_three_caveat():
    text = render([hard_negative(duration=7200.0)])  # 2 hours
    assert "rule-of-three" not in text


# --- data-integrity warnings ---------------------------------------------

def test_missing_videos_are_reported_not_silently_excluded():
    text = render(
        [fall_video(incidents=[incident(10.0, 12.0)])],
        missing_videos=["absent_01.mp4", "absent_02.mp4"],
    )

    assert "2 annotated video(s) were not found on disk" in text
    assert "`absent_01.mp4`" in text
    assert "excluded from every figure below" in text


def test_duration_mismatches_are_surfaced_with_their_detail():
    text = render(
        [fall_video(incidents=[incident(10.0, 12.0)])],
        duration_mismatches=["f1.mp4: annotated 60.0s, decoded 42.0s"],
    )

    assert "Annotated duration disagrees with the decoded file" in text
    assert "annotated 60.0s, decoded 42.0s" in text


# --- results tables --------------------------------------------------------

def test_headline_metrics_carry_their_denominators():
    """'83% recall' means different things at n=6 and n=600."""
    videos = [
        fall_video("f1.mp4", 60.0, [incident(10.0, 12.0)]),
        fall_video("f2.mp4", 60.0, [incident(10.0, 12.0)]),
    ]
    text = render(videos, {"f1.mp4": [Detection(11.0, 0.9)]})

    assert "| Incident recall | 50.0% | 1 of 2 annotated falls |" in text
    assert "| Incident precision | 100.0% | 1 of 1 alerts |" in text


def test_the_per_video_table_lists_every_clip_with_its_status():
    videos = [
        fall_video("detected.mp4", 60.0, [incident(10.0, 12.0)]),
        fall_video("missed.mp4", 60.0, [incident(10.0, 12.0)]),
        hard_negative("noisy.mp4", 600.0, activity="exercising"),
        hard_negative("clean.mp4", 600.0),
    ]
    text = render(videos, {
        "detected.mp4": [Detection(11.0, 0.9)],
        "noisy.mp4": [Detection(100.0, 0.7)],
    })

    assert "| `detected.mp4` | fall |" in text
    assert "detected |" in text
    assert "missed fall |" in text
    assert "false alert |" in text
    assert "clean |" in text


def test_a_median_over_fewer_than_three_samples_says_so():
    """n/a with no explanation would look like a bug; the basis column
    has to distinguish 'not enough data' from 'nothing detected'."""
    videos = [fall_video(f"f{i}.mp4", 60.0, [incident(10.0, 12.0)]) for i in range(2)]
    text = render(videos, {"f0.mp4": [Detection(12.0, 0.9)], "f1.mp4": [Detection(12.0, 0.9)]})

    assert "| Median detection latency | n/a | n<3, not reported |" in text
    assert "| Mean detection latency | 2.00s | n=2 |" in text


def test_false_alerts_are_listed_individually_so_they_can_be_rewatched():
    video = hard_negative(
        "h1.mp4", 600.0, activity="exercising",
        intervals=[ActivityInterval(5.0, 300.0, "sit-ups on a floor mat")],
    )
    text = render([video], {"h1.mp4": [Detection(120.0, 0.83)]})

    assert "| exercising | 1 |" in text
    assert "### Every false alert" in text
    assert "sit-ups on a floor mat" in text
    assert "120.0s" in text
    assert "0.83" in text


def test_missed_falls_note_whether_the_clip_produced_anything_at_all():
    """A miss in a clip that did detect something is a matching-window
    problem, not a blind detector - a different thing to go and fix."""
    video = fall_video("f1.mp4", 200.0, [incident(150.0, 152.0, notes="slow slide from chair")])
    text = render([video], {"f1.mp4": [Detection(5.0, 0.9)]})

    assert "## 6. Missed falls" in text
    assert "slow slide from chair" in text
    assert "150.0s–152.0s" in text
    assert "the alert landed outside the matching window" in text


def test_a_clean_sweep_of_every_fall_says_so_explicitly():
    text = render(
        [fall_video("f1.mp4", 60.0, [incident(10.0, 12.0)])],
        {"f1.mp4": [Detection(11.0, 0.9)]},
    )
    assert "Every annotated fall was detected." in text


# --- threshold sweep --------------------------------------------------------

def test_without_the_sweep_the_section_says_how_to_get_it():
    text = render([hard_negative()])
    assert "Not run. Pass `--sweep`" in text


def test_the_sweep_table_marks_the_setting_currently_in_production():
    """So nobody reads a better-looking row without noticing it is not
    what is deployed."""
    rows = [
        {"min_confidence": 0.25, "min_sustained_seconds": 0.6, "incident_recall": 1.0,
         "incident_precision": 0.5, "false_alerts_per_hour": 4.0, "mean_latency_seconds": 1.2,
         "median_latency_seconds": None, "true_positives": 1, "missed": 0, "false_alerts": 1,
         "is_current": False},
        {"min_confidence": 0.40, "min_sustained_seconds": 0.6, "incident_recall": 1.0,
         "incident_precision": 1.0, "false_alerts_per_hour": 0.0, "mean_latency_seconds": 1.4,
         "median_latency_seconds": None, "true_positives": 1, "missed": 0, "false_alerts": 0,
         "is_current": True},
    ]
    text = render(
        [fall_video("f1.mp4", 60.0, [incident(10.0, 12.0)])],
        {"f1.mp4": [Detection(11.0, 0.9)]},
        sweep_rows=rows,
    )

    assert "| 0.40 ←current | 0.6 |" in text
    assert "| 0.25 | 0.6 |" in text
    assert "exactly equivalent to re-running the model" in text


def test_a_sweep_row_with_unmeasurable_precision_renders_na_not_zero():
    rows = [{
        "min_confidence": 0.70, "min_sustained_seconds": 1.5, "incident_recall": 0.0,
        "incident_precision": None, "false_alerts_per_hour": None,
        "mean_latency_seconds": None, "median_latency_seconds": None,
        "true_positives": 0, "missed": 1, "false_alerts": 0, "is_current": False,
    }]
    text = render(
        [fall_video("f1.mp4", 60.0, [incident(10.0, 12.0)])], {}, sweep_rows=rows,
    )

    assert "| 0.70 | 1.5 | 0.0% | n/a | n/a | n/a | 1 | 0 |" in text


# --- provenance -------------------------------------------------------------

def test_the_report_names_the_exact_checkpoint_that_produced_it():
    """A result that cannot be traced to a checkpoint is not a result."""
    text = render([hard_negative()])

    assert "fall_detector_v1.pt" in text
    assert FakeIdentity.model_sha256[:16] in text
    assert "| Classes | `{'0': 'Fall'}` |" in text
    assert "| Fall detection mode | `model` |" in text
    assert "| `FALL_DETECTOR_MIN_CONFIDENCE` | 0.4 |" in text
    assert "| `FALL_DETECTOR_MIN_SUSTAINED_SECONDS` | 0.6 |" in text


def test_the_report_records_whether_pose_was_run():
    assert "yes (as production)" in render([hard_negative()], run_pose=True)
    assert "no — ignored in model mode" in render([hard_negative()], run_pose=False)


def test_next_steps_point_at_the_activity_producing_the_most_false_alerts():
    videos = [
        hard_negative("h1.mp4", 600.0, activity="exercising"),
        hard_negative("h2.mp4", 600.0, activity="sitting"),
    ]
    text = render(videos, {
        "h1.mp4": [Detection(10.0, 0.8), Detection(200.0, 0.8)],
        "h2.mp4": [Detection(10.0, 0.8)],
    })

    assert "More `exercising` footage" in text


def test_an_empty_corpus_renders_without_crashing():
    """The state the framework ships in, before any footage exists."""
    text = render([])

    assert "# Fall detector — video-level validation report" in text
    assert "No fall incidents in this corpus" in text
