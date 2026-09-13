"""
Tests for the video-level validation engine (ml/validation/).

These use synthetic detector output on purpose. What is under test is the
part where a mistake silently flatters or damns the detector: the incident
matching rule, the latency arithmetic, and the guards that stop an
un-measurable metric being reported as zero. Running a real model here
would add minutes to the suite and test ultralytics rather than any of
that. The real-inference path is exercised by the runner itself (see
ml/validation/README.md), and its refusal-to-run guards are tested below
against the real settings object.

Lives under backend/tests because that is where pytest is configured and
where CI already runs a Python suite; ml/validation imports backend code,
so the dependency direction is the same one the runner uses.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.validation.annotations import (  # noqa: E402
    AnnotationError,
    AnnotatedVideo,
    Corpus,
    Incident,
    load_corpus,
)
from ml.validation.metrics import (  # noqa: E402
    POST_TOLERANCE_SECONDS,
    PRE_TOLERANCE_SECONDS,
    Detection,
    evaluate,
    match_video,
)


def fall_video(filename="fall_01.mp4", duration=30.0, incidents=(), activity="other"):
    return AnnotatedVideo(
        filename=filename,
        duration_seconds=duration,
        category="fall",
        incidents=list(incidents),
        activity=activity,
    )


def hard_negative(filename="hardneg_01.mp4", duration=60.0, activity="sitting", intervals=()):
    return AnnotatedVideo(
        filename=filename,
        duration_seconds=duration,
        category="hard_negative",
        incidents=[],
        activity=activity,
        activity_intervals=list(intervals),
    )


def incident(start, end, id="i1", notes=""):
    return Incident(id=id, start_seconds=start, end_seconds=end, notes=notes)


def detection(t, confidence=0.8):
    return Detection(video_time_seconds=t, confidence=confidence)


# --- matching: the core rule ---------------------------------------------

def test_detection_inside_the_fall_interval_matches():
    video = fall_video(incidents=[incident(10.0, 12.0)])
    result = match_video(video, [detection(11.0)])

    assert len(result.matched) == 1
    assert not result.missed
    assert not result.false_alerts
    assert result.status == "detected"


def test_detection_after_the_fall_still_matches_within_the_post_window():
    """A late alert is a correct alert - the harm is lying there unnoticed."""
    video = fall_video(duration=60.0, incidents=[incident(10.0, 12.0)])
    result = match_video(video, [detection(12.0 + POST_TOLERANCE_SECONDS - 1)])

    assert len(result.matched) == 1
    assert not result.false_alerts


def test_detection_beyond_the_post_window_is_a_false_alert_not_a_late_match():
    """Bounded, so an unrelated detection later in a long clip is not
    silently credited to an earlier fall."""
    video = fall_video(duration=120.0, incidents=[incident(10.0, 12.0)])
    result = match_video(video, [detection(12.0 + POST_TOLERANCE_SECONDS + 5)])

    assert not result.matched
    assert len(result.missed) == 1
    assert len(result.false_alerts) == 1


def test_detection_just_before_the_fall_matches_within_annotation_jitter():
    video = fall_video(incidents=[incident(10.0, 12.0)])
    result = match_video(video, [detection(10.0 - PRE_TOLERANCE_SECONDS + 0.1)])
    assert len(result.matched) == 1


def test_detection_well_before_the_fall_is_a_false_alert():
    """Firing before the fall began is coincidence, not early detection -
    the evidence did not exist yet."""
    video = fall_video(incidents=[incident(10.0, 12.0)])
    result = match_video(video, [detection(3.0)])

    assert not result.matched
    assert len(result.missed) == 1
    assert len(result.false_alerts) == 1


# --- missed falls and false alerts ---------------------------------------

def test_a_fall_with_no_detection_is_missed():
    video = fall_video(incidents=[incident(10.0, 12.0)])
    result = match_video(video, [])

    assert result.missed == [video.incidents[0]]
    assert result.status == "missed fall"


def test_a_hard_negative_with_a_detection_is_a_false_alert():
    video = hard_negative()
    result = match_video(video, [detection(20.0)])

    assert len(result.false_alerts) == 1
    assert not result.matched
    assert result.status == "false alert"


def test_a_clean_hard_negative_reports_clean():
    result = match_video(hard_negative(), [])
    assert result.status == "clean"
    assert not result.false_alerts


def test_a_video_can_be_both_missed_and_false_alerting():
    video = fall_video(duration=200.0, incidents=[incident(100.0, 102.0)])
    result = match_video(video, [detection(5.0)])  # nowhere near the fall

    assert len(result.missed) == 1
    assert len(result.false_alerts) == 1
    assert result.status == "missed + false alert"


# --- multiple incidents and overlapping predictions ----------------------

def test_two_falls_in_one_clip_are_matched_independently():
    video = fall_video(
        duration=120.0,
        incidents=[incident(10.0, 12.0, id="a"), incident(80.0, 82.0, id="b")],
    )
    result = match_video(video, [detection(11.0), detection(81.0)])

    assert {m.incident.id for m in result.matched} == {"a", "b"}
    assert not result.missed
    assert not result.false_alerts


def test_one_of_two_falls_detected_gives_partial_recall():
    video = fall_video(
        duration=200.0,
        incidents=[incident(10.0, 12.0, id="a"), incident(150.0, 152.0, id="b")],
    )
    result = match_video(video, [detection(11.0)])

    assert [m.incident.id for m in result.matched] == ["a"]
    assert [i.id for i in result.missed] == ["b"]


def test_a_repeat_detection_of_the_same_fall_is_a_duplicate_not_a_false_alert():
    """The debounce re-firing on a person still on the ground is a
    nuisance, not a wrong alert - counting it as a false positive would
    understate precision for behaviour that is arguably correct."""
    video = fall_video(duration=60.0, incidents=[incident(10.0, 12.0)])
    result = match_video(video, [detection(11.0), detection(22.0), detection(32.0)])

    assert len(result.matched) == 1
    assert len(result.duplicate_detections) == 2
    assert not result.false_alerts


def test_a_prediction_is_never_credited_to_two_overlapping_incidents():
    video = fall_video(
        duration=60.0,
        incidents=[incident(10.0, 20.0, id="a"), incident(12.0, 22.0, id="b")],
    )
    result = match_video(video, [detection(15.0)])

    # One detection cannot satisfy both annotated falls.
    assert len(result.matched) == 1
    assert len(result.missed) == 1


def test_the_earliest_eligible_detection_is_the_one_matched():
    """Latency is what an operator waits for the FIRST alert."""
    video = fall_video(duration=60.0, incidents=[incident(10.0, 12.0)])
    result = match_video(video, [detection(25.0), detection(11.0)])

    assert result.matched[0].detection.video_time_seconds == 11.0
    assert result.matched[0].latency_seconds == pytest.approx(1.0)


# --- latency ---------------------------------------------------------------

def test_latency_is_measured_from_the_start_of_the_fall():
    video = fall_video(duration=60.0, incidents=[incident(10.0, 12.0)])
    result = match_video(video, [detection(13.5)])
    assert result.matched[0].latency_seconds == pytest.approx(3.5)


def test_a_detection_inside_the_pre_window_reports_negative_latency_unclamped():
    """Not clamped to zero: a systematically negative latency means the
    annotations are offset, and that must stay visible."""
    video = fall_video(incidents=[incident(10.0, 12.0)])
    result = match_video(video, [detection(9.8)])
    assert result.matched[0].latency_seconds == pytest.approx(-0.2)


def test_median_latency_needs_at_least_three_samples():
    videos, detections = [], {}
    for i in range(2):
        name = f"fall_{i}.mp4"
        videos.append(fall_video(filename=name, duration=60.0, incidents=[incident(10.0, 12.0)]))
        detections[name] = [detection(12.0)]

    summary = evaluate(Corpus(videos=videos), detections, 0.4, 0.6)
    assert summary.mean_latency_seconds == pytest.approx(2.0)
    assert summary.median_latency_seconds is None  # n=2

    name = "fall_2.mp4"
    videos.append(fall_video(filename=name, duration=60.0, incidents=[incident(10.0, 12.0)]))
    detections[name] = [detection(14.0)]
    summary = evaluate(Corpus(videos=videos), detections, 0.4, 0.6)
    assert summary.median_latency_seconds == pytest.approx(2.0)


# --- corpus-level aggregation ----------------------------------------------

def test_corpus_metrics_across_multiple_videos():
    videos = [
        fall_video("f1.mp4", 60.0, [incident(10.0, 12.0)]),
        fall_video("f2.mp4", 60.0, [incident(20.0, 22.0)]),
        hard_negative("h1.mp4", 1800.0, activity="exercising"),
        hard_negative("h2.mp4", 1800.0, activity="sitting"),
    ]
    detections = {
        "f1.mp4": [detection(11.0)],          # matched
        "f2.mp4": [],                          # missed
        "h1.mp4": [detection(500.0)],          # false alert
        "h2.mp4": [],
    }
    summary = evaluate(Corpus(videos=videos), detections, 0.4, 0.6)

    assert summary.true_positives == 1
    assert summary.false_negatives == 1
    assert summary.false_positives == 1
    assert summary.incident_recall == pytest.approx(0.5)
    assert summary.incident_precision == pytest.approx(0.5)
    # 1 false alert over 3600s of hard-negative footage = 1.0/hour.
    assert summary.hard_negative_hours == pytest.approx(1.0)
    assert summary.false_alerts_per_hour == pytest.approx(1.0)


def test_false_alert_rate_is_quoted_over_hard_negative_footage_only():
    """Including fall clips would dilute the rate with footage that is
    supposed to trigger an alert."""
    videos = [
        fall_video("f1.mp4", 3600.0, [incident(10.0, 12.0)]),
        hard_negative("h1.mp4", 3600.0, activity="sitting"),
    ]
    detections = {"f1.mp4": [detection(11.0)], "h1.mp4": [detection(100.0), detection(200.0)]}
    summary = evaluate(Corpus(videos=videos), detections, 0.4, 0.6)

    assert summary.hard_negative_hours == pytest.approx(1.0)
    assert summary.false_alerts_per_hour == pytest.approx(2.0)
    # Whole-corpus rate is half that, and is reported separately.
    assert summary.all_false_alerts_per_hour == pytest.approx(1.0)


def test_a_video_absent_from_the_detection_map_counts_as_no_detections():
    videos = [fall_video("f1.mp4", 60.0, [incident(10.0, 12.0)])]
    summary = evaluate(Corpus(videos=videos), {}, 0.4, 0.6)
    assert summary.false_negatives == 1
    assert summary.incident_recall == pytest.approx(0.0)


# --- un-measurable metrics must not read as zero -------------------------

def test_recall_is_none_not_zero_when_there_are_no_falls_to_find():
    """0% and 'not measured' look identical in a table and mean opposite
    things."""
    summary = evaluate(Corpus(videos=[hard_negative()]), {}, 0.4, 0.6)
    assert summary.incident_recall is None
    assert summary.total_incidents == 0


def test_precision_is_none_when_nothing_was_predicted():
    summary = evaluate(Corpus(videos=[hard_negative()]), {}, 0.4, 0.6)
    assert summary.incident_precision is None


def test_false_alert_rate_is_none_with_no_hard_negative_footage():
    videos = [fall_video("f1.mp4", 60.0, [incident(10.0, 12.0)])]
    summary = evaluate(Corpus(videos=videos), {"f1.mp4": [detection(11.0)]}, 0.4, 0.6)
    assert summary.hard_negative_hours == 0
    assert summary.false_alerts_per_hour is None


def test_latency_is_none_with_no_matches():
    summary = evaluate(Corpus(videos=[hard_negative()]), {}, 0.4, 0.6)
    assert summary.mean_latency_seconds is None
    assert summary.max_latency_seconds is None


def test_an_empty_corpus_produces_zeros_not_a_crash():
    summary = evaluate(Corpus(videos=[]), {}, 0.4, 0.6)
    assert summary.total_hours == 0
    assert summary.incident_recall is None
    assert summary.false_alerts_per_hour is None


# --- false-positive attribution -------------------------------------------

def test_false_alerts_are_grouped_by_the_activity_that_produced_them():
    videos = [
        hard_negative("h1.mp4", 600.0, activity="exercising"),
        hard_negative("h2.mp4", 600.0, activity="exercising"),
        hard_negative("h3.mp4", 600.0, activity="sitting"),
    ]
    detections = {
        "h1.mp4": [detection(10.0), detection(100.0)],
        "h2.mp4": [detection(10.0)],
        "h3.mp4": [detection(10.0)],
    }
    summary = evaluate(Corpus(videos=videos), detections, 0.4, 0.6)

    assert summary.false_alert_categories() == {"exercising": 3, "sitting": 1}


def test_a_false_alert_is_attributed_to_the_labelled_moment_when_one_covers_it():
    from ml.validation.annotations import ActivityInterval

    video = hard_negative(
        "h1.mp4", 60.0, activity="exercising",
        intervals=[ActivityInterval(start_seconds=5.0, end_seconds=35.0, label="sit-ups on a mat")],
    )
    summary = evaluate(Corpus(videos=[video]), {"h1.mp4": [detection(20.0)]}, 0.4, 0.6)

    detail = summary.false_alert_details()[0]
    assert detail["activity"] == "exercising"
    assert detail["activity_at_moment"] == "sit-ups on a mat"
    assert detail["video_time_seconds"] == 20.0


def test_missed_fall_details_include_whether_the_clip_produced_anything():
    """A miss in a clip that did produce detections means the alert landed
    outside the window - a different problem from seeing nothing at all."""
    video = fall_video("f1.mp4", 200.0, [incident(150.0, 152.0)])
    summary = evaluate(Corpus(videos=[video]), {"f1.mp4": [detection(5.0)]}, 0.4, 0.6)

    detail = summary.missed_fall_details()[0]
    assert detail["detections_in_video"] == 1
    assert detail["start_seconds"] == 150.0


# --- annotation loading ----------------------------------------------------

def _write(tmp_path, payload):
    import json
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload))
    return path


def test_loading_a_well_formed_corpus(tmp_path):
    path = _write(tmp_path, {"videos": [
        {"filename": "f1.mp4", "duration_seconds": 30.0, "category": "fall",
         "incidents": [{"start_seconds": 5.0, "end_seconds": 7.0}]},
        {"filename": "h1.mp4", "duration_seconds": 60.0, "category": "hard_negative",
         "activity": "sitting", "incidents": []},
    ]})
    corpus = load_corpus(path)

    assert len(corpus.videos) == 2
    assert corpus.total_incidents == 1
    assert corpus.total_duration_seconds == 90.0
    assert corpus.composition() == {"sitting": 1}
    # id is generated when omitted, so incidents are always addressable.
    assert corpus.videos[0].incidents[0].id == "f1-1"


def test_a_fall_clip_with_no_annotated_incident_is_rejected(tmp_path):
    """Would otherwise contribute a guaranteed unmatched detection and
    silently depress precision."""
    path = _write(tmp_path, {"videos": [
        {"filename": "f1.mp4", "duration_seconds": 30.0, "category": "fall", "incidents": []},
    ]})
    with pytest.raises(AnnotationError, match="no incidents are annotated"):
        load_corpus(path)


def test_a_hard_negative_with_an_incident_is_rejected(tmp_path):
    """Would otherwise count a real fall as ordinary footage."""
    path = _write(tmp_path, {"videos": [
        {"filename": "h1.mp4", "duration_seconds": 30.0, "category": "hard_negative",
         "incidents": [{"start_seconds": 5.0, "end_seconds": 7.0}]},
    ]})
    with pytest.raises(AnnotationError, match="belongs in category 'fall'"):
        load_corpus(path)


def test_an_inverted_interval_is_rejected(tmp_path):
    path = _write(tmp_path, {"videos": [
        {"filename": "f1.mp4", "duration_seconds": 30.0, "category": "fall",
         "incidents": [{"start_seconds": 9.0, "end_seconds": 5.0}]},
    ]})
    with pytest.raises(AnnotationError, match="must be after"):
        load_corpus(path)


def test_an_incident_past_the_end_of_its_clip_is_rejected(tmp_path):
    """Usually means the times came from a different cut of the video,
    which would make the fall unmatchable and look like a model failure."""
    path = _write(tmp_path, {"videos": [
        {"filename": "f1.mp4", "duration_seconds": 30.0, "category": "fall",
         "incidents": [{"start_seconds": 95.0, "end_seconds": 97.0}]},
    ]})
    with pytest.raises(AnnotationError, match="past the video's duration"):
        load_corpus(path)


def test_zero_duration_is_rejected(tmp_path):
    path = _write(tmp_path, {"videos": [
        {"filename": "h1.mp4", "duration_seconds": 0, "category": "hard_negative", "incidents": []},
    ]})
    with pytest.raises(AnnotationError, match="must be positive"):
        load_corpus(path)


def test_duplicate_filenames_are_rejected(tmp_path):
    path = _write(tmp_path, {"videos": [
        {"filename": "h1.mp4", "duration_seconds": 10, "category": "hard_negative", "incidents": []},
        {"filename": "h1.mp4", "duration_seconds": 20, "category": "hard_negative", "incidents": []},
    ]})
    with pytest.raises(AnnotationError, match="duplicate filename"):
        load_corpus(path)


def test_an_unknown_category_is_rejected(tmp_path):
    path = _write(tmp_path, {"videos": [
        {"filename": "x.mp4", "duration_seconds": 10, "category": "maybe", "incidents": []},
    ]})
    with pytest.raises(AnnotationError, match="category"):
        load_corpus(path)


def test_a_missing_annotation_file_explains_where_to_look(tmp_path):
    with pytest.raises(AnnotationError, match="README"):
        load_corpus(tmp_path / "nope.json")


def test_the_shipped_example_template_is_valid(tmp_path):
    """The template is the first thing anyone copies; if it does not load,
    the framework is unusable on first contact."""
    import json
    example = REPO_ROOT / "ml" / "validation" / "annotations.example.json"
    payload = json.loads(example.read_text())
    payload.pop("_comment", None)
    corpus = load_corpus(_write(tmp_path, payload))

    assert len(corpus.fall_videos) == 2
    assert len(corpus.hard_negative_videos) == 4
    # Includes a clip with two incidents, so the multi-fall path is exercised.
    assert corpus.total_incidents == 3
    # And it demonstrates every optional field, since a template nobody can
    # copy the taxonomy from is a template that produces untyped corpora.
    assert corpus.fall_taxonomy()["direction"] == {"backward": 1, "forward": 1, "sideways": 1}
    assert corpus.fall_taxonomy()["speed"] == {"fast": 2, "slow": 1}
    assert all(v.conditions for v in corpus.videos), "every example clip shows a conditions block"


# --- F1 -------------------------------------------------------------------

def test_f1_is_the_harmonic_mean_of_precision_and_recall():
    from ml.validation.metrics import f1_score

    assert f1_score(1.0, 1.0) == pytest.approx(1.0)
    assert f1_score(0.5, 0.5) == pytest.approx(0.5)
    assert f1_score(1.0, 0.5) == pytest.approx(2 / 3)


def test_f1_is_unmeasurable_rather_than_zero_when_an_input_is():
    """Same rule as precision and recall: a corpus with no falls has no F1,
    and printing 0.0 would read as total failure rather than not measured."""
    from ml.validation.metrics import f1_score

    assert f1_score(None, 0.9) is None
    assert f1_score(0.9, None) is None
    # But an honest zero stays zero.
    assert f1_score(0.0, 0.0) == 0.0


def test_summary_f1_reflects_the_corpus():
    video = fall_video(incidents=[Incident(id="a", start_seconds=5.0, end_seconds=6.0)])
    negative = hard_negative()
    summary = evaluate(
        Corpus(videos=[video, negative]),
        {
            video.filename: [Detection(video_time_seconds=5.5, confidence=0.9)],
            negative.filename: [Detection(video_time_seconds=10.0, confidence=0.7)],
        },
        min_confidence=0.4, min_sustained_seconds=0.6,
    )

    # 1 TP, 0 FN, 1 FP -> recall 1.0, precision 0.5
    assert summary.incident_recall == pytest.approx(1.0)
    assert summary.incident_precision == pytest.approx(0.5)
    assert summary.incident_f1 == pytest.approx(2 / 3)


def test_f1_is_unmeasurable_on_a_pure_hard_negative_corpus():
    negative = hard_negative()
    summary = evaluate(
        Corpus(videos=[negative]), {negative.filename: []},
        min_confidence=0.4, min_sustained_seconds=0.6,
    )

    assert summary.incident_recall is None
    assert summary.incident_f1 is None


# --- confidence distributions --------------------------------------------

def _distribution_corpus():
    video = fall_video(incidents=[Incident(id="a", start_seconds=5.0, end_seconds=6.0)])
    negative = hard_negative()
    return evaluate(
        Corpus(videos=[video, negative]),
        {
            video.filename: [Detection(video_time_seconds=5.5, confidence=0.88)],
            negative.filename: [
                Detection(video_time_seconds=10.0, confidence=0.52),
                Detection(video_time_seconds=20.0, confidence=0.61),
            ],
        },
        min_confidence=0.4, min_sustained_seconds=0.6,
    )


def test_confidence_distribution_separates_true_detections_from_false_alerts():
    """The diagnostic the sweep cannot give: if the two overlap heavily, no
    threshold separates them and the fix is not the threshold."""
    distribution = _distribution_corpus().confidence_distribution()

    assert distribution["true_positives"]["n"] == 1
    assert distribution["true_positives"]["max"] == pytest.approx(0.88)
    assert distribution["false_alerts"]["n"] == 2
    assert distribution["false_alerts"]["max"] == pytest.approx(0.61)


def test_confidence_histogram_buckets_by_the_documented_edges():
    histogram = _distribution_corpus().confidence_distribution()["false_alerts"]["histogram"]

    assert histogram["0.50-0.60"] == 1
    assert histogram["0.60-0.70"] == 1
    assert sum(histogram.values()) == 2


def test_a_confidence_of_exactly_one_lands_in_the_top_bucket():
    """Half-open buckets everywhere except the top, or a perfect score
    would fall off the end and vanish from the histogram."""
    video = fall_video(incidents=[Incident(id="a", start_seconds=5.0, end_seconds=6.0)])
    summary = evaluate(
        Corpus(videos=[video]),
        {video.filename: [Detection(video_time_seconds=5.5, confidence=1.0)]},
        min_confidence=0.4, min_sustained_seconds=0.6,
    )

    histogram = summary.confidence_distribution()["true_positives"]["histogram"]
    assert histogram["0.90-1.00"] == 1
    assert sum(histogram.values()) == 1


def test_an_empty_distribution_reports_none_rather_than_zero():
    negative = hard_negative()
    summary = evaluate(
        Corpus(videos=[negative]), {negative.filename: []},
        min_confidence=0.4, min_sustained_seconds=0.6,
    )

    stats = summary.confidence_distribution()["true_positives"]
    assert stats["n"] == 0
    assert stats["mean"] is None


# --- per-condition and per-taxonomy breakdowns ----------------------------

def _conditioned(filename, lighting, incidents=(), category="fall", activity="other"):
    return AnnotatedVideo(
        filename=filename,
        duration_seconds=60.0,
        category=category,
        incidents=list(incidents),
        activity=activity,
        conditions={"lighting": lighting},
    )


def test_results_split_by_a_recording_condition():
    """A detector that works in daylight and fails at night has the same
    headline recall as one that works everywhere; only this tells them
    apart."""
    day = _conditioned("day.mp4", "daylight", [Incident(id="d", start_seconds=5.0, end_seconds=6.0)])
    night = _conditioned("night.mp4", "night_ir", [Incident(id="n", start_seconds=5.0, end_seconds=6.0)])
    summary = evaluate(
        Corpus(videos=[day, night]),
        {
            day.filename: [Detection(video_time_seconds=5.5, confidence=0.9)],
            night.filename: [],  # missed
        },
        min_confidence=0.4, min_sustained_seconds=0.6,
    )

    by_lighting = {g.label: g for g in summary.by_condition("lighting")}
    assert by_lighting["daylight"].recall == pytest.approx(1.0)
    assert by_lighting["night_ir"].recall == pytest.approx(0.0)
    assert summary.incident_recall == pytest.approx(0.5), "and the average hides both"


def test_unlabelled_conditions_are_grouped_visibly_not_dropped():
    labelled = _conditioned("day.mp4", "daylight", [Incident(id="d", start_seconds=5.0, end_seconds=6.0)])
    unlabelled = fall_video("plain.mp4", incidents=[Incident(id="p", start_seconds=5.0, end_seconds=6.0)])
    summary = evaluate(
        Corpus(videos=[labelled, unlabelled]),
        {labelled.filename: [], unlabelled.filename: []},
        min_confidence=0.4, min_sustained_seconds=0.6,
    )

    labels = {g.label for g in summary.by_condition("lighting")}
    assert labels == {"daylight", "unlabelled"}


def test_recall_splits_by_fall_direction():
    video = fall_video(incidents=[
        Incident(id="fwd", start_seconds=5.0, end_seconds=6.0, direction="forward"),
        Incident(id="back", start_seconds=20.0, end_seconds=21.0, direction="backward"),
    ])
    summary = evaluate(
        Corpus(videos=[video]),
        {video.filename: [Detection(video_time_seconds=5.5, confidence=0.9)]},
        min_confidence=0.4, min_sustained_seconds=0.6,
    )

    by_direction = {g.label: g for g in summary.by_fall_direction()}
    assert by_direction["forward"].recall == pytest.approx(1.0)
    assert by_direction["backward"].recall == pytest.approx(0.0)


def test_fall_type_groups_do_not_claim_a_precision():
    """A false alert cannot be attributed to a fall direction, because no
    fall happened - so precision within these groups is undefined, not 100%."""
    video = fall_video(incidents=[
        Incident(id="fwd", start_seconds=5.0, end_seconds=6.0, direction="forward"),
    ])
    summary = evaluate(
        Corpus(videos=[video]),
        {video.filename: [Detection(video_time_seconds=5.5, confidence=0.9)]},
        min_confidence=0.4, min_sustained_seconds=0.6,
    )

    group = summary.by_fall_direction()[0]
    assert group.false_positives == 0
    assert group.recall == pytest.approx(1.0)


def test_false_alert_rate_per_activity_is_reported_per_hour():
    negative = hard_negative("exercise.mp4", duration=1800.0, activity="exercising")
    summary = evaluate(
        Corpus(videos=[negative]),
        {negative.filename: [Detection(video_time_seconds=100.0, confidence=0.7)]},
        min_confidence=0.4, min_sustained_seconds=0.6,
    )

    group = {g.label: g for g in summary.by_hard_negative_activity()}["exercising"]
    # One false alert in half an hour is two per hour.
    assert group.false_alerts_per_hour == pytest.approx(2.0)


# --- the corpus taxonomy and its coverage check ---------------------------

def _write_corpus(tmp_path, videos):
    import json
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps({"videos": videos}))
    return path


def test_fall_direction_and_speed_are_parsed_and_normalised(tmp_path):
    path = _write_corpus(tmp_path, [{
        "filename": "f.mp4", "duration_seconds": 30.0, "category": "fall",
        "incidents": [{"start_seconds": 5.0, "end_seconds": 6.0,
                       "direction": "Forward", "speed": " FAST "}],
    }])

    corpus = load_corpus(path)

    assert corpus.videos[0].incidents[0].direction == "forward"
    assert corpus.videos[0].incidents[0].speed == "fast"


@pytest.mark.parametrize("field,value", [("direction", "backwards"), ("speed", "medium")])
def test_an_unknown_fall_direction_or_speed_is_rejected(tmp_path, field, value):
    """These are enumerations, not labels. A typo would silently become a
    bucket of its own, and the coverage check - whose whole job is to say
    "you have no backward falls yet" - would report that you do."""
    path = _write_corpus(tmp_path, [{
        "filename": "f.mp4", "duration_seconds": 30.0, "category": "fall",
        "incidents": [{"start_seconds": 5.0, "end_seconds": 6.0, field: value}],
    }])

    with pytest.raises(AnnotationError, match=field):
        load_corpus(path)


def test_conditions_are_parsed_and_normalised(tmp_path):
    good = _write_corpus(tmp_path, [{
        "filename": "f.mp4", "duration_seconds": 30.0, "category": "fall",
        "incidents": [{"start_seconds": 5.0, "end_seconds": 6.0}],
        "conditions": {"lighting": "Night_IR", "distance": "far"},
    }])
    corpus = load_corpus(good)
    assert corpus.videos[0].conditions == {"lighting": "night_ir", "distance": "far"}
    # Absent dimensions read as "unlabelled", never None, so a report can
    # group by them without special-casing.
    assert corpus.videos[0].condition("occlusion") == "unlabelled"


def test_a_misspelled_condition_key_is_rejected_rather_than_ignored(tmp_path):
    """Ignoring it would make the clip look labelled when it is not - the
    dimension would simply vanish from the per-condition breakdown."""
    path = _write_corpus(tmp_path, [{
        "filename": "f.mp4", "duration_seconds": 30.0, "category": "fall",
        "incidents": [{"start_seconds": 5.0, "end_seconds": 6.0}],
        "conditions": {"lightning": "daylight"},
    }])

    with pytest.raises(AnnotationError, match="lightning"):
        load_corpus(path)


def test_missing_coverage_names_what_the_corpus_still_lacks():
    """The readiness check. An empty result is the only honest basis for
    dropping the "preliminary" framing from a report."""
    corpus = Corpus(videos=[
        AnnotatedVideo(
            filename="f.mp4", duration_seconds=30.0, category="fall",
            incidents=[Incident(id="a", start_seconds=5.0, end_seconds=6.0,
                                direction="forward", speed="fast")],
            conditions={"lighting": "daylight"},
        ),
    ])

    missing = corpus.missing_coverage()

    assert missing["fall_direction"] == ["backward", "sideways"]
    assert missing["fall_speed"] == ["slow"]
    assert "sleeping" in missing["hard_negative_activity"]
    # One labelled value is no better than none for a breakdown - there is
    # nothing to compare it against.
    assert "lighting" in missing["condition_variation"]


def test_a_corpus_meeting_the_specification_reports_nothing_missing():
    from ml.validation.annotations import (
        CONDITION_DIMENSIONS,
        FALL_DIRECTIONS,
        FALL_SPEEDS,
        REQUIRED_HARD_NEGATIVES,
    )

    falls = [
        AnnotatedVideo(
            filename=f"fall_{direction}_{speed}.mp4", duration_seconds=30.0, category="fall",
            incidents=[Incident(id=f"{direction}-{speed}", start_seconds=5.0, end_seconds=6.0,
                                direction=direction, speed=speed)],
            conditions={d: "a" for d in CONDITION_DIMENSIONS},
        )
        for direction in FALL_DIRECTIONS for speed in FALL_SPEEDS
    ]
    negatives = [
        AnnotatedVideo(
            filename=f"hn_{activity}.mp4", duration_seconds=60.0, category="hard_negative",
            incidents=[], activity=activity,
            conditions={d: "b" for d in CONDITION_DIMENSIONS},
        )
        for activity in REQUIRED_HARD_NEGATIVES
    ]

    assert Corpus(videos=falls + negatives).missing_coverage() == {}


def test_condition_coverage_counts_clips_per_value():
    corpus = Corpus(videos=[
        AnnotatedVideo(filename="a.mp4", duration_seconds=30.0, category="hard_negative",
                       incidents=[], conditions={"lighting": "daylight"}),
        AnnotatedVideo(filename="b.mp4", duration_seconds=30.0, category="hard_negative",
                       incidents=[], conditions={"lighting": "daylight"}),
        AnnotatedVideo(filename="c.mp4", duration_seconds=30.0, category="hard_negative",
                       incidents=[], conditions={"lighting": "dim"}),
    ])

    assert corpus.condition_coverage()["lighting"] == {"daylight": 2, "dim": 1}
