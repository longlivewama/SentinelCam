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
    assert len(corpus.hard_negative_videos) == 3
    # Includes a clip with two incidents, so the multi-fall path is exercised.
    assert corpus.total_incidents == 3
