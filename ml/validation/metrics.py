"""
Incident matching and operational metrics.

Deliberately pure: this module takes annotated incidents and predicted
detections as plain data and returns numbers. No model, no video, no
filesystem. That is what makes the matching rule - the part where a
mistake silently flatters or damns the detector - unit-testable against
hand-written cases.

THE MATCHING RULE
-----------------
A prediction is a single instant (the moment the sustained-duration gate
fired), not an interval. Ground truth is an interval [start, end]. So the
rule is a tolerance window around the annotated fall:

    a prediction at time t matches incident [start, end] if
        start - PRE_TOLERANCE <= t <= end + POST_TOLERANCE

with PRE_TOLERANCE small and POST_TOLERANCE generous. The asymmetry is
deliberate and is the crux of the design:

  * Firing BEFORE the fall begins is not early detection, it is a
    coincidence - the detector cannot have seen evidence that does not
    exist yet. A small pre-window (0.5s) absorbs annotation jitter, since
    "when exactly did the fall start" is genuinely fuzzy to a human
    scrubbing a timeline, and nothing more.

  * Firing well AFTER the fall is still a correct detection. The whole
    product premise is that the harm comes from lying there unnoticed,
    so an alert 15 seconds after someone goes down is a success, just a
    slower one. The post-window is generous (30s) because a person who
    has fallen stays fallen; it is bounded rather than infinite so that a
    detection in a *later, unrelated* part of a long clip is not silently
    credited to an earlier fall.

Matching is greedy by time and one-to-one:

  * Each ground-truth incident can be matched by at most one prediction -
    the EARLIEST eligible one, because latency is measured from the first
    alert an operator would actually see.
  * Each prediction matches at most one incident, so a single detection
    inside two overlapping annotated falls cannot be counted twice.
  * Predictions still unmatched after that pass are false alerts.
    Incidents still unmatched are missed falls.

An extra detection during a fall the detector already reported is NOT a
false alert - it is the same incident re-firing after the debounce window,
which is a duplicate-alert nuisance, not a wrong one. It is counted and
reported separately as `duplicate_detections` so the number is visible
without corrupting the precision figure.

LATENCY
-------
Measured from the START of the annotated fall to the matched prediction:
`latency = prediction_time - incident.start_seconds`. It can be slightly
negative when a prediction lands inside the pre-tolerance window; those
are reported as-is rather than clamped, because a systematically negative
latency would mean the annotations are offset and that should be visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence

from .annotations import CONDITION_DIMENSIONS, AnnotatedVideo, Corpus, Incident

# See the module docstring for why these are asymmetric.
PRE_TOLERANCE_SECONDS = 0.5
POST_TOLERANCE_SECONDS = 30.0

SECONDS_PER_HOUR = 3600.0

# Buckets for the confidence histogram. The production floor is 0.40, so
# the edges straddle it: what matters operationally is whether true and
# false detections separate around the threshold, not their shape at the
# extremes.
CONFIDENCE_BINS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def f1_score(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    """Harmonic mean, or None when either input is un-measurable.

    None rather than 0.0, for the same reason precision and recall are:
    a corpus with no falls in it has no F1, and printing 0.0 would read as
    total failure rather than "not measured"."""
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class GroupResult:
    """Incident-level counts for one slice of the corpus - one lighting
    condition, one fall direction, one activity. Carries its own n so a
    reader cannot quote a rate without seeing how thin it is."""

    label: str
    true_positives: int
    false_negatives: int
    false_positives: int
    videos: int
    duration_seconds: float

    @property
    def total_incidents(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def recall(self) -> Optional[float]:
        if self.total_incidents == 0:
            return None
        return self.true_positives / self.total_incidents

    @property
    def precision(self) -> Optional[float]:
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return None
        return self.true_positives / denominator

    @property
    def f1(self) -> Optional[float]:
        return f1_score(self.precision, self.recall)

    @property
    def false_alerts_per_hour(self) -> Optional[float]:
        if self.duration_seconds <= 0:
            return None
        return self.false_positives / (self.duration_seconds / SECONDS_PER_HOUR)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "videos": self.videos,
            "duration_seconds": round(self.duration_seconds, 2),
            "true_positives": self.true_positives,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "total_incidents": self.total_incidents,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "false_alerts_per_hour": self.false_alerts_per_hour,
        }


@dataclass(frozen=True)
class Detection:
    """One fall event the pipeline emitted, on the video's timeline."""

    video_time_seconds: float
    confidence: float
    detector: str = "model"
    track_id: Optional[int] = None
    sustained_seconds: Optional[float] = None


@dataclass
class MatchedIncident:
    incident: Incident
    detection: Detection

    @property
    def latency_seconds(self) -> float:
        return self.detection.video_time_seconds - self.incident.start_seconds


@dataclass
class VideoResult:
    video: AnnotatedVideo
    detections: List[Detection]
    matched: List[MatchedIncident] = field(default_factory=list)
    missed: List[Incident] = field(default_factory=list)
    false_alerts: List[Detection] = field(default_factory=list)
    duplicate_detections: List[Detection] = field(default_factory=list)

    @property
    def ground_truth_falls(self) -> int:
        return len(self.video.incidents)

    @property
    def detected_falls(self) -> int:
        return len(self.matched)

    @property
    def status(self) -> str:
        """A one-word verdict for the per-video table."""
        if self.missed and self.false_alerts:
            return "missed + false alert"
        if self.missed:
            return "missed fall"
        if self.false_alerts:
            return "false alert"
        if self.ground_truth_falls:
            return "detected"
        return "clean"

    @property
    def latencies(self) -> List[float]:
        return [m.latency_seconds for m in self.matched]


@dataclass
class EvaluationSummary:
    results: List[VideoResult]
    # Echoed so a report can never be read without the configuration that
    # produced it.
    min_confidence: float
    min_sustained_seconds: float

    # -- incident-level ------------------------------------------------

    @property
    def true_positives(self) -> int:
        return sum(len(r.matched) for r in self.results)

    @property
    def false_negatives(self) -> int:
        return sum(len(r.missed) for r in self.results)

    @property
    def false_positives(self) -> int:
        return sum(len(r.false_alerts) for r in self.results)

    @property
    def duplicate_detections(self) -> int:
        return sum(len(r.duplicate_detections) for r in self.results)

    @property
    def total_incidents(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def incident_recall(self) -> Optional[float]:
        """None, not 0.0, when there are no falls to find - a corpus of
        pure hard negatives has no recall, and reporting 0% would read as
        catastrophic failure rather than 'not measured'."""
        if self.total_incidents == 0:
            return None
        return self.true_positives / self.total_incidents

    @property
    def incident_precision(self) -> Optional[float]:
        """None when the detector produced nothing at all."""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return None
        return self.true_positives / denominator

    @property
    def incident_f1(self) -> Optional[float]:
        """The single number for comparing threshold settings.

        Reported alongside precision and recall, never instead of them:
        for this product they are not interchangeable. A missed fall is
        someone lying on a floor unnoticed; a false alert is an operator
        checking a camera for nothing. F1 weights them equally, which is
        a convenience for ranking configurations, not a statement that
        the two errors cost the same."""
        return f1_score(self.incident_precision, self.incident_recall)

    # -- operational ----------------------------------------------------

    @property
    def total_duration_seconds(self) -> float:
        return sum(r.video.duration_seconds for r in self.results)

    @property
    def total_hours(self) -> float:
        return self.total_duration_seconds / SECONDS_PER_HOUR

    @property
    def hard_negative_hours(self) -> float:
        """False-alert rate is quoted over hard-negative footage only.
        Including fall clips would dilute it with footage that is supposed
        to trigger an alert, understating the nuisance rate an operator
        would actually experience."""
        return sum(
            r.video.duration_seconds for r in self.results if r.video.is_hard_negative
        ) / SECONDS_PER_HOUR

    @property
    def hard_negative_false_alerts(self) -> int:
        return sum(len(r.false_alerts) for r in self.results if r.video.is_hard_negative)

    @property
    def false_alerts_per_hour(self) -> Optional[float]:
        """Over hard-negative footage. None when there is none - dividing
        by zero hours would report an infinite or zero rate, both lies."""
        hours = self.hard_negative_hours
        if hours <= 0:
            return None
        return self.hard_negative_false_alerts / hours

    @property
    def all_false_alerts_per_hour(self) -> Optional[float]:
        """Over the entire corpus, for completeness."""
        if self.total_hours <= 0:
            return None
        return self.false_positives / self.total_hours

    # -- latency ---------------------------------------------------------

    @property
    def latencies(self) -> List[float]:
        return [latency for r in self.results for latency in r.latencies]

    @property
    def mean_latency_seconds(self) -> Optional[float]:
        values = self.latencies
        return sum(values) / len(values) if values else None

    @property
    def median_latency_seconds(self) -> Optional[float]:
        """None below 3 samples: a median over one or two detections is
        not a summary of anything."""
        values = self.latencies
        return median(values) if len(values) >= 3 else None

    @property
    def max_latency_seconds(self) -> Optional[float]:
        values = self.latencies
        return max(values) if values else None

    # -- false-positive breakdown ---------------------------------------

    def false_alert_categories(self) -> Dict[str, int]:
        """False alerts grouped by the activity of the clip that produced
        them - the answer to 'what kind of ordinary movement is confusing
        this detector'."""
        counts: Dict[str, int] = {}
        for result in self.results:
            if not result.false_alerts:
                continue
            for detection in result.false_alerts:
                label = result.video.activity if result.video.is_hard_negative else "in_fall_clip"
                counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def false_alert_details(self) -> List[dict]:
        """Every false alert, with enough context to go back and watch it."""
        rows = []
        for result in self.results:
            for detection in result.false_alerts:
                rows.append({
                    "video": result.video.filename,
                    "video_time_seconds": round(detection.video_time_seconds, 2),
                    "confidence": round(detection.confidence, 3),
                    "detector": detection.detector,
                    "category": result.video.category,
                    "activity": result.video.activity if result.video.is_hard_negative else "",
                    "activity_at_moment": result.video.activity_at(detection.video_time_seconds) or "",
                    "notes": result.video.notes,
                })
        return sorted(rows, key=lambda r: (r["video"], r["video_time_seconds"]))

    # -- confidence distributions ---------------------------------------

    def _confidence_stats(self, values: Sequence[float]) -> dict:
        ordered = sorted(values)
        if not ordered:
            return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
        return {
            "n": len(ordered),
            "min": round(ordered[0], 3),
            "median": round(median(ordered), 3),
            "mean": round(sum(ordered) / len(ordered), 3),
            "max": round(ordered[-1], 3),
        }

    def _histogram(self, values: Sequence[float]) -> Dict[str, int]:
        """Counts per CONFIDENCE_BINS bucket, labelled by their edges."""
        counts: Dict[str, int] = {}
        for low, high in zip(CONFIDENCE_BINS, CONFIDENCE_BINS[1:]):
            label = f"{low:.2f}-{high:.2f}"
            # Half-open except at the top, so 1.0 lands in the last bucket
            # rather than falling off the end.
            counts[label] = sum(
                1 for v in values
                if low <= v < high or (high == CONFIDENCE_BINS[-1] and v == high)
            )
        return counts

    def confidence_distribution(self) -> dict:
        """How confident the detector was when it was right, versus when it
        was wrong.

        This is the diagnostic the threshold sweep cannot give on its own.
        A sweep says what happens at each threshold; this says WHY - if
        true and false detections overlap heavily, no threshold separates
        them and the fix is the model or the sustain gate, not the number.
        If they separate cleanly, the sweep's best row is a real operating
        point rather than an artefact of a small corpus."""
        true_positive_confidences = [
            m.detection.confidence for r in self.results for m in r.matched
        ]
        false_alert_confidences = [
            d.confidence for r in self.results for d in r.false_alerts
        ]
        return {
            "true_positives": {
                **self._confidence_stats(true_positive_confidences),
                "histogram": self._histogram(true_positive_confidences),
            },
            "false_alerts": {
                **self._confidence_stats(false_alert_confidences),
                "histogram": self._histogram(false_alert_confidences),
            },
            "bins": list(CONFIDENCE_BINS),
        }

    # -- per-condition / per-taxonomy breakdowns --------------------------

    def _group_videos(self, key: Callable[[VideoResult], str]) -> List[GroupResult]:
        """Groups whole videos, so both recall and false-alert rate are
        meaningful within a group (a false alert belongs to the clip it
        fired in, not to any particular incident)."""
        buckets: Dict[str, List[VideoResult]] = {}
        for result in self.results:
            buckets.setdefault(key(result), []).append(result)

        groups = [
            GroupResult(
                label=label,
                true_positives=sum(len(r.matched) for r in rows),
                false_negatives=sum(len(r.missed) for r in rows),
                false_positives=sum(len(r.false_alerts) for r in rows),
                videos=len(rows),
                duration_seconds=sum(r.video.duration_seconds for r in rows),
            )
            for label, rows in buckets.items()
        ]
        return sorted(groups, key=lambda g: g.label)

    def by_condition(self, dimension: str) -> List[GroupResult]:
        """Results split by one recording condition - lighting, camera
        angle, distance, occlusion or resolution.

        A single overall recall hides exactly what an installer needs: a
        detector that works in daylight at 3 m and fails at night at 8 m
        has the same headline number as one that works everywhere."""
        return self._group_videos(lambda r: r.video.condition(dimension))

    def by_all_conditions(self) -> Dict[str, List[GroupResult]]:
        return {dimension: self.by_condition(dimension) for dimension in CONDITION_DIMENSIONS}

    def _group_incidents(self, key: Callable[[Incident], str]) -> List[GroupResult]:
        """Groups individual incidents. Only recall is defined here: a
        false alert cannot be attributed to a fall direction, because no
        fall happened."""
        buckets: Dict[str, List[bool]] = {}
        for result in self.results:
            for matched in result.matched:
                buckets.setdefault(key(matched.incident), []).append(True)
            for missed in result.missed:
                buckets.setdefault(key(missed), []).append(False)

        groups = [
            GroupResult(
                label=label,
                true_positives=sum(1 for hit in hits if hit),
                false_negatives=sum(1 for hit in hits if not hit),
                false_positives=0,
                videos=0,
                duration_seconds=0.0,
            )
            for label, hits in buckets.items()
        ]
        return sorted(groups, key=lambda g: g.label)

    def by_fall_direction(self) -> List[GroupResult]:
        return self._group_incidents(lambda i: i.direction or "unlabelled")

    def by_fall_speed(self) -> List[GroupResult]:
        return self._group_incidents(lambda i: i.speed or "unlabelled")

    def by_hard_negative_activity(self) -> List[GroupResult]:
        """False-alert rate per ordinary activity - the ranked list of what
        actually confuses the detector."""
        return self._group_videos(
            lambda r: r.video.activity if r.video.is_hard_negative else "fall_clip"
        )

    def missed_fall_details(self) -> List[dict]:
        rows = []
        for result in self.results:
            for incident in result.missed:
                rows.append({
                    "video": result.video.filename,
                    "incident_id": incident.id,
                    "start_seconds": round(incident.start_seconds, 2),
                    "end_seconds": round(incident.end_seconds, 2),
                    "detections_in_video": len(result.detections),
                    "direction": incident.direction,
                    "speed": incident.speed,
                    "notes": incident.notes,
                })
        return sorted(rows, key=lambda r: (r["video"], r["start_seconds"]))


def match_video(
    video: AnnotatedVideo,
    detections: Sequence[Detection],
    pre_tolerance: float = PRE_TOLERANCE_SECONDS,
    post_tolerance: float = POST_TOLERANCE_SECONDS,
) -> VideoResult:
    """Applies the matching rule in the module docstring to one video."""
    ordered = sorted(detections, key=lambda d: d.video_time_seconds)
    incidents = sorted(video.incidents, key=lambda i: i.start_seconds)

    consumed: set = set()          # indices into `ordered` already accounted for
    matched: List[MatchedIncident] = []
    missed: List[Incident] = []
    duplicates: List[Detection] = []

    for incident in incidents:
        window_start = incident.start_seconds - pre_tolerance
        window_end = incident.end_seconds + post_tolerance

        eligible = [
            i for i, d in enumerate(ordered)
            if i not in consumed and window_start <= d.video_time_seconds <= window_end
        ]

        if not eligible:
            missed.append(incident)
            continue

        # Earliest eligible: latency is what an operator waits for the
        # FIRST alert, so a later detection of the same fall must not be
        # the one that sets the number.
        first = eligible[0]
        consumed.add(first)
        matched.append(MatchedIncident(incident=incident, detection=ordered[first]))

        # Anything else inside this incident's window is the same fall
        # re-firing past the debounce, not a separate wrong alert.
        for index in eligible[1:]:
            consumed.add(index)
            duplicates.append(ordered[index])

    false_alerts = [d for i, d in enumerate(ordered) if i not in consumed]

    return VideoResult(
        video=video,
        detections=list(ordered),
        matched=matched,
        missed=missed,
        false_alerts=false_alerts,
        duplicate_detections=duplicates,
    )


def evaluate(
    corpus: Corpus,
    detections_by_video: Dict[str, Sequence[Detection]],
    min_confidence: float,
    min_sustained_seconds: float,
    pre_tolerance: float = PRE_TOLERANCE_SECONDS,
    post_tolerance: float = POST_TOLERANCE_SECONDS,
) -> EvaluationSummary:
    """Evaluates a whole corpus. `detections_by_video` is keyed by
    filename; a video with no entry is treated as having produced no
    detections, which is the correct reading of 'the pipeline ran and
    found nothing'."""
    results = [
        match_video(
            video,
            detections_by_video.get(video.filename, ()),
            pre_tolerance=pre_tolerance,
            post_tolerance=post_tolerance,
        )
        for video in corpus.videos
    ]
    return EvaluationSummary(
        results=results,
        min_confidence=min_confidence,
        min_sustained_seconds=min_sustained_seconds,
    )
