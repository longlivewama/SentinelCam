"""
The labelled-corpus format, and its loader.

Design constraint: a human has to be able to write and correct these by
hand after watching a clip, with a text editor and no tooling. That rules
out anything clever - one JSON file, one object per video, times in
seconds as plain numbers.

    {
      "videos": [
        {
          "filename": "fall_kitchen_01.mp4",
          "duration_seconds": 22.5,
          "category": "fall",
          "incidents": [
            {"id": "f1", "start_seconds": 8.2, "end_seconds": 10.0,
             "notes": "trips on rug, lands on side, stays down"}
          ]
        },
        {
          "filename": "hardneg_sitting_01.mp4",
          "duration_seconds": 18.0,
          "category": "hard_negative",
          "activity": "sitting",
          "incidents": [],
          "activity_intervals": [
            {"start_seconds": 3.0, "end_seconds": 7.5, "label": "sits down on chair"}
          ]
        }
      ]
    }

`category` is "fall" or "hard_negative", and it is checked against
`incidents` rather than trusted: a clip labelled "fall" with no annotated
incident, or a "hard_negative" with one, is a labelling mistake that would
quietly corrupt every metric downstream, so loading raises instead.

`activity` on a hard negative is what drives the false-positive category
breakdown in the report - it is the answer to "what kind of ordinary
movement is confusing this detector", which is the whole reason hard
negatives are collected separately from "any video without a fall".

`activity_intervals` are optional and purely descriptive: they let the
report say *which* moment of an 18-second clip produced a false alert
without requiring anyone to annotate every clip that precisely.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CATEGORY_FALL = "fall"
CATEGORY_HARD_NEGATIVE = "hard_negative"
VALID_CATEGORIES = (CATEGORY_FALL, CATEGORY_HARD_NEGATIVE)

# Free-text-ish, but a known vocabulary keeps the report's grouping
# meaningful. Anything outside it is accepted and grouped under "other",
# rather than rejected - refusing a clip because its activity word is new
# would discourage exactly the corpus growth this is here to enable.
KNOWN_ACTIVITIES = (
    "sitting", "standing_up", "crouching", "bending", "kneeling",
    "lying_sofa", "lying_bed", "sleeping", "exercising", "stretching",
    "chair_transfer", "floor_activity", "picking_up", "dropped_object",
    "pet", "occluded", "walking", "other",
)

# How a fall happened. Unlike `activity` these ARE validated against a
# closed set, because they are enumerations rather than labels: a typo of
# "backwards" would silently become a bucket of its own, and the corpus
# coverage report below - which exists to say "you have no backward falls
# yet" - would then quietly report that you do.
FALL_DIRECTIONS = ("forward", "backward", "sideways")
FALL_SPEEDS = ("slow", "fast")

# The recording conditions a result should be broken down by. Keys are
# validated (a typo'd key would vanish from the report); values are not,
# because deployments differ - "1080p", "4k" and "cif" are all legitimate
# answers to `resolution`.
CONDITION_DIMENSIONS = ("lighting", "camera_angle", "distance", "occlusion", "resolution")

# The hard-negative activities a corpus must contain before its
# false-alert rate means anything. Every one of these produces the visual
# signature the detector fires on - a person, horizontal or low, sustained
# - which is exactly what the frame-level training data contains none of.
# See README.md "Why hard negatives matter more than more fall clips".
REQUIRED_HARD_NEGATIVES = (
    "sitting", "lying_sofa", "bending", "crouching", "exercising", "sleeping",
    "picking_up", "dropped_object", "pet", "occluded",
)


class AnnotationError(ValueError):
    """The annotation file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Incident:
    """One ground-truth fall, as an interval on the video's timeline.

    `direction` and `speed` are optional, and describe the fall itself
    rather than the clip - a clip can contain a fast forward fall and a
    slow sideways one, and those are different tests of the detector. They
    drive the per-taxonomy recall breakdown in the report, which is how
    "the model misses backward falls" becomes visible instead of being
    averaged into a single recall figure."""

    id: str
    start_seconds: float
    end_seconds: float
    notes: str = ""
    direction: str = ""
    speed: str = ""

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class ActivityInterval:
    start_seconds: float
    end_seconds: float
    label: str = ""


@dataclass(frozen=True)
class AnnotatedVideo:
    filename: str
    duration_seconds: float
    category: str
    incidents: List[Incident] = field(default_factory=list)
    activity: str = "other"
    activity_intervals: List[ActivityInterval] = field(default_factory=list)
    notes: str = ""
    # Recording conditions, keyed by CONDITION_DIMENSIONS. Optional, but
    # without them a result is a single number that hides the thing an
    # installer actually needs to know: whether it holds at their camera
    # height, in their lighting, at their distance.
    conditions: Dict[str, str] = field(default_factory=dict)

    @property
    def is_hard_negative(self) -> bool:
        return self.category == CATEGORY_HARD_NEGATIVE

    def condition(self, dimension: str) -> str:
        """The labelled value for one dimension, or "unlabelled" - never
        None, so a report can group by it without special-casing."""
        return self.conditions.get(dimension) or "unlabelled"

    def activity_at(self, seconds: float) -> Optional[str]:
        """The labelled activity covering `seconds`, if any interval does.
        Used to attribute a false alert to a specific moment."""
        for interval in self.activity_intervals:
            if interval.start_seconds <= seconds <= interval.end_seconds:
                return interval.label or self.activity
        return None


@dataclass(frozen=True)
class Corpus:
    videos: List[AnnotatedVideo]
    source_path: Optional[Path] = None

    @property
    def total_duration_seconds(self) -> float:
        return sum(v.duration_seconds for v in self.videos)

    @property
    def total_incidents(self) -> int:
        return sum(len(v.incidents) for v in self.videos)

    @property
    def fall_videos(self) -> List[AnnotatedVideo]:
        return [v for v in self.videos if v.category == CATEGORY_FALL]

    @property
    def hard_negative_videos(self) -> List[AnnotatedVideo]:
        return [v for v in self.videos if v.is_hard_negative]

    @property
    def incidents(self) -> List[Incident]:
        return [incident for video in self.videos for incident in video.incidents]

    def composition(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for video in self.hard_negative_videos:
            counts[video.activity] = counts.get(video.activity, 0) + 1
        return dict(sorted(counts.items()))

    def fall_taxonomy(self) -> Dict[str, Dict[str, int]]:
        """Annotated falls counted by direction and by speed, with
        unlabelled ones visible rather than dropped."""
        directions: Dict[str, int] = {}
        speeds: Dict[str, int] = {}
        for incident in self.incidents:
            direction = incident.direction or "unlabelled"
            speed = incident.speed or "unlabelled"
            directions[direction] = directions.get(direction, 0) + 1
            speeds[speed] = speeds.get(speed, 0) + 1
        return {"direction": dict(sorted(directions.items())), "speed": dict(sorted(speeds.items()))}

    def condition_coverage(self) -> Dict[str, Dict[str, int]]:
        """Clips counted by the value of each recording condition."""
        coverage: Dict[str, Dict[str, int]] = {}
        for dimension in CONDITION_DIMENSIONS:
            counts: Dict[str, int] = {}
            for video in self.videos:
                value = video.condition(dimension)
                counts[value] = counts.get(value, 0) + 1
            coverage[dimension] = dict(sorted(counts.items()))
        return coverage

    def missing_coverage(self) -> Dict[str, List[str]]:
        """The buckets the README asks for that this corpus does not yet
        contain.

        This is the readiness check, and it is deliberately mechanical: a
        corpus is not "done" because it is large, it is done when it
        exercises each failure mode the detector is claimed to handle. What
        this returns is the shopping list, and an empty result is the only
        honest basis for dropping the "preliminary" banner from a report.
        """
        taxonomy = self.fall_taxonomy()
        missing: Dict[str, List[str]] = {}

        absent_directions = [d for d in FALL_DIRECTIONS if not taxonomy["direction"].get(d)]
        if absent_directions:
            missing["fall_direction"] = absent_directions

        absent_speeds = [s for s in FALL_SPEEDS if not taxonomy["speed"].get(s)]
        if absent_speeds:
            missing["fall_speed"] = absent_speeds

        present_activities = {v.activity for v in self.hard_negative_videos}
        absent_activities = [a for a in REQUIRED_HARD_NEGATIVES if a not in present_activities]
        if absent_activities:
            missing["hard_negative_activity"] = absent_activities

        unlabelled = [
            dimension for dimension, counts in self.condition_coverage().items()
            # Only one labelled value is no better than none for a
            # breakdown: there is nothing to compare it against.
            if len([v for v in counts if v != "unlabelled"]) < 2
        ]
        if unlabelled:
            missing["condition_variation"] = unlabelled

        return missing


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnnotationError(message)


def _parse_incident(raw: dict, video_filename: str, index: int, duration: float) -> Incident:
    where = f"{video_filename} incident #{index + 1}"
    _require(isinstance(raw, dict), f"{where}: expected an object, got {type(raw).__name__}")

    for key in ("start_seconds", "end_seconds"):
        _require(key in raw, f"{where}: missing {key!r}")
        _require(
            isinstance(raw[key], (int, float)) and not isinstance(raw[key], bool),
            f"{where}: {key!r} must be a number",
        )

    start = float(raw["start_seconds"])
    end = float(raw["end_seconds"])

    _require(start >= 0, f"{where}: start_seconds must not be negative")
    _require(
        end > start,
        f"{where}: end_seconds ({end}) must be after start_seconds ({start})",
    )
    # A fall annotated past the end of its own clip means the times came
    # from the wrong video, which would silently make it unmatchable.
    _require(
        start <= duration,
        f"{where}: start_seconds ({start}) is past the video's duration ({duration})",
    )

    direction = str(raw.get("direction", "")).strip().lower()
    _require(
        not direction or direction in FALL_DIRECTIONS,
        f"{where}: 'direction' must be one of {list(FALL_DIRECTIONS)}, got {direction!r}",
    )
    speed = str(raw.get("speed", "")).strip().lower()
    _require(
        not speed or speed in FALL_SPEEDS,
        f"{where}: 'speed' must be one of {list(FALL_SPEEDS)}, got {speed!r}",
    )

    return Incident(
        id=str(raw.get("id") or f"{Path(video_filename).stem}-{index + 1}"),
        start_seconds=start,
        end_seconds=end,
        notes=str(raw.get("notes", "")),
        direction=direction,
        speed=speed,
    )


def _parse_video(raw: dict, index: int) -> AnnotatedVideo:
    where = f"videos[{index}]"
    _require(isinstance(raw, dict), f"{where}: expected an object")
    _require("filename" in raw, f"{where}: missing 'filename'")
    filename = str(raw["filename"])

    _require("duration_seconds" in raw, f"{filename}: missing 'duration_seconds'")
    _require(
        isinstance(raw["duration_seconds"], (int, float)) and not isinstance(raw["duration_seconds"], bool),
        f"{filename}: 'duration_seconds' must be a number",
    )
    duration = float(raw["duration_seconds"])
    _require(duration > 0, f"{filename}: 'duration_seconds' must be positive")

    category = str(raw.get("category", "")).strip().lower()
    _require(
        category in VALID_CATEGORIES,
        f"{filename}: 'category' must be one of {list(VALID_CATEGORIES)}, got {category!r}",
    )

    raw_incidents = raw.get("incidents", [])
    _require(isinstance(raw_incidents, list), f"{filename}: 'incidents' must be a list")
    incidents = [
        _parse_incident(item, filename, i, duration) for i, item in enumerate(raw_incidents)
    ]

    # Cross-check the category against the annotations rather than
    # trusting it. Either mismatch silently corrupts every metric: a
    # "fall" clip with no incident contributes a guaranteed unmatched
    # detection, and a "hard_negative" with one is a fall counted as
    # ordinary footage.
    if category == CATEGORY_FALL:
        _require(
            incidents,
            f"{filename}: category is 'fall' but no incidents are annotated - "
            "annotate the fall, or label the clip 'hard_negative'",
        )
    else:
        _require(
            not incidents,
            f"{filename}: category is 'hard_negative' but {len(incidents)} incident(s) "
            "are annotated - a clip containing a fall belongs in category 'fall'",
        )

    raw_intervals = raw.get("activity_intervals", [])
    _require(isinstance(raw_intervals, list), f"{filename}: 'activity_intervals' must be a list")
    intervals = []
    for item in raw_intervals:
        _require(isinstance(item, dict), f"{filename}: each activity interval must be an object")
        intervals.append(ActivityInterval(
            start_seconds=float(item.get("start_seconds", 0.0)),
            end_seconds=float(item.get("end_seconds", 0.0)),
            label=str(item.get("label", "")),
        ))

    raw_conditions = raw.get("conditions", {})
    _require(isinstance(raw_conditions, dict), f"{filename}: 'conditions' must be an object")
    unknown = sorted(set(raw_conditions) - set(CONDITION_DIMENSIONS))
    # Rejected rather than ignored: a misspelled dimension would simply
    # disappear from the per-condition breakdown, and the clip would look
    # labelled when it is not.
    _require(
        not unknown,
        f"{filename}: unknown condition key(s) {unknown}; expected any of {list(CONDITION_DIMENSIONS)}",
    )
    conditions = {
        key: str(value).strip().lower()
        for key, value in raw_conditions.items()
        if str(value).strip()
    }

    return AnnotatedVideo(
        filename=filename,
        duration_seconds=duration,
        category=category,
        incidents=incidents,
        activity=str(raw.get("activity", "other")).strip().lower() or "other",
        activity_intervals=intervals,
        notes=str(raw.get("notes", "")),
        conditions=conditions,
    )


def load_corpus(path: "str | Path") -> Corpus:
    """Reads and validates an annotations file. Raises AnnotationError
    with a message naming the offending video - these files are written by
    hand, so a parse failure has to say what to fix."""
    path = Path(path)
    if not path.exists():
        raise AnnotationError(
            f"No annotation file at {path}. See ml/validation/README.md for the "
            "expected layout, and ml/validation/annotations.example.json for a template."
        )

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"{path} is not valid JSON: {exc}") from exc

    _require(isinstance(raw, dict), f"{path}: top level must be an object with a 'videos' key")
    _require("videos" in raw, f"{path}: missing 'videos'")
    _require(isinstance(raw["videos"], list), f"{path}: 'videos' must be a list")

    videos = [_parse_video(item, i) for i, item in enumerate(raw["videos"])]

    filenames = [v.filename for v in videos]
    duplicates = sorted({name for name in filenames if filenames.count(name) > 1})
    _require(not duplicates, f"{path}: duplicate filename(s): {duplicates}")

    return Corpus(videos=videos, source_path=path)
