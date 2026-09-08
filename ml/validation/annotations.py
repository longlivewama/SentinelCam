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
    "lying_sofa", "lying_bed", "exercising", "stretching", "chair_transfer",
    "floor_activity", "walking", "other",
)


class AnnotationError(ValueError):
    """The annotation file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Incident:
    """One ground-truth fall, as an interval on the video's timeline."""

    id: str
    start_seconds: float
    end_seconds: float
    notes: str = ""

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

    @property
    def is_hard_negative(self) -> bool:
        return self.category == CATEGORY_HARD_NEGATIVE

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

    def composition(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for video in self.hard_negative_videos:
            counts[video.activity] = counts.get(video.activity, 0) + 1
        return dict(sorted(counts.items()))


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

    return Incident(
        id=str(raw.get("id") or f"{Path(video_filename).stem}-{index + 1}"),
        start_seconds=start,
        end_seconds=end,
        notes=str(raw.get("notes", "")),
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

    return AnnotatedVideo(
        filename=filename,
        duration_seconds=duration,
        category=category,
        incidents=incidents,
        activity=str(raw.get("activity", "other")).strip().lower() or "other",
        activity_intervals=intervals,
        notes=str(raw.get("notes", "")),
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
