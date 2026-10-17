"""
The per-video tracking record behind the uploaded-video overlay.

While an uploaded video is analysed, every processed frame's tracked
people (box, confidence, keypoints, track id, state) are collected here and
written once, at the end, as one JSON file next to that upload's clips:
`RECORDINGS_DIR/uploads/{upload_id}/tracking.json`.

Why a file and not PostgreSQL rows: this is frame-level data - ~6 rows per
second of video per person - read only when someone replays that one
upload, and meaningless once the upload is gone. A file keeps the database
holding what it already holds (uploads, events, recordings) and is deleted
with the upload by `cascade_delete.stage_delete_video_upload`, the same as
the clips beside it. Its contents are derived from the uploader's own
footage, so it is served only through the upload's existing ownership
check.

Size, measured on the format below at the default stride of 5: 68 KB for a
29 s, 30 fps video with 1-2 people in view (176 processed frames), 27 KB
for an 8 s clip with up to 3. Roughly 0.15 KB per person per processed
frame. `TRACKING_MAX_RECORDED_FRAMES` bounds it for very long videos; past
the cap the record says it is truncated rather than growing without limit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

from app.services.detection.pose import pose_format


TRACKING_RECORD_VERSION = 1


class TrackingRecorder:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float,
        stride: int,
        keypoint_min_confidence: float,
        max_frames: int,
        settings_snapshot: Optional[dict] = None,
    ):
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.fps = float(fps)
        self.stride = max(int(stride), 1)
        self.keypoint_min_confidence = float(keypoint_min_confidence)
        self.max_frames = max(int(max_frames), 0)
        self.settings_snapshot = settings_snapshot or {}
        self.frames: List[dict] = []
        self.fall_events: List[dict] = []
        self.suppressed_events: List[dict] = []
        self.truncated = False

    def add_frame(self, frame_index: int, video_time_seconds: float, persons: Iterable) -> None:
        if len(self.frames) >= self.max_frames:
            self.truncated = True
            return
        self.frames.append({
            "i": int(frame_index),
            "t": round(float(video_time_seconds), 3),
            "persons": [_person(p) for p in persons],
        })

    def add_fall_event(
        self,
        video_time_seconds: float,
        event: dict,
        event_id: Optional[int] = None,
        recording_id: Optional[int] = None,
    ) -> None:
        self.fall_events.append({
            "event_id": event_id,
            "recording_id": recording_id,
            **_event_evidence(video_time_seconds, event),
        })

    def add_suppressed_events(self, video_time_seconds: float, events: Iterable[dict]) -> None:
        for event in events:
            self.suppressed_events.append({
                **_event_evidence(video_time_seconds, event),
                "reason": event.get("suppressed_reason"),
            })

    def to_dict(self, track_summaries: Iterable) -> dict:
        return {
            "version": TRACKING_RECORD_VERSION,
            "video": {
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "fps": self.fps,
                "stride": self.stride,
            },
            "pose_format": pose_format(),
            "keypoint_min_confidence": self.keypoint_min_confidence,
            "settings": self.settings_snapshot,
            "truncated": self.truncated,
            "tracks": [
                {
                    "track_id": s.track_id,
                    "first_seen": round(s.first_seen, 3),
                    "last_seen": round(s.last_seen, 3),
                    "duration_seconds": round(s.duration_seconds, 3),
                    "frames_observed": s.frames_observed,
                    "max_confidence": None if s.max_confidence is None else round(s.max_confidence, 3),
                }
                for s in track_summaries
            ],
            "fall_events": self.fall_events,
            "suppressed_events": self.suppressed_events,
            "frames": self.frames,
        }

    def write(self, path: "str | Path", track_summaries: Iterable) -> None:
        """Atomic: a reader never sees a half-written record, and a crash
        mid-write leaves no file rather than a corrupt one."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(track_summaries), handle, separators=(",", ":"))
        os.replace(tmp, path)


def _person(person) -> dict:
    return {
        "track_id": person.track_id,
        "state": person.state,
        "bbox": [round(float(v), 1) for v in person.bbox],
        "confidence": None if person.confidence is None else round(float(person.confidence), 2),
        "keypoints": [[round(x, 1), round(y, 1), round(c, 2)] for x, y, c in person.keypoints],
    }


def _event_evidence(video_time_seconds: float, event: dict) -> dict:
    bbox = event.get("bbox")
    pose = event.get("pose") or None
    return {
        "video_time_seconds": round(float(video_time_seconds), 2),
        "person_track_id": event.get("person_track_id"),
        "confidence": event.get("confidence"),
        "detector": event.get("detector"),
        "bbox": None if bbox is None else [round(float(v), 1) for v in bbox],
        "pose": pose,
    }
