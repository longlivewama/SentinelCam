"""
Response shapes for pose estimation + multi-person tracking results.

`UploadTrackingOut` is the per-video tracking record
(services/detection/tracking_record.py) as served by
`GET /api/video-uploads/{id}/tracking`. Validating it on the way out keeps
the contract explicit: a record written by an older or newer version that
no longer matches fails loudly server-side instead of drawing nonsense in
the browser.

Coordinates are in the source video's own pixels; `t` is seconds on the
video's own timeline. Track ids are temporary handles inside this one
analysis - not identities.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class PoseFormatOut(BaseModel):
    keypoint_names: List[str]
    # Pairs of 0-based keypoint indices - the model's own skeleton.
    skeleton: List[List[int]]


class TrackedPersonFrameOut(BaseModel):
    track_id: int
    state: str  # "tentative" | "confirmed" | "lost"
    bbox: List[float] = Field(min_length=4, max_length=4)
    confidence: Optional[float] = None
    # [x, y, confidence] per keypoint, or [] when no usable pose was found.
    keypoints: List[List[float]]


class TrackingFrameOut(BaseModel):
    i: int  # source frame index
    t: float  # seconds into the video
    persons: List[TrackedPersonFrameOut]


class TrackSummaryOut(BaseModel):
    track_id: int
    first_seen: float
    last_seen: float
    duration_seconds: float
    frames_observed: int
    max_confidence: Optional[float] = None


class PoseConfirmationOut(BaseModel):
    """What the person did after the detector fired on them while upright
    (deferred pose verification, FALL_POSE_CONFIRM_SECONDS)."""
    # "released_not_upright" | "released_track_ended" | "released_insufficient_evidence"
    # | "released_source_ended" | "withheld_upright"
    outcome: str
    waited_seconds: Optional[float] = None
    post_event_samples: int = 0
    post_event_torso_angle_deg: Optional[float] = None


class PoseEvidenceOut(BaseModel):
    verdict: str  # "upright" | "not_upright" | "no_pose_evidence" | "no_person_association"
    torso_angle_deg: Optional[float] = None
    samples: int = 0
    # Present only on events pose verification held.
    confirmation: Optional[PoseConfirmationOut] = None


class FallEventEvidenceOut(BaseModel):
    event_id: Optional[int] = None
    recording_id: Optional[int] = None
    video_time_seconds: float
    person_track_id: Optional[int] = None
    confidence: Optional[float] = None
    detector: Optional[str] = None
    bbox: Optional[List[float]] = None
    pose: Optional[PoseEvidenceOut] = None


class SuppressedFallEventOut(FallEventEvidenceOut):
    reason: Optional[str] = None


class TrackingVideoOut(BaseModel):
    frame_width: int
    frame_height: int
    fps: float
    stride: int


class UploadTrackingOut(BaseModel):
    version: int
    video: TrackingVideoOut
    pose_format: PoseFormatOut
    keypoint_min_confidence: float
    settings: dict
    truncated: bool
    tracks: List[TrackSummaryOut]
    fall_events: List[FallEventEvidenceOut]
    suppressed_events: List[SuppressedFallEventOut]
    frames: List[TrackingFrameOut]
