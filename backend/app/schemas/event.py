from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: Optional[int] = None
    video_upload_id: Optional[int] = None
    recording_id: Optional[int] = None
    event_type: str
    confidence_score: float
    # Wall-clock time the event was recorded - NOT a position within an
    # uploaded video. Use video_timestamp_seconds for that.
    timestamp: datetime
    # Seconds from the start of the source file, for events from an
    # uploaded video; null for live-camera events.
    video_timestamp_seconds: Optional[float] = None
    # "model" (trained YOLO fall detector) or "heuristic"; null on rows
    # written before this was recorded.
    detector: Optional[str] = None
    triggered_recording: bool
    acknowledged: bool
    acknowledged_by: Optional[int] = None
    acknowledged_at: Optional[datetime] = None
