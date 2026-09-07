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
    timestamp: datetime
    triggered_recording: bool
    acknowledged: bool
    acknowledged_by: Optional[int] = None
    acknowledged_at: Optional[datetime] = None
