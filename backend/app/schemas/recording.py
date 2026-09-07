from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class RecordingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: Optional[int] = None
    video_upload_id: Optional[int] = None
    filename: str
    file_path: str
    duration_seconds: float
    trigger_action: str
    file_size_bytes: int
    event_timestamp: datetime
    created_at: datetime
