from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class VideoUploadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    original_filename: str
    status: str
    progress_percent: int
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    persons_detected: int
    fall_events_count: int
    file_size_bytes: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
