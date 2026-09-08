from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class RecordingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: Optional[int] = None
    video_upload_id: Optional[int] = None
    filename: str
    # file_path is deliberately NOT exposed: it is an absolute path on the
    # server's filesystem, useless to any client (clips are fetched via
    # /api/recordings/{id}/video) and a free disclosure of the deployment's
    # directory layout to every authenticated user.
    duration_seconds: float
    trigger_action: str
    file_size_bytes: int
    event_timestamp: datetime
    created_at: datetime
