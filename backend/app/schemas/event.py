from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: int
    event_type: str
    confidence_score: float
    timestamp: datetime
    triggered_recording: bool
