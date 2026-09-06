from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CameraBase(BaseModel):
    name: str
    url: str
    camera_type: str = "ip"  # "ip" | "usb"
    location: Optional[str] = None
    ai_detection_enabled: bool = False
    crowd_threshold: int = 20
    abandoned_object_seconds: int = 30


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    camera_type: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None
    is_active: Optional[bool] = None
    ai_detection_enabled: Optional[bool] = None
    crowd_threshold: Optional[int] = None
    abandoned_object_seconds: Optional[int] = None


class CameraOut(CameraBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    is_active: bool
    created_at: datetime
