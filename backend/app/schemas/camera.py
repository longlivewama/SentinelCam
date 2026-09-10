from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from app.core.camera_url import InvalidCameraSource, validate_source


def _validate_camera_url(value: str) -> str:
    """Rejects a camera `url` that is not a stream URL or a USB device
    index. Enforced in the schema rather than in the routes so create and
    update cannot drift apart, and so the caller gets a 422 naming the
    problem before anything opens a capture device. See
    core/camera_url.py for why the scheme matters."""
    try:
        return validate_source(value)
    except InvalidCameraSource as exc:
        raise ValueError(str(exc)) from exc


class CameraBase(BaseModel):
    name: str
    url: str
    camera_type: str = "ip"  # "ip" | "usb"
    location: Optional[str] = None
    ai_detection_enabled: bool = False
    crowd_threshold: int = 20
    abandoned_object_seconds: int = 30


class CameraCreate(CameraBase):
    @field_validator("url")
    @classmethod
    def _check_url(cls, v: str) -> str:
        return _validate_camera_url(v)


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

    @field_validator("url")
    @classmethod
    def _check_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return _validate_camera_url(v)


class CameraOut(CameraBase):
    """Note that `url` is NOT the stored value for every caller: the
    cameras routes mask any inline credentials before returning this to a
    non-operator (see api/routes/cameras.py). The field stays required so
    the operator edit form still round-trips the real URL."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    is_active: bool
    created_at: datetime
