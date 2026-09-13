from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from app.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    url = Column(String, nullable=False)  # RTSP URL or numeric USB device index (as string)
    camera_type = Column(String, nullable=False, default="ip")  # "ip" | "usb"
    status = Column(String, nullable=False, default="inactive")  # "active" | "inactive" | "error"
    location = Column(String, nullable=True)
    # Whether the camera is intentionally turned on. User-controlled (see
    # PUT /api/cameras/{id}), independent of deleted_at below - toggling
    # this off stops the live stream + detection but leaves the camera
    # (and its history) fully intact and visible.
    is_active = Column(Boolean, nullable=False, default=True)
    ai_detection_enabled = Column(Boolean, nullable=False, default=False)
    crowd_threshold = Column(Integer, nullable=False, default=20)
    abandoned_object_seconds = Column(Integer, nullable=False, default=30)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Soft-delete marker, set only by DELETE /api/cameras/{id}. Kept
    # separate from is_active so "camera turned off" and "camera deleted"
    # are distinguishable: list_cameras filters on this column, not on
    # is_active, so a merely-off camera stays visible and toggleable.
    deleted_at = Column(DateTime(timezone=True), nullable=True)
