from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.sql import func

from app.database import Base


class Event(Base):
    """A detected event ("alert"). Sourced either from a live camera's
    detection loop (camera_id set, video_upload_id null) or from an
    uploaded video's offline analysis (video_upload_id set, camera_id
    null)."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True, index=True)
    video_upload_id = Column(Integer, ForeignKey("video_uploads.id"), nullable=True, index=True)
    recording_id = Column(Integer, ForeignKey("recordings.id"), nullable=True, index=True)

    event_type = Column(String, nullable=False)  # "fall" | "violence" | "abandoned_object" | "crowd"
    confidence_score = Column(Float, nullable=False, default=0.0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    triggered_recording = Column(Boolean, nullable=False, default=False)

    acknowledged = Column(Boolean, nullable=False, default=False)
    acknowledged_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
