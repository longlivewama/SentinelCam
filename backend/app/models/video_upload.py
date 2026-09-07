from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.sql import func

from app.database import Base

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class VideoUpload(Base):
    """A customer-uploaded video submitted for offline analysis (as opposed
    to a live camera feed). Processed asynchronously by
    services/video_analysis.py, which runs the same detection pipeline as
    live cameras (pose + object models -> fall detector) over the uploaded
    file's frames and records progress/results here."""

    __tablename__ = "video_uploads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    original_filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)

    status = Column(String, nullable=False, default=STATUS_PENDING)
    progress_percent = Column(Integer, nullable=False, default=0)
    error_message = Column(String, nullable=True)

    duration_seconds = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    frame_count = Column(Integer, nullable=True)
    persons_detected = Column(Integer, nullable=False, default=0)
    fall_events_count = Column(Integer, nullable=False, default=0)

    file_size_bytes = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
