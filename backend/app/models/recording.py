from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.sql import func

from app.database import Base


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True, index=True)
    video_upload_id = Column(Integer, ForeignKey("video_uploads.id"), nullable=True, index=True)

    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    duration_seconds = Column(Float, nullable=False, default=0.0)
    trigger_action = Column(String, nullable=False)  # "fall" | "violence" | "abandoned_object" | "crowd"
    file_size_bytes = Column(Integer, nullable=False, default=0)
    # Indexed: every listing of this table orders by it (newest first),
    # and without an index that is a full scan plus a sort on a table that
    # grows for as long as a camera runs.
    event_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
