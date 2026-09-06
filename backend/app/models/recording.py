from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.sql import func

from app.database import Base


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    duration_seconds = Column(Float, nullable=False, default=0.0)
    trigger_action = Column(String, nullable=False)  # "fall" | "violence" | "abandoned_object" | "crowd"
    file_size_bytes = Column(Integer, nullable=False, default=0)
    event_timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
