from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.sql import func

from app.database import Base


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    event_type = Column(String, nullable=False)  # "fall" | "violence" | "abandoned_object" | "crowd"
    confidence_score = Column(Float, nullable=False, default=0.0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    triggered_recording = Column(Boolean, nullable=False, default=False)
