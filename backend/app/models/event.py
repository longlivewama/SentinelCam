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

    # Wall-clock time the event was *recorded*. For a live camera this is
    # also when it happened; for an uploaded video it is when the analysis
    # reached that moment, which says nothing about where in the footage
    # the fall occurred - that is video_timestamp_seconds below. The UI
    # must not present this as a position within the video.
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Offset, in seconds from the start of the source file, at which the
    # event was detected. Only meaningful for events from an uploaded
    # video (video_upload_id set); null for live-camera events, which
    # have no such timeline.
    video_timestamp_seconds = Column(Float, nullable=True)

    # Which detection strategy produced this event - "model" (the trained
    # YOLO fall detector) or "heuristic" (the pose/geometry detectors).
    # Null on rows written before this column existed. Operators need this
    # to interpret a confidence score: the two strategies compute it
    # differently and are not comparable.
    detector = Column(String, nullable=True)

    triggered_recording = Column(Boolean, nullable=False, default=False)

    acknowledged = Column(Boolean, nullable=False, default=False)
    acknowledged_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
