"""
Read-only aggregate statistics for the dashboard/analytics UI. Time-series
bucketing is done in Python rather than via SQL date-trunc functions so the
same code works against both PostgreSQL (production) and SQLite (tests) -
these aggregates are computed over a bounded, recent window (default 14
days) so this is cheap even done in Python rather than in the database.
"""
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.database import get_db
from app.models.camera import Camera
from app.models.event import Event
from app.models.recording import Recording
from app.models.user import User
from app.models.video_upload import VideoUpload

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
def get_summary(
    days: int = 14,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    days = max(1, min(days, 90))
    since = datetime.now(timezone.utc) - timedelta(days=days)

    total_cameras = db.query(func.count(Camera.id)).filter(Camera.is_active == True).scalar() or 0  # noqa: E712
    active_cameras = (
        db.query(func.count(Camera.id))
        .filter(Camera.is_active == True, Camera.status == "active")  # noqa: E712
        .scalar()
        or 0
    )

    total_alerts = db.query(func.count(Event.id)).scalar() or 0
    unacknowledged_alerts = db.query(func.count(Event.id)).filter(Event.acknowledged == False).scalar() or 0  # noqa: E712

    recent_events = (
        db.query(Event.event_type, Event.camera_id, Event.confidence_score, Event.timestamp)
        .filter(Event.timestamp >= since)
        .all()
    )

    falls_by_day = defaultdict(int)
    falls_by_camera_count = defaultdict(int)
    confidence_by_type = defaultdict(list)
    event_type_counts = defaultdict(int)

    camera_names = {c.id: c.name for c in db.query(Camera.id, Camera.name).all()}

    for event_type, camera_id, confidence_score, ts in recent_events:
        event_type_counts[event_type] += 1
        confidence_by_type[event_type].append(confidence_score)
        if event_type == "fall":
            day_key = ts.strftime("%Y-%m-%d") if ts else "unknown"
            falls_by_day[day_key] += 1
            if camera_id is not None:
                falls_by_camera_count[camera_id] += 1

    falls_over_time = []
    for i in range(days - 1, -1, -1):
        day = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
        falls_over_time.append({"date": day, "count": falls_by_day.get(day, 0)})

    falls_by_camera = [
        {"camera_id": cam_id, "camera_name": camera_names.get(cam_id, f"Camera #{cam_id}"), "count": count}
        for cam_id, count in sorted(falls_by_camera_count.items(), key=lambda kv: -kv[1])
    ]

    avg_confidence_by_type = {
        event_type: round(sum(scores) / len(scores), 3) if scores else 0.0
        for event_type, scores in confidence_by_type.items()
    }

    total_recordings = db.query(func.count(Recording.id)).scalar() or 0
    total_storage_bytes = db.query(func.coalesce(func.sum(Recording.file_size_bytes), 0)).scalar() or 0

    upload_status_counts = dict(
        db.query(VideoUpload.status, func.count(VideoUpload.id)).group_by(VideoUpload.status).all()
    )
    total_uploads = sum(upload_status_counts.values())

    return {
        "cameras": {"total": total_cameras, "active": active_cameras},
        "alerts": {
            "total": total_alerts,
            "unacknowledged": unacknowledged_alerts,
            "by_type_recent": dict(event_type_counts),
        },
        "falls": {
            "over_time": falls_over_time,
            "by_camera": falls_by_camera,
        },
        "confidence": {"avg_by_type_recent": avg_confidence_by_type},
        "recordings": {"total": total_recordings, "total_storage_bytes": int(total_storage_bytes)},
        "video_uploads": {"total": total_uploads, "by_status": upload_status_counts},
        "window_days": days,
    }
