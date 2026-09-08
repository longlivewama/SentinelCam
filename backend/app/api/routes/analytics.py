"""
Read-only aggregate statistics for the dashboard/analytics UI. Time-series
bucketing is done in Python rather than via SQL date-trunc functions so the
same code works against both PostgreSQL (production) and SQLite (tests) -
these aggregates are computed over a bounded, recent window (default 14
days) so this is cheap even done in Python rather than in the database.

Every figure here is scoped to what the caller may actually see (see
core/scoping.py): a non-operator's counts cover shared-camera events plus
their own uploads, not other users' analyses. Aggregates are still data -
an unscoped "total alerts" tells a viewer how much other people uploaded
and how many falls were found in it.
"""
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.core.scoping import scope_events, scope_recordings
from app.database import get_db
from app.models.camera import Camera
from app.models.event import Event
from app.models.recording import Recording
from app.models.user import User
from app.models.video_upload import VideoUpload

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _utc_day(ts: datetime) -> str:
    """The UTC calendar day `ts` falls on, as `YYYY-MM-DD`.

    Necessary because `Event.timestamp` is `DateTime(timezone=True)` -
    `timestamptz` - and psycopg2 renders a timestamptz in the database
    SESSION's timezone, not in UTC. The same stored instant therefore
    comes back as a different wall clock depending only on how a
    deployment's Postgres is configured:

        2026-09-08 23:30+00 read under session TZ=UTC   -> 23:30 on the 8th
        ...the very same row under session TZ=+03       -> 02:30 on the 9th

    Taking `.strftime()` off that value directly (as this did) bucketed
    events by the session's local day while the bucket keys below are
    built from UTC, so under any non-UTC session every event in the
    offset window landed in a key that did not exist and vanished from
    the series. Converting first makes the answer depend only on the
    instant, which is the whole point of storing timestamptz.

    UTC is the convention the rest of the application already uses (every
    `datetime.now(timezone.utc)` call site, every timestamp column); there
    is no configured application timezone to defer to instead.
    """
    if ts.tzinfo is None:
        # No session rendering happened (a naive column, or SQLite).
        # These are written as UTC everywhere, so say so explicitly
        # rather than letting astimezone() assume the host's local zone.
        return ts.strftime("%Y-%m-%d")
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d")


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

    visible_events = scope_events(db.query(Event), current_user)
    total_alerts = visible_events.with_entities(func.count(Event.id)).scalar() or 0
    unacknowledged_alerts = (
        visible_events.filter(Event.acknowledged == False)  # noqa: E712
        .with_entities(func.count(Event.id))
        .scalar()
        or 0
    )

    recent_events = (
        visible_events.filter(Event.timestamp >= since)
        .with_entities(Event.event_type, Event.camera_id, Event.confidence_score, Event.timestamp)
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
            day_key = _utc_day(ts) if ts else "unknown"
            falls_by_day[day_key] += 1
            if camera_id is not None:
                falls_by_camera_count[camera_id] += 1

    # Keys are UTC days, and so are falls_by_day's - see _utc_day. Both
    # sides must use the same convention or every lookup below silently
    # misses, which is exactly how this broke.
    now_utc = datetime.now(timezone.utc)
    falls_over_time = []
    for i in range(days - 1, -1, -1):
        day = _utc_day(now_utc - timedelta(days=i))
        falls_over_time.append({"date": day, "count": falls_by_day.get(day, 0)})

    falls_by_camera = [
        {"camera_id": cam_id, "camera_name": camera_names.get(cam_id, f"Camera #{cam_id}"), "count": count}
        for cam_id, count in sorted(falls_by_camera_count.items(), key=lambda kv: -kv[1])
    ]

    avg_confidence_by_type = {
        event_type: round(sum(scores) / len(scores), 3) if scores else 0.0
        for event_type, scores in confidence_by_type.items()
    }

    visible_recordings = scope_recordings(db.query(Recording), current_user)
    total_recordings = visible_recordings.with_entities(func.count(Recording.id)).scalar() or 0
    total_storage_bytes = (
        visible_recordings.with_entities(func.coalesce(func.sum(Recording.file_size_bytes), 0)).scalar() or 0
    )

    uploads_query = db.query(VideoUpload)
    if not current_user.is_operator:
        uploads_query = uploads_query.filter(VideoUpload.user_id == current_user.id)
    upload_status_counts = dict(
        uploads_query.with_entities(VideoUpload.status, func.count(VideoUpload.id))
        .group_by(VideoUpload.status)
        .all()
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
