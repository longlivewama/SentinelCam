"""
Row-level access control for the two entities that can belong to an
individual user: Events (alerts) and Recordings (clips).

Both tables hold rows from two different sources, and the sources have
genuinely different ownership semantics:

  * camera_id set     - produced by a shared surveillance camera. Cameras
                        are shared infrastructure (GET /api/cameras lists
                        every active camera to every authenticated user),
                        so their alerts and clips are visible to every
                        authenticated user too.

  * video_upload_id set - derived from one user's uploaded video. These
                        are that user's data: their footage, their
                        analysis results, and the source filename. They
                        must not be readable by another account.

Before this module existed, /api/alerts and /api/recordings applied no
ownership filter at all, so any authenticated user - including a plain
`viewer` - could list, read, stream and download every other user's
upload-derived alerts and fall clips by ID. These helpers are the single
place that rule is now expressed, so every route enforces the same one.

Operators and admins deliberately see everything: reviewing all alerts
across the deployment is the job the operator role exists for.
"""
from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import Query, Session

from app.models.event import Event
from app.models.recording import Recording
from app.models.user import User
from app.models.video_upload import VideoUpload


def _own_upload_ids(user: User):
    """Correlated subquery of the upload IDs this user owns. A subquery
    (rather than materialising the IDs in Python) keeps the filter to a
    single round trip and cannot go stale between the two queries."""
    return select(VideoUpload.id).where(VideoUpload.user_id == user.id)


def scope_events(query: Query, user: User) -> Query:
    """Restricts an Event query to rows `user` may see."""
    if user.is_operator:
        return query
    return query.filter(
        or_(
            Event.video_upload_id.is_(None),
            Event.video_upload_id.in_(_own_upload_ids(user)),
        )
    )


def scope_recordings(query: Query, user: User) -> Query:
    """Restricts a Recording query to rows `user` may see."""
    if user.is_operator:
        return query
    return query.filter(
        or_(
            Recording.video_upload_id.is_(None),
            Recording.video_upload_id.in_(_own_upload_ids(user)),
        )
    )


def can_access_upload_id(db: Session, upload_id: int | None, user: User) -> bool:
    """Whether `user` may see rows derived from video upload `upload_id`.
    `None` means the row came from a shared camera, which everyone may
    see."""
    if upload_id is None or user.is_operator:
        return True
    owner_id = db.query(VideoUpload.user_id).filter(VideoUpload.id == upload_id).scalar()
    return owner_id == user.id
