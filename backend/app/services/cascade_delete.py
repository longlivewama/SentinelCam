"""
Shared cascade-delete helpers for entities that other rows hold a foreign
key to. Postgres FKs here default to RESTRICT (no `ondelete` was set on
any of them), so naively `db.delete()`-ing a referenced row raises an
IntegrityError - these helpers stage the correct cleanup first. Used by
both the video-uploads route (deleting one's own upload) and the admin
route (deleting a user who owns uploads / acknowledged alerts).
"""
from __future__ import annotations

import logging
import os
from typing import List

from sqlalchemy.orm import Session

from app.config import settings
from app.models.event import Event
from app.models.recording import Recording
from app.models.video_upload import VideoUpload

logger = logging.getLogger(__name__)


def stage_delete_video_upload(db: Session, upload: VideoUpload) -> List[str]:
    """Stages deletion of a VideoUpload and everything derived from it
    (its Events and Recordings - which have no independent meaning
    without the source upload) within the given session, WITHOUT
    committing. Returns the on-disk file paths (recording clips + the
    source video itself) the caller should remove after a successful
    commit."""
    recordings = db.query(Recording).filter(Recording.video_upload_id == upload.id).all()
    paths = [r.file_path for r in recordings]

    db.query(Event).filter(Event.video_upload_id == upload.id).delete()
    db.query(Recording).filter(Recording.video_upload_id == upload.id).delete()

    paths.append(upload.stored_path)
    db.delete(upload)
    return paths


def stage_delete_recording(db: Session, recording: Recording) -> None:
    """Stages deletion of a single Recording, preserving any Event that
    references it (a live-camera alert's history is meaningful even after
    its clip is pruned) by clearing recording_id rather than cascading."""
    db.query(Event).filter(Event.recording_id == recording.id).update({"recording_id": None})
    db.delete(recording)


def unlink_user_from_acknowledged_alerts(db: Session, user_id: int) -> None:
    """Clears acknowledged_by on any alert this user acknowledged, keeping
    acknowledged=True (the alert was still handled - only the attribution
    to a since-deleted account is removed)."""
    db.query(Event).filter(Event.acknowledged_by == user_id).update({"acknowledged_by": None})


def remove_upload_clip_dir(upload_id: int) -> None:
    """Removes the per-upload clip directory once its clips are gone.

    `_write_upload_clip` creates `recordings/uploads/{id}/` for every
    upload that produces a fall, and deleting the upload removes the
    clips inside it but left the directory behind - one empty directory
    per deleted upload, accumulating for the life of the deployment.

    `os.rmdir` rather than a recursive delete, deliberately: if anything
    is still in there the directory stays, and whatever it is survives to
    be noticed. A failure here is not worth failing a delete over, so it
    is logged and swallowed."""
    clip_dir = os.path.join(settings.RECORDINGS_DIR, "uploads", str(upload_id))
    try:
        os.rmdir(clip_dir)
    except FileNotFoundError:
        pass  # No fall was ever recorded for this upload.
    except OSError as exc:
        logger.warning("Left %s in place while deleting upload %s: %s", clip_dir, upload_id, exc)
