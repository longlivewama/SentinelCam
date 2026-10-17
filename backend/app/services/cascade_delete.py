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
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from app.config import settings
from app.models.event import Event
from app.models.recording import Recording
from app.models.video_upload import VideoUpload

logger = logging.getLogger(__name__)


def snapshot_path_for_clip(
    clip_filename: str, camera_id: Optional[int] = None, video_upload_id: Optional[int] = None,
) -> Path:
    """Where the alert snapshot JPEG for one event clip lives.

    The snapshot's path is not stored on the Recording row, so the writers
    (recording_engine, video_analysis) and every deleter derive it from the
    clip's filename through this one function. Before it existed the name
    was spelled inline at each writer and no delete path knew it at all:
    every snapshot - a frame of someone's footage - outlived its clip,
    its upload and its owner's account."""
    prefix = f"upload{video_upload_id}" if video_upload_id is not None else str(camera_id)
    return Path(settings.SNAPSHOTS_DIR) / f"{prefix}_{Path(clip_filename).stem}.jpg"


def upload_clip_dir(upload_id: int) -> Path:
    """`recordings/uploads/{id}/` - the per-upload directory holding its
    fall clips and its tracking record."""
    return Path(settings.RECORDINGS_DIR) / "uploads" / str(upload_id)


def upload_tracking_path(upload_id: int) -> Path:
    """The per-video tracking record (see detection/tracking_record.py).
    Frame-level pose and track data for one upload, kept beside its clips
    and deleted with them."""
    return upload_clip_dir(upload_id) / "tracking.json"


# Containers recording_engine.open_writer can produce for a browser-playable
# codec - the preview's extension depends on which encoder the build has.
_PREVIEW_EXTENSIONS = (".mp4", ".webm")


def upload_preview_paths(upload_id: int) -> List[Path]:
    """Every path a browser-playable re-encode of the source video may sit
    at (see video_analysis._write_playable_preview). Only written when the
    uploaded file's codec is one no browser decodes, e.g. MPEG-4 Part 2."""
    return [upload_clip_dir(upload_id) / f"source_preview{ext}" for ext in _PREVIEW_EXTENSIONS]


def find_upload_preview(upload_id: int) -> Optional[Path]:
    for path in upload_preview_paths(upload_id):
        if path.is_file():
            return path
    return None


def recording_snapshot_path(recording: Recording) -> str:
    return str(snapshot_path_for_clip(
        recording.filename, camera_id=recording.camera_id, video_upload_id=recording.video_upload_id,
    ))


def stage_delete_video_upload(db: Session, upload: VideoUpload) -> List[str]:
    """Stages deletion of a VideoUpload and everything derived from it
    (its Events and Recordings - which have no independent meaning
    without the source upload) within the given session, WITHOUT
    committing. Returns the on-disk file paths (recording clips, their
    snapshots, the tracking record, any playable preview, and the source
    video itself) the
    caller should remove after a successful commit."""
    recordings = db.query(Recording).filter(Recording.video_upload_id == upload.id).all()
    paths = []
    for recording in recordings:
        paths.append(recording.file_path)
        paths.append(recording_snapshot_path(recording))

    db.query(Event).filter(Event.video_upload_id == upload.id).delete()
    db.query(Recording).filter(Recording.video_upload_id == upload.id).delete()

    paths.append(str(upload_tracking_path(upload.id)))
    paths.extend(str(path) for path in upload_preview_paths(upload.id))
    paths.append(upload.stored_path)
    db.delete(upload)
    return paths


def stage_delete_recording(db: Session, recording: Recording) -> List[str]:
    """Stages deletion of a single Recording, preserving any Event that
    references it (a live-camera alert's history is meaningful even after
    its clip is pruned) by clearing recording_id rather than cascading.
    Returns the on-disk paths (the clip and its snapshot) the caller
    should remove after a successful commit."""
    paths = [recording.file_path, recording_snapshot_path(recording)]
    db.query(Event).filter(Event.recording_id == recording.id).update({"recording_id": None})
    db.delete(recording)
    return paths


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
    clip_dir = str(upload_clip_dir(upload_id))
    try:
        os.rmdir(clip_dir)
    except FileNotFoundError:
        pass  # No fall was ever recorded for this upload.
    except OSError as exc:
        logger.warning("Left %s in place while deleting upload %s: %s", clip_dir, upload_id, exc)
