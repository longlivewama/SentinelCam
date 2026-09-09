"""
Access to event clips: listing, metadata, byte-range streaming for the
in-browser player, and download.

Authorization: recordings derived from a user's uploaded video belong to
that user; recordings from a shared camera are visible to everyone
authenticated. Both rules come from core/scoping.py so every route here
enforces the same one - see that module for why.
"""
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_operator
from app.core.ranges import media_type_for, serve_file_range
from app.core.scoping import can_access_upload_id, scope_recordings
from app.database import get_db
from app.models.recording import Recording
from app.services.cascade_delete import stage_delete_recording
from app.models.user import User
from app.schemas.recording import RecordingOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recordings", tags=["recordings"])

# Newest-first with a bound rather than an unbounded fetch: a camera left
# running produces clips indefinitely, and "return every row" turns into a
# multi-megabyte response and a browser rendering thousands of rows. The
# cap is generous enough that no realistic UI hits it; real pagination is
# on the roadmap.
DEFAULT_LIMIT = 500
MAX_LIMIT = 1000



def _get_recording_or_404(recording_id: int, db: Session, user: User) -> Recording:
    """404 rather than 403 for a recording the caller may not see - see
    alerts.py's _get_visible_alert_or_404 for why."""
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if recording is None or not can_access_upload_id(db, recording.video_upload_id, user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")
    return recording


def _existing_file_or_404(recording: Recording) -> str:
    """The clip's path on disk, verified to still exist. A missing file
    is an operational problem, not a client error, so it is logged for
    operators - the caller only learns the clip is gone."""
    if not os.path.exists(recording.file_path):
        logger.error(
            "Recording %s references %s, which is missing on disk",
            recording.id, recording.file_path,
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording file missing on disk")
    return recording.file_path


@router.get("", response_model=List[RecordingOut])
def list_recordings(
    camera_id: Optional[int] = None,
    video_upload_id: Optional[int] = None,
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = scope_recordings(db.query(Recording), current_user)
    if camera_id is not None:
        query = query.filter(Recording.camera_id == camera_id)
    if video_upload_id is not None:
        query = query.filter(Recording.video_upload_id == video_upload_id)
    return query.order_by(Recording.event_timestamp.desc()).limit(limit).all()


@router.get("/{recording_id}", response_model=RecordingOut)
def get_recording(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _get_recording_or_404(recording_id, db, current_user)


@router.get("/{recording_id}/video")
def stream_video(
    recording_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),  # accepts header OR ?token= query param
):
    recording = _get_recording_or_404(recording_id, db, current_user)
    file_path = _existing_file_or_404(recording)
    return serve_file_range(file_path, request)


@router.get("/{recording_id}/download")
def download_recording(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),  # accepts header OR ?token= query param
):
    recording = _get_recording_or_404(recording_id, db, current_user)
    file_path = _existing_file_or_404(recording)

    return FileResponse(
        path=file_path,
        # Follows the container the clip was actually encoded into - see
        # recording_engine.open_writer; not every clip is an MP4.
        media_type=media_type_for(file_path),
        filename=recording.filename,
        content_disposition_type="attachment",
    )


@router.delete("/{recording_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recording(
    recording_id: int,
    db: Session = Depends(get_db),
    operator: User = Depends(require_operator),
):
    recording = _get_recording_or_404(recording_id, db, operator)
    file_path = recording.file_path

    stage_delete_recording(db, recording)
    db.commit()

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    return None
