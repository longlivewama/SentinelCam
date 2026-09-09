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
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.deps import (
    MEDIA_KIND_RECORDING,
    MediaAccess,
    get_current_user,
    issue_media_token,
    require_operator,
)
from app.core.pagination import Page, PageParams, paginate
from app.core.ranges import media_type_for, serve_file_range
from app.core.scoping import can_access_upload_id, scope_recordings
from app.database import get_db
from app.models.recording import Recording
from app.services.cascade_delete import stage_delete_recording
from app.models.user import User
from app.schemas.media import MediaTokenOut
from app.schemas.recording import RecordingOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recordings", tags=["recordings"])

# Reached by <video src> and <a href download>, neither of which can send
# an Authorization header - so they take a short-lived, clip-scoped media
# token in the query string instead (see core/deps.MediaAccess).
_media_access = MediaAccess(MEDIA_KIND_RECORDING, "recording_id")



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


@router.get("", response_model=Page[RecordingOut])
def list_recordings(
    camera_id: Optional[int] = None,
    video_upload_id: Optional[int] = None,
    trigger_action: Optional[str] = None,
    page: PageParams = Depends(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Newest first. A camera left running produces clips indefinitely, so
    this is paged rather than bounded by a `limit` that put older events
    permanently out of reach.

    `trigger_action` filters server-side because paging made client-side
    filtering wrong: filtering the twenty rows that happened to be on the
    current page hides matches on every other page, and leaves the page
    count describing the unfiltered list."""
    # Ownership filter first, and in SQL, so `total` counts only what this
    # user may see and no later branch can widen it.
    query = scope_recordings(db.query(Recording), current_user)
    if camera_id is not None:
        query = query.filter(Recording.camera_id == camera_id)
    if video_upload_id is not None:
        query = query.filter(Recording.video_upload_id == video_upload_id)
    if trigger_action is not None:
        query = query.filter(Recording.trigger_action == trigger_action)
    # id breaks ties: clips written in the same transaction share a
    # timestamp, and without it a row could appear on two pages.
    return paginate(query, page, Recording.event_timestamp.desc(), Recording.id.desc())


@router.get("/{recording_id}", response_model=RecordingOut)
def get_recording(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _get_recording_or_404(recording_id, db, current_user)


@router.post("/{recording_id}/media-token", response_model=MediaTokenOut)
def create_recording_media_token(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mints a token that plays or downloads THIS clip and nothing else.

    Deliberately runs the same `_get_recording_or_404` the streaming
    endpoints run, so a token is never issued for a recording the caller
    could not already fetch - and a caller probing for clips they cannot
    see gets the same 404 here as everywhere else."""
    recording = _get_recording_or_404(recording_id, db, current_user)
    return issue_media_token(current_user, MEDIA_KIND_RECORDING, recording.id)


@router.get("/{recording_id}/video")
def stream_video(
    recording_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(_media_access),
):
    recording = _get_recording_or_404(recording_id, db, current_user)
    file_path = _existing_file_or_404(recording)
    return serve_file_range(file_path, request)


@router.get("/{recording_id}/download")
def download_recording(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(_media_access),
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
