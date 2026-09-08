"""
Customer video upload + offline analysis API.

Upload hardening lives here rather than in the analyser: by the time
video_analysis.py opens the file it is already on disk, so extension,
size, container-format and per-user quota checks all have to happen on
the way in.
"""
import logging
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.core.deps import get_current_user
from app.core.ranges import serve_file_range
from app.core.video_signature import looks_like_supported_video
from app.database import get_db
from app.models.user import User
from app.models.video_upload import VideoUpload
from app.schemas.video_upload import VideoUploadOut
from app.services.cascade_delete import stage_delete_video_upload
from app.services.video_analysis import analyze_video_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video-uploads", tags=["video-uploads"])

CHUNK_SIZE = 1024 * 1024  # 1 MB
MAX_UPLOAD_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
# Enough of the first chunk to cover every container signature we check
# (the longest, Matroska/WebM's EBML DocType, sits within the first 64
# bytes in practice; 256 leaves generous headroom).
HEADER_SNIFF_BYTES = 256


def _get_upload_or_404(upload_id: int, db: Session) -> VideoUpload:
    upload = db.query(VideoUpload).filter(VideoUpload.id == upload_id).first()
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video upload not found")
    return upload


def _ensure_can_view(upload: VideoUpload, user: User):
    if upload.user_id != user.id and not user.is_operator:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this upload")


def _resolved_upload_path(upload: VideoUpload) -> str:
    """The stored file's real path, verified to still sit inside
    UPLOADS_DIR. `stored_path` is server-generated (a UUID under the
    owner's directory) so it cannot be attacker-controlled today, but
    this endpoint reads an arbitrary file off disk based on a database
    value - re-checking containment here means a future bug that lets a
    path into that column cannot turn into arbitrary file disclosure."""
    uploads_root = Path(settings.UPLOADS_DIR).resolve()
    try:
        resolved = Path(upload.stored_path).resolve(strict=True)
        resolved.relative_to(uploads_root)
    except (OSError, ValueError):
        logger.error(
            "VideoUpload %s stored_path %r does not resolve inside %s",
            upload.id, upload.stored_path, uploads_root,
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Uploaded file missing on disk")
    return str(resolved)


@router.post("", response_model=VideoUploadOut, status_code=status.HTTP_201_CREATED)
async def create_video_upload(
    file: UploadFile,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    original_filename = file.filename or "upload.mp4"
    extension = Path(original_filename).suffix.lower()
    if extension not in settings.allowed_video_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{extension}'. Allowed: {', '.join(settings.allowed_video_extensions)}",
        )

    # Per-user storage quota. Without it any account can fill the disk
    # one allowed-size upload at a time, taking down analysis and
    # recording for everyone.
    if settings.MAX_UPLOAD_STORAGE_PER_USER_MB > 0:
        used_bytes = (
            db.query(func.coalesce(func.sum(VideoUpload.file_size_bytes), 0))
            .filter(VideoUpload.user_id == current_user.id)
            .scalar()
            or 0
        )
        quota_bytes = settings.MAX_UPLOAD_STORAGE_PER_USER_MB * 1024 * 1024
        if used_bytes >= quota_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"You have used your {settings.MAX_UPLOAD_STORAGE_PER_USER_MB}MB upload quota. "
                    "Delete an existing video to free up space."
                ),
            )

    user_dir = Path(settings.UPLOADS_DIR) / str(current_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    # Never trust the client-supplied filename for the on-disk path - a
    # crafted filename like "../../etc/passwd" or one containing path
    # separators must not be able to escape user_dir.
    stored_filename = f"{uuid.uuid4().hex}{extension}"
    stored_path = user_dir / stored_filename

    size = 0
    header = b""
    try:
        with open(stored_path, "wb") as out_file:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                if len(header) < HEADER_SNIFF_BYTES:
                    header += chunk[: HEADER_SNIFF_BYTES - len(header)]
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE_BYTES:
                    out_file.close()
                    stored_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB upload limit",
                    )
                out_file.write(chunk)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to save uploaded video for user %s", current_user.id)
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save uploaded file")

    if size == 0:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    # The extension check above only reflects what the client *called* the
    # file. Verify the bytes actually are one of the container formats we
    # accept, so a renamed archive, script or HTML page never reaches the
    # decoder or lands in storage under a video extension.
    if not looks_like_supported_video(header):
        stored_path.unlink(missing_ok=True)
        logger.warning(
            "User %s uploaded %r with a %s extension but an unrecognised container signature",
            current_user.id, original_filename, extension,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "This file does not look like a supported video. Allowed formats: "
                f"{', '.join(settings.allowed_video_extensions)}"
            ),
        )

    upload = VideoUpload(
        user_id=current_user.id,
        original_filename=original_filename,
        stored_path=str(stored_path),
        file_size_bytes=size,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    analyze_video_upload(upload.id)

    return upload


@router.get("", response_model=List[VideoUploadOut])
def list_video_uploads(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(VideoUpload)
    if not current_user.is_operator:
        query = query.filter(VideoUpload.user_id == current_user.id)
    return query.order_by(VideoUpload.created_at.desc()).all()


@router.get("/{upload_id}", response_model=VideoUploadOut)
def get_video_upload(
    upload_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    upload = _get_upload_or_404(upload_id, db)
    _ensure_can_view(upload, current_user)
    return upload


@router.get("/{upload_id}/video")
def stream_video_upload(
    upload_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),  # accepts header OR ?token= query param
):
    upload = _get_upload_or_404(upload_id, db)
    _ensure_can_view(upload, current_user)

    return serve_file_range(_resolved_upload_path(upload), request)


@router.delete("/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video_upload(
    upload_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    upload = _get_upload_or_404(upload_id, db)
    _ensure_can_view(upload, current_user)

    paths = stage_delete_video_upload(db, upload)
    db.commit()

    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    return None
