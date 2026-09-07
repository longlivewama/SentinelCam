import logging
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.core.deps import get_current_user
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


def _get_upload_or_404(upload_id: int, db: Session) -> VideoUpload:
    upload = db.query(VideoUpload).filter(VideoUpload.id == upload_id).first()
    if upload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video upload not found")
    return upload


def _ensure_can_view(upload: VideoUpload, user: User):
    if upload.user_id != user.id and not user.is_operator:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this upload")


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

    user_dir = Path(settings.UPLOADS_DIR) / str(current_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    # Never trust the client-supplied filename for the on-disk path - a
    # crafted filename like "../../etc/passwd" or one containing path
    # separators must not be able to escape user_dir.
    stored_filename = f"{uuid.uuid4().hex}{extension}"
    stored_path = user_dir / stored_filename

    size = 0
    try:
        with open(stored_path, "wb") as out_file:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
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

    if not os.path.exists(upload.stored_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Uploaded file missing on disk")

    file_size = os.path.getsize(upload.stored_path)
    range_header = request.headers.get("range")

    start, end, status_code = 0, file_size - 1, status.HTTP_200_OK
    if range_header:
        status_code = status.HTTP_206_PARTIAL_CONTENT
        range_value = range_header.strip().lower().replace("bytes=", "")
        start_str, _, end_str = range_value.partition("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)

    content_length = max(end - start + 1, 0)

    def iterfile():
        with open(upload.stored_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = f.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
    }
    return StreamingResponse(iterfile(), status_code=status_code, headers=headers, media_type="video/mp4")


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
