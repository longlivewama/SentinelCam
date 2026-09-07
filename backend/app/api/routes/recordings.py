import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_operator
from app.database import get_db
from app.models.recording import Recording
from app.services.cascade_delete import stage_delete_recording
from app.models.user import User
from app.schemas.recording import RecordingOut

router = APIRouter(prefix="/recordings", tags=["recordings"])

CHUNK_SIZE = 1024 * 1024  # 1 MB


def _get_recording_or_404(recording_id: int, db: Session) -> Recording:
    recording = db.query(Recording).filter(Recording.id == recording_id).first()
    if recording is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found")
    return recording


@router.get("", response_model=List[RecordingOut])
def list_recordings(
    camera_id: Optional[int] = None,
    video_upload_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Recording)
    if camera_id is not None:
        query = query.filter(Recording.camera_id == camera_id)
    if video_upload_id is not None:
        query = query.filter(Recording.video_upload_id == video_upload_id)
    return query.order_by(Recording.event_timestamp.desc()).all()


@router.get("/{recording_id}", response_model=RecordingOut)
def get_recording(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _get_recording_or_404(recording_id, db)


@router.get("/{recording_id}/video")
def stream_video(
    recording_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),  # accepts header OR ?token= query param
):
    recording = _get_recording_or_404(recording_id, db)
    file_path = recording.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording file missing on disk")

    file_size = os.path.getsize(file_path)
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
        with open(file_path, "rb") as f:
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


@router.get("/{recording_id}/download")
def download_recording(
    recording_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),  # accepts header OR ?token= query param
):
    recording = _get_recording_or_404(recording_id, db)
    if not os.path.exists(recording.file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording file missing on disk")

    return FileResponse(
        path=recording.file_path,
        media_type="video/mp4",
        filename=recording.filename,
        content_disposition_type="attachment",
    )


@router.delete("/{recording_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recording(
    recording_id: int,
    db: Session = Depends(get_db),
    _operator: User = Depends(require_operator),
):
    recording = _get_recording_or_404(recording_id, db)
    file_path = recording.file_path

    stage_delete_recording(db, recording)
    db.commit()

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

    return None
