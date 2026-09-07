import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.core.deps import get_current_user, require_operator
from app.database import get_db
from app.models.camera import Camera
from app.models.user import User
from app.schemas.camera import CameraCreate, CameraOut, CameraUpdate
from app.services.stream_manager import stream_manager
from app.services.detection.engine import detection_engine

router = APIRouter(prefix="/cameras", tags=["cameras"])


def _sync_detection_state(camera: Camera):
    """Start/stop the background detection loop for this camera to match
    its current is_active / ai_detection_enabled flags."""
    if camera.is_active and camera.ai_detection_enabled:
        detection_engine.ensure_running(camera.id, camera.url)
    else:
        detection_engine.stop(camera.id)


@router.get("", response_model=List[CameraOut])
def list_cameras(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Camera).filter(Camera.is_active == True).all()  # noqa: E712


@router.post("", response_model=CameraOut, status_code=status.HTTP_201_CREATED)
def create_camera(
    payload: CameraCreate,
    db: Session = Depends(get_db),
    _operator: User = Depends(require_operator),
):
    camera = Camera(**payload.model_dump())
    db.add(camera)
    db.commit()
    db.refresh(camera)
    _sync_detection_state(camera)
    return camera


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return camera


@router.put("/{camera_id}", response_model=CameraOut)
def update_camera(
    camera_id: int,
    payload: CameraUpdate,
    db: Session = Depends(get_db),
    _operator: User = Depends(require_operator),
):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(camera, field, value)

    db.commit()
    db.refresh(camera)
    _sync_detection_state(camera)
    return camera


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    _operator: User = Depends(require_operator),
):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    camera.is_active = False
    db.commit()
    detection_engine.stop(camera.id)
    stream = stream_manager.get(camera.id)
    if stream is not None:
        stream.stop()
    return None


@router.get("/{camera_id}/stream")
def stream_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),  # accepts header OR ?token= query param
):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if camera is None or not camera.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    # Lazily start (or reuse) the single capture thread for this camera.
    stream = stream_manager.get_or_create(camera.id, camera.url)

    def generate():
        interval = 1.0 / max(settings.STREAM_FPS, 1)
        boundary = b"--frame"
        while True:
            jpeg = stream.get_latest_jpeg()
            if jpeg is not None:
                yield (
                    boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" +
                    jpeg + b"\r\n"
                )
            time.sleep(interval)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
