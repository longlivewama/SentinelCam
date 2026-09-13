import time
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.core.camera_url import redact_credentials
from app.core.deps import (
    MEDIA_KIND_CAMERA,
    MediaAccess,
    get_current_user,
    issue_media_token,
    require_operator,
)
from app.database import get_db
from app.models.camera import Camera
from app.models.user import User
from app.schemas.camera import CameraCreate, CameraOut, CameraUpdate
from app.schemas.media import MediaTokenOut
from app.services.stream_manager import stream_manager
from app.services.detection.engine import detection_engine

router = APIRouter(prefix="/cameras", tags=["cameras"])

# The MJPEG stream is consumed by <img src>, which cannot send an
# Authorization header - see core/deps.MediaAccess.
_media_access = MediaAccess(MEDIA_KIND_CAMERA, "camera_id")


def _sync_detection_state(camera: Camera):
    """Start/stop the background detection loop - and, when the camera is
    switched off, the underlying capture thread too - to match its current
    is_active / ai_detection_enabled flags.

    A camera turned off must stop ALL live processing, not just AI
    inference: nobody can reach its stream (see stream_camera below, which
    404s once is_active is False) so a capture thread still polling the
    hardware afterwards would just be wasted work. Turning it back on
    needs no symmetric start here - the capture thread starts lazily,
    exactly as it always has, the next time someone opens the stream or
    detection is (re)enabled."""
    if not camera.is_active:
        detection_engine.stop(camera.id)
        stream = stream_manager.get(camera.id)
        if stream is not None:
            stream.stop()
        return

    if camera.ai_detection_enabled:
        detection_engine.ensure_running(camera.id, camera.url)
    else:
        detection_engine.stop(camera.id)


def _visible_camera(camera: Camera, user: User) -> CameraOut:
    """The camera as `user` is allowed to see it.

    Cameras are shared infrastructure - every authenticated user may list
    them and watch their streams - but an IP camera's `url` carries its
    credentials inline (`rtsp://admin:hunter2@host/...`), and that is the
    camera's password, not a description of it. Handing it to a `viewer`
    gave the least-privileged account in the deployment direct RTSP access
    to the hardware, outside this application and outside every check it
    makes; the same credentials usually open the camera's own admin web UI
    too.

    Operators and admins keep the real value: they are the roles that may
    create and update cameras, the edit form is populated from this
    response, and masking it for them would overwrite a working camera's
    configuration with a redaction on the next save."""
    out = CameraOut.model_validate(camera)
    if not user.is_operator:
        out.url = redact_credentials(camera.url)
    return out


@router.get("", response_model=List[CameraOut])
def list_cameras(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Deleted cameras drop out of the list; merely-off ones (is_active is
    # False but deleted_at is unset) stay visible so they can be switched
    # back on - see _sync_detection_state's docstring above.
    cameras = db.query(Camera).filter(Camera.deleted_at.is_(None)).all()
    return [_visible_camera(camera, current_user) for camera in cameras]


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
    camera = db.query(Camera).filter(Camera.id == camera_id, Camera.deleted_at.is_(None)).first()
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return _visible_camera(camera, current_user)


@router.put("/{camera_id}", response_model=CameraOut)
def update_camera(
    camera_id: int,
    payload: CameraUpdate,
    db: Session = Depends(get_db),
    _operator: User = Depends(require_operator),
):
    camera = db.query(Camera).filter(Camera.id == camera_id, Camera.deleted_at.is_(None)).first()
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
    camera = db.query(Camera).filter(Camera.id == camera_id, Camera.deleted_at.is_(None)).first()
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    camera.is_active = False
    camera.deleted_at = datetime.now(timezone.utc)
    db.commit()
    _sync_detection_state(camera)
    return None


@router.post("/{camera_id}/media-token", response_model=MediaTokenOut)
def create_camera_media_token(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mints a token that opens THIS camera's MJPEG stream and nothing
    else. Applies the same active-camera check the stream endpoint
    applies, so a token is never minted for a camera that could not be
    watched anyway."""
    camera = db.query(Camera).filter(Camera.id == camera_id, Camera.deleted_at.is_(None)).first()
    if camera is None or not camera.is_active:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return issue_media_token(current_user, MEDIA_KIND_CAMERA, camera.id)


@router.get("/{camera_id}/stream")
def stream_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(_media_access),
):
    camera = db.query(Camera).filter(Camera.id == camera_id, Camera.deleted_at.is_(None)).first()
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
