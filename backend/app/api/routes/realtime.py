"""
WebSocket endpoint the frontend uses to receive alerts, camera status
changes, and video-upload progress the instant they happen, instead of
polling. See app/services/realtime.py for the publish side (called from
background worker threads in recording_engine.py, stream_manager.py, and
video_analysis.py).

Browsers' native WebSocket API cannot attach custom headers, so - same as
the MJPEG stream and video/download endpoints - the JWT is passed as a
`?token=` query parameter rather than an Authorization header.
"""
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.security import decode_access_token
from app.database import SessionLocal
from app.models.user import User
from app.services.realtime import realtime_broadcaster

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket, token: str):
    payload = decode_access_token(token)
    if payload is None:
        await websocket.close(code=4401)
        return

    user_id = payload.get("user_id")
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        user_is_valid = user is not None and user.is_active

    if not user_is_valid:
        await websocket.close(code=4401)
        return

    await realtime_broadcaster.connect(websocket)
    try:
        while True:
            # The frontend never sends messages on this channel; we just
            # need to block here so the connection stays open and we
            # notice a disconnect promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Unexpected error on realtime websocket")
    finally:
        realtime_broadcaster.disconnect(websocket)
