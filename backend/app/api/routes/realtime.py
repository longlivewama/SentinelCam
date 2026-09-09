"""
WebSocket endpoint the frontend uses to receive alerts, camera status
changes, and video-upload progress the instant they happen, instead of
polling. See app/services/realtime.py for the publish side (called from
background worker threads in recording_engine.py, stream_manager.py, and
video_analysis.py).

Browsers' native WebSocket API cannot attach custom headers, so - same as
the MJPEG stream and video/download endpoints - the JWT is passed as a
`?token=` query parameter rather than an Authorization header.

Two things the handshake establishes and the connection then honours for
its whole lifetime:

  * WHO the subscriber is. Delivery is scoped to that identity (see
    services/realtime.py), so a socket only receives events the same user
    could read over REST.
  * WHEN their token expires. A WebSocket is long-lived, so unlike an HTTP
    request it cannot re-check authorization per call: a socket opened one
    minute before expiry would otherwise keep streaming alerts for as long
    as it stayed open. The connection is closed at the token's own `exp`.
"""
import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.security import decode_access_token, is_media_token
from app.database import SessionLocal
from app.models.user import User
from app.services.realtime import realtime_broadcaster

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])

# Close codes: 4401 mirrors HTTP 401 (bad/absent credentials), 4403 is
# used specifically for "your token expired while connected" so the
# frontend can tell an auth failure apart from a network drop and stop
# retrying with a token it now knows is dead.
WS_UNAUTHORIZED = 4401
WS_TOKEN_EXPIRED = 4403


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket, token: str):
    payload = decode_access_token(token)
    if payload is None:
        await websocket.close(code=WS_UNAUTHORIZED)
        return

    # A media token is a read credential for one clip (see
    # core/security.py). This channel delivers every alert, camera status
    # change and upload progress event the subscriber can see, so honouring
    # one here would make a token handed out for a `<video src>` a live
    # feed of the account - the exact escalation the media/session split
    # exists to prevent. The session token is the only credential for this
    # endpoint.
    if is_media_token(payload):
        await websocket.close(code=WS_UNAUTHORIZED)
        return

    user_id = payload.get("user_id")
    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        user_is_valid = user is not None and user.is_active
        is_operator = bool(user and user.is_operator)

    if not user_is_valid:
        await websocket.close(code=WS_UNAUTHORIZED)
        return

    subscriber = await realtime_broadcaster.connect(websocket, user_id=user_id, is_operator=is_operator)
    expiry_task = asyncio.create_task(_close_at_token_expiry(websocket, payload.get("exp")))
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
        expiry_task.cancel()
        realtime_broadcaster.disconnect(subscriber)


async def _close_at_token_expiry(websocket: WebSocket, exp):
    """Closes the socket once the presented token expires. Cancelled by
    the caller when the client disconnects first."""
    if not exp:
        return
    try:
        seconds_left = float(exp) - datetime.now(timezone.utc).timestamp()
    except (TypeError, ValueError):
        return
    if seconds_left > 0:
        await asyncio.sleep(seconds_left)
    try:
        await websocket.close(code=WS_TOKEN_EXPIRED)
    except Exception:
        pass
