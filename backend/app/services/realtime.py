"""
Thread-safe pub/sub bridge between background worker threads (the
per-camera detection loops in detection/engine.py, the recording engine,
and the video-upload analysis worker - all plain Python threads, not
asyncio tasks) and WebSocket clients connected to /api/ws/events (served
on FastAPI's asyncio event loop).

Worker threads call `realtime_broadcaster.publish(...)`, a plain
synchronous call safe from any thread. Internally it hops onto the bound
event loop via `run_coroutine_threadsafe` to actually push bytes to each
connected socket, so a slow/broken client can never block a detection
thread.

Delivery is scoped per subscriber, not broadcast to everyone. Events
about shared infrastructure (a camera's alerts, a camera's status) go to
every authenticated subscriber, matching the REST API where every user
can list every camera. Events about one user's uploaded video -
`upload.progress`, `upload.completed`, and the `alert.created` for a fall
found in that video, which carries the source filename - go only to that
user and to operators/admins. This mirrors the row-level rules in
core/scoping.py: without it the socket would hand every connected client
another user's filenames and analysis results, quietly undoing the
authorization the REST endpoints enforce.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, NamedTuple, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class Subscriber(NamedTuple):
    """Who is on the other end of a connected socket. Captured at connect
    time from the same JWT the REST API validates."""

    websocket: WebSocket
    user_id: int
    is_operator: bool


class RealtimeBroadcaster:
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: Set[Subscriber] = set()
        self._lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    @property
    def connection_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    async def connect(self, websocket: WebSocket, user_id: int, is_operator: bool) -> Subscriber:
        await websocket.accept()
        subscriber = Subscriber(websocket=websocket, user_id=user_id, is_operator=is_operator)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def disconnect(self, subscriber: Subscriber):
        with self._lock:
            self._subscribers.discard(subscriber)

    def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        owner_user_id: Optional[int] = None,
    ):
        """Safe to call from any thread, including from outside the
        asyncio event loop (e.g. a detection worker thread). No-op if no
        loop is bound yet (app hasn't finished starting) or nobody is
        subscribed.

        `owner_user_id` is the user this event is private to. Leave it
        None only for events about shared infrastructure that every
        authenticated user may see; pass the owning user's ID for
        anything derived from their uploaded video."""
        if self._loop is None:
            return

        with self._lock:
            subscribers = [s for s in self._subscribers if _may_receive(s, owner_user_id)]
        if not subscribers:
            return

        message = json.dumps({"type": event_type, "data": data}, default=str)
        for subscriber in subscribers:
            try:
                asyncio.run_coroutine_threadsafe(self._safe_send(subscriber, message), self._loop)
            except RuntimeError:
                logger.debug("Event loop unavailable while publishing realtime event; dropping")

    async def _safe_send(self, subscriber: Subscriber, message: str):
        try:
            await subscriber.websocket.send_text(message)
        except Exception:
            self.disconnect(subscriber)


def _may_receive(subscriber: Subscriber, owner_user_id: Optional[int]) -> bool:
    if owner_user_id is None:
        return True
    return subscriber.user_id == owner_user_id or subscriber.is_operator


realtime_broadcaster = RealtimeBroadcaster()
