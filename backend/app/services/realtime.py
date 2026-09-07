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
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class RealtimeBroadcaster:
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connections: Set[WebSocket] = set()
        self._lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            self._connections.discard(websocket)

    def publish(self, event_type: str, data: Dict[str, Any]):
        """Safe to call from any thread, including from outside the
        asyncio event loop (e.g. a detection worker thread). No-op if no
        loop is bound yet (app hasn't finished starting) or no clients are
        connected."""
        if self._loop is None:
            return
        with self._lock:
            connections = list(self._connections)
        if not connections:
            return
        message = json.dumps({"type": event_type, "data": data}, default=str)
        for ws in connections:
            try:
                asyncio.run_coroutine_threadsafe(self._safe_send(ws, message), self._loop)
            except RuntimeError:
                logger.debug("Event loop unavailable while publishing realtime event; dropping")

    async def _safe_send(self, websocket: WebSocket, message: str):
        try:
            await websocket.send_text(message)
        except Exception:
            self.disconnect(websocket)


realtime_broadcaster = RealtimeBroadcaster()
