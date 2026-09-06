"""
Per-camera background capture threads.

For each camera that is actively being viewed (via /stream) or has AI
detection enabled, a single daemon thread owns the `cv2.VideoCapture`
connection, continuously reading frames, JPEG-encoding the latest one for
MJPEG streaming, and maintaining a rolling deque of raw frames that
recording_engine uses to build pre-event footage.

Only one capture connection is ever opened per camera regardless of how
many HTTP clients are watching the stream concurrently - every consumer
(the /stream endpoint generator, the detection engine, the recording
engine) reads from the shared, lock-protected "latest frame" holder owned
by this module.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

import cv2

from app.config import settings
from app.database import SessionLocal
from app.models.camera import Camera

logger = logging.getLogger(__name__)


def _resolve_source(url: str):
    """USB webcams are addressed by a numeric device index; RTSP/HTTP
    camera URLs are left as-is. cv2.VideoCapture wants an int for the
    former and a str for the latter."""
    try:
        return int(url)
    except (TypeError, ValueError):
        return url


class CameraStream:
    """Owns the capture loop + shared state for a single camera."""

    def __init__(self, camera_id: int, url: str):
        self.camera_id = camera_id
        self.url = url

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame = None  # raw ndarray, most recently captured
        self._frame_buffer = deque(maxlen=settings.ROLLING_BUFFER_FRAMES)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._last_status_written: Optional[str] = None

    # -- lifecycle -----------------------------------------------------

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camstream-{self.camera_id}")
        self._thread.start()

    def stop(self):
        self._running = False

    # -- consumers -------------------------------------------------------

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def get_buffer_snapshot(self) -> list:
        """Copy of the rolling pre-event buffer (oldest first)."""
        with self._lock:
            return [f.copy() for f in self._frame_buffer]

    @property
    def is_alive(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    # -- capture loop ----------------------------------------------------

    def _set_status(self, status: str):
        if status == self._last_status_written:
            return
        self._last_status_written = status
        try:
            with SessionLocal() as db:
                camera = db.query(Camera).filter(Camera.id == self.camera_id).first()
                if camera and camera.status != status:
                    camera.status = status
                    db.commit()
        except Exception:
            logger.exception("Failed to update status for camera %s", self.camera_id)

    def _run(self):
        source = _resolve_source(self.url)
        backoff = 1.0
        max_backoff = 30.0

        while self._running:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self._consecutive_failures += 1
                logger.warning(
                    "Camera %s: failed to open source %r (attempt %d)",
                    self.camera_id, source, self._consecutive_failures,
                )
                if self._consecutive_failures >= 3:
                    self._set_status("error")
                cap.release()
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            self._set_status("active")
            self._consecutive_failures = 0
            backoff = 1.0
            frame_interval = 1.0 / max(settings.STREAM_FPS, 1)

            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._consecutive_failures += 1
                    logger.warning("Camera %s: frame read failed (attempt %d)", self.camera_id, self._consecutive_failures)
                    if self._consecutive_failures >= 3:
                        self._set_status("error")
                    break

                self._consecutive_failures = 0
                ok, jpeg = cv2.imencode(".jpg", frame)
                with self._lock:
                    self._latest_frame = frame
                    self._frame_buffer.append(frame)
                    if ok:
                        self._latest_jpeg = jpeg.tobytes()

                time.sleep(frame_interval)

            cap.release()
            if self._running:
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)


class StreamManager:
    """Registry of CameraStream instances, keyed by camera id."""

    def __init__(self):
        self._streams: dict[int, CameraStream] = {}
        self._lock = threading.Lock()

    def get_or_create(self, camera_id: int, url: str) -> CameraStream:
        with self._lock:
            stream = self._streams.get(camera_id)
            if stream is None or not stream.is_alive:
                stream = CameraStream(camera_id, url)
                self._streams[camera_id] = stream
                stream.start()
            return stream

    def get(self, camera_id: int) -> Optional[CameraStream]:
        with self._lock:
            return self._streams.get(camera_id)

    def stop_all(self):
        with self._lock:
            for stream in self._streams.values():
                stream.stop()


stream_manager = StreamManager()
