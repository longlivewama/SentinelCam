"""
Rolling-buffer + event-triggered clip writer.

The capture loop in stream_manager.CameraStream continuously maintains a
rolling deque of the last ROLLING_BUFFER_FRAMES raw frames per camera
(roughly ROLLING_BUFFER_FRAMES / STREAM_FPS seconds of pre-event footage,
e.g. 90 frames / 30fps = 3s). When detection/engine.py reports an event,
this module:

  1. Snapshots the current buffer (pre-event frames).
  2. Keeps collecting live frames from the stream for POST_EVENT_SECONDS
     more seconds.
  3. Concatenates pre+post frames and writes them out via cv2.VideoWriter
     to storage/recordings/{camera_id}/{event_timestamp}_{trigger_action}.mp4
  4. Persists a Recording row, a snapshot JPEG, an Event row, and sends an
     alert email.

The collection + write work runs on a background thread so the calling
detection loop is never blocked for POST_EVENT_SECONDS. Per-event
debounce (not re-firing repeatedly for an ongoing condition) is the
responsibility of each detector module, not this one.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from app.config import settings
from app.database import SessionLocal
from app.models.camera import Camera
from app.models.event import Event
from app.models.recording import Recording
from app.services import email_service
from app.services.stream_manager import stream_manager

logger = logging.getLogger(__name__)


def trigger_event(camera_id: int, trigger_action: str, confidence_score: float = 1.0):
    """Fire-and-forget entry point called by detection/engine.py. Spawns a
    worker thread so the detection loop isn't blocked collecting
    post-event frames."""
    thread = threading.Thread(
        target=_handle_event,
        args=(camera_id, trigger_action, confidence_score),
        daemon=True,
        name=f"recording-{camera_id}-{trigger_action}",
    )
    thread.start()


def _handle_event(camera_id: int, trigger_action: str, confidence_score: float):
    event_timestamp = datetime.now(timezone.utc)
    stream = stream_manager.get(camera_id)
    if stream is None:
        logger.warning("No active stream for camera %s; cannot record event %s", camera_id, trigger_action)
        return

    # Pre-event footage: whatever the rolling buffer currently holds.
    pre_frames = stream.get_buffer_snapshot()

    # Post-event footage: keep sampling the live feed for a few more seconds.
    fps = max(settings.STREAM_FPS, 1)
    frame_interval = 1.0 / fps
    post_frames = []
    deadline = time.monotonic() + settings.POST_EVENT_SECONDS
    while time.monotonic() < deadline:
        frame = stream.get_latest_frame()
        if frame is not None:
            post_frames.append(frame)
        time.sleep(frame_interval)

    all_frames = pre_frames + post_frames
    if not all_frames:
        logger.warning("No frames available; skipping recording for camera %s event %s", camera_id, trigger_action)
        return

    try:
        _write_recording(camera_id, trigger_action, confidence_score, event_timestamp, all_frames, pre_frames)
    except Exception:
        logger.exception("Failed to persist recording for camera %s event %s", camera_id, trigger_action)


def _open_writer(path: str, width: int, height: int, fps: int):
    """Try 'avc1' (H.264, browser-compatible) first; fall back to 'mp4v'
    if it fails to open. Whether avc1 is available depends entirely on the
    OpenCV/ffmpeg build in the deployment environment - many stock
    opencv-python wheels do not bundle a licensed H.264 encoder, in which
    case we transparently fall back to MPEG-4 ('mp4v'), which is less
    broadly compatible (notably with Safari) but works everywhere OpenCV
    does. Which codec was actually used is logged for operators."""
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return codec, writer
        writer.release()
    return None, None


def _write_recording(camera_id, trigger_action, confidence_score, event_timestamp, all_frames, pre_frames):
    height, width = all_frames[0].shape[:2]
    fps = max(settings.STREAM_FPS, 1)

    camera_dir = Path(settings.RECORDINGS_DIR) / str(camera_id)
    camera_dir.mkdir(parents=True, exist_ok=True)

    ts_str = event_timestamp.strftime("%Y%m%dT%H%M%S%f")
    filename = f"{ts_str}_{trigger_action}.mp4"
    file_path = camera_dir / filename

    codec_used, writer = _open_writer(str(file_path), width, height, fps)
    if writer is None:
        logger.error("Could not open VideoWriter with any codec for %s", file_path)
        return

    for frame in all_frames:
        writer.write(frame)
    writer.release()

    file_size = file_path.stat().st_size if file_path.exists() else 0
    duration_seconds = len(all_frames) / fps

    # Snapshot: the frame closest to the actual event moment is the last
    # pre-event frame (falls back to the first post-event frame if the
    # rolling buffer happened to be empty).
    snapshot_frame = pre_frames[-1] if pre_frames else all_frames[0]
    snapshots_dir = Path(settings.SNAPSHOTS_DIR)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_filename = f"{camera_id}_{ts_str}_{trigger_action}.jpg"
    snapshot_path = snapshots_dir / snapshot_filename
    cv2.imwrite(str(snapshot_path), snapshot_frame)

    logger.info(
        "Camera %s: wrote recording %s (codec=%s, frames=%d, duration=%.2fs)",
        camera_id, file_path, codec_used, len(all_frames), duration_seconds,
    )

    with SessionLocal() as db:
        recording = Recording(
            camera_id=camera_id,
            filename=filename,
            file_path=str(file_path),
            duration_seconds=duration_seconds,
            trigger_action=trigger_action,
            file_size_bytes=file_size,
            event_timestamp=event_timestamp,
        )
        db.add(recording)

        event = Event(
            camera_id=camera_id,
            event_type=trigger_action,
            confidence_score=confidence_score,
            timestamp=event_timestamp,
            triggered_recording=True,
        )
        db.add(event)
        db.commit()

        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        camera_name = camera.name if camera else f"camera-{camera_id}"

    email_service.send_alert_email(
        event_type=trigger_action,
        camera_name=camera_name,
        timestamp=event_timestamp,
        snapshot_path=str(snapshot_path),
    )
