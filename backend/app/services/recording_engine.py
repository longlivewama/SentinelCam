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
     to storage/recordings/{camera_id}/{event_timestamp}_{trigger_action}.EXT,
     where EXT is the container required by the first codec in
     `_CODEC_LADDER` that this build can actually open - see `open_writer`.
  4. Persists a Recording row, a snapshot JPEG, an Event row, and sends an
     alert email.

The collection + write work runs on a background thread so the calling
detection loop is never blocked for POST_EVENT_SECONDS. Per-event
debounce (not re-firing repeatedly for an ongoing condition) is the
responsibility of each detector module, not this one.
"""
from __future__ import annotations

import logging
import os
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
from app.services.notifications import notification_service
from app.services.realtime import realtime_broadcaster
from app.services.stream_manager import stream_manager

logger = logging.getLogger(__name__)


def trigger_event(
    camera_id: int,
    trigger_action: str,
    confidence_score: float = 1.0,
    detector: str = "heuristic",
):
    """Fire-and-forget entry point called by detection/engine.py. Spawns a
    worker thread so the detection loop isn't blocked collecting
    post-event frames.

    `detector` records which strategy made the call ("model" for the
    trained YOLO fall detector, "heuristic" for the pose/geometry
    detectors) and is stored on the Event row - the two compute
    confidence differently, so a score is not interpretable without it."""
    thread = threading.Thread(
        target=_handle_event,
        args=(camera_id, trigger_action, confidence_score, detector),
        daemon=True,
        name=f"recording-{camera_id}-{trigger_action}",
    )
    thread.start()


def _handle_event(camera_id: int, trigger_action: str, confidence_score: float, detector: str = "heuristic"):
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
        _write_recording(camera_id, trigger_action, confidence_score, event_timestamp, all_frames, pre_frames, detector)
    except Exception:
        logger.exception("Failed to persist recording for camera %s event %s", camera_id, trigger_action)


# Codec ladder, best browser compatibility first. Each entry is the
# fourcc plus the container extension it must be muxed into - FFmpeg
# selects the muxer from the file extension, so the two cannot be chosen
# independently.
#
# Why this order, and why WebM is in it at all: a clip nobody can play is
# not a recording. `avc1` (H.264 in MP4) is the ideal - universally
# playable - but the PyPI `opencv-python-headless` wheels ship an FFmpeg
# built WITHOUT libx264 (it is GPL, so the wheels deliberately omit it),
# and there is no hardware encoder inside a container. Every H.264 fourcc
# therefore fails to open in the default deployment, which is exactly
# what the logs showed:
#
#     h264_v4l2m2m: Could not find a valid device
#     ... fallback to use tag 'mp4v'
#
# `mp4v` (MPEG-4 Part 2) opens fine, which is why it was silently chosen
# - but Chrome, Firefox and Edge cannot DECODE it. The clip downloaded,
# the range requests returned 206, and the <video> element still showed
# nothing. VP9/VP8 in WebM are supported by every current browser AND are
# present in the stock wheel's FFmpeg, so they are the first rungs that
# actually work here. `mp4v` is kept as the last resort: still the wrong
# answer for a browser, but better than failing to record at all, and it
# is now only ever reached if nothing above it opened.
#
# VP8 sits ahead of VP9 on measurement, not preference. Both are equally
# playable in every browser this targets, but libvpx-vp9 through OpenCV's
# VideoWriter has no way to set a speed/deadline, so it runs at its slow
# default. On the reference container (4 CPUs) encoding 180 frames of
# 576x1024 - one fall clip - cost:
#
#     vp09   26.61s   PSNR 39.91 dB   SSIM 0.9722   4.94 MB
#     VP80    5.58s   PSNR 39.64 dB   SSIM 0.9712   4.06 MB
#
# 4.8x faster, 18% smaller, and 0.27 dB apart - well inside the ~1 dB a
# viewer could notice, and VP8's WORST frame is actually the better of
# the two (38.92 vs 38.37 dB). Clip encoding runs inline in the upload
# analyser, so this was ~70% of an upload's wall-clock time.
_CODEC_LADDER = (
    ("avc1", ".mp4"),    # H.264 - ideal, needs a libx264-enabled FFmpeg
    ("VP80", ".webm"),   # VP8  - widest browser support, ~4.8x faster than VP9
    ("vp09", ".webm"),   # VP9  - better compression, far slower to encode
    ("mp4v", ".mp4"),    # MPEG-4 Part 2 - NOT browser-playable, last resort
)


def open_writer(path: str, width: int, height: int, fps: int):
    """Opens a VideoWriter using the most browser-compatible codec this
    OpenCV/FFmpeg build actually supports.

    `path`'s extension is advisory: it is replaced with whichever
    container the chosen codec requires, so callers must use the returned
    path rather than the one they passed in.

    Returns `(codec, writer, path)`, or `(None, None, None)` if no codec
    opened. Which codec was used is logged for operators.
    """
    base = os.path.splitext(path)[0]

    for codec, extension in _CODEC_LADDER:
        candidate = base + extension
        writer = cv2.VideoWriter(candidate, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            if codec == "mp4v":
                logger.warning(
                    "Recording %s with mp4v: no browser-playable encoder was available in this "
                    "build, so this clip will not play in Chrome/Firefox/Edge. Install an FFmpeg "
                    "with libx264 (or VP8/VP9) support to fix playback.",
                    candidate,
                )
            return codec, writer, candidate

        writer.release()
        # A failed open can still leave a zero-byte file behind; removing
        # it stops a stray empty .mp4 sitting next to the real .webm.
        if os.path.exists(candidate) and os.path.getsize(candidate) == 0:
            try:
                os.remove(candidate)
            except OSError:
                pass

    return None, None, None


def _write_recording(camera_id, trigger_action, confidence_score, event_timestamp, all_frames, pre_frames, detector="heuristic"):
    height, width = all_frames[0].shape[:2]
    fps = max(settings.STREAM_FPS, 1)

    camera_dir = Path(settings.RECORDINGS_DIR) / str(camera_id)
    camera_dir.mkdir(parents=True, exist_ok=True)

    ts_str = event_timestamp.strftime("%Y%m%dT%H%M%S%f")
    # Extension is advisory - open_writer swaps in whatever container the
    # codec it actually managed to open requires, and returns the real
    # path. Recording rows must store THAT, not this.
    codec_used, writer, written_path = open_writer(
        str(camera_dir / f"{ts_str}_{trigger_action}.mp4"), width, height, fps,
    )
    if writer is None:
        logger.error("Could not open VideoWriter with any codec for camera %s event %s", camera_id, trigger_action)
        return

    file_path = Path(written_path)
    filename = file_path.name

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
        db.flush()  # assigns recording.id without a full commit

        event = Event(
            camera_id=camera_id,
            recording_id=recording.id,
            event_type=trigger_action,
            confidence_score=confidence_score,
            timestamp=event_timestamp,
            detector=detector,
            triggered_recording=True,
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        db.refresh(recording)

        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        camera_name = camera.name if camera else f"camera-{camera_id}"

        realtime_broadcaster.publish(
            "alert.created",
            {
                "id": event.id,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "recording_id": recording.id,
                "event_type": trigger_action,
                "confidence_score": confidence_score,
                "timestamp": event_timestamp,
                "detector": detector,
            },
        )

    notification_service.notify_alert(
        event_type=trigger_action,
        source_name=camera_name,
        timestamp=event_timestamp,
        snapshot_path=str(snapshot_path),
    )
