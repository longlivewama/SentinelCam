"""
Offline analysis pipeline for customer-uploaded videos (as opposed to the
live-camera pipeline in detection/engine.py + stream_manager.py).

Upload -> validate -> decode frames -> run the same fall-detection
pipeline live cameras use (the trained YOLO fall detector, or the pose
heuristic - see detection/fall_pipeline.py) -> emit fall events with
in-video timestamps/confidence -> write an event clip per fall -> persist
Recording + Event rows (video_upload_id set, camera_id null) -> notify ->
mark the VideoUpload row completed, all while reporting progress over the
same realtime WebSocket channel live-camera alerts use.

Runs on a background thread per upload (fire-and-forget from the API
route, mirroring recording_engine.trigger_event's pattern) so the HTTP
request that accepted the upload returns immediately; the frontend polls
`GET /api/video-uploads/{id}` (or listens on the realtime channel) for
progress.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2

from app.config import settings
from app.database import SessionLocal
from app.models.event import Event
from app.models.recording import Recording
from app.models.user import User
from app.models.video_upload import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PROCESSING,
    VideoUpload,
)
from app.services.detection.engine import detection_engine
from app.services.detection.fall_pipeline import FallPipeline
from app.services.notifications import notification_service
from app.services.realtime import realtime_broadcaster
from app.services import recording_engine as recording_engine_mod

logger = logging.getLogger(__name__)

DEFAULT_FPS = 25.0
PRE_EVENT_SECONDS = 3
POST_EVENT_SECONDS = 3

# Every accepted upload gets its own worker thread immediately, but only
# MAX_CONCURRENT_VIDEO_ANALYSES of them may be decoding + running
# inference at a time; the rest block on this semaphore and stay in
# "pending" until a slot frees up. Without it, N simultaneous uploads
# means N video decoders and N inference streams competing for the same
# CPU, which degrades every analysis at once and can exhaust memory on a
# small host. Module-level so the bound is process-wide.
_analysis_slots = threading.BoundedSemaphore(max(settings.MAX_CONCURRENT_VIDEO_ANALYSES, 1))


def analyze_video_upload(video_upload_id: int):
    """Fire-and-forget entry point called by the video_uploads API route."""
    thread = threading.Thread(
        target=_run, args=(video_upload_id,), daemon=True, name=f"video-analysis-{video_upload_id}",
    )
    thread.start()


def _set_status(video_upload_id: int, **fields):
    with SessionLocal() as db:
        upload = db.query(VideoUpload).filter(VideoUpload.id == video_upload_id).first()
        if upload is None:
            return
        for key, value in fields.items():
            setattr(upload, key, value)
        db.commit()
        db.refresh(upload)
        realtime_broadcaster.publish(
            "upload.progress",
            {
                "id": upload.id,
                "status": upload.status,
                "progress_percent": upload.progress_percent,
                "duration_seconds": upload.duration_seconds,
                "fps": upload.fps,
                "frame_count": upload.frame_count,
            },
            # This upload is one user's data; only they (and operators)
            # should see it move.
            owner_user_id=upload.user_id,
        )


def _run(video_upload_id: int):
    with _analysis_slots:
        _run_guarded(video_upload_id)


def _run_guarded(video_upload_id: int):
    with SessionLocal() as db:
        upload = db.query(VideoUpload).filter(VideoUpload.id == video_upload_id).first()
        if upload is None:
            logger.error("VideoUpload %s not found; aborting analysis", video_upload_id)
            return
        stored_path = upload.stored_path
        user = db.query(User).filter(User.id == upload.user_id).first()
        uploader_email = user.email if user else None
        original_filename = upload.original_filename

    _set_status(video_upload_id, status=STATUS_PROCESSING, started_at=datetime.now(timezone.utc), progress_percent=0)

    try:
        _process(video_upload_id, stored_path, original_filename)
    except Exception as exc:
        logger.exception("Video analysis failed for upload %s", video_upload_id)
        _set_status(video_upload_id, status=STATUS_FAILED, error_message=str(exc)[:500])
        return

    with SessionLocal() as db:
        upload = db.query(VideoUpload).filter(VideoUpload.id == video_upload_id).first()
        fall_count = upload.fall_events_count if upload else 0

    if uploader_email:
        notification_service.notify_video_analysis_complete(uploader_email, original_filename, fall_count)


def _process(video_upload_id: int, stored_path: str, original_filename: str):
    detection_engine.ensure_models()

    cap = cv2.VideoCapture(stored_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video file - it may be corrupt or in an unsupported format")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    if fps <= 0:
        fps = DEFAULT_FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = (frame_count / fps) if frame_count else None

    _set_status(
        video_upload_id,
        fps=fps,
        frame_count=frame_count or None,
        duration_seconds=duration_seconds,
    )

    fall_pipeline = FallPipeline()
    logger.info(
        "Video upload %s: analysing %d frames at %.2f fps, fall detection mode=%r",
        video_upload_id, frame_count, fps, fall_pipeline.mode,
    )
    raw_buffer = deque(maxlen=int(PRE_EVENT_SECONDS * fps) or 1)
    pending_clips = []  # list of dicts: {"remaining": int, "frames": list, "event": dict}

    max_concurrent_persons = 0
    fall_events = []  # list of {"timestamp_seconds", "confidence"}
    frame_index = 0
    last_reported_percent = -1

    stride = max(settings.VIDEO_ANALYSIS_FRAME_STRIDE, 1)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        raw_buffer.append(frame)

        # Advance any clips currently collecting post-event frames.
        still_pending = []
        for clip in pending_clips:
            clip["frames"].append(frame.copy())
            clip["remaining"] -= 1
            if clip["remaining"] <= 0:
                _write_upload_clip(video_upload_id, clip["frames"], fps, clip["event"])
            else:
                still_pending.append(clip)
        pending_clips = still_pending

        if frame_index % stride == 0:
            video_time_seconds = frame_index / fps
            people = detection_engine.extract_people(frame)
            max_concurrent_persons = max(max_concurrent_persons, len(people))

            for fall_event in fall_pipeline.update(frame, people, now=video_time_seconds):
                confidence = fall_event.get("confidence", 1.0)
                detector = fall_event.get("detector")
                fall_events.append({"timestamp_seconds": round(video_time_seconds, 2), "confidence": confidence})
                pending_clips.append({
                    "remaining": int(POST_EVENT_SECONDS * fps) or 1,
                    "frames": list(raw_buffer),
                    "event": {
                        "timestamp_seconds": video_time_seconds,
                        "confidence": confidence,
                        "detector": detector,
                    },
                })

        frame_index += 1
        if frame_count:
            percent = min(99, int((frame_index / frame_count) * 100))
            if percent != last_reported_percent:
                last_reported_percent = percent
                _set_status(video_upload_id, progress_percent=percent)

    # Flush any clips still collecting post-event frames when the video ends.
    for clip in pending_clips:
        _write_upload_clip(video_upload_id, clip["frames"], fps, clip["event"])

    cap.release()

    with SessionLocal() as db:
        upload = db.query(VideoUpload).filter(VideoUpload.id == video_upload_id).first()
        if upload is not None:
            upload.status = STATUS_COMPLETED
            upload.progress_percent = 100
            upload.persons_detected = max_concurrent_persons
            upload.fall_events_count = len(fall_events)
            upload.completed_at = datetime.now(timezone.utc)
            db.commit()
            realtime_broadcaster.publish(
                "upload.completed",
                {
                    "id": upload.id,
                    "status": upload.status,
                    "progress_percent": upload.progress_percent,
                    "fall_events_count": upload.fall_events_count,
                    "persons_detected": upload.persons_detected,
                    "duration_seconds": upload.duration_seconds,
                    "fps": upload.fps,
                    "frame_count": upload.frame_count,
                },
                owner_user_id=upload.user_id,
            )


def _write_upload_clip(video_upload_id: int, frames: list, fps: float, event: dict):
    if not frames:
        return
    height, width = frames[0].shape[:2]
    fps_int = max(int(round(fps)), 1)

    clip_dir = Path(settings.RECORDINGS_DIR) / "uploads" / str(video_upload_id)
    clip_dir.mkdir(parents=True, exist_ok=True)

    event_timestamp = datetime.now(timezone.utc)
    ts_str = event_timestamp.strftime("%Y%m%dT%H%M%S%f")
    filename = f"{ts_str}_fall.mp4"
    file_path = clip_dir / filename

    codec_used, writer = recording_engine_mod.open_writer(str(file_path), width, height, fps_int)
    if writer is None:
        logger.error("Could not open VideoWriter with any codec for %s", file_path)
        return

    for frame in frames:
        writer.write(frame)
    writer.release()

    file_size = file_path.stat().st_size if file_path.exists() else 0
    duration_seconds = len(frames) / fps_int

    snapshot_dir = Path(settings.SNAPSHOTS_DIR)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_filename = f"upload{video_upload_id}_{ts_str}_fall.jpg"
    snapshot_path = snapshot_dir / snapshot_filename
    mid_frame = frames[min(len(frames) // 2, len(frames) - 1)]
    cv2.imwrite(str(snapshot_path), mid_frame)

    logger.info(
        "Video upload %s: wrote event clip %s (codec=%s, frames=%d) at t=%.2fs",
        video_upload_id, file_path, codec_used, len(frames), event["timestamp_seconds"],
    )

    with SessionLocal() as db:
        recording = Recording(
            camera_id=None,
            video_upload_id=video_upload_id,
            filename=filename,
            file_path=str(file_path),
            duration_seconds=duration_seconds,
            trigger_action="fall",
            file_size_bytes=file_size,
            event_timestamp=event_timestamp,
        )
        db.add(recording)
        db.flush()

        event_row = Event(
            camera_id=None,
            video_upload_id=video_upload_id,
            recording_id=recording.id,
            event_type="fall",
            confidence_score=event["confidence"],
            timestamp=event_timestamp,
            # Where in the uploaded file the fall happened. `timestamp`
            # above is only when the worker got there, so without this the
            # UI has nothing truthful to point the operator at.
            video_timestamp_seconds=round(event["timestamp_seconds"], 2),
            detector=event.get("detector"),
            triggered_recording=True,
        )
        db.add(event_row)
        db.commit()
        db.refresh(event_row)
        db.refresh(recording)

        upload = db.query(VideoUpload).filter(VideoUpload.id == video_upload_id).first()
        source_name = upload.original_filename if upload else f"upload-{video_upload_id}"
        owner_user_id = upload.user_id if upload else None

        realtime_broadcaster.publish(
            "alert.created",
            {
                "id": event_row.id,
                "video_upload_id": video_upload_id,
                "source_name": source_name,
                "recording_id": recording.id,
                "event_type": "fall",
                "confidence_score": event["confidence"],
                "timestamp": event_timestamp,
                "video_timestamp_seconds": round(event["timestamp_seconds"], 2),
                "detector": event.get("detector"),
            },
            owner_user_id=owner_user_id,
        )

    notification_service.notify_alert(
        event_type="fall",
        source_name=source_name,
        timestamp=event_timestamp,
        snapshot_path=str(snapshot_path),
    )
