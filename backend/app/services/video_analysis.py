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
import os
import threading
import time
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
from app.services.cascade_delete import remove_upload_clip_dir, stage_delete_video_upload
from app.services.detection.engine import detection_engine
from app.services.detection.fall_pipeline import FallPipeline
from app.services.detection.video_scan import open_video, scan_video
from app.services.notifications import notification_service
from app.services.realtime import realtime_broadcaster
from app.services import recording_engine as recording_engine_mod

logger = logging.getLogger(__name__)

PRE_EVENT_SECONDS = 3
POST_EVENT_SECONDS = 3

# Mid-loop check for services/video_uploads.py's DELETE endpoint having
# flipped deletion_requested, so a long video stops decoding promptly
# after the user asks to delete it rather than running to completion
# first. Throttled (real wall-clock time, not video time or frame count)
# so this never becomes the "database polling" the fix is explicitly
# meant to avoid - one query per second of wall clock regardless of fps,
# stride or video length. This is an optimization, not the correctness
# gate: _finalize_if_deletion_requested (below) is what actually decides
# an upload's fate, under a row lock, at every point that matters.
CANCEL_CHECK_INTERVAL_SECONDS = 1.0

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
        # Already flagged for deletion: don't write progress to it, and
        # above all don't broadcast that progress. The API stopped
        # returning this upload the moment the flag was set, so an
        # `upload.progress` event naming it is at best noise and at worst
        # something a client could use to re-add a row the user just
        # deleted. Free to check - the row is already loaded here.
        if upload.deletion_requested:
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


def _deletion_requested(video_upload_id: int) -> bool:
    """Cheap, throttled, lock-free check used mid-loop purely to stop
    analysing promptly once deletion has been requested. Not the
    correctness gate - see _finalize_if_deletion_requested - just an
    optimization so a long video doesn't keep decoding for several more
    seconds after the user asked to delete it. Treats a row that has
    already vanished the same as one flagged for deletion, since either
    way there is nothing left to analyse for."""
    with SessionLocal() as db:
        flag = (
            db.query(VideoUpload.deletion_requested)
            .filter(VideoUpload.id == video_upload_id)
            .scalar()
        )
    return flag is None or bool(flag)


def _finalize_if_deletion_requested(video_upload_id: int) -> bool:
    """The actual correctness gate. Called at every point the worker is
    about to commit an outcome for this upload - before marking it
    completed, before recording a failure, and (from _write_upload_clip)
    before persisting a fall's Recording/Event rows - so none of those
    writes can land on an upload the user asked to delete.

    Locks the row with SELECT ... FOR UPDATE, which is what actually
    closes the race rather than just narrowing it: the DELETE endpoint's
    flag flip is a plain UPDATE on the same row, so Postgres serializes
    the two against each other automatically. Whichever of the two
    commits first is authoritative, and the other sees that committed
    result once its lock is granted - there is no window in between where
    both proceed on stale information. The lock is held only for this one
    short read-then-maybe-delete transaction, never across a video's
    decode or encode time.

    If the upload is gone or flagged, performs the actual cascade-delete
    the DELETE endpoint deferred (the same helper it uses for an
    already-terminal upload) and returns True. Returns False, having
    touched nothing, if the upload is still live - the caller should
    proceed with its own outcome normally.
    """
    with SessionLocal() as db:
        upload = (
            db.query(VideoUpload)
            .filter(VideoUpload.id == video_upload_id)
            .with_for_update()
            .first()
        )
        if upload is None:
            return True
        if not upload.deletion_requested:
            return False
        paths = stage_delete_video_upload(db, upload)
        db.commit()

    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning(
                    "Could not remove %s while finalizing deleted upload %s", path, video_upload_id,
                )
    remove_upload_clip_dir(video_upload_id)
    return True


def _run(video_upload_id: int):
    with _analysis_slots:
        _run_guarded(video_upload_id)


def _run_guarded(video_upload_id: int):
    # Covers "deleted before analysis started" and "deleted while queued
    # on the concurrency semaphore": if this upload was already flagged
    # (or is already gone) by the time its turn comes up, finish the
    # deletion now and never open the video file at all.
    if _finalize_if_deletion_requested(video_upload_id):
        logger.info("Video upload %s was deleted before analysis started; nothing to do", video_upload_id)
        return

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

    started = time.monotonic()
    try:
        completed = _process(video_upload_id, stored_path, original_filename)
    except Exception as exc:
        logger.exception(
            "Video analysis failed for upload %s after %.1fs", video_upload_id, time.monotonic() - started,
        )
        # The exception may well BE the deletion race (e.g. the source
        # file disappearing mid-decode) rather than a genuine failure -
        # check before recording one, so a deleted upload doesn't
        # resurface with a "failed" status and a "your analysis failed"
        # email nobody should receive for a video that no longer exists.
        if _finalize_if_deletion_requested(video_upload_id):
            logger.info(
                "Video upload %s: analysis raised %s, but the upload had been deleted; cleaned up instead "
                "of recording a failure",
                video_upload_id, type(exc).__name__,
            )
            return
        # str(exc) reaches the user via error_message, so it must stay a
        # description of what went wrong - never a traceback or a path.
        _set_status(video_upload_id, status=STATUS_FAILED, error_message=str(exc)[:500])
        return

    if not completed:
        # _process only returns False after finalizing the deletion
        # itself (see its final lines) - nothing left to do here.
        logger.info("Video upload %s: analysis stopped, upload was deleted mid-analysis", video_upload_id)
        return

    elapsed = time.monotonic() - started

    with SessionLocal() as db:
        upload = db.query(VideoUpload).filter(VideoUpload.id == video_upload_id).first()
        fall_count = upload.fall_events_count if upload else 0

    logger.info(
        "Video analysis complete for upload %s: %d fall event(s) in %.1fs of wall clock",
        video_upload_id, fall_count, elapsed,
    )

    if uploader_email:
        notification_service.notify_video_analysis_complete(uploader_email, original_filename, fall_count)


def _process(video_upload_id: int, stored_path: str, original_filename: str) -> bool:
    """Returns True if analysis reached the end of the video and the
    upload was marked completed; False if it was deleted mid-analysis
    (deletion has already been fully finalized - row, files and all -
    before this returns)."""
    detection_engine.ensure_models()

    # Frame iteration, stride and video-timeline arithmetic live in
    # detection/video_scan.py so the offline validation runner
    # (ml/validation/) drives the exact same loop - its measured recall and
    # false-alert rate are then statements about this code path, not about
    # a re-implementation of it.
    cap, properties = open_video(stored_path)
    fps = properties.fps
    frame_count = properties.frame_count

    _set_status(
        video_upload_id,
        fps=fps,
        frame_count=frame_count or None,
        duration_seconds=properties.duration_seconds,
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
    last_reported_percent = -1
    cancelled = False
    last_cancel_check = time.monotonic()

    for scanned in scan_video(cap, properties, fall_pipeline, detection_engine.extract_people):
        now_wall = time.monotonic()
        if now_wall - last_cancel_check >= CANCEL_CHECK_INTERVAL_SECONDS:
            last_cancel_check = now_wall
            if _deletion_requested(video_upload_id):
                cancelled = True
                break

        frame = scanned.frame
        raw_buffer.append(frame)

        # Advance any clips currently collecting post-event frames.
        still_pending = []
        for clip in pending_clips:
            clip["frames"].append(frame.copy())
            clip["remaining"] -= 1
            if clip["remaining"] <= 0:
                if not _write_upload_clip(video_upload_id, clip["frames"], fps, clip["event"]):
                    # The upload was deleted between the fall firing and
                    # this clip finishing its post-event buffer - the
                    # write was discarded rather than persisted. Nothing
                    # further in this video matters either.
                    cancelled = True
            else:
                still_pending.append(clip)
        pending_clips = still_pending
        if cancelled:
            break

        if scanned.processed:
            max_concurrent_persons = max(max_concurrent_persons, len(scanned.people))

            for fall_event in scanned.fall_events:
                confidence = fall_event.get("confidence", 1.0)
                detector = fall_event.get("detector")
                fall_events.append({
                    "timestamp_seconds": round(scanned.video_time_seconds, 2),
                    "confidence": confidence,
                })
                pending_clips.append({
                    "remaining": int(POST_EVENT_SECONDS * fps) or 1,
                    "frames": list(raw_buffer),
                    "event": {
                        "timestamp_seconds": scanned.video_time_seconds,
                        "confidence": confidence,
                        "detector": detector,
                    },
                })

        frame_index = scanned.index + 1
        if frame_count:
            percent = min(99, int((frame_index / frame_count) * 100))
            if percent != last_reported_percent:
                last_reported_percent = percent
                _set_status(video_upload_id, progress_percent=percent)

    # Flush any clips still collecting post-event frames when the video
    # ends - skipped once cancelled, since every one of them would just
    # be encoded and then discarded by _write_upload_clip's own check.
    if not cancelled:
        for clip in pending_clips:
            if not _write_upload_clip(video_upload_id, clip["frames"], fps, clip["event"]):
                cancelled = True

    cap.release()

    # The authoritative check, regardless of how we got here: whether the
    # loop broke early, ran to completion, or was cancelled right on its
    # last iteration, this is what actually decides - under a row lock -
    # whether the upload still exists to be marked completed. Also covers
    # deletion requested in the instant after the last periodic check.
    if _finalize_if_deletion_requested(video_upload_id):
        return False

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
    return True


def _write_upload_clip(video_upload_id: int, frames: list, fps: float, event: dict) -> bool:
    """Encodes one fall's pre/post-event buffer to disk and records it as
    a Recording + Event. Returns False, having written nothing to the
    database, if the upload was deleted or flagged for deletion - the
    caller (_process) treats that as its own signal to stop analysing.

    This is the exact call site the original race broke: without the
    check below, a fall detected after the user deleted the upload would
    try to INSERT a Recording/Event whose video_upload_id no longer
    existed, raising an IntegrityError from inside the analysis thread.
    """
    if not frames:
        return True

    # Cheap pre-check: skip the encode entirely in the common case where
    # deletion was already noticed (or requested well before this clip's
    # post-event buffer finished filling) - no point spending CPU on
    # frames nobody will keep. Not itself the correctness gate; the
    # locked check below is, for the narrow window this can still miss.
    if _deletion_requested(video_upload_id):
        return False

    height, width = frames[0].shape[:2]
    fps_int = max(int(round(fps)), 1)

    clip_dir = Path(settings.RECORDINGS_DIR) / "uploads" / str(video_upload_id)
    clip_dir.mkdir(parents=True, exist_ok=True)

    event_timestamp = datetime.now(timezone.utc)
    ts_str = event_timestamp.strftime("%Y%m%dT%H%M%S%f")
    # Extension is advisory - open_writer swaps in whatever container the
    # codec it actually opened requires (WebM where this build has no
    # H.264 encoder), and returns the real path. The Recording row must
    # store THAT, or the clip endpoint 404s on a file that is right there
    # under a different suffix.
    codec_used, writer, written_path = recording_engine_mod.open_writer(
        str(clip_dir / f"{ts_str}_fall.mp4"), width, height, fps_int,
    )
    if writer is None:
        logger.error("Could not open VideoWriter with any codec for upload %s", video_upload_id)
        return True

    file_path = Path(written_path)
    filename = file_path.name

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
        # The authoritative check, in the SAME transaction as the INSERTs
        # below and under a row lock: this is what actually closes the
        # race rather than just narrowing it. SELECT ... FOR UPDATE
        # serializes against the DELETE endpoint's flag-flip UPDATE on
        # this same row (a plain UPDATE also takes a row lock), so
        # whichever of the two commits first is authoritative and the
        # other sees that committed result once unblocked - there is no
        # window where both proceed on stale information. The lock is
        # held only for this one short transaction, never across the
        # encode above.
        upload = (
            db.query(VideoUpload)
            .filter(VideoUpload.id == video_upload_id)
            .with_for_update()
            .first()
        )
        if upload is None or upload.deletion_requested:
            db.rollback()
            file_path.unlink(missing_ok=True)
            snapshot_path.unlink(missing_ok=True)
            logger.info(
                "Video upload %s was deleted mid-analysis; discarding this fall clip instead of persisting it",
                video_upload_id,
            )
            return False

        # Captured now, while `upload` is still attached and unexpired -
        # db.commit() below would otherwise force a second round-trip to
        # re-fetch exactly what we already hold the row locked for.
        source_name = upload.original_filename
        owner_user_id = upload.user_id

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
    return True
