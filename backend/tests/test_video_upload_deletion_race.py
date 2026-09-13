"""
Tests for the delete-during-analysis race in the offline video upload
pipeline (services/video_analysis.py + api/routes/video_uploads.py).

THE BUG THIS GUARDS AGAINST
---------------------------
Deleting an upload while its analysis worker thread was still running
raced the worker two ways, both reproduced (before the fix) with a
throwaway script before writing any of this:

  1. If the row was hard-deleted while the worker was still going, a fall
     detected afterwards tried to INSERT a Recording/Event whose
     video_upload_id no longer existed - an IntegrityError raised from
     inside the analysis thread, aborting the rest of that video's
     processing and leaving the just-encoded clip file orphaned on disk
     forever (the DELETE had already run and has no way to know about a
     file that did not exist yet when it ran).
  2. If the worker's INSERT committed first, the DELETE endpoint's own
     `db.commit()` could raise that same IntegrityError instead (the
     upload row now has a live FK reference from a Recording/Event the
     delete's bulk-delete statements ran too early to see), surfacing as
     an unhandled 500 to whoever called DELETE.

THE FIX
-------
DELETE on a pending/processing upload no longer touches the row or file
at all - see api/routes/video_uploads.py. It flips `deletion_requested`
and returns immediately (still a deterministic 204: _get_upload_or_404
makes a flagged row read as gone through every GET from that instant).
The worker is the only thing that ever performs the real cascade-delete,
via `_finalize_if_deletion_requested`, which locks the row with
SELECT ... FOR UPDATE - the same row the flag-flip UPDATE touches - so
Postgres serializes the two rather than either one racing the other's
stale read. That call is threaded through every point the worker is
about to commit an outcome: before persisting a fall's Recording/Event
(_write_upload_clip), before marking the upload completed, before
recording a failure, and before ever opening the video file at all.

WHAT IS TESTED WHERE
---------------------
Most tests below call the worker's private functions directly with
hand-built VideoUpload rows, so they are fast and exercise a specific
interleaving deterministically rather than hoping a real thread happens
to be scheduled at the right moment. Two tests are genuine integration
tests against the real API and a real background thread, because the
lower-level tests alone would not catch a wiring mistake (e.g. the route
never actually calling the deferred path). One test forces a true
concurrent-thread race at the database level, to verify the row lock
itself - not just the application-level flag check - is what closes the
window.
"""
import io
import threading
import time
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import settings
from app.database import SessionLocal
from app.models.event import Event
from app.models.recording import Recording
from app.models.video_upload import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    VideoUpload,
)
from app.services import video_analysis

Box = namedtuple("Box", ["bbox", "confidence"])


# --------------------------------------------------------------------------
# Shared helpers (deliberately duplicated from other test files rather than
# imported - `tests` is not an importable package here, see
# conftest.py's make_user_factory docstring).
# --------------------------------------------------------------------------

def _tiny_video_bytes(frames=8, size=(64, 64), fps=10) -> bytes:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), fill_value=i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    data = Path(path).read_bytes()
    Path(path).unlink(missing_ok=True)
    return data


def _wait_until_fully_gone(db, upload_id, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        db.expire_all()
        if db.query(VideoUpload).filter(VideoUpload.id == upload_id).first() is None:
            return
        time.sleep(0.2)
    raise AssertionError(f"Video upload {upload_id} was not fully cleaned up within {timeout}s")


def _wait_for_terminal_status(db, upload_id, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        db.expire_all()
        upload = db.query(VideoUpload).filter(VideoUpload.id == upload_id).first()
        if upload and upload.status in (STATUS_COMPLETED, STATUS_FAILED):
            return upload
        time.sleep(0.2)
    raise AssertionError(f"Video upload {upload_id} did not reach a terminal status within {timeout}s")


@pytest.fixture()
def uploads_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOADS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def clip_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDINGS_DIR", str(tmp_path / "recordings"))
    monkeypatch.setattr(settings, "SNAPSHOTS_DIR", str(tmp_path / "snapshots"))
    return tmp_path


# --------------------------------------------------------------------------
# _deletion_requested - the cheap, throttled, lock-free check
# --------------------------------------------------------------------------

def test_deletion_requested_is_false_for_a_live_unflagged_upload(db, viewer_user, tmp_path):
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(tmp_path / "x.mp4"),
        status=STATUS_PROCESSING,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    assert video_analysis._deletion_requested(upload.id) is False


def test_deletion_requested_reflects_the_flag_once_set(db, viewer_user, tmp_path):
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(tmp_path / "x.mp4"),
        status=STATUS_PROCESSING,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    upload.deletion_requested = True
    db.commit()

    assert video_analysis._deletion_requested(upload.id) is True


def test_deletion_requested_treats_a_vanished_row_as_requested(db):
    """There is nothing left to analyse for either reason, so a row that
    is simply gone must not be read as 'still live'."""
    assert video_analysis._deletion_requested(999_999_999) is True


# --------------------------------------------------------------------------
# _finalize_if_deletion_requested - the authoritative, locked gate
# --------------------------------------------------------------------------

def test_finalize_touches_nothing_and_returns_false_for_a_live_upload(db, viewer_user, tmp_path):
    stored = tmp_path / "src.mp4"
    stored.write_bytes(b"source video bytes")
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(stored), status=STATUS_PROCESSING,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    assert video_analysis._finalize_if_deletion_requested(upload.id) is False
    assert stored.exists()
    db.expire_all()
    assert db.query(VideoUpload).filter(VideoUpload.id == upload.id).first() is not None


def test_finalize_cascades_the_row_events_recordings_and_files_when_flagged(db, viewer_user, tmp_path):
    stored = tmp_path / "src.mp4"
    stored.write_bytes(b"source video bytes")
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(stored),
        status=STATUS_PROCESSING, deletion_requested=True,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    # Captured now, not read off `upload` after the row is deleted below -
    # `_finalize_if_deletion_requested` deletes it through a different
    # session, so this session's `upload` object would raise
    # ObjectDeletedError on any attribute access once expired.
    upload_id = upload.id

    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"clip bytes")
    recording = Recording(
        video_upload_id=upload_id, filename="clip.mp4", file_path=str(clip_path),
        duration_seconds=1.0, trigger_action="fall", file_size_bytes=10,
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(recording)
    db.commit()
    db.refresh(recording)
    event = Event(
        video_upload_id=upload_id, recording_id=recording.id, event_type="fall",
        confidence_score=0.9, timestamp=datetime.now(timezone.utc), triggered_recording=True,
    )
    db.add(event)
    db.commit()
    recording_id, event_id = recording.id, event.id

    assert video_analysis._finalize_if_deletion_requested(upload_id) is True

    assert not stored.exists()
    assert not clip_path.exists()
    db.expire_all()
    assert db.query(VideoUpload).filter(VideoUpload.id == upload_id).first() is None
    assert db.query(Recording).filter(Recording.id == recording_id).first() is None
    assert db.query(Event).filter(Event.id == event_id).first() is None


def test_finalize_is_a_safe_no_op_when_the_row_is_already_gone(db):
    """A second finalize call (e.g. from a retried check) must not raise
    just because a previous one already did the work."""
    assert video_analysis._finalize_if_deletion_requested(999_999_999) is True


# --------------------------------------------------------------------------
# _write_upload_clip - the exact call site the original race broke
# --------------------------------------------------------------------------

def _frames(n=3, size=8):
    return [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(n)]


def test_write_upload_clip_persists_normally_when_the_upload_is_live(db, viewer_user, clip_dirs):
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(clip_dirs / "src.mp4"),
        status=STATUS_PROCESSING,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    ok = video_analysis._write_upload_clip(
        upload.id, _frames(), fps=10.0,
        event={"timestamp_seconds": 1.0, "confidence": 0.9, "detector": "model"},
    )

    assert ok is True
    assert db.query(Recording).filter(Recording.video_upload_id == upload.id).count() == 1
    assert db.query(Event).filter(Event.video_upload_id == upload.id).count() == 1


def test_write_upload_clip_discards_without_raising_when_flagged_for_deletion(db, viewer_user, clip_dirs):
    """The exact scenario that used to raise IntegrityError from inside
    the analysis thread: a fall's post-event buffer finishes filling
    after the upload has been flagged for deletion."""
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(clip_dirs / "src.mp4"),
        status=STATUS_PROCESSING, deletion_requested=True,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    ok = video_analysis._write_upload_clip(
        upload.id, _frames(), fps=10.0,
        event={"timestamp_seconds": 1.0, "confidence": 0.9, "detector": "model"},
    )

    assert ok is False
    assert db.query(Recording).filter(Recording.video_upload_id == upload.id).count() == 0
    assert db.query(Event).filter(Event.video_upload_id == upload.id).count() == 0
    # No orphaned clip/snapshot files left behind either.
    assert list((clip_dirs / "recordings").rglob("*.mp4")) == []
    assert list((clip_dirs / "snapshots").rglob("*.jpg")) == []


def test_write_upload_clip_discards_without_raising_when_the_row_is_entirely_gone(db, viewer_user, clip_dirs):
    """The OTHER half of the original bug: the row was hard-deleted
    outright (as the pre-fix DELETE endpoint always did), not just
    flagged. This is a direct regression test for the IntegrityError
    reproduced against the pre-fix code before any of this was written."""
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(clip_dirs / "src.mp4"),
        status=STATUS_PROCESSING,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    upload_id = upload.id
    db.delete(upload)
    db.commit()

    ok = video_analysis._write_upload_clip(
        upload_id, _frames(), fps=10.0,
        event={"timestamp_seconds": 1.0, "confidence": 0.9, "detector": "model"},
    )

    assert ok is False
    assert db.query(Recording).filter(Recording.video_upload_id == upload_id).count() == 0
    assert db.query(Event).filter(Event.video_upload_id == upload_id).count() == 0
    assert list((clip_dirs / "recordings").rglob("*.mp4")) == []


def test_write_upload_clip_of_an_empty_frame_list_is_a_harmless_no_op(db, viewer_user, clip_dirs):
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(clip_dirs / "src.mp4"),
        status=STATUS_PROCESSING,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    assert video_analysis._write_upload_clip(upload.id, [], fps=10.0, event={}) is True
    assert db.query(Recording).filter(Recording.video_upload_id == upload.id).count() == 0


# --------------------------------------------------------------------------
# The real database lock, raced with a genuine concurrent thread
# --------------------------------------------------------------------------

def test_the_row_lock_genuinely_serializes_a_concurrent_delete_and_clip_write(db, viewer_user, clip_dirs):
    """Not just 'the flag says no' - this proves SELECT ... FOR UPDATE
    actually blocks a concurrent writer, using commit-order rather than a
    sleep-and-hope: a holder thread acquires the row lock and sits on it
    for a fixed, short window before flipping deletion_requested and
    committing. Meanwhile the real _write_upload_clip is called
    concurrently. If it is genuinely blocked by the lock (as
    with_for_update() must guarantee), its completion timestamp cannot
    precede the holder's release timestamp - regardless of how the OS
    happens to schedule the two threads. A version of the code without
    the lock would let the write finish immediately, well before the
    holder releases, and this assertion would catch that.
    """
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(clip_dirs / "src.mp4"),
        status=STATUS_PROCESSING,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    upload_id = upload.id

    hold_seconds = 0.4
    lock_acquired = threading.Event()
    release_timestamp = {}

    def hold_lock_then_flag_for_deletion():
        session = SessionLocal()
        try:
            row = (
                session.query(VideoUpload)
                .filter(VideoUpload.id == upload_id)
                .with_for_update()
                .first()
            )
            lock_acquired.set()
            time.sleep(hold_seconds)  # fixed, deliberate hold - see docstring
            row.deletion_requested = True
            session.commit()
            release_timestamp["t"] = time.monotonic()
        finally:
            session.close()

    holder = threading.Thread(target=hold_lock_then_flag_for_deletion)
    holder.start()
    assert lock_acquired.wait(timeout=5), "holder thread never acquired the row lock"

    result = video_analysis._write_upload_clip(
        upload_id, _frames(), fps=10.0,
        event={"timestamp_seconds": 1.0, "confidence": 0.9, "detector": "model"},
    )
    completed_timestamp = time.monotonic()
    holder.join(timeout=5)
    assert not holder.is_alive(), "holder thread did not finish"

    # The write could only have completed once it saw the holder's
    # commit - i.e. no earlier than the holder's release.
    assert completed_timestamp >= release_timestamp["t"] - 0.01
    assert result is False
    db.expire_all()
    assert db.query(Recording).filter(Recording.video_upload_id == upload_id).count() == 0
    assert db.query(Event).filter(Event.video_upload_id == upload_id).count() == 0


# --------------------------------------------------------------------------
# _process - the frame loop, with a real (tiny) video and a stubbed
# detector so a fall fires at a known, controllable point.
# --------------------------------------------------------------------------

FPS = 10
TOTAL_FRAMES = 100
FALL_START_FRAME = 40
FALL_END_FRAME = 70


@pytest.fixture()
def source_video(tmp_path):
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 120))
    assert writer.isOpened()
    for i in range(TOTAL_FRAMES):
        frame = np.full((120, 160, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture()
def upload(db, viewer_user, source_video):
    row = VideoUpload(
        user_id=viewer_user.id, original_filename="clip.mp4", stored_path=str(source_video),
        file_size_bytes=source_video.stat().st_size, status=STATUS_PROCESSING,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@pytest.fixture()
def stubbed_models(monkeypatch, clip_dirs):
    monkeypatch.setattr(settings, "VIDEO_ANALYSIS_FRAME_STRIDE", 1)
    monkeypatch.setattr(settings, "FALL_DETECTOR_MIN_SUSTAINED_SECONDS", 0.6)
    monkeypatch.setattr(video_analysis.detection_engine, "ensure_models", lambda: None)
    monkeypatch.setattr(video_analysis.detection_engine, "extract_people", lambda frame: [])

    state = {"frame": -1}

    def fake_detect(frame):
        state["frame"] += 1
        if FALL_START_FRAME <= state["frame"] < FALL_END_FRAME:
            return [Box(bbox=(20.0, 70.0, 120.0, 110.0), confidence=0.82)]
        return []

    monkeypatch.setattr("app.services.detection.fall_pipeline.fall_object_detector.detect", fake_detect)
    monkeypatch.setattr("app.services.detection.fall_pipeline.resolve_mode", lambda: "model")
    return state


def test_process_completes_and_persists_the_fall_when_nothing_is_deleted(db, upload, stubbed_models):
    """Baseline: confirms the fixture reproduces the pre-existing,
    unmodified happy path before the cancellation tests below rely on
    the same setup to prove the opposite outcome."""
    upload_id = upload.id
    completed = video_analysis._process(upload_id, upload.stored_path, upload.original_filename)

    assert completed is True
    assert db.query(Event).filter(Event.video_upload_id == upload_id).count() == 1
    db.expire_all()
    row = db.query(VideoUpload).filter(VideoUpload.id == upload_id).first()
    assert row.status == STATUS_COMPLETED


def test_process_stops_promptly_when_already_flagged_before_it_starts(db, upload, stubbed_models, monkeypatch):
    """Forces the throttled mid-loop check to trip on the very first
    iteration (rather than waiting on real wall-clock time, which a fast
    synthetic test would never accumulate) and confirms the loop actually
    breaks early - it does not merely clean up after decoding the whole
    video anyway."""
    monkeypatch.setattr(video_analysis, "CANCEL_CHECK_INTERVAL_SECONDS", 0)
    # Bound before the row is deleted: _process removes it through a
    # different session, after which this session's `upload` object
    # raises ObjectDeletedError on attribute access.
    upload_id, stored_path, filename = upload.id, upload.stored_path, upload.original_filename
    upload.deletion_requested = True
    db.commit()

    completed = video_analysis._process(upload_id, stored_path, filename)

    assert completed is False
    assert stubbed_models["frame"] < FALL_START_FRAME, "the loop ran past the point it should have stopped at"
    assert db.query(Event).filter(Event.video_upload_id == upload_id).count() == 0
    assert db.query(Recording).filter(Recording.video_upload_id == upload_id).count() == 0
    db.expire_all()
    assert db.query(VideoUpload).filter(VideoUpload.id == upload_id).first() is None


def test_process_discards_a_fall_detected_just_before_deletion_was_requested(
    db, upload, stubbed_models, monkeypatch,
):
    """Deletion arrives AFTER a fall has already fired and been scheduled
    as a pending post-event clip, but before that clip finishes
    buffering and would be persisted. The periodic mid-loop check is
    disabled so this exercises _write_upload_clip's own independent
    check specifically, not the loop-level optimization."""
    monkeypatch.setattr(video_analysis, "CANCEL_CHECK_INTERVAL_SECONDS", 999)
    upload_id, stored_path, filename = upload.id, upload.stored_path, upload.original_filename

    def extract_people_and_request_deletion_once_the_fall_has_fired(frame):
        if stubbed_models["frame"] >= FALL_END_FRAME:
            row = db.query(VideoUpload).filter(VideoUpload.id == upload_id).first()
            if row is not None and not row.deletion_requested:
                row.deletion_requested = True
                db.commit()
        return []

    monkeypatch.setattr(
        video_analysis.detection_engine, "extract_people",
        extract_people_and_request_deletion_once_the_fall_has_fired,
    )

    completed = video_analysis._process(upload_id, stored_path, filename)

    assert completed is False
    assert db.query(Event).filter(Event.video_upload_id == upload_id).count() == 0
    assert db.query(Recording).filter(Recording.video_upload_id == upload_id).count() == 0
    db.expire_all()
    assert db.query(VideoUpload).filter(VideoUpload.id == upload_id).first() is None


# --------------------------------------------------------------------------
# _run_guarded - the outer wrapper, including the exception path
# --------------------------------------------------------------------------

def test_run_guarded_finalizes_deletion_before_it_ever_opens_the_video(db, viewer_user, tmp_path):
    """'Delete before analysis starts' / 'delete while queued': already
    flagged by the time this upload's turn comes up, so _process must
    never even be reached."""
    stored = tmp_path / "src.mp4"
    stored.write_bytes(b"placeholder - must never be opened")
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(stored),
        status=STATUS_PENDING, deletion_requested=True,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    upload_id = upload.id

    video_analysis._run_guarded(upload_id)

    assert not stored.exists()
    db.expire_all()
    assert db.query(VideoUpload).filter(VideoUpload.id == upload_id).first() is None


def test_run_guarded_prefers_deletion_over_recording_a_genuine_failure(db, viewer_user, tmp_path, monkeypatch):
    """Analysis fails for an unrelated reason (e.g. a corrupt video) in
    the same instant a delete request arrives. The upload must vanish
    cleanly - never resurface with status=failed, and never send a
    'your analysis failed' notification for a video that no longer
    exists."""
    stored = tmp_path / "src.mp4"
    stored.write_bytes(b"placeholder - _process is stubbed below")
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(stored), status=STATUS_PROCESSING,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    upload_id = upload.id

    def fails_after_deletion_is_requested(video_upload_id, stored_path, original_filename):
        row = db.query(VideoUpload).filter(VideoUpload.id == video_upload_id).first()
        row.deletion_requested = True
        db.commit()
        raise RuntimeError("simulated genuine analysis failure, unrelated to the deletion")

    monkeypatch.setattr(video_analysis, "_process", fails_after_deletion_is_requested)

    video_analysis._run_guarded(upload_id)

    assert not stored.exists()
    db.expire_all()
    assert db.query(VideoUpload).filter(VideoUpload.id == upload_id).first() is None


def test_run_guarded_still_records_a_genuine_failure_when_nothing_was_deleted(db, viewer_user, tmp_path, monkeypatch):
    """Regression guard for the fix itself: an unrelated failure with NO
    deletion involved must still behave exactly as before - status
    failed, with the error message recorded."""
    stored = tmp_path / "src.mp4"
    stored.write_bytes(b"placeholder - _process is stubbed below")
    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(stored), status=STATUS_PROCESSING,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    upload_id = upload.id

    def boom(video_upload_id, stored_path, original_filename):
        raise RuntimeError("the file is corrupt")

    monkeypatch.setattr(video_analysis, "_process", boom)

    video_analysis._run_guarded(upload_id)

    db.expire_all()
    row = db.query(VideoUpload).filter(VideoUpload.id == upload_id).first()
    assert row is not None
    assert row.status == STATUS_FAILED
    assert "the file is corrupt" in row.error_message


# --------------------------------------------------------------------------
# Full-stack integration: the real API endpoints, a real background
# thread, and (in the second test) a real analysis run.
# --------------------------------------------------------------------------

def test_delete_on_a_pending_upload_defers_and_the_file_survives_until_finalized(
    client, viewer_headers, uploads_dir, db, monkeypatch,
):
    """Analysis is prevented from ever starting, so the row stays pending
    indefinitely - making 'the file must still be there right after
    DELETE returns' a deterministic assertion rather than a race against
    a real thread finishing first. Covers invariant: the source file must
    remain available for as long as active processing could still need
    it, and repeated DELETE calls behave safely."""
    monkeypatch.setattr("app.api.routes.video_uploads.analyze_video_upload", lambda video_upload_id: None)

    video_bytes = _tiny_video_bytes()
    created = client.post(
        "/api/video-uploads", files={"file": ("clip.mp4", io.BytesIO(video_bytes), "video/mp4")},
        headers=viewer_headers,
    )
    assert created.status_code == 201
    upload_id = created.json()["id"]

    row = db.query(VideoUpload).filter(VideoUpload.id == upload_id).first()
    stored_path = Path(row.stored_path)
    assert stored_path.exists()

    first = client.delete(f"/api/video-uploads/{upload_id}", headers=viewer_headers)
    assert first.status_code == 204

    # Deferred, not deleted: the row and its file are both still there -
    # only the flag flipped.
    db.expire_all()
    still_there = db.query(VideoUpload).filter(VideoUpload.id == upload_id).first()
    assert still_there is not None
    assert still_there.deletion_requested is True
    assert still_there.status == STATUS_PENDING
    assert stored_path.exists()

    # But it must already read as fully gone through the API.
    assert client.get(f"/api/video-uploads/{upload_id}", headers=viewer_headers).status_code == 404
    assert upload_id not in [
        u["id"] for u in client.get("/api/video-uploads", headers=viewer_headers).json()["items"]
    ]

    # A second DELETE behaves like deleting an already-gone upload always
    # has - 404, not a second 204 - and does not disturb the pending flag.
    second = client.delete(f"/api/video-uploads/{upload_id}", headers=viewer_headers)
    assert second.status_code == 404
    db.expire_all()
    assert db.query(VideoUpload).filter(VideoUpload.id == upload_id).first().deletion_requested is True

    # Only once the worker actually runs (simulated directly here, since
    # the real one was disabled above) does cleanup happen.
    assert video_analysis._finalize_if_deletion_requested(upload_id) is True
    assert not stored_path.exists()
    db.expire_all()
    assert db.query(VideoUpload).filter(VideoUpload.id == upload_id).first() is None


def test_deleting_immediately_after_upload_races_the_real_background_thread(
    client, viewer_headers, uploads_dir, db,
):
    """The genuine end-to-end version: a real POST starts a real
    background thread running the real (pose-model) pipeline, and DELETE
    is issued essentially immediately after - racing whatever that
    thread has managed to do so far. Which exact interleaving occurs is
    not asserted (and does not need to be, given the interleaving-specific
    tests above); what must hold regardless is the final state: no
    lingering row, no orphaned Event/Recording, no orphaned file on disk,
    and the API reporting the upload as gone right away."""
    video_bytes = _tiny_video_bytes()
    created = client.post(
        "/api/video-uploads", files={"file": ("clip.mp4", io.BytesIO(video_bytes), "video/mp4")},
        headers=viewer_headers,
    )
    assert created.status_code == 201
    upload_id = created.json()["id"]

    resp = client.delete(f"/api/video-uploads/{upload_id}", headers=viewer_headers)
    assert resp.status_code == 204

    # Gone from the API's point of view immediately, regardless of
    # whether the background thread has noticed yet.
    assert client.get(f"/api/video-uploads/{upload_id}", headers=viewer_headers).status_code == 404

    _wait_until_fully_gone(db, upload_id)
    assert db.query(Event).filter(Event.video_upload_id == upload_id).count() == 0
    assert db.query(Recording).filter(Recording.video_upload_id == upload_id).count() == 0

    remaining_files = [p for p in Path(uploads_dir).rglob("*") if p.is_file()]
    assert remaining_files == [], f"orphaned upload file(s) after deletion: {remaining_files}"


def test_repeated_delete_on_a_completed_upload_matches_existing_semantics(
    client, viewer_headers, uploads_dir, db,
):
    """Unchanged from before this fix: deleting an already-terminal
    upload hard-deletes it immediately, and a second delete 404s - the
    same idempotency semantics as everywhere else in this API."""
    video_bytes = _tiny_video_bytes(frames=6)
    created = client.post(
        "/api/video-uploads", files={"file": ("clip.mp4", io.BytesIO(video_bytes), "video/mp4")},
        headers=viewer_headers,
    )
    upload_id = created.json()["id"]
    _wait_for_terminal_status(db, upload_id)

    first = client.delete(f"/api/video-uploads/{upload_id}", headers=viewer_headers)
    assert first.status_code == 204
    second = client.delete(f"/api/video-uploads/{upload_id}", headers=viewer_headers)
    assert second.status_code == 404


def test_admin_cannot_delete_a_user_with_an_upload_still_being_analyzed(
    client, admin_headers, viewer_user, viewer_headers, uploads_dir, db, monkeypatch,
):
    """The same race, reached through a second door: admin-deletes-user
    used to cascade-delete every upload unconditionally, including one a
    worker thread might still be analysing. video_uploads.user_id is
    NOT NULL, so the deferred-flag approach used elsewhere cannot apply
    here (the upload row would keep referencing the about-to-be-deleted
    user) - refusing with a clear, actionable error is the safe choice."""
    monkeypatch.setattr("app.api.routes.video_uploads.analyze_video_upload", lambda video_upload_id: None)

    video_bytes = _tiny_video_bytes()
    created = client.post(
        "/api/video-uploads", files={"file": ("clip.mp4", io.BytesIO(video_bytes), "video/mp4")},
        headers=viewer_headers,
    )
    assert created.status_code == 201

    resp = client.delete(f"/api/admin/users/{viewer_user.id}", headers=admin_headers)
    assert resp.status_code == 409
    assert "still being analyzed" in resp.json()["detail"]

    # The user and their upload are untouched.
    db.expire_all()
    from app.models.user import User
    assert db.query(User).filter(User.id == viewer_user.id).first() is not None
    assert db.query(VideoUpload).filter(VideoUpload.user_id == viewer_user.id).count() == 1


def test_progress_updates_stop_broadcasting_once_deletion_is_requested(db, viewer_user, tmp_path, monkeypatch):
    """Invariant: a deleted upload must not be resurrected by a stale
    background task. The API stops returning the upload the instant the
    flag is set, so a worker still mid-loop must not keep broadcasting
    `upload.progress` events naming it - a client could otherwise use one
    to re-add a row the user just deleted."""
    published = []
    monkeypatch.setattr(
        video_analysis.realtime_broadcaster, "publish",
        lambda event_type, data, **kw: published.append(event_type),
    )

    upload = VideoUpload(
        user_id=viewer_user.id, original_filename="x.mp4", stored_path=str(tmp_path / "x.mp4"),
        status=STATUS_PROCESSING,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    upload_id = upload.id

    video_analysis._set_status(upload_id, progress_percent=10)
    assert published == ["upload.progress"]

    upload.deletion_requested = True
    db.commit()

    video_analysis._set_status(upload_id, progress_percent=20)
    assert published == ["upload.progress"], "progress was still broadcast after deletion was requested"

    db.expire_all()
    assert db.query(VideoUpload).filter(VideoUpload.id == upload_id).first().progress_percent == 10
