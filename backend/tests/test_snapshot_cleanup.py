"""
An event's snapshot JPEG must go wherever its clip goes.

Every fall clip is written alongside a snapshot - the frame emailed to
operators - whose path is not stored on the Recording row. None of the
delete paths knew where it was, so deleting an upload, a recording or a
whole account removed the clip and left a still of the footage behind,
indefinitely. Found by an end-to-end audit: after a successful
`DELETE /api/video-uploads/{id}`, `snapshots/upload{id}_..._fall.jpg` was
the only file of that upload still on disk.

These tests drive the real writers (not hand-made files named by a guess
at the convention), so a writer and a deleter can never quietly disagree
about the snapshot's name again.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.models.camera import Camera
from app.models.event import Event
from app.models.recording import Recording
from app.models.video_upload import STATUS_COMPLETED, VideoUpload
from app.services import recording_engine, video_analysis


@pytest.fixture()
def storage(monkeypatch, tmp_path):
    for name in ("RECORDINGS_DIR", "SNAPSHOTS_DIR", "UPLOADS_DIR"):
        path = tmp_path / name.lower()
        path.mkdir()
        monkeypatch.setattr(settings, name, str(path))
    return tmp_path


def _frames(count=6):
    return [np.full((48, 64, 3), i * 10, dtype=np.uint8) for i in range(count)]


def _snapshots():
    return sorted(Path(settings.SNAPSHOTS_DIR).glob("*.jpg"))


def _completed_upload(db, user):
    source = Path(settings.UPLOADS_DIR) / "source.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"x" * 64)
    upload = VideoUpload(
        user_id=user.id, original_filename="source.mp4", stored_path=str(source),
        status=STATUS_COMPLETED, progress_percent=100, file_size_bytes=source.stat().st_size,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


def _write_upload_fall(upload_id):
    assert video_analysis._write_upload_clip(
        upload_id, _frames(), 10.0, {"timestamp_seconds": 1.0, "confidence": 0.9, "detector": "model"},
    )


def test_deleting_an_upload_removes_its_snapshots(client, db, viewer_user, viewer_headers, storage):
    upload = _completed_upload(db, viewer_user)
    _write_upload_fall(upload.id)
    assert len(_snapshots()) == 1, "the writer should have produced exactly one snapshot"

    assert client.delete(f"/api/video-uploads/{upload.id}", headers=viewer_headers).status_code == 204

    assert _snapshots() == [], "the snapshot outlived its upload"


def test_deleting_a_user_removes_their_uploads_snapshots_and_clip_dirs(
    client, db, viewer_user, admin_headers, storage,
):
    upload = _completed_upload(db, viewer_user)
    _write_upload_fall(upload.id)
    clip_dir = Path(settings.RECORDINGS_DIR) / "uploads" / str(upload.id)
    assert clip_dir.is_dir() and len(_snapshots()) == 1

    assert client.delete(f"/api/admin/users/{viewer_user.id}", headers=admin_headers).status_code == 204

    assert _snapshots() == []
    assert not clip_dir.exists(), "the per-upload clip directory outlived its owner"


@pytest.fixture()
def camera_recording(db, storage):
    """A camera clip written by the live recording engine's own writer."""
    camera = Camera(name="Lobby", url="0", camera_type="usb")
    db.add(camera)
    db.commit()
    frames = _frames()
    recording_engine._write_recording(
        camera.id, "fall", 0.8, datetime.now(timezone.utc) - timedelta(days=40), frames, frames[:3], "model",
    )
    db.expire_all()
    recording = db.query(Recording).filter(Recording.camera_id == camera.id).one()
    assert len(_snapshots()) == 1
    return recording


def test_deleting_a_recording_removes_its_snapshot(client, db, operator_headers, camera_recording):
    assert client.delete(f"/api/recordings/{camera_recording.id}", headers=operator_headers).status_code == 204

    assert _snapshots() == []
    assert not Path(camera_recording.file_path).exists()
    # The alert itself is history and stays; only its clip link is cleared.
    event = db.query(Event).filter(Event.camera_id == camera_recording.camera_id).one()
    db.refresh(event)
    assert event.recording_id is None


def test_retention_cleanup_removes_snapshots_too(monkeypatch, db, camera_recording):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import cleanup_old_recordings

    monkeypatch.setattr(sys, "argv", ["cleanup_old_recordings.py", "--days", "30"])
    cleanup_old_recordings.main()

    assert _snapshots() == []
    assert not Path(camera_recording.file_path).exists()
