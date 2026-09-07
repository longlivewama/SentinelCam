import io
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.models.event import Event
from app.models.recording import Recording
from app.models.video_upload import STATUS_COMPLETED, STATUS_FAILED, VideoUpload


@pytest.fixture()
def uploads_dir(tmp_path, monkeypatch):
    import app.config as config_module

    monkeypatch.setattr(config_module.settings, "UPLOADS_DIR", str(tmp_path))
    return tmp_path


def _tiny_video_bytes(frames=8, size=(64, 64), fps=10) -> bytes:
    path = None
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), fill_value=i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    data = Path(path).read_bytes()
    Path(path).unlink(missing_ok=True)
    return data


def test_upload_rejects_unsupported_extension(client, viewer_headers, uploads_dir):
    resp = client.post(
        "/api/video-uploads",
        files={"file": ("clip.txt", io.BytesIO(b"not a video"), "text/plain")},
        headers=viewer_headers,
    )
    assert resp.status_code == 400


def test_upload_rejects_empty_file(client, viewer_headers, uploads_dir):
    resp = client.post(
        "/api/video-uploads",
        files={"file": ("clip.mp4", io.BytesIO(b""), "video/mp4")},
        headers=viewer_headers,
    )
    assert resp.status_code == 400


def test_upload_ownership_visibility(client, viewer_headers, operator_headers, viewer_user, uploads_dir, db):
    video_bytes = _tiny_video_bytes()
    created = client.post(
        "/api/video-uploads",
        files={"file": ("clip.mp4", io.BytesIO(video_bytes), "video/mp4")},
        headers=viewer_headers,
    )
    assert created.status_code == 201
    upload_id = created.json()["id"]

    # Owner can see their own upload.
    own = client.get(f"/api/video-uploads/{upload_id}", headers=viewer_headers)
    assert own.status_code == 200

    # Operator (not the owner) can see it too - operators have oversight
    # of all uploads.
    as_operator = client.get(f"/api/video-uploads/{upload_id}", headers=operator_headers)
    assert as_operator.status_code == 200

    # A different plain viewer cannot.
    other_viewer_token_resp = client.post(
        "/api/auth/signup", json={"email": "other-viewer@example.com", "password": "password123"}
    )
    other_headers = {"Authorization": f"Bearer {other_viewer_token_resp.json()['access_token']}"}
    forbidden = client.get(f"/api/video-uploads/{upload_id}", headers=other_headers)
    assert forbidden.status_code == 403

    # Let the background analysis thread finish before the test DB is torn
    # down, so nothing writes to a dropped connection after the test ends.
    _wait_for_terminal_status(db, upload_id)


def _wait_for_terminal_status(db, upload_id, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        db.expire_all()
        upload = db.query(VideoUpload).filter(VideoUpload.id == upload_id).first()
        if upload and upload.status in (STATUS_COMPLETED, STATUS_FAILED):
            return upload
        time.sleep(0.5)
    raise AssertionError(f"Video upload {upload_id} did not reach a terminal status within {timeout}s")


def test_upload_is_processed_end_to_end(client, viewer_headers, uploads_dir, db):
    """Full pipeline smoke test: upload a tiny synthetic (person-free)
    video and confirm it's actually decoded, run through the real pose
    model, and marked completed with zero false-positive falls."""
    video_bytes = _tiny_video_bytes(frames=6)
    created = client.post(
        "/api/video-uploads",
        files={"file": ("clip.mp4", io.BytesIO(video_bytes), "video/mp4")},
        headers=viewer_headers,
    )
    assert created.status_code == 201
    upload_id = created.json()["id"]

    upload = _wait_for_terminal_status(db, upload_id)
    assert upload.status == STATUS_COMPLETED
    assert upload.persons_detected == 0
    assert upload.fall_events_count == 0
    assert upload.frame_count == 6


def test_deleting_upload_cascades_to_its_recordings_and_events(client, viewer_headers, uploads_dir, db, tmp_path):
    """Regression test: deleting a VideoUpload that has derived
    Recording/Event rows must not hit their FK constraints - it should
    cascade-delete them (they have no meaning without the source upload)."""
    video_bytes = _tiny_video_bytes()
    created = client.post(
        "/api/video-uploads",
        files={"file": ("clip.mp4", io.BytesIO(video_bytes), "video/mp4")},
        headers=viewer_headers,
    )
    upload_id = created.json()["id"]
    _wait_for_terminal_status(db, upload_id)

    clip_path = tmp_path / "fake_clip.mp4"
    clip_path.write_bytes(b"fake")
    recording = Recording(
        video_upload_id=upload_id,
        filename="fake_clip.mp4",
        file_path=str(clip_path),
        duration_seconds=1.0,
        trigger_action="fall",
        file_size_bytes=4,
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(recording)
    db.commit()
    db.refresh(recording)

    event = Event(
        video_upload_id=upload_id,
        recording_id=recording.id,
        event_type="fall",
        confidence_score=0.9,
        timestamp=datetime.now(timezone.utc),
        triggered_recording=True,
    )
    db.add(event)
    db.commit()
    # Captured before the delete request, since these ORM objects can't be
    # dereferenced afterwards - the API call deletes their rows through a
    # different DB session, which would raise ObjectDeletedError on access.
    recording_id, event_id = recording.id, event.id

    resp = client.delete(f"/api/video-uploads/{upload_id}", headers=viewer_headers)
    assert resp.status_code == 204

    db.expire_all()
    assert db.query(Recording).filter(Recording.id == recording_id).first() is None
    assert db.query(Event).filter(Event.id == event_id).first() is None
    assert not clip_path.exists()
