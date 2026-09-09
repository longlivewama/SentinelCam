from datetime import datetime, timezone

from app.models.camera import Camera
from app.models.event import Event
from app.models.recording import Recording


def _make_recording(db, camera_id, trigger_action="fall"):
    recording = Recording(
        camera_id=camera_id,
        filename="clip.mp4",
        file_path="/tmp/does-not-exist.mp4",
        duration_seconds=6.0,
        trigger_action=trigger_action,
        file_size_bytes=1234,
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(recording)
    db.commit()
    db.refresh(recording)
    return recording


def test_list_recordings_requires_auth(client):
    assert client.get("/api/recordings").status_code == 401


def test_list_and_filter_recordings(client, viewer_headers, db):
    camera = Camera(name="Cam A", url="0", camera_type="usb")
    db.add(camera)
    db.commit()
    db.refresh(camera)
    _make_recording(db, camera.id)

    resp = client.get("/api/recordings", headers=viewer_headers)
    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 1
    assert resp.json()["total"] == 1

    filtered = client.get("/api/recordings", params={"camera_id": camera.id}, headers=viewer_headers)
    assert len(filtered.json()["items"]) == 1

    empty = client.get("/api/recordings", params={"camera_id": 999999}, headers=viewer_headers)
    # An empty page still carries the envelope, and reports no pages at all
    # rather than "page 1 of 1" over nothing.
    assert empty.json()["items"] == []
    assert empty.json()["total"] == 0
    assert empty.json()["pages"] == 0


def test_viewer_cannot_delete_recording(client, viewer_headers, db):
    camera = Camera(name="Cam A", url="0", camera_type="usb")
    db.add(camera)
    db.commit()
    db.refresh(camera)
    recording = _make_recording(db, camera.id)

    resp = client.delete(f"/api/recordings/{recording.id}", headers=viewer_headers)
    assert resp.status_code == 403


def test_operator_can_delete_recording(client, operator_headers, db):
    camera = Camera(name="Cam A", url="0", camera_type="usb")
    db.add(camera)
    db.commit()
    db.refresh(camera)
    recording = _make_recording(db, camera.id)

    resp = client.delete(f"/api/recordings/{recording.id}", headers=operator_headers)
    assert resp.status_code == 204


def test_deleting_recording_linked_to_an_alert_does_not_500(client, operator_headers, db):
    """Regression test: deleting a Recording that an Event.recording_id
    points to must not hit the FK constraint - the alert is kept, only its
    recording_id link is cleared."""
    camera = Camera(name="Cam A", url="0", camera_type="usb")
    db.add(camera)
    db.commit()
    db.refresh(camera)
    recording = _make_recording(db, camera.id)

    event = Event(
        camera_id=camera.id,
        recording_id=recording.id,
        event_type="fall",
        confidence_score=0.9,
        timestamp=datetime.now(timezone.utc),
        triggered_recording=True,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    resp = client.delete(f"/api/recordings/{recording.id}", headers=operator_headers)
    assert resp.status_code == 204

    db.refresh(event)
    assert event.recording_id is None


def test_missing_recording_video_404s(client, viewer_headers, db):
    camera = Camera(name="Cam A", url="0", camera_type="usb")
    db.add(camera)
    db.commit()
    db.refresh(camera)
    recording = _make_recording(db, camera.id)

    # file_path points to a nonexistent file - should 404, not 500.
    resp = client.get(f"/api/recordings/{recording.id}/video", headers=viewer_headers)
    assert resp.status_code == 404
