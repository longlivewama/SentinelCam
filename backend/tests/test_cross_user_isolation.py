"""
Regression tests for the cross-user data exposure found in the
integration audit: /api/alerts and /api/recordings applied no ownership
filter at all, so any authenticated user - including a plain `viewer` -
could list, read, stream and download every other user's upload-derived
alerts and fall clips, and see their counts in /api/analytics/summary.

The rule under test (see app/core/scoping.py): rows derived from a user's
uploaded video belong to that user; rows from a shared camera are visible
to every authenticated user; operators and admins see everything.
"""
from datetime import datetime, timezone

import pytest

from app.models.event import Event
from app.models.recording import Recording
from app.models.user import ROLE_VIEWER
from app.models.video_upload import VideoUpload


@pytest.fixture()
def other_user(make_user_factory):
    return make_user_factory("someone-else@example.com", ROLE_VIEWER)


@pytest.fixture()
def other_users_upload(db, other_user, tmp_path):
    """A completed analysis belonging to `other_user`: the upload, one
    fall clip, and the alert that references it."""
    clip = tmp_path / "other_users_fall.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)

    upload = VideoUpload(
        user_id=other_user.id,
        original_filename="confidential-ward-3.mp4",
        stored_path=str(tmp_path / "source.mp4"),
        status="completed",
        file_size_bytes=1234,
    )
    db.add(upload)
    db.flush()

    recording = Recording(
        video_upload_id=upload.id,
        filename="other_users_fall.mp4",
        file_path=str(clip),
        duration_seconds=6.0,
        trigger_action="fall",
        file_size_bytes=clip.stat().st_size,
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(recording)
    db.flush()

    event = Event(
        video_upload_id=upload.id,
        recording_id=recording.id,
        event_type="fall",
        confidence_score=0.9,
        timestamp=datetime.now(timezone.utc),
        video_timestamp_seconds=4.5,
        detector="model",
        triggered_recording=True,
    )
    db.add(event)
    db.commit()
    db.refresh(upload)
    db.refresh(recording)
    db.refresh(event)
    return {"upload": upload, "recording": recording, "event": event}


# --- alerts ---------------------------------------------------------------

def test_viewer_cannot_list_another_users_upload_alerts(client, viewer_headers, other_users_upload):
    response = client.get("/api/alerts", headers=viewer_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_viewer_cannot_read_another_users_upload_alert_by_id(client, viewer_headers, other_users_upload):
    alert_id = other_users_upload["event"].id
    response = client.get(f"/api/alerts/{alert_id}", headers=viewer_headers)
    # 404, not 403: confirming the ID exists is itself a disclosure.
    assert response.status_code == 404


def test_viewer_cannot_reach_another_users_alerts_via_the_upload_filter(
    client, viewer_headers, other_users_upload,
):
    """The video_upload_id filter must narrow the caller's own rows, not
    escape the ownership scope."""
    upload_id = other_users_upload["upload"].id
    response = client.get("/api/alerts", params={"video_upload_id": upload_id}, headers=viewer_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_operator_can_see_all_upload_alerts(client, operator_headers, other_users_upload):
    response = client.get("/api/alerts", headers=operator_headers)
    assert response.status_code == 200
    assert [a["id"] for a in response.json()] == [other_users_upload["event"].id]


# --- recordings -----------------------------------------------------------

def test_viewer_cannot_list_another_users_upload_recordings(client, viewer_headers, other_users_upload):
    response = client.get("/api/recordings", headers=viewer_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_viewer_cannot_read_another_users_recording_by_id(client, viewer_headers, other_users_upload):
    recording_id = other_users_upload["recording"].id
    assert client.get(f"/api/recordings/{recording_id}", headers=viewer_headers).status_code == 404


def test_viewer_cannot_stream_another_users_recording(client, viewer_headers, other_users_upload):
    recording_id = other_users_upload["recording"].id
    assert client.get(f"/api/recordings/{recording_id}/video", headers=viewer_headers).status_code == 404


def test_viewer_cannot_download_another_users_recording(client, viewer_headers, other_users_upload):
    recording_id = other_users_upload["recording"].id
    assert client.get(f"/api/recordings/{recording_id}/download", headers=viewer_headers).status_code == 404


def test_owner_can_still_read_their_own_upload_results(client, other_user, other_users_upload, auth_headers_for):
    headers = auth_headers_for(other_user)

    alerts = client.get("/api/alerts", headers=headers)
    assert alerts.status_code == 200
    assert [a["id"] for a in alerts.json()] == [other_users_upload["event"].id]

    recordings = client.get("/api/recordings", headers=headers)
    assert recordings.status_code == 200
    assert [r["id"] for r in recordings.json()] == [other_users_upload["recording"].id]

    stream = client.get(f"/api/recordings/{other_users_upload['recording'].id}/video", headers=headers)
    assert stream.status_code == 200


def test_recording_payload_does_not_expose_the_server_file_path(client, other_user, other_users_upload, auth_headers_for):
    response = client.get("/api/recordings", headers=auth_headers_for(other_user))
    assert response.status_code == 200
    assert "file_path" not in response.json()[0]


# --- camera-sourced rows stay shared -------------------------------------

def test_camera_alerts_and_recordings_remain_visible_to_every_user(client, viewer_headers, db, tmp_path):
    """Cameras are shared infrastructure - GET /api/cameras lists them to
    everyone - so their alerts and clips must NOT be locked to one user by
    the ownership filter."""
    from app.models.camera import Camera

    camera = Camera(name="Ward 1", url="rtsp://example.invalid/1", is_active=True)
    db.add(camera)
    db.flush()

    clip = tmp_path / "camera_fall.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64)
    recording = Recording(
        camera_id=camera.id,
        filename="camera_fall.mp4",
        file_path=str(clip),
        duration_seconds=6.0,
        trigger_action="fall",
        file_size_bytes=clip.stat().st_size,
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(recording)
    db.flush()
    db.add(Event(
        camera_id=camera.id,
        recording_id=recording.id,
        event_type="fall",
        confidence_score=0.8,
        timestamp=datetime.now(timezone.utc),
        triggered_recording=True,
    ))
    db.commit()

    assert len(client.get("/api/alerts", headers=viewer_headers).json()) == 1
    assert len(client.get("/api/recordings", headers=viewer_headers).json()) == 1


# --- analytics ------------------------------------------------------------

def test_analytics_does_not_count_another_users_uploads(client, viewer_headers, other_users_upload):
    summary = client.get("/api/analytics/summary", headers=viewer_headers).json()
    assert summary["alerts"]["total"] == 0
    assert summary["recordings"]["total"] == 0
    assert summary["video_uploads"]["total"] == 0


def test_analytics_counts_everything_for_an_operator(client, operator_headers, other_users_upload):
    summary = client.get("/api/analytics/summary", headers=operator_headers).json()
    assert summary["alerts"]["total"] == 1
    assert summary["recordings"]["total"] == 1
    assert summary["video_uploads"]["total"] == 1
