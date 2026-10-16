def test_admin_endpoints_require_admin_role(client, viewer_headers, operator_headers):
    for headers in (viewer_headers, operator_headers):
        assert client.get("/api/admin/users", headers=headers).status_code == 403


def test_admin_can_list_create_and_delete_users(client, admin_headers):
    listed = client.get("/api/admin/users", headers=admin_headers)
    assert listed.status_code == 200

    created = client.post(
        "/api/admin/users",
        json={"email": "operator2@example.com", "password": "password123", "role": "operator"},
        headers=admin_headers,
    )
    assert created.status_code == 201
    user_id = created.json()["id"]
    assert created.json()["role"] == "operator"

    toggled = client.put(f"/api/admin/users/{user_id}/toggle", headers=admin_headers)
    assert toggled.status_code == 200
    assert toggled.json()["is_active"] is False

    role_updated = client.put(
        f"/api/admin/users/{user_id}/role", json={"role": "admin"}, headers=admin_headers
    )
    assert role_updated.status_code == 200
    assert role_updated.json()["role"] == "admin"

    deleted = client.delete(f"/api/admin/users/{user_id}", headers=admin_headers)
    assert deleted.status_code == 204


def test_admin_cannot_delete_or_demote_self(client, admin_headers, admin_user):
    assert client.delete(f"/api/admin/users/{admin_user.id}", headers=admin_headers).status_code == 400
    assert client.put(
        f"/api/admin/users/{admin_user.id}/toggle", headers=admin_headers
    ).status_code == 400
    assert client.put(
        f"/api/admin/users/{admin_user.id}/role", json={"role": "viewer"}, headers=admin_headers
    ).status_code == 400


def test_create_user_rejects_invalid_role(client, admin_headers):
    resp = client.post(
        "/api/admin/users",
        json={"email": "bad-role@example.com", "password": "password123", "role": "superuser"},
        headers=admin_headers,
    )
    assert resp.status_code == 422


def test_create_user_rejects_duplicate_email(client, admin_headers, viewer_user):
    resp = client.post(
        "/api/admin/users",
        json={"email": viewer_user.email, "password": "password123", "role": "viewer"},
        headers=admin_headers,
    )
    assert resp.status_code == 400


def test_deleting_user_with_uploads_and_acknowledged_alerts_does_not_500(
    client, admin_headers, viewer_user, operator_user, db,
):
    """Regression test: a user who has uploaded a video and/or
    acknowledged an alert is referenced by foreign keys on VideoUpload,
    Event.acknowledged_by, and PasswordResetToken - deleting them must
    cascade/clear those instead of hitting the FK constraints."""
    from datetime import datetime, timezone

    from app.models.camera import Camera
    from app.models.event import Event
    from app.models.password_reset_token import PasswordResetToken
    from app.models.video_upload import VideoUpload

    upload = VideoUpload(user_id=viewer_user.id, original_filename="f.mp4", stored_path="/tmp/does-not-exist.mp4", file_size_bytes=1)
    db.add(upload)

    camera = Camera(name="Cam A", url="0", camera_type="usb")
    db.add(camera)
    db.commit()
    db.refresh(camera)

    event = Event(
        camera_id=camera.id, event_type="fall", confidence_score=0.9,
        timestamp=datetime.now(timezone.utc), triggered_recording=True,
        acknowledged=True, acknowledged_by=operator_user.id, acknowledged_at=datetime.now(timezone.utc),
    )
    db.add(event)

    reset_token = PasswordResetToken(
        user_id=operator_user.id, token_hash="x" * 64,
        expires_at=datetime.now(timezone.utc),
    )
    db.add(reset_token)
    db.commit()

    resp = client.delete(f"/api/admin/users/{operator_user.id}", headers=admin_headers)
    assert resp.status_code == 204

    db.expire_all()
    refreshed_event = db.query(Event).filter(Event.id == event.id).first()
    assert refreshed_event is not None
    assert refreshed_event.acknowledged is True
    assert refreshed_event.acknowledged_by is None


def test_deleting_the_owner_of_analysed_uploads_removes_them(client, admin_headers, viewer_user, db, tmp_path):
    """The test above deletes a user who owns no uploads, so it never
    reached the cascade it is named after. Deleting the owner of finished
    analyses - uploads with recordings and fall events - used to 500 with
    a video_uploads.user_id FK violation, because nothing ordered the
    staged upload deletes before the user's own."""
    from datetime import datetime, timezone

    from app.models.event import Event
    from app.models.recording import Recording
    from app.models.user import User
    from app.models.video_upload import VideoUpload

    for index in range(3):
        upload = VideoUpload(
            user_id=viewer_user.id, original_filename=f"{index}.mp4",
            stored_path=str(tmp_path / f"{index}.mp4"), status="completed",
        )
        db.add(upload)
        db.flush()
        recording = Recording(
            video_upload_id=upload.id, filename=f"{index}_fall.mp4", file_path=str(tmp_path / f"{index}_fall.mp4"),
            trigger_action="fall", event_timestamp=datetime.now(timezone.utc),
        )
        db.add(recording)
        db.flush()
        db.add(Event(video_upload_id=upload.id, recording_id=recording.id, event_type="fall"))
    db.commit()
    user_id = viewer_user.id

    resp = client.delete(f"/api/admin/users/{user_id}", headers=admin_headers)
    assert resp.status_code == 204

    db.expire_all()
    assert db.query(User).filter(User.id == user_id).count() == 0
    assert db.query(VideoUpload).filter(VideoUpload.user_id == user_id).count() == 0
    assert db.query(Recording).filter(Recording.video_upload_id.isnot(None)).count() == 0
    assert db.query(Event).filter(Event.video_upload_id.isnot(None)).count() == 0
