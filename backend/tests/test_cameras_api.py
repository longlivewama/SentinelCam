CAMERA_PAYLOAD = {"name": "Front Door", "url": "0", "camera_type": "usb", "location": "Entrance"}


def test_list_cameras_requires_auth(client):
    assert client.get("/api/cameras").status_code == 401


def test_viewer_can_list_but_not_create_camera(client, viewer_headers):
    listed = client.get("/api/cameras", headers=viewer_headers)
    assert listed.status_code == 200
    assert listed.json() == []

    created = client.post("/api/cameras", json=CAMERA_PAYLOAD, headers=viewer_headers)
    assert created.status_code == 403


def test_operator_can_create_update_delete_camera(client, operator_headers):
    created = client.post("/api/cameras", json=CAMERA_PAYLOAD, headers=operator_headers)
    assert created.status_code == 201
    camera_id = created.json()["id"]

    updated = client.put(
        f"/api/cameras/{camera_id}", json={"location": "Lobby"}, headers=operator_headers
    )
    assert updated.status_code == 200
    assert updated.json()["location"] == "Lobby"

    deleted = client.delete(f"/api/cameras/{camera_id}", headers=operator_headers)
    assert deleted.status_code == 204

    # Soft-deleted cameras drop out of the list.
    listed = client.get("/api/cameras", headers=operator_headers)
    assert all(c["id"] != camera_id for c in listed.json())


def test_admin_can_manage_cameras_too(client, admin_headers):
    created = client.post("/api/cameras", json=CAMERA_PAYLOAD, headers=admin_headers)
    assert created.status_code == 201


def test_get_nonexistent_camera_404s(client, viewer_headers):
    resp = client.get("/api/cameras/999999", headers=viewer_headers)
    assert resp.status_code == 404


def test_stream_requires_auth(client):
    # Must 401 before ever touching stream_manager / opening a capture
    # device - no video hardware is available in the test environment.
    resp = client.get("/api/cameras/1/stream")
    assert resp.status_code == 401


# --- Camera on/off, kept distinct from delete (deleted_at) -----------------


def test_switching_a_camera_off_keeps_it_in_the_list_unlike_delete(client, operator_headers):
    created = client.post("/api/cameras", json=CAMERA_PAYLOAD, headers=operator_headers)
    camera_id = created.json()["id"]

    off = client.put(f"/api/cameras/{camera_id}", json={"is_active": False}, headers=operator_headers)
    assert off.status_code == 200
    assert off.json()["is_active"] is False

    # Unlike delete, turning a camera off must not remove it from the list -
    # it has to stay visible (and reachable) to be switched back on.
    listed = client.get("/api/cameras", headers=operator_headers)
    assert any(c["id"] == camera_id for c in listed.json())

    back_on = client.put(f"/api/cameras/{camera_id}", json={"is_active": True}, headers=operator_headers)
    assert back_on.status_code == 200
    assert back_on.json()["is_active"] is True


def test_deleted_camera_is_unreachable_by_id_afterwards(client, operator_headers):
    created = client.post("/api/cameras", json=CAMERA_PAYLOAD, headers=operator_headers)
    camera_id = created.json()["id"]
    assert client.delete(f"/api/cameras/{camera_id}", headers=operator_headers).status_code == 204

    assert client.get(f"/api/cameras/{camera_id}", headers=operator_headers).status_code == 404
    assert client.put(
        f"/api/cameras/{camera_id}", json={"location": "Lobby"}, headers=operator_headers
    ).status_code == 404
    # Deleting an already-deleted camera 404s too, rather than silently
    # succeeding a second time.
    assert client.delete(f"/api/cameras/{camera_id}", headers=operator_headers).status_code == 404


def test_turning_a_camera_off_stops_detection_and_the_capture_thread(client, operator_headers, monkeypatch):
    from app.api.routes import cameras as cameras_route

    calls = {"detection": [], "stream_stopped": 0}

    class FakeStream:
        def stop(self):
            calls["stream_stopped"] += 1

    monkeypatch.setattr(
        cameras_route.detection_engine, "ensure_running",
        lambda cid, url: calls["detection"].append(("start", cid)),
    )
    monkeypatch.setattr(
        cameras_route.detection_engine, "stop",
        lambda cid: calls["detection"].append(("stop", cid)),
    )
    monkeypatch.setattr(cameras_route.stream_manager, "get", lambda cid: FakeStream())

    payload = {**CAMERA_PAYLOAD, "ai_detection_enabled": True}
    created = client.post("/api/cameras", json=payload, headers=operator_headers)
    camera_id = created.json()["id"]
    # An active camera created with AI detection enabled starts detection
    # via the same _sync_detection_state path a later update goes through.
    assert ("start", camera_id) in calls["detection"]

    off = client.put(f"/api/cameras/{camera_id}", json={"is_active": False}, headers=operator_headers)
    assert off.status_code == 200
    assert ("stop", camera_id) in calls["detection"]
    assert calls["stream_stopped"] == 1, "turning the camera off must also stop the capture thread"


def test_ai_detection_can_be_turned_off_while_the_camera_stays_active(client, operator_headers, monkeypatch):
    from app.api.routes import cameras as cameras_route

    calls = []
    monkeypatch.setattr(
        cameras_route.detection_engine, "ensure_running", lambda cid, url: calls.append(("start", cid)),
    )
    monkeypatch.setattr(cameras_route.detection_engine, "stop", lambda cid: calls.append(("stop", cid)))

    payload = {**CAMERA_PAYLOAD, "ai_detection_enabled": True}
    created = client.post("/api/cameras", json=payload, headers=operator_headers)
    camera_id = created.json()["id"]
    assert ("start", camera_id) in calls

    resp = client.put(
        f"/api/cameras/{camera_id}", json={"ai_detection_enabled": False}, headers=operator_headers,
    )
    assert resp.status_code == 200
    # AI detection is off, but the camera itself is untouched - still on.
    assert resp.json()["is_active"] is True
    assert resp.json()["ai_detection_enabled"] is False
    assert ("stop", camera_id) in calls
