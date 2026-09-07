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
