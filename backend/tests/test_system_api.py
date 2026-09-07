def test_system_status_requires_operator(client, viewer_headers):
    assert client.get("/api/system/status", headers=viewer_headers).status_code == 403


def test_system_status_shape(client, operator_headers):
    resp = client.get("/api/system/status", headers=operator_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    assert "detection" in body
    assert "notifications" in body
