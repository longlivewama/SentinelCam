from datetime import datetime, timezone

from app.models.camera import Camera
from app.models.event import Event


def _make_camera(db, name="Cam A"):
    camera = Camera(name=name, url="0", camera_type="usb")
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


def _make_event(db, camera_id, event_type="fall", confidence=0.9):
    event = Event(
        camera_id=camera_id,
        event_type=event_type,
        confidence_score=confidence,
        timestamp=datetime.now(timezone.utc),
        triggered_recording=True,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def test_list_alerts_requires_auth(client):
    assert client.get("/api/alerts").status_code == 401


def test_list_and_filter_alerts(client, viewer_headers, db):
    camera = _make_camera(db)
    _make_event(db, camera.id, event_type="fall")
    _make_event(db, camera.id, event_type="crowd")

    all_alerts = client.get("/api/alerts", headers=viewer_headers)
    assert all_alerts.status_code == 200
    assert len(all_alerts.json()) == 2

    falls_only = client.get("/api/alerts", params={"event_type": "fall"}, headers=viewer_headers)
    assert len(falls_only.json()) == 1
    assert falls_only.json()[0]["event_type"] == "fall"


def test_viewer_cannot_acknowledge_alert(client, viewer_headers, db):
    camera = _make_camera(db)
    event = _make_event(db, camera.id)

    resp = client.put(f"/api/alerts/{event.id}/acknowledge", headers=viewer_headers)
    assert resp.status_code == 403


def test_operator_can_acknowledge_alert(client, operator_headers, operator_user, db):
    camera = _make_camera(db)
    event = _make_event(db, camera.id)

    resp = client.put(f"/api/alerts/{event.id}/acknowledge", headers=operator_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["acknowledged"] is True
    assert body["acknowledged_by"] == operator_user.id
    assert body["acknowledged_at"] is not None


def test_acknowledge_missing_alert_404s(client, operator_headers):
    resp = client.put("/api/alerts/999999/acknowledge", headers=operator_headers)
    assert resp.status_code == 404
