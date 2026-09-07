from datetime import datetime, timezone

from app.models.camera import Camera
from app.models.event import Event


def test_analytics_summary_requires_auth(client):
    assert client.get("/api/analytics/summary").status_code == 401


def test_analytics_summary_shape_and_counts(client, viewer_headers, db):
    camera = Camera(name="Cam A", url="0", camera_type="usb", status="active")
    db.add(camera)
    db.commit()
    db.refresh(camera)

    for _ in range(3):
        db.add(Event(
            camera_id=camera.id, event_type="fall", confidence_score=0.8,
            timestamp=datetime.now(timezone.utc), triggered_recording=True,
        ))
    db.commit()

    resp = client.get("/api/analytics/summary", headers=viewer_headers)
    assert resp.status_code == 200
    body = resp.json()

    assert body["cameras"]["total"] == 1
    assert body["cameras"]["active"] == 1
    assert body["alerts"]["total"] == 3
    assert body["alerts"]["unacknowledged"] == 3
    assert sum(day["count"] for day in body["falls"]["over_time"]) == 3
    assert body["falls"]["by_camera"][0]["camera_id"] == camera.id
    assert body["falls"]["by_camera"][0]["count"] == 3
    assert body["confidence"]["avg_by_type_recent"]["fall"] == 0.8
