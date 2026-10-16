"""
Editing a running camera's URL must take effect.

`StreamManager.get_or_create` only replaced a stream that had died, and a
capture thread pointed at an unreachable address never dies - it retries
with backoff forever. So correcting a mistyped RTSP URL through
`PUT /api/cameras/{id}` changed the database row and nothing else: the old
thread kept retrying the old address until the process restarted, and the
detection loop kept a reference to that stream regardless.
"""
import pytest

from app.models.camera import Camera
from app.services import stream_manager as stream_manager_mod
from app.services.detection import engine as engine_mod


class FakeStream:
    """Stands in for CameraStream: no capture thread, no device."""

    def __init__(self, camera_id, url):
        self.camera_id = camera_id
        self.url = url
        self.alive = False

    def start(self):
        self.alive = True

    def stop(self):
        self.alive = False

    @property
    def is_alive(self):
        return self.alive


@pytest.fixture()
def manager(monkeypatch):
    monkeypatch.setattr(stream_manager_mod, "CameraStream", FakeStream)
    manager = stream_manager_mod.StreamManager()
    monkeypatch.setattr(stream_manager_mod, "stream_manager", manager)
    monkeypatch.setattr(engine_mod, "stream_manager", manager)
    from app.api.routes import cameras as cameras_route
    monkeypatch.setattr(cameras_route, "stream_manager", manager)
    return manager


def test_the_same_url_reuses_the_running_stream(manager):
    first = manager.get_or_create(1, "rtsp://old")
    assert manager.get_or_create(1, "rtsp://old") is first


def test_a_new_url_replaces_the_running_stream(manager):
    old = manager.get_or_create(1, "rtsp://old")

    new = manager.get_or_create(1, "rtsp://new")

    assert new is not old and new.url == "rtsp://new" and new.is_alive
    assert not old.is_alive, "the old capture thread must be stopped, not left retrying"
    assert manager.get(1) is new


def test_an_already_running_detection_loop_gets_the_new_stream(manager, monkeypatch):
    """ensure_running used to return before touching the stream whenever
    the loop's thread was alive - which it always is while detection is on."""
    detection = engine_mod.DetectionEngine()

    class AliveThread:
        def is_alive(self):
            return True

    manager.get_or_create(7, "rtsp://old")
    detection._threads[7] = AliveThread()

    detection.ensure_running(7, "rtsp://new")

    assert manager.get(7).url == "rtsp://new"


def test_editing_a_watched_camera_s_url_through_the_api_restarts_its_capture(
    client, db, operator_headers, manager,
):
    camera = Camera(name="Gate", url="rtsp://old", camera_type="ip", ai_detection_enabled=False)
    db.add(camera)
    db.commit()
    manager.get_or_create(camera.id, "rtsp://old")  # someone has the live view open

    response = client.put(f"/api/cameras/{camera.id}", headers=operator_headers, json={"url": "rtsp://new"})

    assert response.status_code == 200
    assert manager.get(camera.id).url == "rtsp://new"


def test_an_edit_does_not_start_capture_for_a_camera_nobody_is_watching(client, db, operator_headers, manager):
    """Capture starts lazily; an edit must not change that."""
    camera = Camera(name="Yard", url="rtsp://old", camera_type="ip", ai_detection_enabled=False)
    db.add(camera)
    db.commit()

    client.put(f"/api/cameras/{camera.id}", headers=operator_headers, json={"url": "rtsp://new"})

    assert manager.get(camera.id) is None
