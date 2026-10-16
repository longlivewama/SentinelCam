"""
A live MJPEG view must not pin server resources for as long as it lasts.

Found by an audit after a Playwright run left uvicorn unable to shut down:
`GET /api/cameras/{id}/stream` kept its request's database session checked
out (idle in transaction) for the whole life of the stream, and produced
frames from a sync generator. Measured against a real server: 15 views of
an offline camera left exactly 15 connections idle in transaction - the
entire SQLAlchemy pool - after which every database-backed endpoint hung,
while /health (no database) still answered. With no frame to send, the sync
generator's next() never returned, so a viewer's disconnect was never
noticed and those connections never came back.
"""
import asyncio
import inspect

import pytest

from app.api.routes import cameras as cameras_route
from app.database import SessionLocal, engine
from app.models.camera import Camera


def _run(coro):
    """Runs `coro` on a private loop. Not asyncio.run(): that clears the
    main thread's current loop on exit, and app startup (exercised by
    other tests in the session) still calls asyncio.get_event_loop()."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class FramelessStream:
    """An offline camera: the capture thread never produces a frame."""

    url = "rtsp://offline"

    def get_latest_jpeg(self):
        return None


@pytest.fixture()
def offline_camera(db, monkeypatch):
    camera = Camera(name="Offline", url="rtsp://offline", camera_type="ip")
    db.add(camera)
    db.commit()
    stream = FramelessStream()
    monkeypatch.setattr(cameras_route.stream_manager, "get_or_create", lambda cid, url: stream)
    monkeypatch.setattr(cameras_route.stream_manager, "get", lambda cid: stream)
    return camera


class FakeRequest:
    def __init__(self):
        self.disconnected = False

    async def is_disconnected(self):
        return self.disconnected


def _open_stream(camera_id, viewer_user, request=None):
    session = SessionLocal()
    response = cameras_route.stream_camera(
        camera_id, request=request or FakeRequest(), db=session, current_user=viewer_user,
    )
    return session, response


def test_an_open_stream_does_not_hold_a_database_connection(offline_camera, viewer_user):
    before = engine.pool.checkedout()

    session, _ = _open_stream(offline_camera.id, viewer_user)

    assert engine.pool.checkedout() == before, "the stream kept its pooled connection checked out"
    session.close()


def test_many_open_streams_leave_the_pool_usable(offline_camera, viewer_user):
    """More concurrent views than the pool has connections."""
    pool_capacity = engine.pool.size() + engine.pool._max_overflow
    opened = [_open_stream(offline_camera.id, viewer_user) for _ in range(pool_capacity + 5)]

    with SessionLocal() as other_request:
        assert other_request.query(Camera).count() >= 1

    for session, _ in opened:
        session.close()


def test_a_frameless_stream_can_be_cancelled_by_a_disconnect(offline_camera, viewer_user):
    """The body is async, so waiting for a frame is an await a disconnect
    can interrupt - not a blocking call on a worker thread."""
    session, response = _open_stream(offline_camera.id, viewer_user)
    assert inspect.isasyncgen(response.body_iterator)

    async def first_chunk():
        return await response.body_iterator.__anext__()

    async def disconnect_after_a_moment():
        task = asyncio.ensure_future(first_chunk())
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(asyncio.wait_for(disconnect_after_a_moment(), timeout=5))
    session.close()


def test_a_frameless_stream_ends_once_the_viewer_disconnects(offline_camera, viewer_user):
    """Under ASGI 2.4 Starlette only notices a departed client when a send
    fails, and a camera with no frame never sends - so the stream has to
    ask. Before this, the loop (and uvicorn's graceful shutdown waiting
    on it) never ended."""
    request = FakeRequest()
    session, response = _open_stream(offline_camera.id, viewer_user, request)

    async def viewer_leaves_then_drain():
        async def leave():
            await asyncio.sleep(0.1)
            request.disconnected = True

        asyncio.ensure_future(leave())
        return [chunk async for chunk in response.body_iterator]

    assert _run(asyncio.wait_for(viewer_leaves_then_drain(), timeout=5)) == []
    session.close()


def test_through_the_whole_middleware_stack_a_disconnect_ends_the_stream(offline_camera, viewer_headers):
    """End to end at the ASGI level, so any middleware between the route
    and the server is in the path. A BaseHTTPMiddleware in the stack
    swallowed the disconnect and this never returned."""
    from app.main import app

    token = viewer_headers["Authorization"].encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": f"/api/cameras/{offline_camera.id}/stream",
        "raw_path": f"/api/cameras/{offline_camera.id}/stream".encode(), "query_string": b"",
        "root_path": "", "headers": [(b"host", b"test"), (b"authorization", token)],
        "client": ("127.0.0.1", 1), "server": ("test", 80),
    }
    sent = []

    async def run():
        messages = [{"type": "http.request", "body": b"", "more_body": False}]
        left = asyncio.Event()

        async def receive():
            if messages:
                return messages.pop(0)
            await left.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        async def viewer_leaves():
            await asyncio.sleep(0.3)
            left.set()

        asyncio.ensure_future(viewer_leaves())
        await app(scope, receive, send)

    _run(asyncio.wait_for(run(), timeout=10))
    assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 200
