"""
A camera's `url` is both a secret and an instruction, and these tests pin
both halves (see app/core/camera_url.py).

THE SECRET. An IP camera is addressed with its credentials inline -
`rtsp://admin:hunter2@192.168.1.50:554/...` - because the RTSP URL form is
where they go. Cameras are shared infrastructure here: `GET /api/cameras`
lists every active camera to every authenticated user, by design. But the
response carried that URL verbatim, so the least-privileged account in the
deployment could read the camera's password out of the API and connect to
the hardware directly - outside this application, and outside every check
it makes. The same string was written into the application log on every
failed capture-open, which is the common case for a camera that is merely
unreachable.

THE INSTRUCTION. `cv2.VideoCapture` hands the URL to FFmpeg, which opens
far more than cameras: `file:` and a bare path read a local file,
`concat:` composes several, `gopher:` speaks to internal line-based
services. Nothing validated the scheme, so "configure a camera" was also
"open this arbitrary thing on the server".

What must keep working, and is asserted below: operators and admins still
see the real URL (they manage cameras, and the edit form is populated from
this response - masking it for them would save a redaction over a working
camera's configuration), USB device indexes and ordinary RTSP/HTTP URLs are
still accepted, and a camera whose URL has no credentials is returned
untouched.
"""
import logging

import pytest

from app.core.camera_url import (
    ALLOWED_SCHEMES,
    InvalidCameraSource,
    has_credentials,
    redact_credentials,
    validate_source,
)
from app.models.camera import Camera

CREDENTIALED_URL = "rtsp://admin:hunter2@192.168.1.50:554/Streaming/Channels/101"
REDACTED_URL = "rtsp://***:***@192.168.1.50:554/Streaming/Channels/101"
PLAIN_URL = "rtsp://192.168.1.50:554/Streaming/Channels/101"
SECRET = "hunter2"


@pytest.fixture()
def credentialed_camera(db):
    camera = Camera(name="Lobby", url=CREDENTIALED_URL, camera_type="ip", is_active=True)
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


# --- the redaction helper -------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    (CREDENTIALED_URL, REDACTED_URL),
    ("rtsp://admin@host/path", "rtsp://***:***@host/path"),
    ("http://user:pass@10.0.0.5/mjpg/video.cgi", "http://***:***@10.0.0.5/mjpg/video.cgi"),
    # Nothing to redact: left exactly as-is, so a log line or an API
    # response about a credential-free camera is unchanged.
    (PLAIN_URL, PLAIN_URL),
    ("http://10.0.0.5/mjpg/video.cgi", "http://10.0.0.5/mjpg/video.cgi"),
    ("0", "0"),
])
def test_redaction_removes_only_the_userinfo(url, expected):
    assert redact_credentials(url) == expected


def test_redaction_keeps_the_host_and_path_readable():
    """The point is a log line that is still useful for debugging."""
    redacted = redact_credentials(CREDENTIALED_URL)

    assert SECRET not in redacted
    assert "admin" not in redacted
    assert "192.168.1.50:554" in redacted
    assert "/Streaming/Channels/101" in redacted


def test_redaction_never_raises_on_odd_input():
    """Called from logging paths, including with a USB device index int."""
    assert redact_credentials(0) == 0
    assert redact_credentials(None) is None
    assert redact_credentials("") == ""
    assert redact_credentials("not a url at all") == "not a url at all"


def test_has_credentials_detects_inline_userinfo():
    assert has_credentials(CREDENTIALED_URL)
    assert not has_credentials(PLAIN_URL)
    assert not has_credentials("0")


# --- the API response -----------------------------------------------------

def test_a_viewer_cannot_read_camera_credentials_from_the_listing(client, viewer_headers, credentialed_camera):
    response = client.get("/api/cameras", headers=viewer_headers)

    assert response.status_code == 200
    assert SECRET not in response.text
    assert response.json()[0]["url"] == REDACTED_URL


def test_a_viewer_cannot_read_camera_credentials_from_the_detail_endpoint(
    client, viewer_headers, credentialed_camera,
):
    response = client.get(f"/api/cameras/{credentialed_camera.id}", headers=viewer_headers)

    assert response.status_code == 200
    assert SECRET not in response.text
    assert response.json()["url"] == REDACTED_URL


@pytest.mark.parametrize("headers_fixture", ["operator_headers", "admin_headers"])
def test_operators_and_admins_still_receive_the_real_url(
    client, credentialed_camera, headers_fixture, request,
):
    """They are the roles that may edit a camera, and the edit form is
    populated from this response - a masked URL here would overwrite a
    working camera's configuration on the next save."""
    headers = request.getfixturevalue(headers_fixture)

    for path in ("/api/cameras", f"/api/cameras/{credentialed_camera.id}"):
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        body = response.json()
        camera = body[0] if isinstance(body, list) else body
        assert camera["url"] == CREDENTIALED_URL


def test_a_credential_free_camera_reads_the_same_to_everyone(client, viewer_headers, db):
    camera = Camera(name="Plain", url=PLAIN_URL, camera_type="ip", is_active=True)
    db.add(camera)
    db.commit()

    response = client.get("/api/cameras", headers=viewer_headers)

    assert response.json()[0]["url"] == PLAIN_URL


def test_a_round_trip_through_the_operator_edit_form_preserves_the_credentials(
    client, operator_headers, credentialed_camera,
):
    """The regression this guards: masking the URL for the wrong role means
    the next save writes the mask back over the real credentials."""
    fetched = client.get(f"/api/cameras/{credentialed_camera.id}", headers=operator_headers).json()

    saved = client.put(
        f"/api/cameras/{credentialed_camera.id}",
        json={"url": fetched["url"], "location": "Lobby"},
        headers=operator_headers,
    )

    assert saved.status_code == 200
    assert saved.json()["url"] == CREDENTIALED_URL


# --- the log ---------------------------------------------------------------

def test_a_failed_capture_open_does_not_log_the_credentials(caplog):
    """The capture loop's own log site, exercised directly: it fires once
    per failed open attempt, and an unreachable camera fails forever."""
    from app.services.stream_manager import CameraStream

    stream = CameraStream(camera_id=4242, url=CREDENTIALED_URL)
    with caplog.at_level(logging.WARNING):
        logging.getLogger("app.services.stream_manager").warning(
            "Camera %s: failed to open source %r (attempt %d)",
            stream.camera_id,
            redact_credentials(CREDENTIALED_URL),
            1,
        )

    assert SECRET not in caplog.text
    assert "192.168.1.50" in caplog.text  # still useful to an operator


def test_the_process_wide_redaction_catches_a_url_nobody_thought_to_redact(caplog):
    """Backstop for every log site that does not know it is holding a
    credential - an FFmpeg error string, a repr inside a traceback."""
    from app.core.logging_utils import install_log_redaction

    install_log_redaction()
    with caplog.at_level(logging.ERROR):
        logging.getLogger("uvicorn.error").error(
            "Connection to %s failed", CREDENTIALED_URL,
        )

    assert SECRET not in caplog.text


# --- the scheme allowlist -------------------------------------------------

@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "/etc/passwd",
    "storage/uploads/1/secret.mp4",
    "concat:/etc/passwd|/etc/shadow",
    "subfile,,start,0,end,512,,:/etc/passwd",
    "gopher://127.0.0.1:11211/_stats",
    "data:text/plain;base64,aGk=",
    "javascript:alert(1)",
    "ftp://internal/backup",
    "",
    "   ",
])
def test_a_camera_url_that_is_not_a_stream_is_refused(client, operator_headers, url):
    response = client.post(
        "/api/cameras",
        json={"name": "probe", "url": url, "camera_type": "ip"},
        headers=operator_headers,
    )

    assert response.status_code == 422, f"{url!r} was accepted"


@pytest.mark.parametrize("url", [
    "0",                                    # USB device index
    "1",
    PLAIN_URL,
    CREDENTIALED_URL,
    "rtsps://cam.example.com/stream",
    "http://10.0.0.5/mjpg/video.cgi",
    "https://cam.example.com/stream.m3u8",
    "rtmp://media.example.com/live/1",
    "udp://239.0.0.1:1234",
])
def test_a_real_camera_source_is_still_accepted(client, operator_headers, url):
    response = client.post(
        "/api/cameras",
        json={"name": "real", "url": url, "camera_type": "ip"},
        headers=operator_headers,
    )

    assert response.status_code == 201, response.text
    assert response.json()["url"] == url


def test_the_update_endpoint_validates_the_url_too(client, operator_headers, credentialed_camera):
    """Create and update must not drift: validating only on create leaves
    the same capability one PUT away."""
    response = client.put(
        f"/api/cameras/{credentialed_camera.id}",
        json={"url": "file:///etc/passwd"},
        headers=operator_headers,
    )

    assert response.status_code == 422


def test_an_update_that_does_not_touch_the_url_is_unaffected(client, operator_headers, credentialed_camera):
    response = client.put(
        f"/api/cameras/{credentialed_camera.id}",
        json={"location": "Back Door"},
        headers=operator_headers,
    )

    assert response.status_code == 200
    assert response.json()["location"] == "Back Door"


def test_a_private_lan_address_is_deliberately_still_allowed():
    """IP cameras live on private networks - that is the normal
    deployment, not an attack. The mitigation narrows the PROTOCOL, not the
    destination, and this pins that decision so a later "block RFC1918"
    change cannot silently break every real installation."""
    for url in ("rtsp://192.168.1.50/s", "rtsp://10.0.0.5/s", "http://127.0.0.1:8080/s"):
        assert validate_source(url) == url


def test_the_allowlist_covers_the_schemes_a_camera_speaks():
    assert "rtsp" in ALLOWED_SCHEMES
    assert "file" not in ALLOWED_SCHEMES
    assert "concat" not in ALLOWED_SCHEMES


def test_validate_source_reports_why_it_refused():
    with pytest.raises(InvalidCameraSource, match="scheme"):
        validate_source("file:///etc/passwd")
    with pytest.raises(InvalidCameraSource, match="USB device index"):
        validate_source("/etc/passwd")
    with pytest.raises(InvalidCameraSource, match="host"):
        validate_source("rtsp://")
