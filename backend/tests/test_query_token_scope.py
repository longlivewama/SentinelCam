"""
What a `?token=` credential is, and where it is accepted.

Browsers cannot attach an Authorization header to `<video src>`,
`<img src>` or a plain download link, so those endpoints have to take a
credential in the URL. A URL is the one place a credential is guaranteed
to be written down - the access log of every server and proxy on the
path, the browser's history, any `Referer` sent onward - so two separate
narrowings apply, and these tests pin both.

WHERE: only the media endpoints look at `?token=` at all. Every JSON
endpoint refuses it, so a credential recovered from a URL cannot be
replayed against the rest of the API. (Pinned since the previous fix.)

WHAT: the media endpoints no longer accept a session JWT in the query
string either. They accept only a media token - minted per clip, expiring
in minutes, carrying no role, and refused everywhere except the one
resource it names. This is the part that changed: `<video src>` used to
publish a full-privilege, day-long credential into every URL.

Media-token behaviour itself (expiry, forgery, cross-user, cross-resource)
is covered in test_media_tokens.py.
"""
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import settings

from app.core.security import create_access_token
from app.models.recording import Recording
from app.models.video_upload import VideoUpload


@pytest.fixture()
def operator_token(operator_user):
    """A full session JWT - the thing that used to appear in `<video src>`
    URLs and must no longer be accepted from one."""
    return create_access_token(operator_user.id, operator_user.email)


@pytest.fixture()
def uploaded_video(db, operator_user):
    """Written inside UPLOADS_DIR on purpose: the route refuses a
    stored_path that resolves outside it, which is a path-traversal guard
    worth not working around."""
    source_dir = Path(settings.UPLOADS_DIR) / f"qtoken-{operator_user.id}"
    source_dir.mkdir(parents=True, exist_ok=True)
    source = source_dir / "source.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"payload" * 128)
    row = VideoUpload(
        user_id=operator_user.id, original_filename="source.mp4", stored_path=str(source),
        status="completed", progress_percent=100, persons_detected=1, fall_events_count=1,
        file_size_bytes=source.stat().st_size,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    yield row
    shutil.rmtree(source_dir, ignore_errors=True)


@pytest.fixture()
def clip(db, tmp_path):
    path = tmp_path / "clip.webm"
    path.write_bytes(b"\x1a\x45\xdf\xa3" + b"payload" * 128)
    row = Recording(
        camera_id=None, video_upload_id=None, filename="clip.webm", file_path=str(path),
        duration_seconds=2.0, trigger_action="fall", file_size_bytes=path.stat().st_size,
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _media_token(client, headers, path: str) -> str:
    response = client.post(path, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()["token"]


# --- the endpoints that genuinely cannot send a header --------------------

def test_a_clip_streams_with_a_media_token(client, operator_headers, clip):
    """<video src> has no other way to authenticate."""
    token = _media_token(client, operator_headers, f"/api/recordings/{clip.id}/media-token")

    assert client.get(f"/api/recordings/{clip.id}/video?token={token}").status_code == 200


def test_a_clip_downloads_with_a_media_token(client, operator_headers, clip):
    """<a href download> likewise - and the same token covers both, since
    they serve identical bytes and differ only in Content-Disposition."""
    token = _media_token(client, operator_headers, f"/api/recordings/{clip.id}/media-token")

    assert client.get(f"/api/recordings/{clip.id}/download?token={token}").status_code == 200


def test_the_uploaded_source_video_streams_with_a_media_token(client, operator_headers, uploaded_video):
    token = _media_token(client, operator_headers, f"/api/video-uploads/{uploaded_video.id}/media-token")

    assert client.get(f"/api/video-uploads/{uploaded_video.id}/video?token={token}").status_code == 200


# --- a SESSION token is no longer a media credential ----------------------

@pytest.mark.parametrize("path_template", [
    "/api/recordings/{clip}/video",
    "/api/recordings/{clip}/download",
])
def test_media_endpoints_refuse_a_session_jwt_in_the_query_string(
    client, operator_token, clip, path_template,
):
    """The change this file exists for. The session JWT is valid, and its
    owner may absolutely read this clip - but not by putting a
    full-privilege, day-long credential in a URL."""
    path = path_template.format(clip=clip.id)

    assert client.get(f"{path}?token={operator_token}").status_code == 401


def test_the_uploaded_source_video_refuses_a_session_jwt(client, operator_token, uploaded_video):
    resp = client.get(f"/api/video-uploads/{uploaded_video.id}/video?token={operator_token}")

    assert resp.status_code == 401


# --- and the rest of the API, which can --------------------------------

@pytest.mark.parametrize(
    "path",
    [
        "/api/recordings",
        "/api/alerts",
        "/api/video-uploads",
        "/api/cameras",
        "/api/auth/me",
        "/api/analytics/summary",
    ],
)
def test_json_endpoints_refuse_a_query_token(client, operator_token, path):
    """These are all reached by axios, which sends the header. Accepting
    a URL credential here would let a token leaked from a video link be
    replayed against the whole API."""
    resp = client.get(f"{path}?token={operator_token}")

    assert resp.status_code == 401, f"{path} accepted a query-param credential"


def test_reading_one_recording_refuses_a_query_token(client, operator_token, clip):
    """The metadata endpoint sits right next to the streaming one and is
    the easy one to widen by accident."""
    assert client.get(f"/api/recordings/{clip.id}?token={operator_token}").status_code == 401


def test_a_mutating_endpoint_refuses_a_query_token(client, operator_headers, operator_token, clip):
    """The worst case for a URL credential: state change from a link."""
    resp = client.delete(f"/api/recordings/{clip.id}?token={operator_token}")
    assert resp.status_code == 401

    media_token = _media_token(client, operator_headers, f"/api/recordings/{clip.id}/media-token")
    assert client.get(f"/api/recordings/{clip.id}/video?token={media_token}").status_code == 200, \
        "and the clip must still be there"


# --- the header keeps working everywhere ---------------------------------

@pytest.mark.parametrize("path", ["/api/recordings", "/api/alerts", "/api/video-uploads", "/api/auth/me"])
def test_the_authorization_header_still_works(client, operator_headers, path):
    assert client.get(path, headers=operator_headers).status_code == 200


def test_the_media_endpoints_still_accept_a_bearer_header(client, operator_headers, clip):
    """Unchanged for API clients and the rest of this suite: a header is
    not written into logs or history, so there was never a reason to stop
    accepting one here."""
    assert client.get(f"/api/recordings/{clip.id}/video", headers=operator_headers).status_code == 200


def test_the_header_is_preferred_when_both_are_present(client, operator_headers, operator_token, clip):
    """A media endpoint given a valid header and a junk query token must
    honour the header rather than failing on the query string."""
    resp = client.get(f"/api/recordings/{clip.id}/video?token=not-a-jwt", headers=operator_headers)

    assert resp.status_code == 200


def test_a_garbage_query_token_is_still_rejected_on_media(client, clip):
    assert client.get(f"/api/recordings/{clip.id}/video?token=not-a-jwt").status_code == 401
