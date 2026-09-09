"""
Where a `?token=` credential is accepted, and where it must not be.

Browsers cannot attach an Authorization header to `<video src>`,
`<img src>` or a plain download link, so those endpoints have to take the
JWT in the URL. That is a real constraint, but it was being satisfied by
letting EVERY endpoint accept `?token=`, which is a much bigger promise
than the constraint needs.

A token in a URL ends up in browser history, in `Referer` headers, and in
every access log and proxy on the path - this repository's own QA capture
is an example, having recorded full-privilege JWTs from video URLs. Broad
acceptance means such a leaked token can then be replayed against the
whole API by pasting a URL. Narrow acceptance means it only reaches the
four endpoints that already stream the media it was leaked from.

These tests pin both halves: the media endpoints must keep working, and
the rest of the API must refuse.
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
    """The raw JWT, as it would appear in a `<video src>` URL."""
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


# --- the endpoints that genuinely cannot send a header --------------------

def test_a_clip_streams_with_a_query_token(client, operator_token, clip):
    """<video src> has no other way to authenticate."""
    assert client.get(f"/api/recordings/{clip.id}/video?token={operator_token}").status_code == 200


def test_a_clip_downloads_with_a_query_token(client, operator_token, clip):
    """<a href download> likewise."""
    assert client.get(f"/api/recordings/{clip.id}/download?token={operator_token}").status_code == 200


def test_the_uploaded_source_video_streams_with_a_query_token(client, operator_token, uploaded_video):
    resp = client.get(f"/api/video-uploads/{uploaded_video.id}/video?token={operator_token}")
    assert resp.status_code == 200


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


def test_a_mutating_endpoint_refuses_a_query_token(client, operator_token, clip):
    """The worst case for a URL credential: state change from a link."""
    resp = client.delete(f"/api/recordings/{clip.id}?token={operator_token}")

    assert resp.status_code == 401
    assert client.get(f"/api/recordings/{clip.id}/video?token={operator_token}").status_code == 200, \
        "and the clip must still be there"


# --- the header keeps working everywhere ---------------------------------

@pytest.mark.parametrize("path", ["/api/recordings", "/api/alerts", "/api/video-uploads", "/api/auth/me"])
def test_the_authorization_header_still_works(client, operator_headers, path):
    assert client.get(path, headers=operator_headers).status_code == 200


def test_the_header_is_preferred_when_both_are_present(client, operator_headers, operator_token, clip):
    """A media endpoint given both must not be confused by the pair."""
    resp = client.get(f"/api/recordings/{clip.id}/video?token={operator_token}", headers=operator_headers)
    assert resp.status_code == 200


def test_a_garbage_query_token_is_still_rejected_on_media(client, clip):
    assert client.get(f"/api/recordings/{clip.id}/video?token=not-a-jwt").status_code == 401
