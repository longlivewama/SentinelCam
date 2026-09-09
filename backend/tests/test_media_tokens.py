"""
The media token: what it can do, and - mostly - what it cannot.

A browser cannot put a header on `<video src>`, so something has to travel
in the URL, and a URL is written down everywhere. The design answer is to
make the thing in the URL nearly worthless: scoped to one clip, expiring
in minutes, carrying no role and no email, and useless anywhere except the
one endpoint it names.

"Nearly worthless" is only true if every one of those limits actually
holds, so each gets a test here. The interesting cases are the negative
ones - a token that works where it should not is the whole risk.
"""
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jose import jwt

from app.config import settings
from app.core.security import create_access_token, create_media_token, media_resource_id
from app.models.camera import Camera
from app.models.recording import Recording
from app.models.video_upload import VideoUpload

CONTENT = bytes(range(256)) * 8  # 2048 deterministic bytes, enough to range over


@pytest.fixture()
def upload(db, viewer_user):
    """An upload owned by viewer_user, with its file inside UPLOADS_DIR so
    the route's path-containment check passes."""
    source_dir = Path(settings.UPLOADS_DIR) / f"mediatok-{viewer_user.id}"
    source_dir.mkdir(parents=True, exist_ok=True)
    source = source_dir / "source.mp4"
    source.write_bytes(CONTENT)
    row = VideoUpload(
        user_id=viewer_user.id, original_filename="source.mp4", stored_path=str(source),
        status="completed", progress_percent=100, persons_detected=1, fall_events_count=1,
        file_size_bytes=source.stat().st_size,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    yield row
    shutil.rmtree(source_dir, ignore_errors=True)


@pytest.fixture()
def clip(db, tmp_path, upload):
    """A fall clip belonging to `upload`, and therefore to viewer_user."""
    path = tmp_path / "clip.webm"
    path.write_bytes(CONTENT)
    row = Recording(
        camera_id=None, video_upload_id=upload.id, filename="clip.webm", file_path=str(path),
        duration_seconds=6.0, trigger_action="fall", file_size_bytes=len(CONTENT),
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@pytest.fixture()
def other_clip(db, tmp_path):
    """A second clip, from no upload at all - visible to everyone, so it
    isolates "wrong resource" from "not allowed to see it"."""
    path = tmp_path / "other.webm"
    path.write_bytes(CONTENT)
    row = Recording(
        camera_id=None, video_upload_id=None, filename="other.webm", file_path=str(path),
        duration_seconds=6.0, trigger_action="fall", file_size_bytes=len(CONTENT),
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@pytest.fixture()
def camera(db):
    row = Camera(name="Lobby", url="rtsp://example.invalid/stream", is_active=True)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def mint(client, headers, path: str) -> dict:
    response = client.post(path, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _signed(payload: dict, key: str | None = None) -> str:
    return jwt.encode(payload, key or settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


# --- minting --------------------------------------------------------------

def test_minting_requires_a_real_session_and_returns_a_scoped_token(client, viewer_headers, clip):
    body = mint(client, viewer_headers, f"/api/recordings/{clip.id}/media-token")

    assert body["resource"] == f"recording:{clip.id}"
    assert body["expires_in"] == settings.MEDIA_TOKEN_EXPIRE_SECONDS
    assert body["token"]


def test_minting_is_refused_without_authentication(client, clip):
    assert client.post(f"/api/recordings/{clip.id}/media-token").status_code == 401


def test_a_media_token_cannot_be_used_to_mint_another_one(client, viewer_headers, clip):
    """Otherwise the short lifetime would be decorative: a leaked token
    could renew itself indefinitely, and (worse) mint tokens for other
    resources."""
    token = mint(client, viewer_headers, f"/api/recordings/{clip.id}/media-token")["token"]

    resp = client.post(
        f"/api/recordings/{clip.id}/media-token", headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 401


def test_minting_for_someone_elses_upload_is_a_404(client, make_user_factory, auth_headers_for, clip):
    """The mint endpoint runs the same ownership check the stream endpoint
    runs, so it cannot become a way to obtain access the streamer would
    have refused."""
    stranger = make_user_factory("stranger@example.com")

    resp = client.post(f"/api/recordings/{clip.id}/media-token", headers=auth_headers_for(stranger))

    assert resp.status_code == 404


# --- the happy path, including Range --------------------------------------

def test_valid_media_access_streams_the_clip(client, viewer_headers, clip):
    token = mint(client, viewer_headers, f"/api/recordings/{clip.id}/media-token")["token"]

    response = client.get(f"/api/recordings/{clip.id}/video?token={token}")

    assert response.status_code == 200
    assert response.content == CONTENT


def test_range_requests_still_return_206_with_a_media_token(client, viewer_headers, clip):
    """Seeking in the player is a Range request on the same URL, so the
    media token has to survive one - a 206 path that only worked under a
    bearer header would break scrubbing for every user."""
    token = mint(client, viewer_headers, f"/api/recordings/{clip.id}/media-token")["token"]

    response = client.get(
        f"/api/recordings/{clip.id}/video?token={token}", headers={"Range": "bytes=100-199"},
    )

    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 100-199/{len(CONTENT)}"
    assert response.content == CONTENT[100:200]


def test_the_source_video_streams_and_ranges_with_its_own_token(client, viewer_headers, upload):
    token = mint(client, viewer_headers, f"/api/video-uploads/{upload.id}/media-token")["token"]

    whole = client.get(f"/api/video-uploads/{upload.id}/video?token={token}")
    partial = client.get(
        f"/api/video-uploads/{upload.id}/video?token={token}", headers={"Range": "bytes=-64"},
    )

    assert whole.status_code == 200
    assert partial.status_code == 206
    assert partial.content == CONTENT[-64:]


# --- expiry ---------------------------------------------------------------

def test_an_expired_media_token_is_refused(client, viewer_user, clip):
    """Minted in the past rather than by waiting: the production lifetime
    is minutes, and a test that sleeps for it would be untestable."""
    past = datetime.now(timezone.utc) - timedelta(seconds=10)
    token = _signed({
        "sub": str(viewer_user.id), "user_id": viewer_user.id, "typ": "media",
        "res": media_resource_id("recording", clip.id),
        "iat": past - timedelta(seconds=300), "exp": past,
    })

    assert client.get(f"/api/recordings/{clip.id}/video?token={token}").status_code == 401


def test_a_token_minted_now_is_not_already_expired(client, viewer_user, clip):
    """Guards the arithmetic in create_media_token: an off-by-one on the
    sign would make every token dead on arrival, which the expiry test
    above would happily pass."""
    token, expires_in = create_media_token(viewer_user.id, media_resource_id("recording", clip.id))

    assert expires_in > 0
    assert client.get(f"/api/recordings/{clip.id}/video?token={token}").status_code == 200


# --- forgery and malformed input ------------------------------------------

def test_a_forged_media_token_is_refused(client, viewer_user, clip):
    """Same claims, wrong signing key - what an attacker who has read a
    token and knows the format, but not the secret, would produce."""
    forged = _signed({
        "sub": str(viewer_user.id), "user_id": viewer_user.id, "typ": "media",
        "res": media_resource_id("recording", clip.id),
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    }, key="not-the-real-signing-key")

    assert client.get(f"/api/recordings/{clip.id}/video?token={forged}").status_code == 401


@pytest.mark.parametrize("token", ["", "not-a-jwt", "a.b.c", "eyJhbGciOiJub25lIn0..", "null"])
def test_invalid_media_tokens_are_refused(client, clip, token):
    assert client.get(f"/api/recordings/{clip.id}/video?token={token}").status_code == 401


def test_a_media_token_without_a_resource_claim_is_refused(client, viewer_user, clip):
    """A correctly-signed media token whose `res` is missing must not fall
    through to "any resource"."""
    token = _signed({
        "sub": str(viewer_user.id), "user_id": viewer_user.id, "typ": "media",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
    })

    assert client.get(f"/api/recordings/{clip.id}/video?token={token}").status_code == 401


# --- scope: wrong resource ------------------------------------------------

def test_a_token_for_one_clip_cannot_read_another(client, viewer_headers, clip, other_clip):
    """`other_clip` is readable by this user with a proper token, so this
    isolates scope from authorization: the refusal is about the token
    naming the wrong resource, not about who is asking."""
    token = mint(client, viewer_headers, f"/api/recordings/{clip.id}/media-token")["token"]

    assert client.get(f"/api/recordings/{other_clip.id}/video?token={token}").status_code == 401
    assert client.get(f"/api/recordings/{other_clip.id}/download?token={token}").status_code == 401


def test_a_recording_token_cannot_read_an_upload(client, viewer_headers, clip, upload):
    """Both belong to the same user, so only the kind differs."""
    token = mint(client, viewer_headers, f"/api/recordings/{clip.id}/media-token")["token"]

    assert client.get(f"/api/video-uploads/{upload.id}/video?token={token}").status_code == 401


def test_an_upload_token_cannot_read_a_recording(client, viewer_headers, clip, upload):
    token = mint(client, viewer_headers, f"/api/video-uploads/{upload.id}/media-token")["token"]

    assert client.get(f"/api/recordings/{clip.id}/video?token={token}").status_code == 401


def test_a_camera_token_cannot_read_a_recording(client, viewer_headers, camera, clip):
    token = mint(client, viewer_headers, f"/api/cameras/{camera.id}/media-token")["token"]

    assert client.get(f"/api/recordings/{clip.id}/video?token={token}").status_code == 401


# --- scope: wrong user ----------------------------------------------------

def test_a_media_token_for_another_users_clip_is_refused(
    client, make_user_factory, auth_headers_for, viewer_user, clip,
):
    """Even a perfectly-formed token, signed by this server, naming this
    clip: if it was issued to a user who cannot see the clip, the
    ownership check still runs on every request."""
    stranger = make_user_factory("stranger2@example.com")
    token, _ = create_media_token(stranger.id, media_resource_id("recording", clip.id))

    response = client.get(f"/api/recordings/{clip.id}/video?token={token}")

    assert response.status_code == 404, "the clip is not the stranger's to see"
    assert client.get(
        f"/api/recordings/{clip.id}/video", headers=auth_headers_for(viewer_user),
    ).status_code == 200, "and its real owner is unaffected"


def test_a_media_token_for_a_deactivated_user_stops_working(client, db, viewer_user, clip):
    """The token is still unexpired and still correctly scoped; the
    account behind it is not. Authorization is re-resolved per request
    rather than baked into the token, which is what makes this hold."""
    token, _ = create_media_token(viewer_user.id, media_resource_id("recording", clip.id))
    assert client.get(f"/api/recordings/{clip.id}/video?token={token}").status_code == 200

    viewer_user.is_active = False
    db.commit()

    assert client.get(f"/api/recordings/{clip.id}/video?token={token}").status_code == 401


# --- a media token is not an API credential -------------------------------

@pytest.mark.parametrize("path", [
    "/api/recordings",
    "/api/alerts",
    "/api/video-uploads",
    "/api/cameras",
    "/api/auth/me",
    "/api/analytics/summary",
    "/api/admin/users",
    "/api/system/status",
])
def test_a_media_token_cannot_reach_the_json_api_as_a_bearer(client, viewer_user, clip, path):
    """The point of the split. A media token recovered from a URL must not
    become a session, so it is refused as a bearer credential everywhere -
    including on the endpoints that never look at `?token=` at all."""
    token, _ = create_media_token(viewer_user.id, media_resource_id("recording", clip.id))

    response = client.get(path, headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401, f"{path} accepted a media token as a session"


def test_a_media_token_cannot_mutate_state_as_a_bearer(client, viewer_user, clip):
    token, _ = create_media_token(viewer_user.id, media_resource_id("recording", clip.id))

    response = client.delete(
        f"/api/recordings/{clip.id}", headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401


def test_a_media_token_cannot_open_the_realtime_socket(client, viewer_user, clip):
    """The WebSocket takes its credential in the query string too, and it
    delivers every alert the user can see - a clip token must not be a way
    in."""
    from starlette.websockets import WebSocketDisconnect

    token, _ = create_media_token(viewer_user.id, media_resource_id("recording", clip.id))

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(f"/api/ws/events?token={token}") as ws:
            ws.receive_text()

    assert excinfo.value.code == 4401


# --- the session token is unchanged everywhere else -----------------------

def test_a_session_token_still_authenticates_the_whole_api(client, viewer_user):
    """The other half of "do not weaken authentication": adding the `typ`
    claim must not have disturbed ordinary bearer auth."""
    token = create_access_token(viewer_user.id, viewer_user.email)
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get("/api/auth/me", headers=headers).status_code == 200
    assert client.get("/api/recordings", headers=headers).status_code == 200


def test_a_session_token_still_opens_the_realtime_socket(client, viewer_user):
    token = create_access_token(viewer_user.id, viewer_user.email)

    with client.websocket_connect(f"/api/ws/events?token={token}") as ws:
        ws.send_text("ping")  # the server ignores client messages; this just proves it is open
