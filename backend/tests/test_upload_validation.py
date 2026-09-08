"""
Upload-acceptance hardening: the route used to trust the filename
extension alone, so any file renamed to `.mp4` was stored and later
served back with `Content-Type: video/mp4`. It also had no per-user
storage quota, so one account could fill the disk one within-limit
upload at a time.
"""
import io

import pytest

from app.config import settings
from app.core.video_signature import looks_like_supported_video
from app.models.video_upload import VideoUpload

# Minimal but genuine container headers.
MP4_HEADER = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41"
MKV_HEADER = b"\x1a\x45\xdf\xa3\x01\x00\x00\x00\x00\x00\x00\x1f"
AVI_HEADER = b"RIFF\x24\x00\x00\x00AVI LIST"


@pytest.fixture(autouse=True)
def _isolate_upload_storage(tmp_path, monkeypatch):
    """Keep test uploads out of the real storage directory."""
    monkeypatch.setattr(settings, "UPLOADS_DIR", str(tmp_path / "uploads"))
    yield


@pytest.fixture(autouse=True)
def _no_background_analysis(monkeypatch):
    """The route kicks off a real analysis thread on success; these tests
    are about acceptance, not inference."""
    monkeypatch.setattr("app.api.routes.video_uploads.analyze_video_upload", lambda _id: None)
    yield


def _post(client, headers, filename, content):
    return client.post(
        "/api/video-uploads",
        headers=headers,
        files={"file": (filename, io.BytesIO(content), "video/mp4")},
    )


# --- signature sniffing ---------------------------------------------------

@pytest.mark.parametrize("header", [MP4_HEADER, MKV_HEADER, AVI_HEADER])
def test_real_container_headers_are_accepted(header):
    assert looks_like_supported_video(header + b"\x00" * 64)


@pytest.mark.parametrize("payload", [
    b"PK\x03\x04" + b"\x00" * 64,                       # zip / office doc / jar
    b"<!DOCTYPE html><html><body>hi</body></html>",     # html
    b"#!/bin/sh\nrm -rf /\n" + b"\x00" * 64,            # script
    b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64,          # linux executable
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,                # image, not a video
    b"short",                                           # too short to classify
])
def test_non_video_payloads_are_rejected(payload):
    assert not looks_like_supported_video(payload)


# --- endpoint behaviour ---------------------------------------------------

def test_upload_with_a_video_extension_but_non_video_bytes_is_rejected(client, viewer_headers, db):
    response = _post(client, viewer_headers, "totally-a-video.mp4", b"PK\x03\x04" + b"\x00" * 128)
    assert response.status_code == 400
    assert "supported video" in response.json()["detail"]
    # And nothing was persisted for it.
    assert db.query(VideoUpload).count() == 0


def test_upload_with_a_real_mp4_header_is_accepted(client, viewer_headers):
    response = _post(client, viewer_headers, "clip.mp4", MP4_HEADER + b"\x00" * 512)
    assert response.status_code == 201
    assert response.json()["original_filename"] == "clip.mp4"


def test_disallowed_extension_is_still_rejected_first(client, viewer_headers):
    response = _post(client, viewer_headers, "payload.exe", MP4_HEADER + b"\x00" * 512)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_empty_upload_is_rejected(client, viewer_headers):
    response = _post(client, viewer_headers, "empty.mp4", b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_traversal_filename_cannot_escape_the_user_directory(client, viewer_headers, db, tmp_path):
    response = _post(client, viewer_headers, "../../../../etc/passwd.mp4", MP4_HEADER + b"\x00" * 512)
    assert response.status_code == 201

    upload = db.query(VideoUpload).one()
    uploads_root = (tmp_path / "uploads").resolve()
    from pathlib import Path
    # Stored under the uploads root with a generated name, not the client's path.
    assert Path(upload.stored_path).resolve().is_relative_to(uploads_root)
    assert "passwd" not in Path(upload.stored_path).name
    # The original name is preserved for display only.
    assert upload.original_filename == "../../../../etc/passwd.mp4"


def test_per_user_storage_quota_is_enforced(client, viewer_headers, db, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_STORAGE_PER_USER_MB", 1)

    first = _post(client, viewer_headers, "one.mp4", MP4_HEADER + b"\x00" * (1024 * 1024))
    assert first.status_code == 201

    second = _post(client, viewer_headers, "two.mp4", MP4_HEADER + b"\x00" * 512)
    assert second.status_code == 413
    assert "quota" in second.json()["detail"].lower()


def test_quota_is_per_user_not_global(client, viewer_headers, make_user_factory, auth_headers_for, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_STORAGE_PER_USER_MB", 1)

    assert _post(client, viewer_headers, "one.mp4", MP4_HEADER + b"\x00" * (1024 * 1024)).status_code == 201

    other = make_user_factory("quota-neighbour@example.com")
    assert _post(client, auth_headers_for(other), "mine.mp4", MP4_HEADER + b"\x00" * 512).status_code == 201
