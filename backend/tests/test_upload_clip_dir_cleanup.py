"""
Deleting an upload must not leave its clip directory behind.

`_write_upload_clip` creates `recordings/uploads/{id}/` for every upload
that produces a fall. Both delete paths - the endpoint, and the worker
finishing a deletion requested mid-analysis - removed the clips inside
but not the directory, so a deployment accumulated one empty directory
per deleted upload forever. Found by QA: the database held 3 uploads
while the storage tree held 13 clip directories.
"""
import shutil
from pathlib import Path

import pytest

from app.config import settings
from app.services.cascade_delete import remove_upload_clip_dir


def _clip_dir(upload_id) -> Path:
    return Path(settings.RECORDINGS_DIR) / "uploads" / str(upload_id)


@pytest.fixture()
def clip_dir():
    created = []

    def make(upload_id, files=()):
        path = _clip_dir(upload_id)
        path.mkdir(parents=True, exist_ok=True)
        for name in files:
            (path / name).write_bytes(b"clip")
        created.append(path)
        return path

    yield make
    for path in created:
        for child in path.glob("*"):
            child.unlink(missing_ok=True)
        try:
            path.rmdir()
        except OSError:
            pass


@pytest.fixture()
def upload_source(operator_user):
    """A source file inside UPLOADS_DIR (the route refuses paths outside
    it), cleaned up afterwards so the suite leaves no litter in the real
    storage tree."""
    source_dir = Path(settings.UPLOADS_DIR) / f"cleanup-test-{operator_user.id}"
    source_dir.mkdir(parents=True, exist_ok=True)

    def make(name):
        path = source_dir / name
        path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"x" * 64)
        return path

    yield make
    shutil.rmtree(source_dir, ignore_errors=True)


def test_an_emptied_clip_directory_is_removed(clip_dir):
    path = clip_dir(999001)

    remove_upload_clip_dir(999001)

    assert not path.exists()


def test_an_upload_that_never_produced_a_clip_is_not_an_error():
    """Most uploads detect no falls, so no directory was ever created."""
    remove_upload_clip_dir(999002)  # must not raise


def test_a_directory_that_still_holds_something_is_left_alone(clip_dir):
    """rmdir, not a recursive delete: if a file is unexpectedly still
    there, it survives to be noticed rather than being destroyed by a
    cleanup step."""
    path = clip_dir(999003, files=["unexpected.webm"])

    remove_upload_clip_dir(999003)

    assert path.exists()
    assert (path / "unexpected.webm").exists()


def test_only_that_upload_s_directory_is_touched(clip_dir):
    keep = clip_dir(999004)
    drop = clip_dir(999005)

    remove_upload_clip_dir(999005)

    assert keep.exists() and not drop.exists()


def test_the_shared_uploads_parent_is_never_removed(clip_dir):
    """The bug's worst possible over-correction: taking the parent with
    it and breaking every future upload."""
    clip_dir(999006)
    parent = Path(settings.RECORDINGS_DIR) / "uploads"

    remove_upload_clip_dir(999006)

    assert parent.is_dir()


def test_deleting_an_upload_through_the_api_removes_its_clip_directory(
    client, operator_headers, operator_user, db, clip_dir, upload_source,
):
    """End to end through the endpoint, which is where it actually
    regressed."""
    from app.models.video_upload import VideoUpload

    source = upload_source("source.mp4")

    upload = VideoUpload(
        user_id=operator_user.id, original_filename="source.mp4", stored_path=str(source),
        status="completed", progress_percent=100, persons_detected=1, fall_events_count=0,
        file_size_bytes=source.stat().st_size,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    path = clip_dir(upload.id)
    assert path.exists()

    assert client.delete(f"/api/video-uploads/{upload.id}", headers=operator_headers).status_code == 204
    assert not path.exists(), "the clip directory outlived its upload"
