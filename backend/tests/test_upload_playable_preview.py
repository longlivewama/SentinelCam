"""
An uploaded video whose codec no browser can decode still gets a playable
results player.

MPEG-4 Part 2 in an .mp4 is a perfectly ordinary upload - OpenCV, many
cameras and public datasets write it - and analysis handles it fine. But
Chrome/Firefox/Edge cannot decode it, so the results player beside the
pose/tracking overlay stayed blank. Analysis now writes a browser-playable
re-encode with the same frames and timing, and the stream endpoint serves
it in place of the original.
"""
import cv2
import numpy as np
import pytest

from app.config import settings
from app.core.video_codec import inspect_clip
from app.models.video_upload import STATUS_COMPLETED, VideoUpload
from app.services import video_analysis
from app.services.cascade_delete import find_upload_preview, stage_delete_video_upload

FPS = 10
TOTAL_FRAMES = 30
SIZE = (160, 120)


def _write_video(path, fourcc):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), FPS, SIZE)
    if not writer.isOpened():
        pytest.skip(f"this OpenCV build cannot write {fourcc}")
    for i in range(TOTAL_FRAMES):
        writer.write(np.full((SIZE[1], SIZE[0], 3), (i * 8) % 256, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture()
def storage(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "RECORDINGS_DIR", str(tmp_path / "recordings"))
    monkeypatch.setattr(settings, "SNAPSHOTS_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setattr(settings, "UPLOADS_DIR", str(tmp_path / "uploads"))
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(video_analysis.detection_engine, "ensure_models", lambda: None)
    monkeypatch.setattr(video_analysis.detection_engine, "extract_people", lambda frame: [])
    monkeypatch.setattr("app.services.detection.fall_pipeline.fall_object_detector.detect", lambda frame: [])
    monkeypatch.setattr("app.services.detection.fall_pipeline.resolve_mode", lambda: "model")
    return tmp_path


def _upload(db, user, path):
    row = VideoUpload(
        user_id=user.id,
        original_filename="clip.mp4",
        stored_path=str(path),
        file_size_bytes=path.stat().st_size,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def test_an_unplayable_upload_gets_a_frame_aligned_playable_preview(db, viewer_user, storage):
    source = _write_video(storage / "uploads" / "source.mp4", "mp4v")
    assert inspect_clip(str(source)).needs_transcode
    upload = _upload(db, viewer_user, source)

    assert video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    preview = find_upload_preview(upload.id)
    assert preview is not None
    assert inspect_clip(str(preview)).playable is True

    # Same frame count, rate and size: the tracking record's per-frame times
    # and pixel coordinates apply to the preview unchanged.
    cap = cv2.VideoCapture(str(preview))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(FPS, abs=0.01)
        assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == SIZE
        frames = 0
        while cap.read()[0]:
            frames += 1
    finally:
        cap.release()
    assert frames == TOTAL_FRAMES

    # Nothing half-written is left behind.
    assert [p.name for p in preview.parent.iterdir() if "partial" in p.name] == []
    db.refresh(upload)
    assert upload.status == STATUS_COMPLETED


def test_a_playable_upload_is_not_re_encoded(db, viewer_user, storage):
    source = _write_video(storage / "uploads" / "source.webm", "VP80")
    upload = _upload(db, viewer_user, source)

    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    assert find_upload_preview(upload.id) is None


def test_the_stream_endpoint_serves_the_preview_and_delete_removes_it(
    client, db, viewer_user, viewer_headers, storage,
):
    source = _write_video(storage / "uploads" / "source.mp4", "mp4v")
    upload = _upload(db, viewer_user, source)
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)
    preview = find_upload_preview(upload.id)
    assert preview is not None

    response = client.get(f"/api/video-uploads/{upload.id}/video", headers=viewer_headers)
    assert response.status_code == 200
    assert response.content == preview.read_bytes()

    paths = stage_delete_video_upload(db, upload)
    assert str(preview) in paths
