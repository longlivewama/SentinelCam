"""
End-to-end test of the offline upload analyser: a real video file is
decoded, the real _process() loop runs over it, and the Recording/Event
rows it writes are checked.

The two YOLO models are stubbed out (the pose model, and the trained fall
detector's `detect`) so this stays hermetic and fast - it must not
download weights or run inference in CI. Everything else is the
production code path: frame decoding, the sustained-duration gate, the
pre/post-event clip buffer, clip writing, database persistence, and the
realtime publish.

What it pins down, beyond "it runs":

  * `video_timestamp_seconds` is the position in the FOOTAGE. The
    analyser computed this per fall and then threw it away, leaving the
    UI to show the wall-clock `timestamp` - which is when the worker
    happened to reach that frame, not where in the video the fall is.
  * `detector` records which strategy fired, so a confidence score is
    interpretable.
  * Upload-derived realtime events are addressed to the uploader, not
    broadcast to every connected client.
"""
from collections import namedtuple

import cv2
import numpy as np
import pytest

from app.config import settings
from app.models.event import Event
from app.models.recording import Recording
from app.models.video_upload import VideoUpload
from app.services import video_analysis

Box = namedtuple("Box", ["bbox", "confidence"])

FPS = 10
TOTAL_FRAMES = 100          # 10 seconds
FALL_START_FRAME = 40       # 4.0s into the video
FALL_END_FRAME = 70         # 7.0s into the video


@pytest.fixture()
def source_video(tmp_path):
    """A real, decodable 10-second video. Content is irrelevant - the
    detector is stubbed - but the file has to be genuinely readable by
    OpenCV for the analyser to get past its "corrupt file" guard."""
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 120))
    assert writer.isOpened(), "OpenCV could not open a VideoWriter for the test fixture"
    for i in range(TOTAL_FRAMES):
        frame = np.full((120, 160, 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0
    return path


@pytest.fixture()
def upload(db, viewer_user, source_video):
    row = VideoUpload(
        user_id=viewer_user.id,
        original_filename="ward-3-morning.mp4",
        stored_path=str(source_video),
        file_size_bytes=source_video.stat().st_size,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@pytest.fixture()
def stubbed_models(monkeypatch, tmp_path):
    """Stubs the two models and redirects clip/snapshot output into the
    test's temp directory."""
    monkeypatch.setattr(settings, "RECORDINGS_DIR", str(tmp_path / "recordings"))
    monkeypatch.setattr(settings, "SNAPSHOTS_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setattr(settings, "VIDEO_ANALYSIS_FRAME_STRIDE", 1)
    monkeypatch.setattr(settings, "FALL_DETECTOR_MIN_SUSTAINED_SECONDS", 0.6)

    monkeypatch.setattr(video_analysis.detection_engine, "ensure_models", lambda: None)
    monkeypatch.setattr(video_analysis.detection_engine, "extract_people", lambda frame: [])

    # The trained detector "sees" a fallen person for a contiguous window
    # in the middle of the clip. Driven by a call counter rather than the
    # frame content, since the stub gets identical synthetic frames.
    state = {"frame": -1}

    def fake_detect(frame):
        state["frame"] += 1
        if FALL_START_FRAME <= state["frame"] < FALL_END_FRAME:
            return [Box(bbox=(20.0, 70.0, 120.0, 110.0), confidence=0.82)]
        return []

    monkeypatch.setattr(
        "app.services.detection.fall_pipeline.fall_object_detector.detect", fake_detect,
    )
    monkeypatch.setattr(
        "app.services.detection.fall_pipeline.resolve_mode", lambda: "model",
    )
    return state


def test_analysis_persists_a_fall_at_its_position_in_the_video(
    db, upload, stubbed_models, viewer_user,
):
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    events = db.query(Event).filter(Event.video_upload_id == upload.id).all()
    assert len(events) == 1
    event = events[0]

    # The fall starts 4.0s in and the gate needs 0.6s of it, so the event
    # lands at ~4.6s - NOT at 0s, and NOT at wall-clock "now".
    assert event.video_timestamp_seconds == pytest.approx(4.6, abs=0.3)
    assert 0 < event.video_timestamp_seconds < TOTAL_FRAMES / FPS
    assert event.event_type == "fall"
    assert event.detector == "model"
    assert event.confidence_score == pytest.approx(0.82, abs=0.01)
    assert event.camera_id is None


def test_the_in_video_timestamp_is_not_the_wall_clock_timestamp(db, upload, stubbed_models):
    """These are different quantities and the UI must be able to tell them
    apart; conflating them is what made the results screen misleading."""
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    event = db.query(Event).filter(Event.video_upload_id == upload.id).one()
    assert event.video_timestamp_seconds is not None
    assert event.timestamp is not None
    # Wall-clock is a real datetime "now"; the in-video offset is seconds
    # from the start of a 10s file.
    assert event.video_timestamp_seconds < 11
    assert event.timestamp.year >= 2024


def test_analysis_writes_a_playable_clip_and_links_it_to_the_event(db, upload, stubbed_models):
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    recording = db.query(Recording).filter(Recording.video_upload_id == upload.id).one()
    event = db.query(Event).filter(Event.video_upload_id == upload.id).one()

    assert event.recording_id == recording.id
    assert recording.trigger_action == "fall"
    assert recording.file_size_bytes > 0

    cap = cv2.VideoCapture(recording.file_path)
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok and frame is not None
    finally:
        cap.release()


def test_upload_row_is_completed_with_the_fall_count(db, upload, stubbed_models):
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    db.refresh(upload)
    assert upload.status == "completed"
    assert upload.progress_percent == 100
    assert upload.fall_events_count == 1
    assert upload.completed_at is not None
    assert upload.fps == pytest.approx(FPS, abs=0.5)


def test_a_video_with_no_detections_completes_with_zero_falls(db, upload, stubbed_models, monkeypatch):
    """A clean video must complete successfully reporting nothing found -
    not fail, and not invent an event."""
    monkeypatch.setattr(
        "app.services.detection.fall_pipeline.fall_object_detector.detect", lambda frame: [],
    )

    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    db.refresh(upload)
    assert upload.status == "completed"
    assert upload.fall_events_count == 0
    assert db.query(Event).filter(Event.video_upload_id == upload.id).count() == 0


def test_a_corrupt_video_fails_the_upload_with_a_message(db, viewer_user, tmp_path, stubbed_models):
    """The decoder must not take the worker down; the user gets a reason."""
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\xff" * 512)

    row = VideoUpload(
        user_id=viewer_user.id,
        original_filename="broken.mp4",
        stored_path=str(broken),
        file_size_bytes=broken.stat().st_size,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    video_analysis._run_guarded(row.id)

    db.refresh(row)
    assert row.status == "failed"
    assert row.error_message


def test_upload_events_are_addressed_to_the_uploader_only(db, upload, stubbed_models, monkeypatch, viewer_user):
    """Every realtime event this analysis publishes must name the owning
    user, so the broadcaster can withhold it from other clients."""
    published = []
    monkeypatch.setattr(
        video_analysis.realtime_broadcaster,
        "publish",
        lambda event_type, data, owner_user_id=None: published.append((event_type, owner_user_id)),
    )

    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    assert published, "expected the analysis to publish progress events"
    for event_type, owner_user_id in published:
        assert owner_user_id == viewer_user.id, f"{event_type} was published unscoped"
