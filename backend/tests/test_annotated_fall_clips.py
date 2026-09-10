"""
The annotation layer running inside the real upload analyser.

Same shape as test_video_analysis_pipeline.py - a real video file is
decoded, the real `_process()` loop runs, and real clips are encoded -
with the two models stubbed so this stays hermetic. What is added here is
the annotation: that the clip a user downloads actually has a box drawn
on the person who fell, that the box follows them, and that nobody else
is labelled.

The other half of what these tests protect is the layer's boundary. A
fall event's timestamp, confidence, detector and rows must be identical
whether or not anything was drawn, and a failure inside the annotation
must cost the clip its boxes and nothing more.
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
from app.services.detection.annotated_clip_renderer import COLOR_MATCHED, COLOR_UNMATCHED

Box = namedtuple("Box", ["bbox", "confidence"])
Person = namedtuple("Person", ["bbox", "keypoints"])

FPS = 10
TOTAL_FRAMES = 100          # 10 seconds
WIDTH, HEIGHT = 320, 240
FALL_START_FRAME = 40       # 4.0s into the video
FALL_END_FRAME = 70         # 7.0s into the video

# Three people in their own lanes. The third one is the one who falls.
LANES = [10, 120, 230]
PERSON_WIDTH, PERSON_HEIGHT = 60, 150
PERSON_TOP = 40
FALLER_LANE = 2


def person_box(lane_index, drift=0):
    x = LANES[lane_index] + (drift if lane_index == FALLER_LANE else 0)
    return (float(x), float(PERSON_TOP), float(x + PERSON_WIDTH), float(PERSON_TOP + PERSON_HEIGHT))


def fall_box_on_faller(drift=0):
    """A sub-region of the faller's box - the shape the trained detector
    actually produces mid-fall."""
    x = LANES[FALLER_LANE] + drift
    return (float(x + 5), 110.0, float(x + 55), 185.0)


def colored_pixels(frame, color, tolerance=60):
    diff = np.abs(frame.astype(int) - np.array(color, dtype=int))
    return int((diff.max(axis=2) <= tolerance).sum())


@pytest.fixture()
def source_video(tmp_path):
    """A real, decodable 10-second video. Deliberately dark and flat so
    the annotation's saturated colours are unambiguous when the encoded
    clip is decoded back."""
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), "OpenCV could not open a VideoWriter for the test fixture"
    for _ in range(TOTAL_FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), 24, dtype=np.uint8))
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
def scene(monkeypatch, tmp_path):
    """Stubs both models and redirects clip/snapshot output.

    Returns a mutable dict the individual tests reshape: `fall_box` picks
    what the trained detector reports, `people` picks who the pose model
    reports.
    """
    monkeypatch.setattr(settings, "RECORDINGS_DIR", str(tmp_path / "recordings"))
    monkeypatch.setattr(settings, "SNAPSHOTS_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setattr(settings, "VIDEO_ANALYSIS_FRAME_STRIDE", 1)
    monkeypatch.setattr(settings, "FALL_DETECTOR_MIN_SUSTAINED_SECONDS", 0.6)

    monkeypatch.setattr(video_analysis.detection_engine, "ensure_models", lambda: None)

    state = {
        "frame": -1,
        "fall_box": fall_box_on_faller,
        "people": lambda drift: [Person(bbox=person_box(i, drift), keypoints=[]) for i in range(3)],
        "drift_per_frame": 1,
    }

    # scan_video calls the pose model first and the fall pipeline second
    # for each frame, so the counter advances in the pose stub and both
    # stubs then agree on which frame they are looking at.
    def fake_extract_people(frame):
        state["frame"] += 1
        return state["people"](state["frame"] * state["drift_per_frame"])

    def fake_detect(frame):
        index = state["frame"]
        if FALL_START_FRAME <= index < FALL_END_FRAME:
            return [Box(bbox=state["fall_box"](index * state["drift_per_frame"]), confidence=0.82)]
        return []

    monkeypatch.setattr(video_analysis.detection_engine, "extract_people", fake_extract_people)
    monkeypatch.setattr(
        "app.services.detection.fall_pipeline.fall_object_detector.detect", fake_detect,
    )
    monkeypatch.setattr(
        "app.services.detection.fall_pipeline.resolve_mode", lambda: "model",
    )
    return state


@pytest.fixture()
def rendered(monkeypatch):
    """Captures what the renderer was asked to draw, without replacing it -
    the real drawing still happens and the real clip is still written."""
    calls = []
    real = video_analysis.iter_annotated_frames

    def spy(frames, first_frame_index, timeline, label):
        calls.append({
            "first_frame_index": first_frame_index,
            "timeline": timeline,
            "label": label,
            "frame_count": len(frames),
        })
        return real(frames, first_frame_index, timeline, label)

    monkeypatch.setattr(video_analysis, "iter_annotated_frames", spy)
    return calls


def clip_frames(recording):
    cap = cv2.VideoCapture(recording.file_path)
    try:
        assert cap.isOpened(), f"the written clip did not open: {recording.file_path}"
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames
    finally:
        cap.release()


# --- the happy path -------------------------------------------------------

def test_the_fall_clip_is_written_with_a_box_drawn_on_it(db, upload, scene):
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    recording = db.query(Recording).filter(Recording.video_upload_id == upload.id).one()
    frames = clip_frames(recording)

    assert frames, "the clip decoded to no frames"
    annotated = [i for i, f in enumerate(frames) if colored_pixels(f, COLOR_MATCHED) > 150]
    assert len(annotated) > len(frames) * 0.8, (
        f"only {len(annotated)} of {len(frames)} clip frames carry an annotation"
    )


def test_the_box_is_drawn_over_the_person_who_fell_and_not_the_others(db, upload, scene, rendered):
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    assert len(rendered) == 1
    timeline = rendered[0]["timeline"]
    first_index = rendered[0]["first_frame_index"]

    boxes = [timeline.box_at(first_index + offset) for offset in range(rendered[0]["frame_count"])]
    drawn = [b for b in boxes if b is not None]
    assert drawn, "nothing was drawn at all"

    # Lane 3 starts at x=230 and drifts right; lanes 1 and 2 sit at x=10
    # and x=120 and never move.
    for x1, _y1, x2, _y2 in drawn:
        assert x1 >= LANES[FALLER_LANE] - 5, "the box drifted onto another lane"
        assert x2 > LANES[1] + PERSON_WIDTH


def test_the_label_names_the_track_and_the_events_own_confidence(db, upload, scene, rendered):
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    event = db.query(Event).filter(Event.video_upload_id == upload.id).one()
    label = rendered[0]["label"]

    assert label.headline == "FALL DETECTED"
    assert label.matched is True
    assert label.detail.startswith("ID ")
    assert label.detail.endswith(f"{event.confidence_score:.2f}")


def test_the_box_follows_the_person_instead_of_sitting_still(db, upload, scene, rendered):
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    timeline = rendered[0]["timeline"]
    first_index = rendered[0]["first_frame_index"]
    xs = [
        timeline.box_at(first_index + offset)[0]
        for offset in range(rendered[0]["frame_count"])
        if timeline.box_at(first_index + offset) is not None
    ]

    assert len(set(xs)) > 5, "the same box was drawn on every frame"
    assert xs[-1] > xs[0], "the box did not follow the subject"


def test_the_annotation_spans_the_pre_event_footage_too(db, upload, scene, rendered):
    """A clip starts three seconds before the fall; those frames are part
    of what the operator watches."""
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    timeline = rendered[0]["timeline"]
    first_index = rendered[0]["first_frame_index"]

    assert timeline.box_at(first_index) is not None
    assert timeline.box_at(first_index + 5) is not None


# --- the event itself is untouched ---------------------------------------

def test_the_fall_event_is_exactly_what_it_was_without_annotation(db, upload, scene):
    """The annotation layer is downstream of the decision: same
    timestamp, same confidence, same detector, same rows."""
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    event = db.query(Event).filter(Event.video_upload_id == upload.id).one()
    recording = db.query(Recording).filter(Recording.video_upload_id == upload.id).one()

    assert event.video_timestamp_seconds == pytest.approx(4.6, abs=0.3)
    assert event.event_type == "fall"
    assert event.detector == "model"
    assert event.confidence_score == pytest.approx(0.82, abs=0.01)
    assert event.camera_id is None
    assert event.recording_id == recording.id
    assert recording.trigger_action == "fall"
    assert recording.file_size_bytes > 0

    db.refresh(upload)
    assert upload.status == "completed"
    assert upload.fall_events_count == 1
    # Three people were on screen; the annotation's own deduplication must
    # not have touched this figure.
    assert upload.persons_detected == 3


def test_a_video_with_no_fall_writes_no_clip_and_draws_nothing(db, upload, scene, rendered, monkeypatch):
    monkeypatch.setattr(
        "app.services.detection.fall_pipeline.fall_object_detector.detect", lambda frame: [],
    )

    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    db.refresh(upload)
    assert upload.status == "completed"
    assert upload.fall_events_count == 0
    assert db.query(Event).filter(Event.video_upload_id == upload.id).count() == 0
    assert db.query(Recording).filter(Recording.video_upload_id == upload.id).count() == 0
    assert rendered == []


# --- two people falling in turn -------------------------------------------

def test_two_people_falling_in_turn_are_annotated_on_their_own_tracks(db, upload, scene, rendered, monkeypatch):
    """One event per faller, each clip boxed on the right person. A
    single global "somebody fell" label on every visible person is
    exactly the failure this rules out."""
    first_window = range(15, 40)     # lane 0 goes down at ~1.5s
    second_window = range(60, 95)    # lane 2 goes down at ~6.0s

    def two_falls(frame):
        index = scene["frame"]
        if index in first_window:
            x = LANES[0]
            return [Box(bbox=(float(x + 5), 110.0, float(x + 55), 185.0), confidence=0.61)]
        if index in second_window:
            x = LANES[2]
            return [Box(bbox=(float(x + 5), 110.0, float(x + 55), 185.0), confidence=0.88)]
        return []

    scene["drift_per_frame"] = 0
    monkeypatch.setattr(
        "app.services.detection.fall_pipeline.fall_object_detector.detect", two_falls,
    )

    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    events = (
        db.query(Event)
        .filter(Event.video_upload_id == upload.id)
        .order_by(Event.video_timestamp_seconds)
        .all()
    )
    assert len(events) == 2
    assert events[0].confidence_score == pytest.approx(0.61, abs=0.01)
    assert events[1].confidence_score == pytest.approx(0.88, abs=0.01)

    assert len(rendered) == 2
    labels = [call["label"] for call in rendered]
    assert all(label.matched for label in labels), [label.detail for label in labels]
    assert labels[0].detail != labels[1].detail, "both clips named the same person"

    # First clip's boxes sit in lane 0, second clip's in lane 2.
    def drawn_boxes(call):
        return [
            call["timeline"].box_at(call["first_frame_index"] + offset)
            for offset in range(call["frame_count"])
            if call["timeline"].box_at(call["first_frame_index"] + offset) is not None
        ]

    assert all(box[2] <= LANES[1] for box in drawn_boxes(rendered[0]))
    assert all(box[0] >= LANES[FALLER_LANE] - 5 for box in drawn_boxes(rendered[1]))


# --- fail-safe paths ------------------------------------------------------

def test_a_fall_on_nobody_is_drawn_as_an_unmatched_region(db, upload, scene, rendered):
    """The detector fires on something that is not one of the tracked
    people. Naming the nearest person would be a confident false claim,
    so the clip gets the detector's own region and no track id."""
    scene["fall_box"] = lambda drift: (150.0, 2.0, 200.0, 32.0)   # above everyone

    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    assert len(rendered) == 1
    label = rendered[0]["label"]
    assert label.matched is False
    assert "ID" not in label.detail
    assert label.detail.startswith("UNMATCHED")

    recording = db.query(Recording).filter(Recording.video_upload_id == upload.id).one()
    frames = clip_frames(recording)
    assert any(colored_pixels(f, COLOR_UNMATCHED) > 150 for f in frames)
    assert all(colored_pixels(f, COLOR_MATCHED) < 150 for f in frames)

    # The event is still a normal fall event.
    event = db.query(Event).filter(Event.video_upload_id == upload.id).one()
    assert event.event_type == "fall"
    assert event.confidence_score == pytest.approx(0.82, abs=0.01)


def test_the_clip_still_lands_when_the_annotation_blows_up(db, upload, scene, monkeypatch):
    """A bug in the drawing must cost this clip its boxes, never the clip
    itself, the event, the alert or the analysis."""
    def explode(*args, **kwargs):
        raise RuntimeError("synthetic renderer failure")

    monkeypatch.setattr(video_analysis, "iter_annotated_frames", explode)

    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    db.refresh(upload)
    assert upload.status == "completed"
    event = db.query(Event).filter(Event.video_upload_id == upload.id).one()
    recording = db.query(Recording).filter(Recording.video_upload_id == upload.id).one()
    assert event.video_timestamp_seconds == pytest.approx(4.6, abs=0.3)
    assert recording.file_size_bytes > 0
    assert clip_frames(recording), "the unannotated clip should still be playable"


def test_no_person_detections_at_all_still_produces_an_annotated_clip(db, upload, scene, rendered):
    """Dark footage where the pose model sees nobody: the fall detector's
    own region is still real, tracked output and is still worth drawing."""
    scene["people"] = lambda drift: []

    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    assert len(rendered) == 1
    assert rendered[0]["label"].matched is False

    db.refresh(upload)
    assert upload.persons_detected == 0
    assert upload.fall_events_count == 1


def test_the_snapshot_carries_the_annotation_too(db, upload, scene, tmp_path):
    """The snapshot is the frame that reaches the operator by email, so it
    should show what the clip shows."""
    video_analysis._process(upload.id, upload.stored_path, upload.original_filename)

    snapshots = list((tmp_path / "snapshots").glob("*.jpg"))
    assert len(snapshots) == 1
    image = cv2.imread(str(snapshots[0]))
    assert image is not None
    assert colored_pixels(image, COLOR_MATCHED) > 150
