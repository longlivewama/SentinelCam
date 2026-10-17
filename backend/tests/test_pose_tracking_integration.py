"""
Pose + tracking wired into fall detection, live cameras and uploads.

  * FallPipeline: events keep coming from the trained-detector gate
    exactly as before; tracking only attributes them to a person, and the
    two optional assist steps (pose verification, track dedupe) only ever
    withhold an already-confirmed event - visibly, with a reason.
  * Live cameras: the detection loop publishes each processed frame's
    tracked people, and clears them when the loop stops.
  * Uploads: the real analyser writes a per-video tracking record whose
    fall events point at the right person and the right DB rows; the API
    serves it under the upload's own access rules; deleting the upload
    deletes it.

Models are stubbed (the real-inference test lives in
test_pose_estimation.py), so these are about wiring, not model quality.
"""
import json
from collections import namedtuple

import cv2
import numpy as np
import pytest

from app.config import settings
from app.models.event import Event
from app.models.video_upload import VideoUpload
from app.services import video_analysis
from app.services.cascade_delete import upload_clip_dir, upload_tracking_path
from app.services.detection import engine as engine_mod
from app.services.detection.engine import DetectionEngine
from app.services.detection.fall_pipeline import (
    POSE_NO_EVIDENCE,
    POSE_NO_PERSON,
    POSE_NOT_UPRIGHT,
    POSE_UPRIGHT,
    FallPipeline,
    build_scene_tracker,
)

Box = namedtuple("Box", ["bbox", "confidence"])
Person = namedtuple("Person", ["bbox", "keypoints", "confidence"])

LANES = [10.0, 120.0, 230.0]


def upright_keypoints(x, top=40.0):
    """17 keypoints of a standing person whose box starts at x."""
    points = [(x + 30, top + 5 + i, 0.9) for i in range(17)]
    points[5], points[6] = (x + 20, top + 30, 0.9), (x + 40, top + 30, 0.9)    # shoulders
    points[11], points[12] = (x + 22, top + 90, 0.9), (x + 38, top + 90, 0.9)  # hips
    return points


def lying_keypoints(x, top=40.0):
    points = [(x + 5 + i, top + 100, 0.9) for i in range(17)]
    points[5], points[6] = (x + 50, top + 95, 0.9), (x + 50, top + 105, 0.9)  # shoulders right
    points[11], points[12] = (x + 10, top + 95, 0.9), (x + 10, top + 105, 0.9)  # hips left
    return points


def people(pose_for_faller="upright"):
    out = []
    for lane, x in enumerate(LANES):
        if lane == 2 and pose_for_faller == "lying":
            kps = lying_keypoints(x)
        elif lane == 2 and pose_for_faller == "none":
            kps = []
        else:
            kps = upright_keypoints(x)
        out.append(Person(bbox=(x, 40.0, x + 60.0, 190.0), keypoints=kps, confidence=0.9))
    return out


FALL_ON_LANE_2 = Box(bbox=(235.0, 110.0, 285.0, 185.0), confidence=0.82)
FALL_ON_NOBODY = Box(bbox=(10.0, 200.0, 60.0, 235.0), confidence=0.82)


class ScriptedDetector:
    def __init__(self, boxes_by_frame):
        self.boxes_by_frame = boxes_by_frame
        self.frame = -1

    def detect(self, frame):
        self.frame += 1
        return list(self.boxes_by_frame(self.frame))


def run_pipeline(pipeline, frames=12, pose="upright", step=0.2):
    """`pose` is the faller's pose on every frame, or a function of the
    frame index returning it. Ends like an uploaded video: with flush()."""
    events, withheld = [], []
    for index in range(frames):
        pose_now = pose(index) if callable(pose) else pose
        events += pipeline.update(None, people(pose_now), now=index * step, frame_index=index)
        withheld += pipeline.last_suppressed_events
    events += pipeline.flush()
    withheld += pipeline.last_suppressed_events
    return events, withheld


def fall_from_frame(start, box=FALL_ON_LANE_2):
    return ScriptedDetector(lambda f: [box] if f >= start else [])


@pytest.fixture(autouse=True)
def _verification_off_unless_asked(monkeypatch):
    """The stub people stand upright throughout, so with the production
    default (deferred pose verification on) every scripted fall would be
    withheld. Tests here are about wiring; the ones about verification
    pass `pose_verification=True` or set the setting themselves."""
    monkeypatch.setattr(settings, "FALL_POSE_VERIFICATION", False)


# --- FallPipeline -------------------------------------------------------------

def test_tracking_changes_no_event_only_attributes_it():
    baseline, _ = run_pipeline(FallPipeline(mode="model", scene=None, fall_detector=fall_from_frame(2)))
    tracked, _ = run_pipeline(FallPipeline(mode="model", scene=build_scene_tracker(), fall_detector=fall_from_frame(2)))

    assert [(e["confidence"], e["sustained_seconds"], e["bbox"]) for e in baseline] == \
        [(e["confidence"], e["sustained_seconds"], e["bbox"]) for e in tracked]
    assert "person_track_id" not in baseline[0]
    assert tracked[0]["person_track_id"] == 3  # the person in lane 2 (ids are 1-based, in lane order)


def test_a_fall_with_nobody_under_it_is_kept_but_not_attributed():
    events, _ = run_pipeline(FallPipeline(mode="model", scene=build_scene_tracker(), fall_detector=fall_from_frame(2, FALL_ON_NOBODY)))
    assert len(events) == 1
    assert events[0]["person_track_id"] is None
    assert events[0]["pose"]["verdict"] == POSE_NO_PERSON


def test_pose_evidence_is_recorded_without_changing_anything_when_verification_is_off():
    events, withheld = run_pipeline(FallPipeline(
        mode="model", scene=build_scene_tracker(), pose_verification=False, fall_detector=fall_from_frame(2),
    ))
    assert withheld == []
    assert events[0]["pose"]["verdict"] == POSE_UPRIGHT
    assert events[0]["pose"]["torso_angle_deg"] < 10


def test_pose_verification_withholds_an_event_whose_person_is_upright():
    # Long enough for the confirmation window (fires t=1.0, decided t=3.0).
    events, withheld = run_pipeline(FallPipeline(
        mode="model", scene=build_scene_tracker(), pose_verification=True, fall_detector=fall_from_frame(2),
    ), frames=20)
    assert events == []
    assert len(withheld) == 1
    assert withheld[0]["suppressed_reason"] == "pose_upright"
    assert withheld[0]["person_track_id"] == 3


def test_pose_verification_keeps_a_lying_person():
    events, _ = run_pipeline(FallPipeline(
        mode="model", scene=build_scene_tracker(), pose_verification=True, fall_detector=fall_from_frame(2),
    ), pose="lying")
    assert len(events) == 1
    assert events[0]["pose"]["verdict"] == POSE_NOT_UPRIGHT


def test_no_pose_evidence_never_vetoes():
    events, _ = run_pipeline(FallPipeline(
        mode="model", scene=build_scene_tracker(), pose_verification=True, fall_detector=fall_from_frame(2),
    ), pose="none")
    assert len(events) == 1
    assert events[0]["pose"]["verdict"] == POSE_NO_EVIDENCE


# --- deferred pose verification (FALL_POSE_CONFIRM_SECONDS) -------------------
# The gate fires ~0.6 s into a fall, often while the torso is still near
# vertical. These are the failure cases measured on the URFD corpus and the
# i-LIDS station clip: an immediate veto lost real falls caught mid-fall;
# the deferred check keeps them and still withholds people who walk on.

def verifying_pipeline():
    return FallPipeline(
        mode="model", scene=build_scene_tracker(), pose_verification=True, fall_detector=fall_from_frame(2),
    )


def test_a_fall_caught_while_still_upright_is_released_once_the_person_is_down():
    # Gate fires on frame 5 (t=1.0) with the faller upright; down from frame 6.
    events, withheld = run_pipeline(
        verifying_pipeline(), frames=20, pose=lambda i: "lying" if i >= 6 else "upright",
    )
    assert withheld == []
    assert len(events) == 1
    assert events[0]["person_track_id"] == 3
    assert events[0]["pose"]["verdict"] == POSE_UPRIGHT  # at fire time
    confirmation = events[0]["pose"]["confirmation"]
    assert confirmation["outcome"] == "released_not_upright"
    assert 0 < confirmation["waited_seconds"] < 2.0


def test_a_released_event_fires_on_the_frame_the_posture_turned():
    pipeline = verifying_pipeline()
    fired_at = []
    for index in range(20):
        pose = "lying" if index >= 8 else "upright"
        if pipeline.update(None, people(pose), now=index * 0.2, frame_index=index):
            fired_at.append(index)
    assert fired_at == [9]  # second consecutive non-upright measurement


def test_a_person_who_stays_upright_is_withheld_after_the_window():
    events, withheld = run_pipeline(verifying_pipeline(), frames=30)
    assert events == []
    assert len(withheld) == 1
    assert withheld[0]["suppressed_reason"] == "pose_upright"
    assert withheld[0]["pose"]["confirmation"]["outcome"] == "withheld_upright"
    assert withheld[0]["pose"]["confirmation"]["waited_seconds"] == pytest.approx(2.0)


def test_nothing_is_decided_before_the_window_runs_out():
    pipeline = verifying_pipeline()
    for index in range(10):  # fires at t=1.0; window ends at t=3.0
        assert pipeline.update(None, people("upright"), now=index * 0.2, frame_index=index) == []
        assert pipeline.last_suppressed_events == []


def test_a_held_event_is_released_when_the_persons_track_ends():
    pipeline = verifying_pipeline()
    events = []
    for index in range(30):
        crowd = people("upright")
        if index >= 6:
            crowd = crowd[:2]  # the faller drops out of the pose model's view
        events += pipeline.update(None, crowd, now=index * 0.2, frame_index=index)
    assert len(events) == 1
    assert events[0]["pose"]["confirmation"]["outcome"] == "released_track_ended"


def test_no_posture_after_the_event_releases_it():
    events, withheld = run_pipeline(
        verifying_pipeline(), frames=30, pose=lambda i: "none" if i >= 5 else "upright",
    )
    assert withheld == []
    assert len(events) == 1
    assert events[0]["pose"]["confirmation"]["outcome"] == "released_insufficient_evidence"


def test_a_video_ending_mid_window_releases_without_enough_evidence():
    # Fires on the video's last processed frame: no post-event posture at all.
    events, withheld = run_pipeline(verifying_pipeline(), frames=6)
    assert withheld == []
    assert len(events) == 1
    assert events[0]["pose"]["confirmation"]["outcome"] == "released_source_ended"


def test_a_video_ending_mid_window_releases_even_an_upright_person():
    # 0.6 s of upright posture is not the 2 s look the veto requires.
    events, withheld = run_pipeline(verifying_pipeline(), frames=9)
    assert withheld == []
    assert events[0]["pose"]["confirmation"]["outcome"] == "released_source_ended"


def test_confirm_seconds_zero_is_the_immediate_veto(monkeypatch):
    monkeypatch.setattr(settings, "FALL_POSE_CONFIRM_SECONDS", 0.0)
    pipeline = verifying_pipeline()
    withheld_at = []
    for index in range(8):
        pipeline.update(None, people("upright"), now=index * 0.2, frame_index=index)
        if pipeline.last_suppressed_events:
            withheld_at.append(index)
    assert withheld_at == [5]


def test_an_unattributed_event_is_never_held():
    events, withheld = run_pipeline(FallPipeline(
        mode="model", scene=build_scene_tracker(), pose_verification=True,
        fall_detector=fall_from_frame(2, FALL_ON_NOBODY),
    ))
    assert withheld == []
    assert len(events) == 1
    assert "confirmation" not in events[0]["pose"]


class _RepeatingGate:
    """Emits a fall on the same box at the given times - what the detector's
    own gate does when its box tracker splits one incident in two."""

    def __init__(self, at_times):
        self.at_times = at_times

    def update(self, detections, now=None):
        if any(abs(now - t) < 1e-6 for t in self.at_times):
            return [{"confidence": 0.8, "sustained_seconds": 0.6, "detector": "model", "bbox": FALL_ON_LANE_2.bbox}]
        return []


def test_track_dedupe_suppresses_a_repeat_for_the_same_person(monkeypatch):
    monkeypatch.setattr(settings, "FALL_TRACK_DEDUPE_SECONDS", 10.0)
    pipeline = FallPipeline(mode="model", scene=build_scene_tracker(), track_dedupe=True,
                            fall_detector=ScriptedDetector(lambda f: [FALL_ON_LANE_2]))
    pipeline._model_gate = _RepeatingGate([0.6, 1.4])
    events, withheld = run_pipeline(pipeline)
    assert len(events) == 1
    assert [w["suppressed_reason"] for w in withheld] == ["duplicate_for_track"]


def test_without_dedupe_both_repeats_are_alerts():
    pipeline = FallPipeline(mode="model", scene=build_scene_tracker(), track_dedupe=False,
                            fall_detector=ScriptedDetector(lambda f: [FALL_ON_LANE_2]))
    pipeline._model_gate = _RepeatingGate([0.6, 1.4])
    events, withheld = run_pipeline(pipeline)
    assert len(events) == 2 and withheld == []


def test_a_tracker_failure_costs_attribution_not_the_event(monkeypatch):
    scene = build_scene_tracker()
    pipeline = FallPipeline(mode="model", scene=scene, fall_detector=fall_from_frame(2))

    def broken(*args, **kwargs):
        raise RuntimeError("tracker exploded")

    monkeypatch.setattr(scene, "update", broken)
    events, _ = run_pipeline(pipeline)
    assert len(events) == 1
    assert events[0]["person_track_id"] is None


def test_tracking_disabled_by_setting(monkeypatch):
    monkeypatch.setattr(settings, "PERSON_TRACKING_ENABLED", False)
    assert FallPipeline(mode="model").scene is None
    monkeypatch.setattr(settings, "PERSON_TRACKING_ENABLED", True)
    assert FallPipeline(mode="model").scene is not None


# --- live cameras -------------------------------------------------------------

def _capture_publish(monkeypatch):
    published = []
    monkeypatch.setattr(
        "app.services.detection.engine.realtime_broadcaster.publish",
        lambda event_type, data, owner_user_id=None: published.append((event_type, data)),
    )
    return published


def test_live_frames_publish_tracked_people_and_attribute_falls(monkeypatch):
    engine = DetectionEngine()
    monkeypatch.setattr(engine, "extract_people", lambda frame: people())
    monkeypatch.setattr(engine, "extract_objects", lambda frame: ([], []))
    alerts = []
    monkeypatch.setattr(
        "app.services.detection.engine.recording_engine.trigger_event",
        lambda *a, **k: alerts.append(k),
    )
    published = _capture_publish(monkeypatch)

    class _Quiet:
        def update(self, *a):
            return [] if len(a) == 1 else None

    pipeline = FallPipeline(mode="model", scene=build_scene_tracker(), fall_detector=fall_from_frame(2))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for index in range(12):  # 0.1 s apart: the 0.6 s sustain gate fires at frame 8
        monkeypatch.setattr(engine_mod.time, "time", lambda index=index: 1000.0 + index * 0.1)
        engine._process_frame(1, frame, pipeline, _Quiet(), _Quiet(), 20, index)

    tracks = [data for kind, data in published if kind == "tracks.update"]
    assert len(tracks) == 12
    latest = tracks[-1]
    assert latest["camera_id"] == 1 and latest["active"] is True
    assert (latest["frame_width"], latest["frame_height"]) == (320, 240)
    assert sorted(p["track_id"] for p in latest["persons"]) == [1, 2, 3]
    assert all(len(p["keypoints"]) == 17 for p in latest["persons"])
    assert latest["pose_format"]["skeleton"][0] == [15, 13]

    assert alerts and alerts[0]["person_track_id"] == 3
    fall_box = [data for kind, data in published if kind == "fall.bbox"][0]
    assert fall_box["person_track_id"] == 3


def test_a_stopping_live_loop_clears_its_overlay(monkeypatch):
    engine = DetectionEngine()
    monkeypatch.setattr(engine, "ensure_models", lambda: None)
    published = _capture_publish(monkeypatch)
    monkeypatch.setattr(engine_mod, "FallPipeline", lambda: FallPipeline(mode="model", scene=build_scene_tracker()))

    class _NoCamera:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def query(self, *a):
            return self

        def filter(self, *a):
            return self

        def first(self):
            return None  # camera deleted -> loop exits at once

    monkeypatch.setattr(engine_mod, "SessionLocal", _NoCamera)
    import threading

    engine._run_loop(7, threading.Event())
    assert ("tracks.update", published[-1][1]) == published[-1]
    assert published[-1][1]["camera_id"] == 7
    assert published[-1][1]["active"] is False and published[-1][1]["persons"] == []


# --- uploads ------------------------------------------------------------------

FPS, TOTAL_FRAMES, WIDTH, HEIGHT = 10, 100, 320, 240
FALL_START, FALL_END = 40, 70


@pytest.fixture()
def analysed_upload(db, viewer_user, monkeypatch, tmp_path):
    """Runs the real upload analyser over a real decoded video, with the
    two models stubbed: three people in their lanes, a fall box on lane 2
    from 4.0 s to 7.0 s."""
    return _analyse(db, viewer_user, monkeypatch, tmp_path)


def _analyse(db, viewer_user, monkeypatch, tmp_path, faller_pose=lambda frame: "upright",
             fall_start=FALL_START, fall_end=FALL_END):
    monkeypatch.setattr(settings, "RECORDINGS_DIR", str(tmp_path / "recordings"))
    monkeypatch.setattr(settings, "SNAPSHOTS_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setattr(settings, "VIDEO_ANALYSIS_FRAME_STRIDE", 1)
    monkeypatch.setattr(video_analysis.detection_engine, "ensure_models", lambda: None)
    monkeypatch.setattr("app.services.detection.fall_pipeline.resolve_mode", lambda: "model")

    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for _ in range(TOTAL_FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), 24, dtype=np.uint8))
    writer.release()

    state = {"frame": -1}

    def fake_extract_people(frame):
        state["frame"] += 1
        return people(faller_pose(state["frame"]))

    def fake_detect(frame):
        return [FALL_ON_LANE_2] if fall_start <= state["frame"] < fall_end else []

    monkeypatch.setattr(video_analysis.detection_engine, "extract_people", fake_extract_people)
    monkeypatch.setattr("app.services.detection.fall_pipeline.fall_object_detector.detect", fake_detect)

    upload = VideoUpload(user_id=viewer_user.id, original_filename="ward.mp4", stored_path=str(path),
                         file_size_bytes=path.stat().st_size, status="processing")
    db.add(upload)
    db.commit()
    db.refresh(upload)
    assert video_analysis._process(upload.id, str(path), "ward.mp4") is True
    db.expire_all()
    return db.query(VideoUpload).filter(VideoUpload.id == upload.id).one()


def test_the_analyser_writes_a_tracking_record_tied_to_the_fall_events(db, analysed_upload):
    record = json.loads(upload_tracking_path(analysed_upload.id).read_text())

    assert record["video"] == {"frame_width": WIDTH, "frame_height": HEIGHT, "fps": float(FPS), "stride": 1}
    assert len(record["frames"]) == TOTAL_FRAMES
    assert sorted(t["track_id"] for t in record["tracks"]) == [1, 2, 3]
    assert all(t["frames_observed"] == TOTAL_FRAMES for t in record["tracks"])
    first_frame = record["frames"][0]
    assert first_frame["t"] == 0.0 and len(first_frame["persons"]) == 3

    events = db.query(Event).filter(Event.video_upload_id == analysed_upload.id).all()
    assert len(events) == 1 and len(record["fall_events"]) == 1
    fall = record["fall_events"][0]
    assert fall["event_id"] == events[0].id
    assert fall["recording_id"] == events[0].recording_id
    assert fall["person_track_id"] == 3
    assert fall["video_time_seconds"] == pytest.approx(events[0].video_timestamp_seconds, abs=0.01)


def test_the_analyser_alerts_on_a_held_fall_once_the_person_is_down(
    client, db, viewer_user, viewer_headers, monkeypatch, tmp_path,
):
    # Production default: deferred pose verification. The gate fires at
    # 4.7 s with the faller still upright; they are lying from 5.0 s.
    monkeypatch.setattr(settings, "FALL_POSE_VERIFICATION", True)
    upload = _analyse(db, viewer_user, monkeypatch, tmp_path,
                      faller_pose=lambda f: "lying" if f >= 50 else "upright")

    events = db.query(Event).filter(Event.video_upload_id == upload.id).all()
    assert upload.fall_events_count == 1 and len(events) == 1
    # Placed where the gate FIRED, not where verification released it
    # (5.1 s, the 2nd lying frame): the timestamp, the overlay highlight and
    # the "at 0:04 in video" link all point at the fall itself.
    assert events[0].video_timestamp_seconds == pytest.approx(4.7, abs=0.01)
    record = json.loads(upload_tracking_path(upload.id).read_text())
    assert record["fall_events"][0]["person_track_id"] == 3
    assert record["fall_events"][0]["event_id"] == events[0].id

    # The API is what the event card reads: the confirmation must survive
    # its response schema, or a real fall is labelled "torso upright".
    body = client.get(f"/api/video-uploads/{upload.id}/tracking", headers=viewer_headers).json()
    pose = body["fall_events"][0]["pose"]
    assert pose["verdict"] == POSE_UPRIGHT
    assert pose["confirmation"]["outcome"] == "released_not_upright"
    assert pose["confirmation"]["waited_seconds"] == pytest.approx(0.4, abs=0.01)
    assert pose["confirmation"]["post_event_torso_angle_deg"] > 30


def test_the_analyser_withholds_a_fall_on_someone_who_stays_upright(db, viewer_user, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "FALL_POSE_VERIFICATION", True)
    upload = _analyse(db, viewer_user, monkeypatch, tmp_path)

    assert upload.status == "completed" and upload.fall_events_count == 0
    assert db.query(Event).filter(Event.video_upload_id == upload.id).count() == 0
    record = json.loads(upload_tracking_path(upload.id).read_text())
    assert record["fall_events"] == []
    assert [e["reason"] for e in record["suppressed_events"]] == ["pose_upright"]


def test_the_analyser_releases_a_fall_still_held_when_the_video_ends(db, viewer_user, monkeypatch, tmp_path):
    # Fall box from 9.0 s; the gate fires at 9.7 s, the last 0.2 s are not
    # enough posture to judge - the event must not vanish with the video.
    monkeypatch.setattr(settings, "FALL_POSE_VERIFICATION", True)
    upload = _analyse(db, viewer_user, monkeypatch, tmp_path, fall_start=90, fall_end=TOTAL_FRAMES)

    events = db.query(Event).filter(Event.video_upload_id == upload.id).all()
    assert upload.fall_events_count == 1 and len(events) == 1
    # At the fire point, not the last frame (9.9 s) it was flushed on.
    assert events[0].video_timestamp_seconds == pytest.approx(9.7, abs=0.01)
    assert events[0].recording_id is not None


def test_the_tracking_api_follows_the_uploads_access_rules(
    client, analysed_upload, viewer_headers, operator_headers, make_user_factory, auth_headers_for,
):
    url = f"/api/video-uploads/{analysed_upload.id}/tracking"
    response = client.get(url, headers=viewer_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["fall_events"][0]["person_track_id"] == 3
    assert body["pose_format"]["skeleton"][0] == [15, 13]
    assert len(body["frames"][0]["persons"][0]["keypoints"]) == 17

    assert client.get(url, headers=operator_headers).status_code == 200
    stranger = make_user_factory("stranger@example.com", "viewer")
    assert client.get(url, headers=auth_headers_for(stranger)).status_code == 403
    assert client.get(url).status_code in (401, 403)


def test_no_record_is_a_clean_404(client, db, viewer_user, viewer_headers):
    upload = VideoUpload(user_id=viewer_user.id, original_filename="old.mp4", stored_path="/nonexistent.mp4",
                         file_size_bytes=1, status="completed")
    db.add(upload)
    db.commit()
    response = client.get(f"/api/video-uploads/{upload.id}/tracking", headers=viewer_headers)
    assert response.status_code == 404
    assert response.json()["detail"] == "No tracking data for this upload"


def test_deleting_the_upload_deletes_its_tracking_record(client, analysed_upload, viewer_headers):
    path = upload_tracking_path(analysed_upload.id)
    assert path.exists()
    assert client.delete(f"/api/video-uploads/{analysed_upload.id}", headers=viewer_headers).status_code == 204
    assert not path.exists()
    assert not upload_clip_dir(analysed_upload.id).exists()


def test_with_tracking_disabled_the_analysis_is_unchanged_and_writes_no_record(db, viewer_user, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PERSON_TRACKING_ENABLED", False)
    monkeypatch.setattr(settings, "RECORDINGS_DIR", str(tmp_path / "recordings"))
    monkeypatch.setattr(settings, "SNAPSHOTS_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setattr(settings, "VIDEO_ANALYSIS_FRAME_STRIDE", 1)
    monkeypatch.setattr(video_analysis.detection_engine, "ensure_models", lambda: None)
    monkeypatch.setattr("app.services.detection.fall_pipeline.resolve_mode", lambda: "model")
    state = {"frame": -1}

    def fake_extract_people(frame):
        state["frame"] += 1
        return people()

    monkeypatch.setattr(video_analysis.detection_engine, "extract_people", fake_extract_people)
    monkeypatch.setattr(
        "app.services.detection.fall_pipeline.fall_object_detector.detect",
        lambda frame: [FALL_ON_LANE_2] if FALL_START <= state["frame"] < FALL_END else [],
    )
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for _ in range(TOTAL_FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), 24, dtype=np.uint8))
    writer.release()
    upload = VideoUpload(user_id=viewer_user.id, original_filename="w.mp4", stored_path=str(path),
                         file_size_bytes=path.stat().st_size, status="processing")
    db.add(upload)
    db.commit()

    assert video_analysis._process(upload.id, str(path), "w.mp4") is True
    assert db.query(Event).filter(Event.video_upload_id == upload.id).count() == 1
    assert not upload_tracking_path(upload.id).exists()
