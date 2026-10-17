"""
Per-camera detection orchestration.

For every camera with ai_detection_enabled=True (and is_active=True), a
background daemon thread:
  - ensures the camera's stream_manager capture thread is running (so
    detection works even if nobody currently has the /stream endpoint
    open),
  - pulls the latest raw frame at a throttled rate (every
    DETECTION_FRAME_STRIDE-th frame - running full detection on every
    single frame is unnecessary and expensive on CPU; this is a tunable),
  - runs the shared pose model once (feeds both fall + violence
    detection) and the shared object model once (feeds crowd detection)
    per processed frame - plus, when the trained fall detector is
    enabled (see detection/fall_pipeline.py), a third shared model that
    produces the fall signal directly,
  - feeds the results to each of the three detector modules,
  - and calls recording_engine.trigger_event(...) whenever a detector
    fires, which takes care of writing the clip, persisting the Recording
    + Event rows, and sending the alert email.

Both YOLO models are loaded once (lazily, on first camera started) and
shared across every camera's detection thread - reloading per-frame or
per-camera would be prohibitively slow on CPU.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import namedtuple
from datetime import datetime, timezone
from typing import Dict, Optional

from ultralytics import YOLO

from app.config import settings
from app.database import SessionLocal
from app.models.camera import Camera
from app.services import recording_engine
from app.services.realtime import realtime_broadcaster
from app.services.stream_manager import stream_manager
from app.services.detection.fall_pipeline import FallPipeline
from app.services.detection.pose import pose_format
from app.services.detection.scene_tracking import STATE_LOST
from app.services.detection.violence_detection import ViolenceDetector
from app.services.detection.crowd_detection import CrowdDetector

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO class index for "person", shared by both pose and object models

# Shared shape consumed by FallDetector, ViolenceDetector and the scene
# tracker. `confidence` is the pose model's person-box score; it defaults
# to None so code that builds a PersonDetection from (bbox, keypoints)
# alone keeps working.
PersonDetection = namedtuple("PersonDetection", ["bbox", "keypoints", "confidence"], defaults=(None,))


class DetectionEngine:
    def __init__(self):
        self._pose_model = None
        self._object_model = None
        self._model_lock = threading.Lock()
        # One model instance is shared by every camera loop and every upload
        # analysis. Ultralytics does not document `predict` on one instance
        # as safe from several threads at once, so calls are serialized per
        # model. On CPU this costs little: torch already spreads a single
        # inference across every core.
        self._pose_inference_lock = threading.Lock()
        self._object_inference_lock = threading.Lock()
        self._pose_load_attempted = False
        self._object_load_attempted = False
        self.pose_model_error: Optional[str] = None
        self.object_model_error: Optional[str] = None
        self._threads: Dict[int, threading.Thread] = {}
        self._stop_flags: Dict[int, threading.Event] = {}

    @property
    def models_loaded(self) -> bool:
        return self._pose_model is not None and self._object_model is not None

    @property
    def pose_available(self) -> bool:
        return self._pose_model is not None

    @property
    def active_camera_count(self) -> int:
        return sum(1 for t in self._threads.values() if t.is_alive())

    # -- model loading (once, shared across all cameras) -----------------

    def ensure_models(self):
        """Loads the shared pose and object models, once each.

        A model that fails to load is recorded (`pose_model_error`,
        `object_model_error`, surfaced by /api/system/status) and not
        retried on every frame. The pipeline then keeps running without
        it: the trained fall detector needs neither, so an unavailable pose
        model costs tracking and overlays, not fall detection."""
        if self._pose_load_attempted and self._object_load_attempted:
            return
        with self._model_lock:
            # Each flag is set only once its load attempt has finished: the
            # unlocked fast path above must not let a second caller through
            # (and run inference against a None model) while the first is
            # still inside YOLO(...).
            if not self._pose_load_attempted:
                try:
                    logger.info("Loading pose model %s", settings.YOLO_POSE_MODEL)
                    self._pose_model = YOLO(settings.YOLO_POSE_MODEL)
                    self.pose_model_error = None
                except Exception as exc:
                    self.pose_model_error = f"failed to load {settings.YOLO_POSE_MODEL}: {exc}"
                    logger.exception("Pose model unavailable; continuing without pose estimation")
                finally:
                    self._pose_load_attempted = True
            if not self._object_load_attempted:
                try:
                    logger.info("Loading object model %s", settings.YOLO_MODEL)
                    self._object_model = YOLO(settings.YOLO_MODEL)
                    self.object_model_error = None
                except Exception as exc:
                    self.object_model_error = f"failed to load {settings.YOLO_MODEL}: {exc}"
                    logger.exception("Object model unavailable; continuing without crowd detection")
                finally:
                    self._object_load_attempted = True

    def should_run_pose(self, fall_pipeline) -> bool:
        """For uploaded-video analysis: pose runs when it is enabled, or when
        the fall pipeline cannot work without keypoints (heuristic / hybrid
        modes). Live cameras always run it - see _process_frame."""
        return bool(settings.POSE_ESTIMATION_ENABLED or getattr(fall_pipeline, "needs_pose", False))

    # -- lifecycle ---------------------------------------------------------

    def ensure_running(self, camera_id: int, url: str):
        """Idempotent: safe to call repeatedly (e.g. on every camera
        update). Starts a detection thread only if one isn't already
        running for this camera."""
        # Before the early return: an already-running loop still needs its
        # stream swapped if the camera's URL was just edited.
        stream_manager.get_or_create(camera_id, url)
        existing = self._threads.get(camera_id)
        if existing is not None and existing.is_alive():
            return
        stop_flag = threading.Event()
        self._stop_flags[camera_id] = stop_flag
        thread = threading.Thread(
            target=self._run_loop, args=(camera_id, stop_flag), daemon=True, name=f"detect-{camera_id}",
        )
        self._threads[camera_id] = thread
        thread.start()

    def stop(self, camera_id: int):
        flag = self._stop_flags.get(camera_id)
        if flag is not None:
            flag.set()

    # -- main loop -----------------------------------------------------

    def _run_loop(self, camera_id: int, stop_flag: threading.Event):
        self.ensure_models()

        # All per-subject state - the fall gates and the person tracker -
        # lives in this pipeline, which is local to this loop: it belongs to
        # exactly one camera session and is released when the loop exits.
        fall_pipeline = FallPipeline()
        logger.info(
            "Camera %s: fall detection running in %r mode, tracking %s",
            camera_id, fall_pipeline.mode, "on" if fall_pipeline.scene is not None else "off",
        )
        violence_detector = ViolenceDetector()
        crowd_detector = CrowdDetector()

        frame_index = 0
        processed_index = 0
        poll_interval = 1.0 / max(settings.STREAM_FPS, 1)

        try:
            while not stop_flag.is_set():
                with SessionLocal() as db:
                    camera = db.query(Camera).filter(Camera.id == camera_id).first()
                    if camera is None or not camera.is_active or not camera.ai_detection_enabled:
                        logger.info("Camera %s detection disabled/removed; stopping detection loop", camera_id)
                        return
                    crowd_threshold = camera.crowd_threshold

                # Looked up every iteration rather than once: editing the
                # camera's URL replaces its stream (see stream_manager), and a
                # reference held from startup would keep reading the stopped
                # one's last frame forever.
                stream = stream_manager.get(camera_id)
                if stream is None:
                    logger.error("No stream for camera %s; detection loop exiting", camera_id)
                    return

                # Always the newest frame - there is no queue to fall behind
                # on. Whatever arrived while inference ran is simply skipped.
                frame = stream.get_latest_frame()
                if frame is None:
                    time.sleep(poll_interval)
                    continue

                frame_index += 1
                if frame_index % settings.DETECTION_FRAME_STRIDE != 0:
                    time.sleep(poll_interval)
                    continue

                try:
                    self._process_frame(
                        camera_id, frame, fall_pipeline, violence_detector,
                        crowd_detector, crowd_threshold, processed_index,
                    )
                except Exception:
                    logger.exception("Detection error on camera %s", camera_id)
                processed_index += 1

                time.sleep(poll_interval)
        finally:
            # Clears any overlay a viewer still shows for this camera; the
            # pipeline and its tracker go out of scope with this frame.
            if fall_pipeline.scene is not None:
                self._broadcast_tracks_cleared(camera_id)

    def _process_frame(
        self, camera_id, frame, fall_pipeline, violence_detector,
        crowd_detector, crowd_threshold, processed_index=0,
    ):
        # Always on a live camera, whatever POSE_ESTIMATION_ENABLED says:
        # the violence detector below works from keypoints and has no other
        # input, so skipping pose here would silently switch it off.
        people = self.extract_people(frame)
        now = time.time()

        for fall_event in fall_pipeline.update(frame, people, now=now, frame_index=processed_index):
            logger.info("Camera %s: FALL detected %s", camera_id, fall_event)
            recording_engine.trigger_event(
                camera_id, "fall", fall_event.get("confidence", 1.0),
                detector=fall_event.get("detector", "heuristic"),
                person_track_id=fall_event.get("person_track_id"),
            )
            self._broadcast_fall_bbox(camera_id, frame, fall_event, fall_pipeline.scene)

        if fall_pipeline.scene is not None:
            self._broadcast_tracks(camera_id, frame, fall_pipeline.scene, now)

        for violence_event in violence_detector.update(people):
            logger.info("Camera %s: VIOLENCE detected %s", camera_id, violence_event)
            recording_engine.trigger_event(camera_id, "violence", violence_event.get("confidence", 1.0))

        person_boxes, _ = self.extract_objects(frame)

        crowd_event = crowd_detector.update(len(person_boxes), crowd_threshold)
        if crowd_event:
            logger.info("Camera %s: CROWD detected %s", camera_id, crowd_event)
            recording_engine.trigger_event(camera_id, "crowd", crowd_event.get("confidence", 1.0))

    def _broadcast_tracks(self, camera_id, frame, scene, now: float):
        """Publishes this frame's tracked people to live-view clients
        (`tracks.update`). Shared-camera data, so not owner-scoped - the
        same rule as `fall.bbox` and the MJPEG stream itself. Nothing is
        persisted: the live feed is the only consumer."""
        height, width = frame.shape[:2]
        realtime_broadcaster.publish(
            "tracks.update",
            {
                "camera_id": camera_id,
                "active": True,
                "frame_width": int(width),
                "frame_height": int(height),
                "keypoint_min_confidence": float(settings.POSE_KEYPOINT_MIN_CONFIDENCE),
                "pose_format": pose_format(),
                "persons": [serialize_person(person, now) for person in scene.current],
                "timestamp": datetime.now(timezone.utc),
            },
        )

    def _broadcast_tracks_cleared(self, camera_id):
        try:
            realtime_broadcaster.publish(
                "tracks.update",
                {"camera_id": camera_id, "active": False, "persons": [], "timestamp": datetime.now(timezone.utc)},
            )
        except Exception:
            logger.exception("Could not clear live tracks for camera %s", camera_id)

    def _broadcast_fall_bbox(self, camera_id, frame, fall_event: dict, scene=None):
        """Publishes the box a fall fired on to any connected live-view
        client, over the same realtime channel as `alert.created`/
        `camera.status` (shared infrastructure - every authenticated user
        may watch a camera's stream, so no owner_user_id scoping).

        The box is exactly `fall_event["bbox"]` - the region the gate in
        detection/fall_detection.py already decided on, in this frame's
        own pixel coordinates - plus the frame's dimensions so a client can
        scale it onto whatever size it is displaying the MJPEG stream at.
        Nothing is computed or inferred here; a fall event with no bbox
        (should not happen for either strategy - see fall_detection.py)
        simply isn't drawable, so it's skipped rather than guessed at.

        One exception: an event held by pose verification is released up
        to FALL_POSE_CONFIRM_SECONDS after it fired, and its fire-time box
        drawn on the live view then hangs where the person USED to be. When
        its attributed person is still tracked, their current box is drawn
        instead."""
        bbox = fall_event.get("bbox")
        if "fired_at" in fall_event and scene is not None:
            person_id = fall_event.get("person_track_id")
            for person in scene.current:
                if person.track_id == person_id and person.state != STATE_LOST:
                    bbox = person.bbox
                    break
        if bbox is None:
            return
        height, width = frame.shape[:2]
        realtime_broadcaster.publish(
            "fall.bbox",
            {
                "camera_id": camera_id,
                "bbox": [float(v) for v in bbox],
                "frame_width": int(width),
                "frame_height": int(height),
                "confidence": fall_event.get("confidence", 1.0),
                "track_id": fall_event.get("track_id"),
                "person_track_id": fall_event.get("person_track_id"),
                "pose": fall_event.get("pose"),
                "detector": fall_event.get("detector", "heuristic"),
                "timestamp": datetime.now(timezone.utc),
            },
        )

    # -- model inference helpers -----------------------------------------

    def extract_people(self, frame):
        """Every person in `frame` - box, box confidence and 17 COCO
        keypoints - from one pose-model pass. [] when the pose model is
        unavailable or finds nobody."""
        if self._pose_model is None:
            return []
        with self._pose_inference_lock:
            results = self._pose_model.predict(
                frame, device=settings.MODEL_DEVICE, imgsz=settings.POSE_INFERENCE_IMGSZ, verbose=False,
            )
        return parse_pose_results(results)

    def extract_objects(self, frame):
        person_boxes = []
        other_objects = []
        if self._object_model is None:
            return person_boxes, other_objects
        with self._object_inference_lock:
            results = self._object_model.predict(frame, device=settings.MODEL_DEVICE, verbose=False)
        if not results:
            return person_boxes, other_objects
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return person_boxes, other_objects

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, classes, confs):
            bbox = tuple(box.tolist())
            if int(cls_id) == PERSON_CLASS_ID:
                person_boxes.append(bbox)
            else:
                other_objects.append((int(cls_id), float(conf), bbox))
        return person_boxes, other_objects


def parse_pose_results(results) -> list:
    """Converts an Ultralytics pose `Results` list into PersonDetections.

    Tolerates every shape a real result can take - no results, no boxes,
    boxes without keypoints - by returning [] rather than raising, so a
    frame with nobody in it is just an empty frame."""
    if not results:
        return []
    result = results[0]
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy() if getattr(result.boxes, "conf", None) is not None else None
    keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3): x, y, conf
    frame_shape = getattr(result, "orig_shape", None)  # (height, width)

    people = []
    for index, (box, kps) in enumerate(zip(boxes, keypoints)):
        confidence = float(confidences[index]) if confidences is not None else None
        people.append(PersonDetection(
            bbox=tuple(box.tolist()),
            keypoints=[mark_missing_keypoint(tuple(p), frame_shape) for p in kps.tolist()],
            confidence=confidence,
        ))
    return people


def mark_missing_keypoint(point, frame_shape=None):
    """Zeroes the confidence of a keypoint the model placed on the frame's
    edge or outside it.

    The pose model reports a keypoint it could not place (typically a limb
    outside the frame) at coordinate 0 - sometimes with a confidence above
    any sensible floor, which drew a "limb" from the person to the frame's
    corner. Ultralytics' own plotter skips exactly these: a coordinate that
    is negative or an exact multiple of the frame size
    (`Annotator.kpts`: `x % width != 0 and y % height != 0`). Same rule here,
    applied once where keypoints enter the system so the overlay, the
    tracker's posture evidence and the pose heuristic all agree."""
    x, y = point[0], point[1]
    conf = point[2] if len(point) > 2 else 1.0
    missing = x <= 0 or y <= 0
    if not missing and frame_shape is not None and len(frame_shape) >= 2:
        height, width = frame_shape[0], frame_shape[1]
        missing = (width and x % width == 0) or (height and y % height == 0)
    return (x, y, 0.0 if missing else conf)


def serialize_person(person, now: Optional[float] = None) -> dict:
    """A TrackedPerson as sent to clients: compact (1 px / 0.01 precision),
    every keypoint kept with its confidence so the client can apply the
    visibility floor it is told about."""
    return {
        "track_id": person.track_id,
        "state": person.state,
        "bbox": [round(float(v), 1) for v in person.bbox],
        "confidence": None if person.confidence is None else round(float(person.confidence), 2),
        "keypoints": [[round(x, 1), round(y, 1), round(c, 2)] for x, y, c in person.keypoints],
        "torso_angle_deg": None if person.torso_angle is None else round(person.torso_angle, 1),
        "duration_seconds": round(person.duration_seconds, 2),
        "seconds_since_seen": None if now is None else round(max(0.0, now - person.last_seen), 2),
    }


detection_engine = DetectionEngine()
