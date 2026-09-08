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
    detection) and the shared object model once (feeds both crowd +
    abandoned-object detection) per processed frame - plus, when the
    trained fall detector is enabled (see detection/fall_pipeline.py), a
    third shared model that produces the fall signal directly,
  - feeds the results to each of the four detector modules,
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
from typing import Dict

from ultralytics import YOLO

from app.config import settings
from app.database import SessionLocal
from app.models.camera import Camera
from app.services import recording_engine
from app.services.stream_manager import stream_manager
from app.services.detection.fall_pipeline import FallPipeline
from app.services.detection.violence_detection import ViolenceDetector
from app.services.detection.crowd_detection import CrowdDetector
from app.services.detection.abandoned_object_detection import AbandonedObjectDetector

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO class index for "person", shared by both pose and object models

# Shared shape consumed by both FallDetector and ViolenceDetector.
PersonDetection = namedtuple("PersonDetection", ["bbox", "keypoints"])


class DetectionEngine:
    def __init__(self):
        self._pose_model = None
        self._object_model = None
        self._model_lock = threading.Lock()
        self._threads: Dict[int, threading.Thread] = {}
        self._stop_flags: Dict[int, threading.Event] = {}

    @property
    def models_loaded(self) -> bool:
        return self._pose_model is not None and self._object_model is not None

    @property
    def active_camera_count(self) -> int:
        return sum(1 for t in self._threads.values() if t.is_alive())

    # -- model loading (once, shared across all cameras) -----------------

    def ensure_models(self):
        if self._pose_model is not None and self._object_model is not None:
            return
        with self._model_lock:
            if self._pose_model is None:
                logger.info("Loading pose model %s", settings.YOLO_POSE_MODEL)
                self._pose_model = YOLO(settings.YOLO_POSE_MODEL)
            if self._object_model is None:
                logger.info("Loading object model %s", settings.YOLO_MODEL)
                self._object_model = YOLO(settings.YOLO_MODEL)

    # -- lifecycle ---------------------------------------------------------

    def ensure_running(self, camera_id: int, url: str):
        """Idempotent: safe to call repeatedly (e.g. on every camera
        update). Starts a detection thread only if one isn't already
        running for this camera."""
        existing = self._threads.get(camera_id)
        if existing is not None and existing.is_alive():
            return
        stream_manager.get_or_create(camera_id, url)
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
        stream = stream_manager.get(camera_id)
        if stream is None:
            logger.error("No stream for camera %s; detection loop exiting", camera_id)
            return

        fall_pipeline = FallPipeline()
        logger.info("Camera %s: fall detection running in %r mode", camera_id, fall_pipeline.mode)
        violence_detector = ViolenceDetector()
        crowd_detector = CrowdDetector()
        abandoned_detector = AbandonedObjectDetector()

        frame_index = 0
        poll_interval = 1.0 / max(settings.STREAM_FPS, 1)

        while not stop_flag.is_set():
            with SessionLocal() as db:
                camera = db.query(Camera).filter(Camera.id == camera_id).first()
                if camera is None or not camera.is_active or not camera.ai_detection_enabled:
                    logger.info("Camera %s detection disabled/removed; stopping detection loop", camera_id)
                    return
                crowd_threshold = camera.crowd_threshold
                abandoned_object_seconds = camera.abandoned_object_seconds

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
                    crowd_detector, abandoned_detector,
                    crowd_threshold, abandoned_object_seconds,
                )
            except Exception:
                logger.exception("Detection error on camera %s", camera_id)

            time.sleep(poll_interval)

    def _process_frame(
        self, camera_id, frame, fall_pipeline, violence_detector,
        crowd_detector, abandoned_detector, crowd_threshold, abandoned_object_seconds,
    ):
        people = self.extract_people(frame)

        for fall_event in fall_pipeline.update(frame, people):
            logger.info("Camera %s: FALL detected %s", camera_id, fall_event)
            recording_engine.trigger_event(
                camera_id, "fall", fall_event.get("confidence", 1.0),
                detector=fall_event.get("detector", "heuristic"),
            )

        for violence_event in violence_detector.update(people):
            logger.info("Camera %s: VIOLENCE detected %s", camera_id, violence_event)
            recording_engine.trigger_event(camera_id, "violence", violence_event.get("confidence", 1.0))

        person_boxes, other_objects = self.extract_objects(frame)

        crowd_event = crowd_detector.update(len(person_boxes), crowd_threshold)
        if crowd_event:
            logger.info("Camera %s: CROWD detected %s", camera_id, crowd_event)
            recording_engine.trigger_event(camera_id, "crowd", crowd_event.get("confidence", 1.0))

        for abandoned_event in abandoned_detector.update(other_objects, person_boxes, abandoned_object_seconds):
            logger.info("Camera %s: ABANDONED_OBJECT detected %s", camera_id, abandoned_event)
            recording_engine.trigger_event(camera_id, "abandoned_object", abandoned_event.get("confidence", 1.0))

    # -- model inference helpers -----------------------------------------

    def extract_people(self, frame):
        results = self._pose_model.predict(frame, device=settings.MODEL_DEVICE, verbose=False)
        if not results:
            return []
        result = results[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()  # (N, 17, 3): x, y, conf

        people = []
        for box, kps in zip(boxes, keypoints):
            people.append(PersonDetection(bbox=tuple(box.tolist()), keypoints=[tuple(p) for p in kps.tolist()]))
        return people

    def extract_objects(self, frame):
        results = self._object_model.predict(frame, device=settings.MODEL_DEVICE, verbose=False)
        person_boxes = []
        other_objects = []
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


detection_engine = DetectionEngine()
