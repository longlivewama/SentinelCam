"""
Loader + inference wrapper for the fine-tuned single-class YOLO "Fall"
detector (`ml/exported/fall_detector_v1.pt`, produced by
`ml/detector/train.py` - see `ml/MODEL_CARD.md` for the dataset, training
configuration and measured metrics).

This is the *primary* fall signal, and it is a different kind of model
from the one in `fall_classifier.py`: that one is a small keypoint MLP
that only ever nudges the confidence of an event the pose heuristic
already decided to fire, whereas this one detects fallen people directly
and, when loaded, decides on its own whether a frame contains a fall.

Responsibilities are deliberately narrow: load the checkpoint once,
run it on a frame, return boxes above a confidence floor. All temporal
reasoning (the sustained-duration gate and per-subject debounce that
turn per-frame boxes into events) lives in
`fall_detection.ModelFallDetector`, so it is testable without torch.

Fails safe everywhere: a missing file, missing dependency, or an
inference error yields `is_available=False` / an empty detection list
rather than raising, so a misconfigured model can never take down live
camera detection or an upload analysis.
"""
from __future__ import annotations

import logging
import threading
from collections import namedtuple
from pathlib import Path
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# One detected fallen subject. `bbox` is absolute-pixel (x1, y1, x2, y2)
# in the frame's own coordinates, matching engine.PersonDetection.bbox.
FallBox = namedtuple("FallBox", ["bbox", "confidence"])

# The trained detector is single-class; index 0 is "Fall". Asserted
# against the checkpoint's own `names` at load time rather than assumed,
# so swapping in a differently-trained checkpoint fails loudly at startup
# instead of silently reporting the wrong class as a fall.
EXPECTED_CLASS_NAMES = {0: "Fall"}


class FallObjectDetector:
    def __init__(self):
        self._model = None
        self._load_attempted = False
        self._lock = threading.Lock()
        self._model_path: Optional[str] = None
        self._class_names: Optional[dict] = None
        self._load_error: Optional[str] = None

    # -- lifecycle -------------------------------------------------------

    @property
    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        """Human-readable reason the model isn't loaded, for /api/system/status."""
        self._ensure_loaded()
        return self._load_error

    @property
    def model_info(self) -> dict:
        self._ensure_loaded()
        return {
            "configured_path": settings.FALL_DETECTOR_MODEL_PATH or None,
            "loaded": self._model is not None,
            "class_names": self._class_names,
            "min_confidence": settings.FALL_DETECTOR_MIN_CONFIDENCE,
            "error": self._load_error,
        }

    def _ensure_loaded(self):
        if self._load_attempted:
            return
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True

            model_path = settings.FALL_DETECTOR_MODEL_PATH
            if not model_path:
                self._load_error = "FALL_DETECTOR_MODEL_PATH is not set"
                return

            path = Path(model_path)
            if not path.exists():
                self._load_error = f"model file not found: {model_path}"
                logger.warning(
                    "FALL_DETECTOR_MODEL_PATH=%s does not exist; falling back to the pose heuristic",
                    model_path,
                )
                return

            try:
                from ultralytics import YOLO
            except ImportError:
                self._load_error = "ultralytics is not installed"
                logger.warning("ultralytics not installed; trained fall detector disabled")
                return

            try:
                model = YOLO(str(path))
                names = dict(getattr(model, "names", {}) or {})
            except Exception as exc:
                self._load_error = f"failed to load checkpoint: {exc}"
                logger.exception("Failed to load trained fall detector from %s", model_path)
                return

            if names != EXPECTED_CLASS_NAMES:
                # Refuse rather than guess: mapping the wrong class index
                # to "fall" would produce confidently wrong alerts.
                self._load_error = (
                    f"unexpected class names {names}; expected {EXPECTED_CLASS_NAMES}"
                )
                logger.error(
                    "Trained fall detector at %s has class names %s, expected %s; detector disabled",
                    model_path, names, EXPECTED_CLASS_NAMES,
                )
                return

            self._model = model
            self._model_path = str(path)
            self._class_names = names
            self._load_error = None
            logger.info("Loaded trained fall detector from %s (classes=%s)", model_path, names)

    # -- inference -------------------------------------------------------

    def detect(self, frame) -> List[FallBox]:
        """Returns every "Fall" box in `frame` above
        FALL_DETECTOR_MIN_CONFIDENCE, highest confidence first. Empty list
        if the model isn't loaded or inference fails - never raises."""
        if not self.is_available:
            return []

        try:
            results = self._model.predict(
                frame,
                device=settings.MODEL_DEVICE,
                conf=settings.FALL_DETECTOR_MIN_CONFIDENCE,
                verbose=False,
            )
        except Exception:
            logger.exception("Trained fall detector inference failed; treating frame as no-detection")
            return []

        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        try:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
        except Exception:
            logger.exception("Could not read boxes off the trained fall detector's result")
            return []

        detections = [
            FallBox(bbox=tuple(float(v) for v in box), confidence=float(conf))
            for box, conf in zip(xyxy, confs)
        ]
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


fall_object_detector = FallObjectDetector()
