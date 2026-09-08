"""
Single entry point both fall-detection callers use - the per-camera loop
in `engine.py` and the offline upload analyser in `video_analysis.py` -
so the choice between the trained detector and the pose heuristic is made
in exactly one place instead of being duplicated (and drifting) at each
call site.

Strategies, selected by `FALL_DETECTION_MODE`:

  auto (default) The trained YOLO "Fall" detector when it loads,
                 otherwise the pose heuristic. Degrades gracefully: a
                 deployment without the checkpoint still detects falls.
  model          Trained detector only. If it cannot be loaded, no fall
                 events are produced at all - use when a silent fallback
                 to the weaker heuristic would be worse than an outage
                 you can see in /api/system/status.
  heuristic      Pose heuristic only, ignoring the trained detector.
  hybrid         Both, reporting the union of their events. Higher recall
                 at the cost of the heuristic's false-positive rate; the
                 debounce in each gate keeps a single fall from being
                 double-reported by the same strategy, but a fall both
                 strategies see can still produce two events.

The mode is resolved once per process (at first use) rather than per
frame, so `active_mode` is stable for the lifetime of the app and can be
reported to operators.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.config import settings
from app.services.detection.fall_classifier import fall_classifier
from app.services.detection.fall_detection import FallDetector, ModelFallDetector
from app.services.detection.fall_object_detector import fall_object_detector

logger = logging.getLogger(__name__)

MODE_AUTO = "auto"
MODE_MODEL = "model"
MODE_HEURISTIC = "heuristic"
MODE_HYBRID = "hybrid"
VALID_MODES = (MODE_AUTO, MODE_MODEL, MODE_HEURISTIC, MODE_HYBRID)


def resolve_mode() -> str:
    """The strategy actually in effect, with `auto` collapsed to the
    concrete choice. An unrecognised configured value falls back to
    `auto` with a warning rather than crashing the app on boot."""
    configured = (settings.FALL_DETECTION_MODE or MODE_AUTO).strip().lower()
    if configured not in VALID_MODES:
        logger.warning(
            "FALL_DETECTION_MODE=%r is not one of %s; using %r",
            settings.FALL_DETECTION_MODE, list(VALID_MODES), MODE_AUTO,
        )
        configured = MODE_AUTO

    if configured != MODE_AUTO:
        return configured
    return MODE_MODEL if fall_object_detector.is_available else MODE_HEURISTIC


class FallPipeline:
    """One instance per camera detection loop or per video-analysis job -
    it holds per-subject tracking state, so instances must not be shared
    across concurrent video sources."""

    def __init__(self, mode: Optional[str] = None):
        self.mode = mode or resolve_mode()
        self._heuristic = (
            FallDetector() if self.mode in (MODE_HEURISTIC, MODE_HYBRID) else None
        )
        self._model_gate = (
            ModelFallDetector(min_sustained_seconds=settings.FALL_DETECTOR_MIN_SUSTAINED_SECONDS)
            if self.mode in (MODE_MODEL, MODE_HYBRID)
            else None
        )

    @property
    def uses_trained_model(self) -> bool:
        return self._model_gate is not None

    def update(self, frame, people, now: Optional[float] = None) -> List[dict]:
        """Runs whichever strategies are enabled over one processed frame
        and returns the fall events they produced. `people` is the pose
        model's output for this frame (reused rather than re-inferred);
        `frame` is only touched when the trained detector is enabled.

        Every returned event carries a `detector` key ("model" or
        "heuristic") identifying which strategy fired it, so the alert
        that reaches the operator says what actually made the call."""
        events: List[dict] = []

        if self._model_gate is not None:
            detections = fall_object_detector.detect(frame)
            events.extend(self._model_gate.update(detections, now=now))

        if self._heuristic is not None:
            classifier_scores = None
            if fall_classifier.is_available:
                frame_height, frame_width = frame.shape[:2]
                classifier_scores = fall_classifier.score_people(people, frame_width, frame_height)
            for event in self._heuristic.update(people, now=now, classifier_scores=classifier_scores):
                event.setdefault("detector", "heuristic")
                events.append(event)

        return events
