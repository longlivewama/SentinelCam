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
import statistics
import time
from typing import Dict, List, Optional

from app.config import settings
from app.services.detection.fall_classifier import fall_classifier
from app.services.detection.fall_detection import FallDetector, ModelFallDetector
from app.services.detection.fall_object_detector import fall_object_detector
from app.services.detection.person_tracker import PersonTracker
from app.services.detection.scene_tracking import SceneTracker

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
    across concurrent video sources.

    Optionally owns a `SceneTracker` (pose + multi-person tracking, see
    detection/scene_tracking.py). With one, every event the gates fire is
    attributed to the tracked person it belongs to (`person_track_id`,
    None when no person can be credibly named) and carries the pose
    evidence measured for that person (`pose`). Two further steps can act
    on those already-confirmed events (pose verification on by default,
    track dedupe off), both evaluated in the README before those defaults
    were chosen:

      pose verification  an event whose person was measured upright over
                         the sustain window is held, not vetoed: it is
                         released as soon as that person's posture turns
                         non-upright (or their track ends), and withheld
                         only if they are still measured upright when
                         FALL_POSE_CONFIRM_SECONDS runs out
                         (FALL_POSE_VERIFICATION)
      track dedupe       one event per tracked person per window, on top
                         of the gate's own per-box debounce
                         (FALL_TRACK_DEDUPE)

    Neither can create an event, and with neither enabled the events are
    exactly the ones the gates produced, plus the two annotation keys.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        scene: "SceneTracker | None | bool" = True,
        pose_verification: Optional[bool] = None,
        track_dedupe: Optional[bool] = None,
        fall_detector=None,
    ):
        self.mode = mode or resolve_mode()
        self._heuristic = (
            FallDetector() if self.mode in (MODE_HEURISTIC, MODE_HYBRID) else None
        )
        self._model_gate = (
            ModelFallDetector(
                min_sustained_seconds=settings.FALL_DETECTOR_MIN_SUSTAINED_SECONDS,
                min_consecutive_frames=settings.FALL_DETECTOR_MIN_CONSECUTIVE_FRAMES,
            )
            if self.mode in (MODE_MODEL, MODE_HYBRID)
            else None
        )
        # Injectable so the offline evaluation can replay cached detector
        # output through this exact class; production always uses the
        # shared trained detector.
        self._fall_detector = fall_detector or fall_object_detector
        # Every `Fall` box the trained detector produced on the most
        # recent update(), whether or not it fired an event. Published
        # for the clip annotation layer (detection/fall_annotation.py),
        # which needs the detector's per-frame regions and must not pay
        # for a second inference pass to get them. Empty in heuristic
        # mode, where there is no such model. Read-only for callers -
        # nothing here feeds back into a detection decision.
        self.last_fall_boxes: tuple = ()

        # `True` builds a tracker when PERSON_TRACKING_ENABLED is set, an
        # explicit SceneTracker is used as given (the upload analyser hands
        # in the one its clip annotator follows), and None/False disables
        # tracking for this pipeline.
        if scene is True:
            scene = build_scene_tracker() if settings.PERSON_TRACKING_ENABLED else None
        self.scene: Optional[SceneTracker] = scene or None

        self.pose_verification = (
            settings.FALL_POSE_VERIFICATION if pose_verification is None else bool(pose_verification)
        )
        self.track_dedupe = settings.FALL_TRACK_DEDUPE if track_dedupe is None else bool(track_dedupe)
        self._last_event_time_by_track: Dict[int, float] = {}
        self._frame_counter = 0
        # Events held by pose verification awaiting post-event posture:
        # [{"event": dict, "fired_at": float}]. See _resolve_pending.
        self._pending: List[dict] = []

        # Events the gates confirmed but an assist step withheld, on the
        # most recent update() - kept so the decision is reviewable rather
        # than silent.
        self.last_suppressed_events: List[dict] = []

    @property
    def needs_pose(self) -> bool:
        """Whether this pipeline cannot work without the pose model: the
        heuristic consumes keypoints directly. Everything else here only
        benefits from it."""
        return self._heuristic is not None

    def update(self, frame, people, now: Optional[float] = None, frame_index: Optional[int] = None) -> List[dict]:
        """Runs whichever strategies are enabled over one processed frame
        and returns the fall events they produced. `people` is the pose
        model's output for this frame (reused rather than re-inferred);
        `frame` is only touched when the trained detector is enabled.

        Every returned event carries a `detector` key ("model" or
        "heuristic") identifying which strategy fired it, so the alert
        that reaches the operator says what actually made the call."""
        if now is None:
            # Live cameras run on wall clock. Resolved once here so the
            # gates and the tracker agree on the frame's time.
            now = time.time()
        if frame_index is None:
            frame_index = self._frame_counter
        self._frame_counter = frame_index + 1

        events: List[dict] = []
        self.last_fall_boxes = ()
        self.last_suppressed_events = []

        if self._model_gate is not None:
            detections = self._fall_detector.detect(frame)
            self.last_fall_boxes = tuple(detections)
            events.extend(self._model_gate.update(detections, now=now))

        if self._heuristic is not None:
            classifier_scores = None
            if fall_classifier.is_available:
                frame_height, frame_width = frame.shape[:2]
                classifier_scores = fall_classifier.score_people(people, frame_width, frame_height)
            for event in self._heuristic.update(people, now=now, classifier_scores=classifier_scores):
                event.setdefault("detector", "heuristic")
                events.append(event)

        if self.scene is None:
            return events

        # Tracking must see this frame before its events are attributed,
        # so an event is matched against the people visible when it fired.
        try:
            self.scene.update(people or (), now, frame_index)
            self.scene.record_fall_evidence(self.last_fall_boxes, frame_index)
        except Exception:
            # Tracking is an annotation of the gates' output, never a
            # precondition for it: a tracker failure costs this frame its
            # attribution, not its fall events.
            logger.exception("Person tracking failed on frame %s; events kept unattributed", frame_index)
            for event in events:
                event.setdefault("person_track_id", None)
            return events

        kept: List[dict] = []
        # Earlier held events first: they were confirmed by the gate before
        # anything firing on this frame.
        for event in self._resolve_pending(now, final=False):
            self._admit(event, now, kept)
        for event in events:
            self._attribute(event)
            if self._should_hold(event):
                # Recorded on the event itself: a held event is returned on
                # a LATER frame, and consumers that place it in the footage
                # (the upload analyser's timestamp, clip and overlay) need
                # the moment the gate fired, not the moment it was released.
                event["fired_at"] = now
                event["fired_frame_index"] = frame_index
                self._pending.append({"event": event, "fired_at": now})
                continue
            self._admit(event, now, kept)
        return kept

    def flush(self) -> List[dict]:
        """Decides every event still held by pose verification, for a
        source that has ended (the last frame of an uploaded video). One
        whose window had not run out is released, not dropped. Returns the released events; withheld ones are on
        `last_suppressed_events`, as after update()."""
        self.last_suppressed_events = []
        now = self.scene.last_time if self.scene is not None else None
        kept: List[dict] = []
        for event in self._resolve_pending(now, final=True):
            self._admit(event, now if now is not None else 0.0, kept)
        return kept

    def _admit(self, event: dict, now: float, kept: List[dict]) -> None:
        reason = self._suppression_reason(event, now)
        if reason is None:
            track_id = event.get("person_track_id")
            if track_id is not None:
                self._last_event_time_by_track[track_id] = now
            kept.append(event)
        else:
            self._withhold(event, reason, now)

    def _withhold(self, event: dict, reason: str, now: Optional[float]) -> None:
        event["suppressed_reason"] = reason
        self.last_suppressed_events.append(event)
        logger.info(
            "Fall event at t=%s withheld (%s): person track %s, pose %s",
            "?" if now is None else f"{now:.2f}", reason, event.get("person_track_id"), event.get("pose"),
        )

    # -- pose / tracking assistance ---------------------------------------

    def _attribute(self, event: dict) -> None:
        """Names the tracked person the event belongs to - through
        `SceneTracker.attribute`, the same rule the clip annotation uses
        (containment + IoU + centre + continuity, refusing when ambiguous) -
        and records the pose evidence for that person. Never guesses: no
        box or no confident match => `person_track_id` is None."""
        event["person_track_id"] = None
        evidence = event.get("bbox")
        if evidence is None:
            event["pose"] = {"verdict": POSE_NO_PERSON, "torso_angle_deg": None, "samples": 0}
            return

        match = self.scene.attribute(evidence, event.get("sustained_seconds"))
        if match is None:
            event["pose"] = {"verdict": POSE_NO_PERSON, "torso_angle_deg": None, "samples": 0}
            return

        event["person_track_id"] = match.track_id
        window = max(float(settings.FALL_DETECTOR_MIN_SUSTAINED_SECONDS), 0.0) + POSE_WINDOW_MARGIN_SECONDS
        angles = self.scene.recent_torso_angles(match.track_id, window)
        if len(angles) < POSE_MIN_SAMPLES:
            event["pose"] = {"verdict": POSE_NO_EVIDENCE, "torso_angle_deg": None, "samples": len(angles)}
            return
        median = statistics.median(angles)
        upright = median < float(settings.FALL_POSE_UPRIGHT_MAX_DEGREES)
        event["pose"] = {
            "verdict": POSE_UPRIGHT if upright else POSE_NOT_UPRIGHT,
            "torso_angle_deg": round(median, 1),
            "samples": len(angles),
        }

    def _should_hold(self, event: dict) -> bool:
        return (
            self.pose_verification
            and float(settings.FALL_POSE_CONFIRM_SECONDS) > 0
            and event.get("person_track_id") is not None
            and (event.get("pose") or {}).get("verdict") == POSE_UPRIGHT
        )

    def _resolve_pending(self, now: Optional[float], final: bool) -> List[dict]:
        """Releases or withholds held events from the posture their person
        showed AFTER the gate fired. The detector typically fires within
        ~0.6 s of a fall starting, so the pre-event window can show a
        person still standing; what separates a fall from someone walking
        is what their posture does next.

          released  the last POSE_MIN_SAMPLES post-event measurements are
                    all non-upright, or the track ended (a person lying on
                    the floor is often lost by the pose model)
          withheld  the window elapsed with at least POSE_MIN_SAMPLES
                    post-event measurements whose median is upright
          released  the window elapsed without enough measurements, or the
                    source ended first - no evidence is never grounds for
                    a veto"""
        if not self._pending:
            return []
        threshold = float(settings.FALL_POSE_UPRIGHT_MAX_DEGREES)
        window = float(settings.FALL_POSE_CONFIRM_SECONDS)
        released: List[dict] = []
        still: List[dict] = []
        for held in self._pending:
            event, fired_at = held["event"], held["fired_at"]
            angles = self.scene.torso_angles_after(event["person_track_id"], fired_at)
            waited = None if now is None else round(max(now - fired_at, 0.0), 2)
            recent = (angles or [])[-POSE_MIN_SAMPLES:]
            if angles is None:
                outcome = "released_track_ended"
            elif len(recent) >= POSE_MIN_SAMPLES and all(a >= threshold for a in recent):
                outcome = "released_not_upright"
            elif now is not None and now - fired_at >= window:
                if len(angles) >= POSE_MIN_SAMPLES and statistics.median(angles) < threshold:
                    outcome = "withheld_upright"
                else:
                    outcome = "released_insufficient_evidence"
            elif final:
                # The source ended before the window did: a shorter look
                # than the window promises is not grounds for a veto.
                outcome = "released_source_ended"
            else:
                still.append(held)
                continue
            event["pose"] = dict(event.get("pose") or {}, confirmation={
                "outcome": outcome,
                "waited_seconds": waited,
                "post_event_samples": len(angles or []),
                "post_event_torso_angle_deg": (
                    round(statistics.median(angles), 1) if angles else None
                ),
            })
            if outcome == "withheld_upright":
                self._withhold(event, "pose_upright", now)
            else:
                released.append(event)
        self._pending = still
        return released

    def _suppression_reason(self, event: dict, now: float) -> Optional[str]:
        if (
            self.pose_verification
            and float(settings.FALL_POSE_CONFIRM_SECONDS) <= 0
            and (event.get("pose") or {}).get("verdict") == POSE_UPRIGHT
        ):
            return "pose_upright"
        track_id = event.get("person_track_id")
        if self.track_dedupe and track_id is not None:
            last = self._last_event_time_by_track.get(track_id)
            if last is not None and (now - last) <= float(settings.FALL_TRACK_DEDUPE_SECONDS):
                return "duplicate_for_track"
        return None


# Pose evidence verdicts attached to every attributed event.
POSE_UPRIGHT = "upright"
POSE_NOT_UPRIGHT = "not_upright"
POSE_NO_EVIDENCE = "no_pose_evidence"
POSE_NO_PERSON = "no_person_association"

# The posture window is the gate's own sustain window plus this margin, so
# it covers the frames that actually led to the event.
POSE_WINDOW_MARGIN_SECONDS = 0.4
# Fewer torso measurements than this is not evidence either way.
POSE_MIN_SAMPLES = 2


def build_scene_tracker(history_length: Optional[int] = None) -> SceneTracker:
    """A SceneTracker configured from settings. `history_length` sizes the
    per-track box history, which the clip annotator needs to cover its
    pre-event window."""
    kwargs = {"max_age_seconds": float(settings.TRACKING_MAX_AGE_SECONDS)}
    if history_length is not None:
        kwargs["history_length"] = history_length
    return SceneTracker(
        tracker=PersonTracker(**kwargs),
        min_hits=int(settings.TRACKING_MIN_HITS),
        keypoint_min_confidence=float(settings.POSE_KEYPOINT_MIN_CONFIDENCE),
        max_age_seconds=float(settings.TRACKING_MAX_AGE_SECONDS),
    )
