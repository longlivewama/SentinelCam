"""
Fall-event gating. Two strategies live here, both turning per-frame
evidence into debounced, sustained-duration-gated fall events:

  * `FallDetector`     - the pose-keypoint heuristic documented below.
  * `ModelFallDetector` - the gate for the fine-tuned YOLO "Fall"
                          detector (bottom of this file).

Neither loads a model: both take already-extracted per-frame detections,
which keeps the temporal logic - the part that decides whether an alert
is raised - unit-testable without torch. `detection/fall_pipeline.py`
picks between them based on FALL_DETECTION_MODE.

--- The pose heuristic ---

Fall detection using YOLOv8-Pose keypoints (17 COCO keypoints).

Signals evaluated per detected person, each processed frame:
  (a) bounding-box aspect ratio (width/height) > 1.3 -> a "wide" box,
      since a fallen body tends to lie horizontally rather than vertically.
  (b) vertical alignment of shoulder (5,6) vs hip (11,12) keypoints - a
      standing person has shoulder_y clearly above hip_y; a fallen person
      has shoulder_y and hip_y bunched into roughly the same horizontal
      band.
  (c) downward velocity of the hip midpoint > 20 px/processed-frame,
      tracked across consecutive processed frames per person via a
      lightweight nearest-centroid tracker (no external tracking library -
      good enough for single-camera frame-to-frame association).

Why a single frame isn't enough (false-positive fix): signals (a) and (b)
alone are a snapshot of *posture*, and plenty of ordinary movements produce
that exact snapshot for a fraction of a second - bending down to pick
something up, sitting down quickly, tying a shoe, stretching. The original
implementation fired the instant any single processed frame had >=2/3
signals true, which is exactly what made those transient posture changes
read as falls.

The fix: a "fall" requires the *sustained on-ground posture* (signals (a)
AND (b) both true, i.e. lying-down aspect ratio AND bunched shoulder/hip)
to hold continuously, with no recovery gap, for >= FALL_MIN_SUSTAINED_SECONDS
of wall-clock time for the same tracked person. A person who bends over
and immediately stands back up never accumulates that sustained duration,
so it's never flagged; a person who actually goes down and stays down
is. The downward-velocity signal (c) is inherently transient (it's a
one-frame derivative) so it isn't part of the sustain gate - it's kept
only as a confidence contributor for the eventual event.

A debounce window additionally prevents re-firing (and re-recording) every
frame while the person remains on the ground after the first firing.

Callers (detection/engine.py) construct one FallDetector per camera (or
per video-upload analysis job) and call update() once per processed frame
with the list of people detected in that frame.
"""
from __future__ import annotations

import time
from collections import namedtuple
from typing import Dict, List, Optional

# Duck-typed: any object with .bbox (x1, y1, x2, y2) and .keypoints
# (17 (x, y, conf) tuples in COCO order) works here.
PersonDetection = namedtuple("PersonDetection", ["bbox", "keypoints"])

FALL_DEBOUNCE_SECONDS = 10.0
ASPECT_RATIO_THRESHOLD = 1.3
ALIGNMENT_BAND_PX = 40.0            # shoulder/hip considered "bunched" within this many px
VERTICAL_VELOCITY_THRESHOLD = 20.0  # px/processed-frame, downward
MATCH_DISTANCE_PX = 80.0            # max hip movement between frames to count as same person
STALE_SECONDS = 5.0                 # drop tracks not seen for this long

# Minimum continuous wall-clock time the on-ground posture (aspect ratio +
# shoulder/hip bunching) must hold, uninterrupted, before it counts as a
# fall rather than a transient posture change. This is the main
# false-positive guard - see module docstring.
FALL_MIN_SUSTAINED_SECONDS = 1.2

# Threshold above which the optional trained classifier (see
# fall_classifier.py) is considered to "corroborate" a heuristic-detected
# fall. Only used to adjust confidence, never to gate firing - see
# update()'s docstring.
CLASSIFIER_CORROBORATION_THRESHOLD = 0.5

KP_SHOULDERS = (5, 6)
KP_HIPS = (11, 12)
KP_KNEES = (13, 14)

MIN_KEYPOINT_CONF = 0.3


def _avg_point(keypoints, indices, min_conf: float = MIN_KEYPOINT_CONF) -> Optional[tuple]:
    pts = [(kp[0], kp[1]) for i in indices if (kp := keypoints[i])[2] >= min_conf]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


class _Track:
    __slots__ = ("hip", "last_seen", "last_fall_time", "on_ground_since")

    def __init__(self, hip, now: float):
        self.hip = hip
        self.last_seen = now
        # None (not 0.0) means "never fired yet" - using 0.0 as that
        # sentinel only worked by accident when `now` came from
        # time.time() (always far more than FALL_DEBOUNCE_SECONDS past
        # epoch 0); it silently breaks for video-upload analysis, which
        # times frames from 0 at the start of each video.
        self.last_fall_time: Optional[float] = None
        self.on_ground_since: Optional[float] = None


class FallDetector:
    """Stateful, per-camera (or per-video-analysis-job). Instantiate once
    and call update() once per processed frame."""

    def __init__(self):
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0

    def update(
        self,
        people: List[PersonDetection],
        now: Optional[float] = None,
        classifier_scores: Optional[List[float]] = None,
    ) -> List[dict]:
        """`now` is injectable (as a monotonic-ish timestamp) so offline
        video analysis can drive the sustain timer using the video's own
        timeline instead of wall-clock time; defaults to time.time() for
        live camera use.

        `classifier_scores`, if given, is a list of P(fall) aligned
        index-for-index with `people` (see
        detection/fall_classifier.py) - an optional trained-model signal
        that only ever adjusts the reported confidence of an event this
        heuristic already decided to fire. It can never make this method
        fire earlier or skip the sustained-duration gate; see
        fall_classifier.py's docstring for why (short version: it's a
        single-frame classifier with unvalidated real-world generalization,
        so it must never be the sole reason a fall is reported)."""
        if now is None:
            now = time.time()
        events: List[dict] = []
        matched_ids = set()

        for person_index, person in enumerate(people):
            hip = _avg_point(person.keypoints, KP_HIPS)
            if hip is None:
                continue

            track_id = self._match(hip, matched_ids)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(hip, now)
            matched_ids.add(track_id)

            track = self._tracks[track_id]
            prev_hip = track.hip

            # (a) bounding-box aspect ratio
            x1, y1, x2, y2 = person.bbox
            width, height = x2 - x1, y2 - y1
            wide_box = (width / height) > ASPECT_RATIO_THRESHOLD if height > 0 else False

            # (b) vertical alignment - shoulders bunched near hips
            shoulder = _avg_point(person.keypoints, KP_SHOULDERS)
            fallen_alignment = shoulder is not None and abs(shoulder[1] - hip[1]) < ALIGNMENT_BAND_PX

            # (c) downward velocity of hip midpoint (transient - confidence only, not a sustain gate)
            fast_downward = False
            if prev_hip is not None:
                fast_downward = (hip[1] - prev_hip[1]) > VERTICAL_VELOCITY_THRESHOLD

            on_ground_now = wide_box and fallen_alignment

            if on_ground_now:
                if track.on_ground_since is None:
                    track.on_ground_since = now
            else:
                track.on_ground_since = None

            sustained_seconds = (now - track.on_ground_since) if track.on_ground_since is not None else 0.0
            signal_count = sum([wide_box, fallen_alignment, fast_downward])

            track.hip = hip
            track.last_seen = now

            debounced = track.last_fall_time is not None and (now - track.last_fall_time) <= FALL_DEBOUNCE_SECONDS
            if (
                on_ground_now
                and sustained_seconds >= FALL_MIN_SUSTAINED_SECONDS
                and not debounced
            ):
                track.last_fall_time = now
                confidence = round(min(1.0, (signal_count / 3.0) * 0.7 + 0.3), 2)

                classifier_score = None
                if classifier_scores is not None and person_index < len(classifier_scores):
                    classifier_score = classifier_scores[person_index]
                    if classifier_score >= CLASSIFIER_CORROBORATION_THRESHOLD:
                        # Corroborated: nudge confidence up, capped at 1.0.
                        confidence = round(min(1.0, confidence + (1 - confidence) * 0.3), 2)
                    else:
                        # Classifier disagrees - still fires (heuristic
                        # remains the source of truth per its docstring),
                        # but slightly tempers the reported confidence.
                        confidence = round(max(0.5, confidence - 0.1), 2)

                events.append({
                    "track_id": track_id,
                    "confidence": confidence,
                    "sustained_seconds": round(sustained_seconds, 2),
                    # The box this event was decided on. Carried purely so
                    # the annotation layer (detection/fall_annotation.py)
                    # can draw the clip's box over the right subject; it
                    # is read by nothing that decides anything.
                    "bbox": tuple(float(v) for v in person.bbox),
                    **({"classifier_score": round(classifier_score, 3)} if classifier_score is not None else {}),
                })

        self._prune(now)
        return events

    def _match(self, hip, already_matched: set) -> Optional[int]:
        best_id, best_dist = None, MATCH_DISTANCE_PX
        for track_id, track in self._tracks.items():
            if track_id in already_matched:
                continue
            dist = ((track.hip[0] - hip[0]) ** 2 + (track.hip[1] - hip[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id

    def _prune(self, now: float):
        stale = [tid for tid, t in self._tracks.items() if now - t.last_seen > STALE_SECONDS]
        for tid in stale:
            del self._tracks[tid]


# ---------------------------------------------------------------------------
# Trained-detector path
# ---------------------------------------------------------------------------
# Everything above turns pose keypoints into fall events via a geometric
# proxy. The class below does the same job for the fine-tuned single-class
# YOLO "Fall" detector (see fall_object_detector.py), which recognises the
# posture directly instead of inferring it.
#
# The temporal contract is deliberately identical - a "Fall" box has to
# persist for the same tracked subject across consecutive processed frames
# before an event fires, and the same subject is then debounced - because
# the failure mode is the same for both: a single frame's opinion about
# posture is not evidence of a fall. At the model's measured operating
# point (precision 0.81 / recall 0.71 per image, see ml/MODEL_CARD.md) a
# per-frame trigger would raise an alert on any one-frame false positive,
# while a real fall persists for seconds and so survives the gate easily.
#
# Kept in this module, next to the heuristic, so both fall gates are read
# and reasoned about together - and, like the heuristic, this class takes
# plain boxes rather than a model, so it is unit-testable without torch.

MODEL_MATCH_MIN_DISTANCE_PX = 60.0   # floor for the box-size-relative match radius below
MODEL_MATCH_DIAGONAL_FRACTION = 0.6  # a subject may move this fraction of its own box between frames


def _box_centroid(bbox) -> tuple:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_diagonal(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


class _ModelTrack:
    __slots__ = ("centroid", "diagonal", "last_seen", "last_fall_time", "detected_since", "peak_confidence")

    def __init__(self, centroid, diagonal, now: float, confidence: float):
        self.centroid = centroid
        self.diagonal = diagonal
        self.last_seen = now
        self.last_fall_time: Optional[float] = None
        self.detected_since: Optional[float] = now
        self.peak_confidence = confidence


class ModelFallDetector:
    """Stateful, one instance per camera or per video-analysis job.

    Call update() once per processed frame with the "Fall" boxes the
    trained detector produced for that frame (already filtered to the
    configured confidence floor by fall_object_detector.detect)."""

    def __init__(
        self,
        min_sustained_seconds: float = 0.6,
        debounce_seconds: float = FALL_DEBOUNCE_SECONDS,
    ):
        self._tracks: Dict[int, _ModelTrack] = {}
        self._next_id = 0
        self._min_sustained_seconds = min_sustained_seconds
        self._debounce_seconds = debounce_seconds

    def update(self, detections: List, now: Optional[float] = None) -> List[dict]:
        """`detections` is any sequence of objects with `.bbox`
        (x1, y1, x2, y2) and `.confidence`. `now` is injectable so offline
        video analysis can drive the sustain timer off the video's own
        timeline (seconds from the start of the file) rather than
        wall-clock time, exactly as FallDetector.update does."""
        if now is None:
            now = time.time()

        events: List[dict] = []
        matched_ids = set()

        for detection in detections:
            centroid = _box_centroid(detection.bbox)
            diagonal = _box_diagonal(detection.bbox)
            confidence = float(detection.confidence)

            track_id = self._match(centroid, diagonal, matched_ids)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _ModelTrack(centroid, diagonal, now, confidence)
            matched_ids.add(track_id)

            track = self._tracks[track_id]
            if track.detected_since is None:
                track.detected_since = now
                track.peak_confidence = confidence
            else:
                track.peak_confidence = max(track.peak_confidence, confidence)

            track.centroid = centroid
            track.diagonal = diagonal
            track.last_seen = now

            sustained_seconds = now - track.detected_since
            debounced = (
                track.last_fall_time is not None
                and (now - track.last_fall_time) <= self._debounce_seconds
            )

            if sustained_seconds >= self._min_sustained_seconds and not debounced:
                track.last_fall_time = now
                events.append({
                    "track_id": track_id,
                    # Peak rather than latest: the reported number should
                    # reflect the strongest evidence seen for this fall,
                    # not whichever frame happened to cross the gate.
                    "confidence": round(track.peak_confidence, 2),
                    "sustained_seconds": round(sustained_seconds, 2),
                    "detector": "model",
                    # The box this event was decided on - see the same key
                    # in FallDetector.update above. Annotation only.
                    "bbox": tuple(float(v) for v in detection.bbox),
                })

        # A track the model stopped seeing has to restart its sustain
        # timer, so a flickering detection can't accumulate credit across
        # the gaps and trip the gate without ever being continuously
        # present.
        for track_id, track in self._tracks.items():
            if track_id not in matched_ids:
                track.detected_since = None

        self._prune(now)
        return events

    def _match(self, centroid, diagonal, already_matched: set) -> Optional[int]:
        best_id, best_dist = None, None
        for track_id, track in self._tracks.items():
            if track_id in already_matched:
                continue
            dist = (
                (track.centroid[0] - centroid[0]) ** 2 + (track.centroid[1] - centroid[1]) ** 2
            ) ** 0.5
            # Radius scales with subject size: a subject filling the frame
            # legitimately moves many more pixels between frames than a
            # distant one, so a single fixed pixel threshold either merges
            # far-apart people or splits one person into two tracks.
            radius = max(
                MODEL_MATCH_MIN_DISTANCE_PX,
                MODEL_MATCH_DIAGONAL_FRACTION * max(diagonal, track.diagonal),
            )
            if dist < radius and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_id = track_id
        return best_id

    def _prune(self, now: float):
        stale = [tid for tid, t in self._tracks.items() if now - t.last_seen > STALE_SECONDS]
        for tid in stale:
            del self._tracks[tid]
