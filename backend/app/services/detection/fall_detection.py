"""
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
