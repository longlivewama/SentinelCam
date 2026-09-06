"""
Violence detection.

# MVP heuristic — swap in a trained temporal action-recognition model
# (CNN+LSTM/3D-CNN on RWF-2000) via VIOLENCE_MODEL_PATH for production
# accuracy.

IMPORTANT SCOPE NOTE: a real CNN+LSTM/3D-CNN model trained on RWF-2000 is
not available in this environment (no dataset/GPU training here), so this
module implements a documented heuristic instead, reusing the same pose
keypoints already computed for fall detection:

For each pair of people whose bounding boxes are within PROXIMITY_PX of
each other, we track the frame-to-frame velocity of their wrist/elbow
keypoints (COCO indices 9, 10, 7, 8) over a rolling window of the last
VELOCITY_WINDOW processed frames per person. If both people in the pair
show high velocity magnitude AND high directional variance (erratic
motion, as opposed to the steady heading of walking/running) sustained
across several consecutive frames, a "violence" event fires for that pair.

Callers (detection/engine.py) construct one ViolenceDetector per camera
and call update() once per processed frame with the list of people
detected in that frame (the same PersonDetection-shaped list handed to
FallDetector).
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, List, Optional

VIOLENCE_DEBOUNCE_SECONDS = 10.0
VELOCITY_WINDOW = 16
PROXIMITY_PX = 150.0
MATCH_DISTANCE_PX = 80.0
STALE_SECONDS = 5.0
PAIR_DEBOUNCE_PRUNE_SECONDS = VIOLENCE_DEBOUNCE_SECONDS * 3

SPEED_THRESHOLD = 15.0              # px/processed-frame, "high activity" limb speed
DIRECTION_VARIANCE_THRESHOLD = 1.2  # erratic-motion threshold (see _circular_variance)
MIN_ACTIVE_FRAMES = 8               # of the last VELOCITY_WINDOW frames that must be high-speed

KP_ELBOWS_WRISTS = (7, 8, 9, 10)
MIN_KEYPOINT_CONF = 0.3


def _limb_centroid(keypoints, min_conf: float = MIN_KEYPOINT_CONF) -> Optional[tuple]:
    pts = [(kp[0], kp[1]) for i in KP_ELBOWS_WRISTS if (kp := keypoints[i])[2] >= min_conf]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _bbox_center(bbox) -> tuple:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_distance(b1, b2) -> float:
    c1, c2 = _bbox_center(b1), _bbox_center(b2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def _circular_variance(angles: List[float]) -> float:
    """0 = every angle identical (steady heading); ~2 = angles scattered
    uniformly around the circle (erratic)."""
    if not angles:
        return 0.0
    n = len(angles)
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    mean_resultant_length = math.hypot(sin_sum, cos_sum) / n
    return (1.0 - mean_resultant_length) * 2.0


class _Track:
    def __init__(self, bbox_center, limb, bbox, now: float):
        self.bbox_center = bbox_center  # identity anchor, used for frame-to-frame matching
        self.limb = limb                # wrist/elbow centroid, used for velocity
        self.bbox = bbox
        self.last_seen = now
        self.history = deque(maxlen=VELOCITY_WINDOW)  # (speed, angle) per processed frame

    def is_agitated(self) -> bool:
        if len(self.history) < MIN_ACTIVE_FRAMES:
            return False
        speeds = [s for s, _ in self.history]
        angles = [a for _, a in self.history]
        if sum(1 for s in speeds if s > SPEED_THRESHOLD) < MIN_ACTIVE_FRAMES:
            return False
        return _circular_variance(angles) > DIRECTION_VARIANCE_THRESHOLD


class ViolenceDetector:
    """Stateful, per-camera. Instantiate once per camera and call update()
    once per processed frame."""

    def __init__(self):
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0
        self._pair_debounce: Dict[frozenset, float] = {}

    def update(self, people: List) -> List[dict]:
        now = time.time()
        matched_ids = set()
        current_ids = []

        for person in people:
            limb = _limb_centroid(person.keypoints)
            if limb is None:
                continue

            bbox_center = _bbox_center(person.bbox)
            track_id = self._match(bbox_center, matched_ids)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(bbox_center, limb, person.bbox, now)
            matched_ids.add(track_id)
            current_ids.append(track_id)

            track = self._tracks[track_id]
            dx, dy = limb[0] - track.limb[0], limb[1] - track.limb[1]
            speed = math.hypot(dx, dy)
            angle = math.atan2(dy, dx) if speed > 0 else 0.0
            track.history.append((speed, angle))
            track.bbox_center = bbox_center
            track.limb = limb
            track.bbox = person.bbox
            track.last_seen = now

        events: List[dict] = []
        for i in range(len(current_ids)):
            for j in range(i + 1, len(current_ids)):
                id_a, id_b = current_ids[i], current_ids[j]
                track_a, track_b = self._tracks[id_a], self._tracks[id_b]
                if _bbox_distance(track_a.bbox, track_b.bbox) > PROXIMITY_PX:
                    continue
                if not (track_a.is_agitated() and track_b.is_agitated()):
                    continue

                pair_key = frozenset((id_a, id_b))
                last_fired = self._pair_debounce.get(pair_key, 0.0)
                if now - last_fired > VIOLENCE_DEBOUNCE_SECONDS:
                    self._pair_debounce[pair_key] = now
                    events.append({"track_ids": (id_a, id_b), "confidence": 0.7})

        self._prune(now)
        return events

    def _match(self, bbox_center, already_matched: set) -> Optional[int]:
        best_id, best_dist = None, MATCH_DISTANCE_PX
        for track_id, track in self._tracks.items():
            if track_id in already_matched:
                continue
            dist = math.hypot(track.bbox_center[0] - bbox_center[0], track.bbox_center[1] - bbox_center[1])
            if dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id

    def _prune(self, now: float):
        stale = [tid for tid, t in self._tracks.items() if now - t.last_seen > STALE_SECONDS]
        for tid in stale:
            del self._tracks[tid]
        stale_pairs = [k for k, t in self._pair_debounce.items() if now - t > PAIR_DEBOUNCE_PRUNE_SECONDS]
        for k in stale_pairs:
            del self._pair_debounce[k]
