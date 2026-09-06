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

A "fall" event fires when >= 2 of the 3 signals are true simultaneously
for the same tracked person. A debounce window prevents re-firing (and
re-recording) every frame while the person remains on the ground.

Callers (detection/engine.py) construct one FallDetector per camera and
call update() once per processed frame with the list of people detected
in that frame.
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
    __slots__ = ("hip", "last_seen", "last_fall_time")

    def __init__(self, hip, now: float):
        self.hip = hip
        self.last_seen = now
        self.last_fall_time = 0.0


class FallDetector:
    """Stateful, per-camera. Instantiate once per camera and call update()
    once per processed frame."""

    def __init__(self):
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0

    def update(self, people: List[PersonDetection]) -> List[dict]:
        now = time.time()
        events: List[dict] = []
        matched_ids = set()

        for person in people:
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

            # (c) downward velocity of hip midpoint
            fast_downward = False
            if prev_hip is not None:
                fast_downward = (hip[1] - prev_hip[1]) > VERTICAL_VELOCITY_THRESHOLD

            signal_count = sum([wide_box, fallen_alignment, fast_downward])

            track.hip = hip
            track.last_seen = now

            if signal_count >= 2 and (now - track.last_fall_time) > FALL_DEBOUNCE_SECONDS:
                track.last_fall_time = now
                events.append({"track_id": track_id, "confidence": round(signal_count / 3.0, 2)})

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
