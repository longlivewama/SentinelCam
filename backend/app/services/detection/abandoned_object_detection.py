"""
Abandoned object detection.

Uses YOLOv8 object detection (excluding the "person" class) plus a simple
custom centroid tracker implemented here (no DeepSORT/ByteTrack
dependency): each detected non-person object is matched frame-to-frame to
an existing tracked object by nearest centroid within MATCH_DISTANCE_PX
(otherwise a new tracked object is started with first_seen=now); a match
updates last_seen.

Every tick, for every tracked object whose (now - first_seen) exceeds the
camera's abandoned_object_seconds, we check whether any currently-detected
person's bounding box is within PERSON_PROXIMITY_PX of the object's
centroid: if so, the object is "attended" and its timer resets; if not,
and the unattended duration has passed the threshold, an
"abandoned_object" event fires once per tracked object (a `fired` flag
prevents re-firing every frame the object stays put; it re-arms if the
object becomes attended and then unattended again).

Tracked objects not seen for PRUNE_AFTER_SECONDS are dropped to avoid
unbounded growth from momentary detections.

Extension point: DeepSORT/ByteTrack would give far more robust identity
persistence through occlusion than this nearest-centroid tracker (e.g. a
bag briefly hidden behind a passerby won't currently survive that gap
gracefully - it will be treated as a new object once it reappears).

Callers (detection/engine.py) construct one AbandonedObjectDetector per
camera and call update() once per processed frame.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional

PERSON_CLASS_ID = 0  # COCO class index for "person"

MATCH_DISTANCE_PX = 50.0
PERSON_PROXIMITY_PX = 150.0
PRUNE_AFTER_SECONDS = 5.0


def _bbox_center(bbox) -> tuple:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class _TrackedObject:
    def __init__(self, centroid, bbox, cls_id, now: float):
        self.centroid = centroid
        self.bbox = bbox
        self.cls_id = cls_id
        self.first_seen = now
        self.last_seen = now
        self.fired = False


class AbandonedObjectDetector:
    """Stateful, per-camera. Instantiate once per camera and call update()
    once per processed frame."""

    def __init__(self):
        self._objects: Dict[int, _TrackedObject] = {}
        self._next_id = 0

    def update(
        self,
        object_detections: Iterable,
        person_boxes: Iterable,
        abandoned_object_seconds: int,
    ) -> List[dict]:
        """
        object_detections: iterable of (cls_id, conf, bbox) for non-person
                            objects detected this frame.
        person_boxes: iterable of bbox for people detected this frame.
        """
        now = time.time()
        person_boxes = list(person_boxes)
        matched_ids = set()

        for cls_id, _conf, bbox in object_detections:
            if int(cls_id) == PERSON_CLASS_ID:
                continue  # people are handled separately, by proximity below
            centroid = _bbox_center(bbox)
            track_id = self._match(centroid, matched_ids)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._objects[track_id] = _TrackedObject(centroid, bbox, cls_id, now)
            matched_ids.add(track_id)

            obj = self._objects[track_id]
            obj.centroid = centroid
            obj.bbox = bbox
            obj.last_seen = now

        events: List[dict] = []
        for track_id, obj in self._objects.items():
            if track_id not in matched_ids:
                continue  # not detected this frame - leave its timer as-is

            person_nearby = any(
                _distance(obj.centroid, _bbox_center(pbox)) < PERSON_PROXIMITY_PX
                for pbox in person_boxes
            )
            if person_nearby:
                # Attended: reset/pause the abandonment timer.
                obj.first_seen = now
                obj.fired = False
                continue

            unattended_duration = now - obj.first_seen
            if unattended_duration >= abandoned_object_seconds and not obj.fired:
                obj.fired = True
                events.append({
                    "track_id": track_id,
                    "confidence": min(unattended_duration / max(abandoned_object_seconds, 1), 1.0),
                })

        self._prune(now)
        return events

    def _match(self, centroid, already_matched: set) -> Optional[int]:
        best_id, best_dist = None, MATCH_DISTANCE_PX
        for track_id, obj in self._objects.items():
            if track_id in already_matched:
                continue
            dist = _distance(obj.centroid, centroid)
            if dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id

    def _prune(self, now: float):
        stale = [tid for tid, o in self._objects.items() if now - o.last_seen > PRUNE_AFTER_SECONDS]
        for tid in stale:
            del self._objects[tid]
