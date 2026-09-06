"""
Crowd detection.

SCOPE NOTE: CSRNet (density-map crowd counting trained on ShanghaiTech) has
no readily pip-installable pretrained checkpoint available in this
environment. Crowd counting is instead implemented via plain YOLOv8 object
detection filtered to the "person" class (COCO class 0): count detections
per processed frame and smooth the count over a short rolling window to
reduce flicker from momentary misses/occlusions.

Extension point: swap `count_people` / CrowdDetector for a CSRNet-based
density-map estimator if/when accuracy in very dense, heavily-occluded
crowds (where individual bounding-box detection breaks down) becomes a
requirement - density estimation degrades much more gracefully than
per-instance detection as people start to overlap heavily.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Iterable, Optional

PERSON_CLASS_ID = 0  # COCO class index for "person"

SMOOTHING_WINDOW = 5
CROWD_DEBOUNCE_SECONDS = 60.0  # don't refire more than once per 60s while still over threshold


def count_people(detections: Iterable) -> int:
    """detections: iterable of (cls_id, conf, bbox). Returns the number of
    "person"-class detections in the frame."""
    return sum(1 for cls_id, *_ in detections if int(cls_id) == PERSON_CLASS_ID)


class CrowdDetector:
    """Stateful, per-camera. Instantiate once per camera and call update()
    once per processed frame with that frame's person count."""

    def __init__(self):
        self._counts = deque(maxlen=SMOOTHING_WINDOW)
        self._last_fired = 0.0

    def update(self, person_count: int, threshold: int) -> Optional[dict]:
        now = time.time()
        self._counts.append(person_count)
        smoothed = sum(self._counts) / len(self._counts)

        if smoothed > threshold and (now - self._last_fired) > CROWD_DEBOUNCE_SECONDS:
            self._last_fired = now
            return {
                "count": round(smoothed, 1),
                "confidence": min(smoothed / max(threshold, 1), 1.0),
            }
        return None
