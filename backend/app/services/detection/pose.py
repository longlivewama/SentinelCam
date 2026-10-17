"""
The keypoint format the pose model actually produces, and the small amount
of geometry the rest of the system reads off it.

The model is Ultralytics' `yolov8n-pose` (see `settings.YOLO_POSE_MODEL`),
trained on COCO-Keypoints: 17 keypoints per person, each `(x, y, conf)` in
the frame's own pixel coordinates (`kpt_shape: [17, 3]` in the model's own
config). Nothing here is invented:

  * `COCO_KEYPOINT_NAMES` is the COCO-Keypoints order the model emits.
  * `COCO_SKELETON` is Ultralytics' own drawing topology
    (`ultralytics.utils.plotting.Annotator.skeleton`), converted from its
    1-based indices to 0-based. `tests/test_pose_estimation.py` asserts it
    still matches the installed Ultralytics, so an upgrade that changes the
    topology fails a test rather than silently mis-drawing limbs.

No inference happens in this module - `detection/engine.py` runs the model
and this only interprets the output, which keeps it testable without torch.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

Keypoint = Tuple[float, float, float]  # (x, y, confidence)

COCO_KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# Ultralytics' Annotator.skeleton (1-based), copied verbatim:
#   [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
#    [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
_ULTRALYTICS_SKELETON_1_BASED = (
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13), (6, 7), (6, 8),
    (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
)
COCO_SKELETON: Tuple[Tuple[int, int], ...] = tuple((a - 1, b - 1) for a, b in _ULTRALYTICS_SKELETON_1_BASED)

KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12

# Below this a keypoint is treated as not visible. Matches the floor the
# pose heuristic in fall_detection.py has always used (MIN_KEYPOINT_CONF),
# so "visible" means the same thing everywhere in the pipeline.
DEFAULT_KEYPOINT_MIN_CONFIDENCE = 0.3


def pose_format() -> dict:
    """The format description sent to clients alongside keypoints, so the
    frontend draws the model's real topology instead of hard-coding one."""
    return {
        "keypoint_names": list(COCO_KEYPOINT_NAMES),
        "skeleton": [list(edge) for edge in COCO_SKELETON],
    }


def normalize_keypoints(raw: Optional[Iterable]) -> List[Keypoint]:
    """Coerces whatever the model handed back into a list of
    `(x, y, conf)` floats. Anything that is not a well-formed list of
    COCO keypoints yields `[]` - "no usable pose" - rather than raising:
    a malformed pose must never cost the frame its fall detection."""
    if raw is None:
        return []
    points: List[Keypoint] = []
    try:
        for point in raw:
            values = list(point)
            if len(values) < 2:
                return []
            x, y = float(values[0]), float(values[1])
            conf = float(values[2]) if len(values) > 2 else 1.0
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(conf)):
                conf = 0.0
            elif x <= 0 or y <= 0:
                # Where the model puts a keypoint it could not place - see
                # engine.mark_missing_keypoint, which applies the full rule.
                conf = 0.0
            points.append((x, y, conf))
    except (TypeError, ValueError):
        return []
    if len(points) != len(COCO_KEYPOINT_NAMES):
        return []
    return points


def visible(point: Keypoint, min_confidence: float) -> bool:
    return point[2] >= min_confidence


def visible_edges(keypoints: Sequence[Keypoint], min_confidence: float) -> List[Tuple[int, int]]:
    """The skeleton edges whose both endpoints are visible. A partially
    visible person gets a partial skeleton; nothing is drawn to a point
    the model was not confident about."""
    if len(keypoints) != len(COCO_KEYPOINT_NAMES):
        return []
    return [
        (a, b) for a, b in COCO_SKELETON
        if visible(keypoints[a], min_confidence) and visible(keypoints[b], min_confidence)
    ]


def _midpoint(keypoints: Sequence[Keypoint], indices: Sequence[int], min_confidence: float) -> Optional[Tuple[float, float]]:
    points = [keypoints[i] for i in indices if visible(keypoints[i], min_confidence)]
    if not points:
        return None
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def torso_angle_degrees(keypoints: Sequence[Keypoint], min_confidence: float = DEFAULT_KEYPOINT_MIN_CONFIDENCE) -> Optional[float]:
    """Angle of the hip-to-shoulder axis from image vertical, in degrees:
    0 is upright, 90 is horizontal, 180 is inverted.

    Uses the midpoint of whichever shoulders and hips are visible, so one
    occluded side does not discard the measurement. Returns None when no
    shoulder or no hip is visible, or the two midpoints coincide - "no
    evidence", which callers must not read as either upright or fallen.

    This is a 2-D image-plane measurement. It is meaningful for a roughly
    side-on or elevated camera; for a camera looking straight down a
    standing person can project as "horizontal". It is evidence about
    posture, never proof of a fall."""
    if len(keypoints) != len(COCO_KEYPOINT_NAMES):
        return None
    shoulders = _midpoint(keypoints, (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER), min_confidence)
    hips = _midpoint(keypoints, (KP_LEFT_HIP, KP_RIGHT_HIP), min_confidence)
    if shoulders is None or hips is None:
        return None
    dx = shoulders[0] - hips[0]
    dy = shoulders[1] - hips[1]  # image y grows downward; upright => dy < 0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, -dy / length))))
