"""
Shared feature engineering: turns a single person's YOLO-Pose-style
17-keypoint annotation (COCO order, (x, y, v) each, already normalized to
[0, 1] by image width/height) plus its bounding box into a fixed-length
feature vector for the fall/not-fall classifier.

This mirrors (and extends) the geometric signals already used by the
production heuristic in
backend/app/services/detection/fall_detection.py, so the classifier is
learning a nonlinear/soft-weighted version of the same underlying
intuition (posture geometry) rather than something unrelated, while
adding a few extra scale-invariant ratios the hand-tuned thresholds don't
use. All features are translation/scale invariant (normalized by bbox
diagonal or bbox height) so the model generalizes across camera distance
and framing rather than memorizing absolute pixel positions.

COCO-17 keypoint order: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear,
4 right_ear, 5 left_shoulder, 6 right_shoulder, 7 left_elbow,
8 right_elbow, 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

FEATURE_NAMES = [
    "aspect_ratio",
    "shoulder_hip_gap_norm",
    "hip_knee_gap_norm",
    "knee_ankle_gap_norm",
    "shoulder_ankle_gap_norm",
    "head_hip_gap_norm",
    "shoulder_width_norm",
    "hip_width_norm",
    "mean_keypoint_conf",
    "visible_keypoint_frac",
]

KP_SHOULDERS = (5, 6)
KP_HIPS = (11, 12)
KP_KNEES = (13, 14)
KP_ANKLES = (15, 16)
KP_HEAD = (0, 1, 2, 3, 4)

MIN_VIS = 0.5  # YOLO-pose visibility is {0, 1, 2}; treat >=1 as "usable"


def _avg_point(keypoints: Sequence[Tuple[float, float, float]], indices) -> Optional[Tuple[float, float]]:
    pts = [(keypoints[i][0], keypoints[i][1]) for i in indices if keypoints[i][2] >= MIN_VIS]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def extract_features(
    bbox: Tuple[float, float, float, float],
    keypoints: Sequence[Tuple[float, float, float]],
) -> Optional[List[float]]:
    """bbox = (cx, cy, w, h), all normalized [0,1]. keypoints = 17 x (x, y, v),
    normalized [0,1]. Returns None if too few keypoints are usable to
    compute a meaningful feature vector (mirrors the production heuristic's
    behaviour of skipping people with insufficient pose confidence)."""
    _, _, w, h = bbox
    if h <= 1e-6:
        return None
    diag = (w ** 2 + h ** 2) ** 0.5 or 1e-6

    shoulder = _avg_point(keypoints, KP_SHOULDERS)
    hip = _avg_point(keypoints, KP_HIPS)
    knee = _avg_point(keypoints, KP_KNEES)
    ankle = _avg_point(keypoints, KP_ANKLES)
    head = _avg_point(keypoints, KP_HEAD)

    if hip is None:
        return None

    def gap(a, b):
        if a is None or b is None:
            return 0.0
        return abs(a[1] - b[1]) / h

    def width(indices):
        pts = [keypoints[i] for i in indices if keypoints[i][2] >= MIN_VIS]
        if len(pts) < 2:
            return 0.0
        return abs(pts[0][0] - pts[1][0]) / w if w > 0 else 0.0

    aspect_ratio = w / h
    shoulder_hip_gap_norm = gap(shoulder, hip)
    hip_knee_gap_norm = gap(hip, knee)
    knee_ankle_gap_norm = gap(knee, ankle)
    shoulder_ankle_gap_norm = gap(shoulder, ankle)
    head_hip_gap_norm = gap(head, hip)
    shoulder_width_norm = width(KP_SHOULDERS)
    hip_width_norm = width(KP_HIPS)

    confs = [kp[2] for kp in keypoints]
    mean_keypoint_conf = sum(confs) / len(confs) if confs else 0.0
    visible_keypoint_frac = sum(1 for c in confs if c >= MIN_VIS) / len(confs) if confs else 0.0

    return [
        aspect_ratio,
        shoulder_hip_gap_norm,
        hip_knee_gap_norm,
        knee_ankle_gap_norm,
        shoulder_ankle_gap_norm,
        head_hip_gap_norm,
        shoulder_width_norm,
        hip_width_norm,
        mean_keypoint_conf,
        visible_keypoint_frac,
    ]


def parse_yolo_pose_label_line(line: str) -> Optional[Tuple[int, Tuple[float, float, float, float], List[Tuple[float, float, float]]]]:
    """Parses one line of a YOLO-Pose label file:
    `class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ... (17 keypoints)`.
    Returns (class_id, bbox, keypoints) or None if malformed."""
    parts = line.split()
    if len(parts) != 1 + 4 + 17 * 3:
        return None
    class_id = int(parts[0])
    bbox = tuple(float(x) for x in parts[1:5])
    kp_values = [float(x) for x in parts[5:]]
    keypoints = [tuple(kp_values[i:i + 3]) for i in range(0, len(kp_values), 3)]
    return class_id, bbox, keypoints
