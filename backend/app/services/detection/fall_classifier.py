"""
Optional corroborating signal for fall detection: a small MLP (10
keypoint-derived geometric features -> P(fall)) trained separately under
`ml/` (see ml/README.md for the full pipeline, dataset, and honest
evaluation caveats). Training code lives entirely outside the running
app; this module only ever loads an exported, frozen ONNX artifact.

Disabled by default (empty `FALL_CLASSIFIER_MODEL_PATH`). When enabled,
`fall_classifier.score_people(...)` is used by detection/engine.py and
video_analysis.py to compute a per-person P(fall) alongside the pose
model's keypoints, which fall_detection.py folds into its confidence
score - but per ml/README.md's integration guidance, this signal only
ever *corroborates* the heuristic's sustained-on-ground-posture gate, it
never replaces or bypasses it: the classifier was trained on single
frames with no temporal information, evaluated on a small two-domain test
set whose 1.000 metrics are flagged in ml/reports/eval_report.md as
likely inflated by domain-shortcut learning, and has not been validated
against real "person crouching/sitting on the floor" hard negatives.
Treat it as a soft confidence adjustment, not a source of truth.

Fails safe everywhere: any missing dependency, missing file, or bad
input yields is_available=False / score 0.0 rather than raising, so a
misconfigured or absent classifier never takes down live detection.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

# COCO-17 keypoint indices, matching fall_detection.py and ml/scripts/features.py.
KP_SHOULDERS = (5, 6)
KP_HIPS = (11, 12)
KP_KNEES = (13, 14)
KP_ANKLES = (15, 16)
KP_HEAD = (0, 1, 2, 3, 4)

FEATURE_ORDER = [
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

# Fallback normalization stats (fall_classifier_v1, see
# ml/exported/fall_classifier_v1.metadata.json) used only if a
# `<model>.metadata.json` sidecar isn't found next to the configured model
# file. Keeping these in sync with a redeployed model is the sidecar
# file's job; this is just a safety net so the classifier still works if
# only the bare .onnx was copied into a deployment.
_FALLBACK_FEATURE_MEAN = [
    0.806239664554596, 0.30263805389404297, 0.15447962284088135, 0.168961301445961,
    0.4620896577835083, 0.38284364342689514, 0.28207117319107056, 0.2028028815984726,
    1.524376392364502, 0.7917011380195618,
]
_FALLBACK_FEATURE_STD = [
    0.6063536405563354, 0.18205823004245758, 0.11481809616088867, 0.11595578491687775,
    0.2972799837589264, 0.24013756215572357, 0.2273719608783722, 0.14202940464019775,
    0.44227465987205505, 0.20026598870754242,
]

# The training data's keypoint "visibility" values are YOLO-Pose-style
# {0, 1, 2} (unlabeled / occluded / visible), not the continuous [0,1]
# confidence score YOLOv8-Pose inference produces in engine.py. To keep
# `mean_keypoint_conf` / `visible_keypoint_frac` on the distribution the
# model was actually trained on, inference-time confidences are binarized
# to {0.0, 2.0} at this same threshold before those two features are
# computed (see _binarize_conf below) - every other feature only uses
# MIN_CONF as a presence/absence filter, where this distinction doesn't
# matter.
MIN_CONF = 0.5


def _avg_point(keypoints: Sequence[Tuple[float, float, float]], indices) -> Optional[Tuple[float, float]]:
    pts = [(keypoints[i][0], keypoints[i][1]) for i in indices if keypoints[i][2] >= MIN_CONF]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _binarize_conf(v: float) -> float:
    return 2.0 if v >= MIN_CONF else 0.0


def extract_features(
    bbox_xyxy: Tuple[float, float, float, float],
    keypoints: Sequence[Tuple[float, float, float]],
    frame_width: float,
    frame_height: float,
) -> Optional[List[float]]:
    """bbox_xyxy: absolute-pixel (x1, y1, x2, y2), as produced by
    engine.py's PersonDetection.bbox. keypoints: 17 absolute-pixel
    (x, y, conf) tuples. Normalizes both by frame size before computing
    the same 10 features ml/scripts/features.py::extract_features does.
    Returns None if there's not enough usable keypoint data (mirrors the
    production heuristic skipping low-confidence detections)."""
    if frame_width <= 0 or frame_height <= 0:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    w = (x2 - x1) / frame_width
    h = (y2 - y1) / frame_height
    if h <= 1e-6:
        return None

    norm_kps = [(x / frame_width, y / frame_height, conf) for x, y, conf in keypoints]

    shoulder = _avg_point(norm_kps, KP_SHOULDERS)
    hip = _avg_point(norm_kps, KP_HIPS)
    knee = _avg_point(norm_kps, KP_KNEES)
    ankle = _avg_point(norm_kps, KP_ANKLES)
    head = _avg_point(norm_kps, KP_HEAD)

    if hip is None:
        return None

    def gap(a, b):
        if a is None or b is None:
            return 0.0
        return abs(a[1] - b[1]) / h

    def width(indices):
        pts = [norm_kps[i] for i in indices if norm_kps[i][2] >= MIN_CONF]
        if len(pts) < 2:
            return 0.0
        return abs(pts[0][0] - pts[1][0]) / w if w > 0 else 0.0

    confs = [_binarize_conf(kp[2]) for kp in norm_kps]

    return [
        w / h,
        gap(shoulder, hip),
        gap(hip, knee),
        gap(knee, ankle),
        gap(shoulder, ankle),
        gap(head, hip),
        width(KP_SHOULDERS),
        width(KP_HIPS),
        sum(confs) / len(confs) if confs else 0.0,
        sum(1 for c in confs if c >= MIN_CONF) / len(confs) if confs else 0.0,
    ]


class FallClassifier:
    def __init__(self):
        self._session = None
        self._input_name = "features"
        self._output_name = "fall_probability"
        self._feature_mean = _FALLBACK_FEATURE_MEAN
        self._feature_std = _FALLBACK_FEATURE_STD
        self._load_attempted = False
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._session is not None

    def _ensure_loaded(self):
        if self._load_attempted:
            return
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True

            model_path = settings.FALL_CLASSIFIER_MODEL_PATH
            if not model_path:
                return
            path = Path(model_path)
            if not path.exists():
                logger.warning("FALL_CLASSIFIER_MODEL_PATH=%s does not exist; classifier disabled", model_path)
                return

            try:
                import onnxruntime as ort
            except ImportError:
                logger.warning("onnxruntime not installed; fall classifier disabled (add it to requirements.txt)")
                return

            metadata_path = path.with_suffix("").with_suffix(".metadata.json")
            if metadata_path.exists():
                try:
                    meta = json.loads(metadata_path.read_text())
                    self._input_name = meta["input"]["name"]
                    self._output_name = meta["output"]["name"]
                    self._feature_mean = meta["input"]["feature_mean"]
                    self._feature_std = meta["input"]["feature_std"]
                except Exception:
                    logger.exception("Failed to parse %s; using built-in fallback normalization stats", metadata_path)

            try:
                self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                logger.info("Loaded fall classifier from %s", model_path)
            except Exception:
                logger.exception("Failed to load fall classifier ONNX session; classifier disabled")
                self._session = None

    def score_people(
        self,
        people: Sequence,
        frame_width: float,
        frame_height: float,
    ) -> List[float]:
        """Returns one P(fall) per entry in `people` (same order), each in
        [0, 1]. Returns all-zero if the classifier isn't configured/loaded
        or on any error - never raises."""
        if not self.is_available:
            return [0.0] * len(people)

        import numpy as np

        rows = []
        row_indices = []
        for i, person in enumerate(people):
            features = extract_features(person.bbox, person.keypoints, frame_width, frame_height)
            if features is None:
                continue
            standardized = [
                (f - m) / s if s > 1e-9 else 0.0
                for f, m, s in zip(features, self._feature_mean, self._feature_std)
            ]
            rows.append(standardized)
            row_indices.append(i)

        scores = [0.0] * len(people)
        if not rows:
            return scores

        try:
            batch = np.array(rows, dtype=np.float32)
            outputs = self._session.run([self._output_name], {self._input_name: batch})
            probs = outputs[0].reshape(-1)
            for idx, prob in zip(row_indices, probs):
                scores[idx] = float(prob)
        except Exception:
            logger.exception("Fall classifier inference failed; falling back to heuristic-only for this frame")
            return [0.0] * len(people)

        return scores


fall_classifier = FallClassifier()
