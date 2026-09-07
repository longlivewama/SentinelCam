"""
Evaluates the trained fall classifier (ml/exported/fall_classifier_v1.pt)
on the held-out TEST split (ml/data/test.csv), which was never touched
during training or model selection (best-epoch selection used only the
val split). Writes ml/reports/eval_report.md with precision/recall/F1/
confusion matrix/ROC-AUC and a few concrete false-positive/false-negative
examples.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train import FallClassifierMLP

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exported")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

FEATURE_COLUMNS = [
    "aspect_ratio", "shoulder_hip_gap_norm", "hip_knee_gap_norm",
    "knee_ankle_gap_norm", "shoulder_ankle_gap_norm", "head_hip_gap_norm",
    "shoulder_width_norm", "hip_width_norm", "mean_keypoint_conf",
    "visible_keypoint_frac",
]


def load_csv(path):
    features, labels, sources = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append([float(row[k]) for k in FEATURE_COLUMNS])
            labels.append(float(row["label"]))
            sources.append(row["source_image"])
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32), sources


def main():
    checkpoint = torch.load(os.path.join(EXPORT_DIR, "fall_classifier_v1.pt"), weights_only=False)
    mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
    std = np.array(checkpoint["feature_std"], dtype=np.float32)

    model = FallClassifierMLP()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    X_test, y_test, sources = load_csv(os.path.join(DATA_DIR, "test.csv"))
    X_test_norm = (X_test - mean) / std

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_norm))
        probs = torch.sigmoid(logits).squeeze(1).numpy()

    preds = (probs >= 0.5).astype(np.float32)

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, probs)
    except ValueError:
        roc_auc = float("nan")  # only one class present in y_test

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Concrete failure examples for the report.
    false_positives = [
        (sources[i], probs[i]) for i in range(len(y_test)) if y_test[i] == 0 and preds[i] == 1
    ]
    false_negatives = [
        (sources[i], probs[i]) for i in range(len(y_test)) if y_test[i] == 1 and preds[i] == 0
    ]

    n_fall = int(y_test.sum())
    n_not_fall = int(len(y_test) - n_fall)

    # Domain breakdown: the two not_fall sources are visually very
    # different (synthetic CCTV renders vs. natural COCO photos), so a
    # model could trivially separate fall/not_fall by domain statistics
    # alone rather than genuine posture semantics. The Simuletic
    # "standing" instances are the harder, more informative negative
    # (same synthetic domain/rendering as the fall examples) - report
    # them separately rather than let them get diluted into the
    # COCO-heavy aggregate.
    simuletic_not_fall_idx = [
        i for i in range(len(y_test)) if y_test[i] == 0 and sources[i].startswith("simuletic/")
    ]
    coco_not_fall_idx = [
        i for i in range(len(y_test)) if y_test[i] == 0 and sources[i].startswith("coco_val2017/")
    ]
    simuletic_acc = (
        float(np.mean(preds[simuletic_not_fall_idx] == 0)) if simuletic_not_fall_idx else float("nan")
    )
    coco_acc = (
        float(np.mean(preds[coco_not_fall_idx] == 0)) if coco_not_fall_idx else float("nan")
    )

    report = f"""# Fall Classifier Evaluation Report

Model: `ml/exported/fall_classifier_v1.pt` (small MLP, 10 keypoint-derived
features -> 16 -> 8 -> 1 logit). Evaluated on `ml/data/test.csv`, a
held-out split never used for training or epoch/model selection (that
used only `ml/data/val.csv`).

## Test set composition

- Total instances: {len(y_test)}
- Fall: {n_fall}
- Not fall: {n_not_fall}

**Caveat: this is a small test set ({len(y_test)} instances). Metrics below
have high variance - a handful of different examples could move
precision/recall by several points. Treat these as directional, not a
tight production SLA.**

## Metrics (threshold = 0.5)

| Metric | Value |
|---|---|
| Precision | {precision:.3f} |
| Recall | {recall:.3f} |
| F1 | {f1:.3f} |
| ROC-AUC | {roc_auc:.3f} |

## Confusion matrix

|  | Predicted not_fall | Predicted fall |
|---|---|---|
| **Actual not_fall** | {tn} (TN) | {fp} (FP) |
| **Actual fall** | {fn} (FN) | {tp} (TP) |

## Domain breakdown of the not_fall class (important caveat)

Perfect aggregate metrics on a mixed test set can hide a shortcut: if the
two not_fall sources are trivially distinguishable from the fall class by
domain statistics alone (rendering style, resolution, compression) rather
than genuine posture semantics, aggregate precision/recall/F1 will look
great without the model having learned anything robust. Breaking out
not_fall accuracy by source:

| Source | n | Correctly predicted not_fall |
|---|---|---|
| Simuletic "standing" (same synthetic domain as fall examples - the harder, more informative case) | {len(simuletic_not_fall_idx)} | {simuletic_acc:.3f} |
| COCO natural photos (different domain - easier case) | {len(coco_not_fall_idx)} | {coco_acc:.3f} |

The Simuletic-domain row is the one that actually tests whether the model
distinguishes "standing" from "lying down" posture within the same
rendering domain, rather than just detecting which dataset an instance
came from. It has very few examples ({len(simuletic_not_fall_idx)}) in
this test split, so treat it as a sanity check, not a reliable estimate.

## False positives ({len(false_positives)})

Not-fall instances the model scored >= 0.5 (predicted fall):

{chr(10).join(f"- `{src}` -> P(fall)={p:.3f}" for src, p in false_positives) or "- none"}

## False negatives ({len(false_negatives)})

Fall instances the model scored < 0.5 (predicted not_fall):

{chr(10).join(f"- `{src}` -> P(fall)={p:.3f}" for src, p in false_negatives) or "- none"}

## Analysis

False positives on this feature set typically come from COCO instances of
people crouching, bending over, or sitting on the ground with their legs
extended - postures that are genuinely geometrically close to "lying
down" from a single static frame with no motion/duration context. This is
exactly why the production system does not rely on this classifier alone:
`FALL_MIN_SUSTAINED_SECONDS` in `fall_detection.py` already requires the
posture to persist for >=1.2s before firing, which filters out the
momentary crouch/bend cases that most resemble these false positives.
False negatives tend to be fall instances with several low-confidence/
occluded keypoints (partial visibility lowers `mean_keypoint_conf` and
`visible_keypoint_frac`, weakening the other geometric ratios too).

## Known limitations

- Test set is small ({len(y_test)} instances) and comes from only two
  source datasets (synthetic CCTV renders + COCO everyday photos) - no
  real elderly-care or hospital-room footage, no low-light/IR camera
  frames, no partial-occlusion-by-furniture cases specific to the
  target deployment environment.
- COCO instances are all labeled not_fall by construction (see
  `scripts/prepare_dataset.py` docstring) - a small number could
  plausibly depict someone actually on the ground (e.g. a photo of
  someone doing yoga on a mat), which would be a label-noise false
  positive rather than a genuine model error. Manual spot-check of the
  false positives above is recommended before trusting this number
  precisely.
- No video/temporal data was used in training - the classifier scores a
  single static posture, consistent with how it is meant to be used in
  production (as a per-frame corroborating signal alongside the
  sustained-duration heuristic gate, not a replacement for it).
"""

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "eval_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} roc_auc={roc_auc:.3f}")
    print(f"confusion matrix: tn={tn} fp={fp} fn={fn} tp={tp}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
