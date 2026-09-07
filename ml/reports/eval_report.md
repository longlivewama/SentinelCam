# Fall Classifier Evaluation Report

Model: `ml/exported/fall_classifier_v1.pt` (small MLP, 10 keypoint-derived
features -> 16 -> 8 -> 1 logit). Evaluated on `ml/data/test.csv`, a
held-out split never used for training or epoch/model selection (that
used only `ml/data/val.csv`).

## Test set composition

- Total instances: 64
- Fall: 12
- Not fall: 52

**Caveat: this is a small test set (64 instances). Metrics below
have high variance - a handful of different examples could move
precision/recall by several points. Treat these as directional, not a
tight production SLA.**

## Metrics (threshold = 0.5)

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |
| ROC-AUC | 1.000 |

## Confusion matrix

|  | Predicted not_fall | Predicted fall |
|---|---|---|
| **Actual not_fall** | 52 (TN) | 0 (FP) |
| **Actual fall** | 0 (FN) | 12 (TP) |

## Domain breakdown of the not_fall class (important caveat)

Perfect aggregate metrics on a mixed test set can hide a shortcut: if the
two not_fall sources are trivially distinguishable from the fall class by
domain statistics alone (rendering style, resolution, compression) rather
than genuine posture semantics, aggregate precision/recall/F1 will look
great without the model having learned anything robust. Breaking out
not_fall accuracy by source:

| Source | n | Correctly predicted not_fall |
|---|---|---|
| Simuletic "standing" (same synthetic domain as fall examples - the harder, more informative case) | 1 | 1.000 |
| COCO natural photos (different domain - easier case) | 51 | 1.000 |

The Simuletic-domain row is the one that actually tests whether the model
distinguishes "standing" from "lying down" posture within the same
rendering domain, rather than just detecting which dataset an instance
came from. It has very few examples (1) in
this test split, so treat it as a sanity check, not a reliable estimate.

## False positives (0)

Not-fall instances the model scored >= 0.5 (predicted fall):

- none

## False negatives (0)

Fall instances the model scored < 0.5 (predicted not_fall):

- none

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

- Test set is small (64 instances) and comes from only two
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
