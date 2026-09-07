# SentinelCam ML: Fall Classifier Training Pipeline

A real, executed, end-to-end training pipeline for a corroborating fall
classifier: dataset acquisition -> preprocessing -> train/val/test split
-> training -> held-out evaluation -> ONNX export. Completely separate
from the production inference service under `backend/` - this directory
is never imported by the running app; it only produces an artifact
(`exported/fall_classifier_v1.onnx`) that the backend can optionally load.

## Problem framing

The production fall detector
(`backend/app/services/detection/fall_detection.py`) is a hand-tuned
geometric heuristic on YOLOv8-Pose keypoints: bounding-box aspect ratio,
shoulder/hip vertical alignment, and downward hip velocity, gated by a
sustained-duration requirement (recently added) so momentary posture
changes (bending, sitting quickly) don't fire false positives.

This pipeline trains a small **corroborating classifier** on the same
kind of keypoint data: given one person's 17 COCO keypoints + bounding
box in a single frame, predict P(fall). It is explicitly *not* meant to
replace the heuristic's temporal-sustain logic (a single frame can never
tell you "has this posture been held for over a second" - that requires
state across frames, which the heuristic already tracks per-person). It's
meant to be combined with it: e.g. require both the sustained-posture
gate *and* classifier_score >= threshold before firing, or use the
classifier score to adjust the reported confidence. See "Integration
contract" below.

## Why keypoint features, not an image classifier

The original brief considered fine-tuning a YOLO image classifier
(`yolov8n-cls`) on cropped person images. That was rejected in favor of
engineered keypoint features for three concrete reasons:

1. **The production system already computes 17-keypoint pose per person
   every frame** (that's what the heuristic runs on). A keypoint-feature
   classifier reuses that same output with zero extra model inference -
   an image classifier would need a second forward pass (crop + resize +
   run a CNN) per person per frame, roughly doubling inference cost on an
   already CPU-only pipeline.
2. **Keypoint geometry is scale/translation-invariant by construction**
   once normalized by bbox size, so it generalizes across camera distance
   and framing without needing thousands of images to learn that
   invariance the way a CNN would.
3. **The available data is small** (~400 labeled instances after
   augmentation from two sources - see below). A tiny MLP on 10 engineered
   features trains to convergence in well under a second and doesn't have
   enough capacity to overfit; an image CNN fine-tuned on 400 images,
   mostly synthetic, would be far more prone to memorizing rendering
   artifacts instead of posture.

## Data sources and licenses

Two sources were combined (see `scripts/prepare_dataset.py` for the exact
loading/labeling code):

| Source | What it provides | License | Instances used |
|---|---|---|---|
| [Simuletic CCTV Incident Dataset (Fall/Lying-Down Detection)](https://huggingface.co/datasets/Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection) | Synthetic, privacy-safe overhead-CCTV renders of people, YOLO-Pose format (17 COCO keypoints + bbox), 2-class: `laying` (fallen) / `standing` | CC BY 4.0 | 111 `laying`->fall, 10 `standing`->not_fall |
| [COCO 2017 person keypoints](https://cocodataset.org) (`val2017` split, re-hosted as YOLO-Pose label files at [Mai0313/coco-pose-2017](https://huggingface.co/datasets/Mai0313/coco-pose-2017)) | Everyday photos of people (standing, walking, sitting, sports, etc.) - a person lying flat on the ground is a vanishing minority of this dataset's "person" category, so every instance is treated as not_fall | CC BY 4.0 (annotations) | 300 (capped) -> 310 not_fall total after dedup |

**Only the numeric keypoint/bbox annotations were used from COCO - no
COCO images were downloaded.** Training operates entirely on
keypoint-derived geometric features, so no image data was needed at all;
this also sidesteps any per-image-license ambiguity in the original COCO
photos (only the CC-BY-4.0-licensed annotation set was touched).

Total: **421 person instances across 230 image groups** (111 fall / 310
not_fall - see the class-imbalance handling in `train.py`).

`scripts/download_images.py` exists to fetch the Simuletic source images
for qualitative visual review, but was **not run** in this iteration -
the failure-case analysis in `reports/eval_report.md` references
`source_image` filenames as text identifiers, not rendered images, since
training and evaluation never needed pixels.

## Split strategy (leakage avoidance)

Splits are grouped by `(dataset, source_image)`, not by individual
instance, so if one image contains multiple people, all of that image's
people stay in the same split. This matters more for COCO (some photos
have several people) than for Simuletic; Simuletic images are
independently-rendered synthetic scenes rather than frames sampled from a
continuous video, so - unlike a video-frame dataset - there's no risk of
near-duplicate adjacent frames leaking across a random per-image split.
Groups are shuffled with a fixed seed (42) and assigned to train
(70%) / val (15%) / test (15%) by running instance-count threshold; see
the per-split class counts printed by `prepare_dataset.py` and reproduced
below - the split is not perfectly stratified by class (documented
limitation, not hidden).

```
Total: 421 instances across 230 image groups (111 fall / 310 not_fall)
  - Simuletic groups: 111
  - COCO groups: 119
train: 292 rows (87 fall / 205 not_fall)
val:   65 rows (12 fall / 53 not_fall)
test:  64 rows (12 fall / 52 not_fall)
```

## Preprocessing / feature engineering

See `scripts/features.py`. From each person's bbox `(cx, cy, w, h)` and 17
`(x, y, visibility)` keypoints (all normalized [0,1] by image size), 10
scale-invariant features are computed: bbox aspect ratio, four vertical
"gap" ratios (shoulder-hip, hip-knee, knee-ankle, shoulder-ankle,
head-hip), two horizontal "width" ratios (shoulder width, hip width), and
two pose-quality signals (mean keypoint confidence, fraction of keypoints
above the visibility threshold). No image augmentation is applicable
here (there are no images in the training loop) - the "augmentation" that
matters is dataset diversity (two independent sources, different
rendering domains), which is what actually stresses the model's
generalization; see the domain-breakdown caveat in the eval report.

## Training

`train.py`: a 3-layer MLP (10 -> 16 -> 8 -> 1 logit), trained with Adam
(lr=0.01) for 200 epochs on the train split, with `BCEWithLogitsLoss`
class-weighted by the train split's fall/not_fall ratio (~2.4:1) rather
than resampling, so every real example stays in play. Features are
standardized using **train-split statistics only** (mean/std saved into
the checkpoint, reused unchanged at val/test/inference time - never
refit). The epoch with the lowest val-split loss is checkpointed (not
necessarily the final epoch), i.e. this is genuine early-stopping-by-val,
not "whatever epoch 200 happened to land on."

Actual run: **200 epochs, wall-clock 0.1 seconds** (CPU, no GPU used or
needed - the whole model has 3 small linear layers and the dataset is
292 training rows). Final: `best_val_loss=0.0746`.

## Evaluation

`evaluate.py` runs ONLY on `data/test.csv`, never touched during training
or epoch selection. Full report: **`reports/eval_report.md`** (regenerated
by re-running the script - do not hand-edit it).

Headline result: precision = recall = F1 = ROC-AUC = **1.000** on the
64-instance test split (52 not_fall / 12 fall).

**Do not take that at face value** - the eval report includes a domain
breakdown specifically because a perfect score on a two-source dataset is
a classic shortcut-learning red flag: it's easy for a model to learn "is
this a COCO photo or a synthetic CCTV render" instead of "is this person
lying down," since 51 of the 52 not_fall test instances are COCO (a
different visual domain from the fall examples) and only **1** is a
same-domain Simuletic "standing" instance. That one same-domain instance
was also classified correctly, but n=1 tells you almost nothing on its
own - it's reported as a sanity check, not proof of robust
within-domain discrimination. See "Known limitations" below and in the
eval report for the honest read on what this metric does and doesn't
demonstrate.

## Export

`export.py` exports the checkpoint to ONNX (opset 17, via PyTorch's
legacy TorchScript-based exporter - the newer dynamo-based default
exporter in torch>=2.x requires the `onnxscript` package, which pulls in
numpy>=2.0 and would conflict with `backend/venv`'s ultralytics pin
`numpy<2.0.0`; since this pipeline reuses that venv, the legacy exporter
was used instead to avoid touching backend's dependency set at all).
Runs a PyTorch-vs-ONNXRuntime equivalence check on a random batch before
writing metadata (actual run: max abs diff `5.96e-08`, well under the
`1e-4` acceptance threshold) - if that check fails the script raises
instead of silently shipping a broken export.

Output files (committed - small, see `.gitignore`):
- `exported/fall_classifier_v1.onnx` (2.3 KB)
- `exported/fall_classifier_v1.pt` (PyTorch checkpoint, includes feature
  mean/std - 4.7 KB)
- `exported/fall_classifier_v1.metadata.json` (exact input/output
  contract + training run stats)
- `exported/train_run_metadata.json`

## How to reproduce

```bash
cd backend && source venv/bin/activate   # reuses the backend venv (torch/numpy/ultralytics already there)
pip install -r ../ml/requirements.txt    # scikit-learn, onnx, onnxruntime
cd ../ml

# 1. Download raw label data (no images needed - see README above)
cd scripts
python3 download_simuletic_labels.py
python3 download_coco_pose_labels.py

# 2. Build the feature dataset (writes ../data/{train,val,test}.csv)
python3 prepare_dataset.py
cd ..

# 3. Train
python3 train.py

# 4. Evaluate (writes reports/eval_report.md)
python3 evaluate.py

# 5. Export to ONNX (writes exported/*.onnx + metadata)
python3 export.py
```

## Integration contract (for the backend team - not implemented here)

To use `exported/fall_classifier_v1.onnx` as a corroborating signal in
`backend/app/services/detection/fall_detection.py` or `engine.py`:

1. Add `onnxruntime` to `backend/requirements.txt` (CPU inference only,
   no GPU needed - the model is 3 tiny linear layers).
2. Config: the reserved `FALL_CLASSIFIER_MODEL_PATH` setting in
   `backend/app/config.py` should point at
   `ml/exported/fall_classifier_v1.onnx` (or wherever it's deployed) -
   empty string keeps the current heuristic-only behavior.
3. At startup (lazily, like the pose/object models in
   `detection/engine.py`), if `FALL_CLASSIFIER_MODEL_PATH` is set and the
   file exists, create one `onnxruntime.InferenceSession` and share it
   across all cameras (same pattern as `_pose_model`/`_object_model`).
4. Per person per processed frame (you already have the 17 keypoints +
   bbox from the pose model - this needs zero extra model inference):
   - Compute the 10 features exactly as `ml/scripts/features.py::extract_features`
     does (bbox as `(cx, cy, w, h)` normalized [0,1] by frame width/height,
     keypoints as `(x, y, visibility)` normalized the same way - note
     production's `PersonDetection.keypoints` are currently in absolute
     pixel coordinates, so normalize by frame width/height first).
   - Standardize: `(feature - feature_mean[i]) / feature_std[i]` using
     the arrays in `fall_classifier_v1.metadata.json` (`input.feature_mean`
     / `input.feature_std`) - do not refit these.
   - Run the ONNX session: `session.run(["fall_probability"], {"features": batch})`,
     input shape `(N, 10)` float32, output shape `(N, 1)` float32, already
     sigmoid-activated (a probability in [0, 1], no further transform).
5. Suggested combination rule (not prescriptive - the backend team should
   validate against real footage before shipping): only raise confidence
   or shorten `FALL_MIN_SUSTAINED_SECONDS` when `classifier_score >= 0.5`
   corroborates the heuristic's `on_ground_now` signal; do not let the
   classifier fire alone without the heuristic's sustained-duration gate,
   since (per the domain-breakdown caveat above) this classifier has not
   been validated on real, non-synthetic "person crouching/sitting on the
   floor" hard negatives at any meaningful sample size.

## Known limitations

- **Small, narrow test set.** 64 test instances from exactly two source
  domains (synthetic renders + COCO everyday photos). No real elderly-care
  footage, no low-light/IR frames, no furniture-occlusion cases specific
  to the actual deployment environment (the scenario this product
  targets).
- **The headline 1.000 metrics are almost certainly inflated by
  domain shortcut learning**, not genuine posture understanding - see
  the domain breakdown in `reports/eval_report.md`. The only same-domain
  (Simuletic standing) test example is n=1.
- **COCO labels are not_fall by construction**, not by human review of
  each instance - a small number could depict a person actually on the
  ground (e.g. yoga, gardening) and be genuine label noise, not a model
  error, in the false-positive/negative breakdown.
- **No temporal information was used at all.** This is a deliberate
  design choice (see "Problem framing"), but it means this model cannot
  and should not replace the sustained-duration heuristic gate - it can
  only corroborate a single frame's posture reading.
- **CPU-only, no hyperparameter search.** One architecture, one learning
  rate, one epoch budget were chosen and worked well enough to be
  reportable honestly; no tuning sweep was run given the time budget for
  this pipeline.
