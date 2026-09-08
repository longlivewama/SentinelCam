# Fall-detection dataset research (Roboflow `tfmbigdata2025/voc-fall-person_falls` v1)

Produced by `ml/detector/analyze_dataset.py` plus the targeted probes described
below. Raw numbers: `fall_detector_dataset_analysis.json`. Annotated sample
grids: `dataset_samples/{train,valid,test}_samples.jpg`.

**This analysis was run before any training.** Two findings below are serious
enough to change the training design, so they are documented here first.

## 1. Basic structure — clean

| Split | Images | Labels | Instances | Fall | Person |
|---|---|---|---|---|---|
| train | 10,767 | 10,767 | 16,087 | 3,902 | 12,185 |
| valid | 2,344 | 2,344 | 3,458 | 834 | 2,624 |
| test | 2,328 | 2,328 | 4,286 | 834 | 3,452 |

- Classes: `0=Fall`, `1=Person`. License CC BY 4.0.
- **Zero** missing labels, orphan labels, empty labels, corrupt images, or
  malformed/out-of-range annotations across all 15,439 images. Formatting is
  faultless.
- All images are exactly 640x640 (Roboflow pre-resized, letterboxed).
- Median 1 box/image (mean 1.49 train, 1.84 test); max 18.
- Object sizes at 640px: 82% large, 15% medium, **only 2.5% small**. This is
  not a small-object problem, which materially relaxes the input-resolution
  and model-capacity requirements.
- Class imbalance ~3.1:1 Person:Fall in train.

## 2. FINDING A — the two classes never co-occur (structural domain split)

`images_with_multiple_classes: 0`. Not one image out of 15,439 contains both a
`Fall` and a `Person` box. Grouping filenames into source families makes the
cause obvious — every family is 100% single-class:

| Source family | Origin | Fall imgs | Person imgs |
|---|---|---|---|
| `2007_000032`-style (`#_#`) | PASCAL VOC | 0 | 6,255 |
| `people-###-` | fall dataset | 3,261 | 0 |
| `cam#####` | CCTV fall footage | 274 | 0 |
| `crop######`, `person_#`, `person_and_bike_#` | VOC crops | 0 | 610 |
| `Screenshot-*`, `fall_vid*`, `A-/C-/D-/F-/H-*` | fall sources | ~240 | 0 |

`Person` is *entirely* PASCAL VOC; `Fall` is *entirely* separate fall datasets.
A 2-class detector trained on this can reach high mAP by learning **"which
dataset is this image from"** rather than "is this person on the ground" —
the same domain-shortcut failure already documented for the keypoint
classifier in `ml/README.md`, but here it is baked into the label structure.

## 3. FINDING B — Fall images have severely incomplete `Person` annotations

Measured objectively (not eyeballed): ran the pretrained COCO detector
(`backend/yolov8n.pt`, conf 0.5, person class) over 250 random train images
from each domain and counted confident person detections with **no**
ground-truth box of either class at IoU >= 0.3.

| Domain | GT boxes | Person detections | Detections with no matching GT | Images with >=1 |
|---|---|---|---|---|
| Fall-domain | 255 | 398 | **316 (79.4%)** | **106/250 (42.4%)** |
| Person-domain (VOC) | 439 | 146 | 26 (17.8%) | 21/250 (8.4%) |

The VOC figure (17.8%) is the natural baseline of detector false positives plus
VOC's own known annotation gaps. The Fall-domain figure is **4.5x** that. This
is real: Fall images are street/CCTV scenes where the fallen person is boxed and
**the standing bystanders are simply not annotated**. The top-left tile of
`dataset_samples/train_samples.jpg` shows one labelled `Fall` beside roughly
eight unlabelled standing people.

Training a 2-class detector on this actively teaches the model that upright
people in CCTV scenes are background, which would wreck `Person` recall in
exactly the deployment domain SentinelCam runs in.

## 4. FINDING C — train/val/test leakage, concentrated entirely in the Fall class

64-bit dHash near-duplicate search across splits:

| Comparison | radius 0 | radius <=3 | radius <=6 |
|---|---|---|---|
| valid images with a near-dup in train | 25 (1.1%) | 109 (4.7%) | 155 (6.6%) |
| test images with a near-dup in train | 17 (0.7%) | 96 (4.1%) | 152 (6.5%) |

**100% of leaked images are Fall-domain.** Since valid/test contain only ~832
Fall images each, that is **155/832 = 18.6% of the validation Fall set** and
**152/833 = 18.2% of the test Fall set** duplicated from training data. Cause:
the fall sources are video frames (`cam11914` vs `cam11902`,
`Screenshot-2024-02-14-170147` vs `...170153`) split randomly per-frame instead
of per-video. Fall AP measured on the stock split is optimistically biased.

## 5. Image characteristics (sampled, n=400/split)

Brightness and sharpness are bimodal, consistent with the two-domain structure:
VOC consumer photos are bright and sharp; CCTV fall frames are darker, softer,
and often heavily compressed. Scenes span indoor rooms, corridors, streets and
outdoor pavements; camera angles range from eye-level (VOC) to elevated CCTV
(fall sources). See the JSON for the per-split percentile tables.

## 6. Consequences for training design

- The `Person` class in this dataset is **worse** than what SentinelCam already
  has. The app runs COCO-pretrained `yolov8n.pt` and `yolov8n-pose.pt`, trained
  on fully-annotated COCO persons. Re-learning `Person` from a dataset whose
  fall images omit most people would be a downgrade.
- Findings A and B are both **artifacts of the `Person` class**. Dropping it
  makes them disappear rather than needing to be worked around: for a
  single-class `Fall` detector, an unannotated standing bystander is correctly
  background, and the VOC half becomes 6,887 richly-varied **hard-negative**
  images of upright people — precisely the supervision that suppresses
  false-positive falls, which is the production heuristic's known weak spot.
- Residual noise under that framing, measured: only 2.0% of VOC person boxes are
  "lying-shaped" (w/h > 1.3, the production heuristic's own threshold),
  affecting 3.2% of train images — and most of those are wide group boxes or
  leaning poses, not genuine unlabelled falls. Acceptable and documented.
- Finding C is fixed independently by de-duplicating valid/test against train
  before reporting any metric.

