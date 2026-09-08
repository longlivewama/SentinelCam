# Model card — `fall_detector_v1`

The fall detector SentinelCam actually runs in production. Machine-readable
counterpart: [`exported/fall_detector_v1.metadata.json`](exported/fall_detector_v1.metadata.json).

| | |
|---|---|
| **Task** | Single-class object detection — locate fallen people in a frame |
| **Architecture** | YOLOv8n, fine-tuned from COCO weights |
| **Classes** | `0: Fall` (one class only) |
| **Input** | 640×640 BGR frame |
| **Size** | 5.4 MB · 2.68 M fused parameters · 6.8 GFLOPs |
| **Artifact** | `ml/exported/fall_detector_v1.pt` (sha256 `da38c55f…a06ba0`) |
| **Training run** | `ml/runs/fall_yolov8n` — 80 epochs, 18.5 h on Apple Silicon (MPS) |
| **Licence** | Fine-tuned from AGPL-3.0 Ultralytics weights on a CC BY 4.0 dataset — see [`NOTICE.md`](../NOTICE.md) |

---

## What it does, and what it replaced

This model detects fallen people **directly**. Everything before it inferred a
fall from a geometric proxy: run a pose model, check whether the bounding box is
wide and the shoulders have bunched down near the hips, and if that posture holds
for 1.2 s, call it a fall. That heuristic still exists (and is still the fallback
when this model can't be loaded), but it cannot distinguish a person lying on the
floor from a person doing push-ups, and it needs confident keypoints to work at all.

A second, earlier artifact — `fall_classifier_v1`, a 10-feature keypoint MLP —
is **not** this model and has a much weaker standing: it was only ever wired in
as a confidence nudge, and its 1.000 test metrics are flagged in
[`reports/eval_report.md`](reports/eval_report.md) as likely domain-shortcut
artifacts. It remains disabled by default.

---

## Data

**Source:** Roboflow `tfmbigdata2025/voc-fall-person_falls` v1, CC BY 4.0.
Not redistributed here — `ml/detector/download_dataset.py` fetches it.

**Preparation** (`ml/detector/prepare_dataset.py`, manifest in
[`reports/fall_detector_prepare_manifest.json`](reports/fall_detector_prepare_manifest.json)):

| Split | Images | Fall instances | Background images |
|---|---:|---:|---:|
| train | 11,074 | 4,210 | 6,887 |
| valid | 2,189 | 679 | 1,512 |
| test | 2,176 | 681 | 1,495 |

Two decisions came out of the pre-training dataset analysis
([`reports/fall_detector_dataset_analysis.md`](reports/fall_detector_dataset_analysis.md))
and both materially shape what this model is:

1. **The raw export's second class, `Person`, was dropped.** Not one image out
   of 15,439 contained both a `Fall` and a `Person` box — `Person` came entirely
   from PASCAL VOC, `Fall` entirely from separate fall datasets. A two-class
   detector could have scored well by learning *which dataset an image came
   from* rather than whether someone is on the ground. Dropping `Person` and
   keeping the VOC images as **background negatives** removes that shortcut and
   turns those images into what they are actually useful for: 6,887 examples of
   upright people that must not fire.

2. **307 near-duplicate images were moved out of valid/test into train.** The
   raw split leaked visually near-identical frames across the boundary, which
   would have inflated the reported metrics.

---

## Training configuration

`yolov8n` at the dataset's native 640 px, 80 epochs, batch 32, seed 42,
`patience=20` (never triggered — the run went the full 80).

The augmentation choices are the non-obvious part, and they follow from one
property of the task: **this class is defined by orientation relative to
gravity.** Any augmentation that can rotate an upright person toward horizontal
manufactures false positives in the training signal.

| Setting | Value | Why |
|---|---|---|
| `flipud` | **0.0** | Vertically flipping a person inverts gravity — it would teach the model that "head below hips" is an ordinary view, destroying the exact cue that defines the class |
| `degrees` | **5.0** | Real CCTV cameras tilt a few degrees; the usual 10–15° would rotate standing people into lying poses while keeping the "not a fall" label |
| `shear`, `perspective` | **0.0** | Mild shear mimics rotation's orientation damage |
| `fliplr` | 0.5 | A fall is left–right symmetric — free 2× diversity |
| `mosaic` | 1.0 | The highest-value augmentation here. Since no source image contains both a fall and an upright person, mosaic composites 4 images into one and puts fall content and hard-negative upright-people content into the *same* image with the *same* global statistics — synthesising the co-occurrence the dataset structurally lacks |
| `close_mosaic` | 10 | Last 10 epochs train on undistorted, real-layout images |
| `hsv_h/s/v` | 0.015 / 0.7 / 0.4 | Measured brightness is strongly bimodal (bright VOC photos vs. dark compressed CCTV) and the product runs day and night |

Full argument dump: `ml/runs/fall_yolov8n/args.yaml`.

---

## Results

Measured with the exported artifact at 640 px on CPU. The validation figures
reproduce the training run's final epoch exactly, which confirms the committed
file is the checkpoint that produced them.

| | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Validation (2,189 images) | 0.812 | 0.710 | 0.826 | 0.528 |
| **Held-out test (2,176 images)** | **0.846** | **0.800** | **0.877** | **0.557** |

Test scores slightly *above* validation, with the gap consistent across all four
metrics — no sign of overfitting to the validation split, and the leakage
cleanup above means the test set is genuinely held out.

Loss curves, PR/F1 curves and confusion matrices:
[`reports/fall_detector/`](reports/fall_detector/).

**Spot check through the application's own inference path** (40 random
labelled-fall and 40 random background test images, at the deployed 0.4
confidence floor):

- 36 / 40 labelled-fall frames produced at least one `Fall` box
- 0 / 40 background frames produced any box

### How to read these numbers honestly

- **mAP@50-95 of 0.557 is modest, and that is expected.** It rewards tight box
  regression; a "fallen person" has genuinely ambiguous extents (do sprawled
  limbs count?) and annotators disagree. mAP@50 — "did it find the right person
  in roughly the right place" — is the metric that matches what the product
  needs, and it is 0.877.
- **Recall of 0.800 is per *image*, not per *incident*.** A real fall persists
  for seconds and is sampled every 5th frame, so an incident gets dozens of
  independent chances. Per-incident recall is much higher than 0.80 — but it is
  **not measured**, because that requires labelled *video*, and this dataset is
  stills. See Limitations.
- Precision of 0.846 means roughly **1 in 6.5 raw per-frame detections is
  wrong**. The application never alerts on a single frame for exactly this
  reason — see Integration below.

---

## Limitations

Stated plainly, because deploying this into a care setting without knowing them
would be irresponsible.

1. **No video-level evaluation.** Everything above is per-frame on stills. The
   end-to-end quantity that matters — falls detected per fall that happened,
   and false alerts per hour of ordinary footage — has not been measured,
   because no labelled fall *video* corpus is wired into this pipeline. This is
   the single largest gap and the top item in the roadmap.
2. **No hard-negative validation for floor-level activity.** The background
   images are PASCAL VOC — ordinary photographs of upright people. They do not
   contain someone doing sit-ups, a child playing on the floor, a person
   crouching to reach a low shelf, or someone lying on a sofa. Those are the
   realistic false-positive sources in a real deployment and the test set says
   nothing about them.
3. **The source data is not care-home footage.** It mixes web-sourced fall
   images with some CCTV frames. Camera height, lens, lighting, and occlusion in
   an actual installation will differ, and detection quality will differ with it.
   Expect to fine-tune on site data before trusting it operationally.
4. **Single frame, no motion.** The model sees posture, not dynamics. It cannot
   distinguish a fall from lying down deliberately. The application's
   sustained-duration gate deliberately does not try to either — it treats
   "person is on the ground and stays there" as the alertable condition, which
   is a product decision, not a model capability.
5. **Small objects are underrepresented.** Only 2.5% of training objects are
   "small" by COCO convention at 640 px. A subject far from the camera in a
   large room is out of distribution.
6. **`yolov8s` was never trained as a comparison.** `ml/detector/train.py`
   supports `--model yolov8s.pt` and the size choice is argued from the dataset
   (one class, one object per image, 82% large objects, CPU deployment target)
   — but the argument is not backed by a measured A/B.

**This is decision-support software, not a medical device.** It must not be the
only thing standing between a person and help.

---

## Integration

The model is the primary fall signal; nothing about it is a placeholder.

```
frame ─▶ fall_object_detector.detect()      # YOLO inference, conf ≥ 0.4
      ─▶ ModelFallDetector.update()          # per-subject tracking + sustain gate + debounce
      ─▶ Event(detector="model", confidence, video_timestamp_seconds)
```

| Piece | Location |
|---|---|
| Checkpoint loading, class-name verification, fail-safe inference | `backend/app/services/detection/fall_object_detector.py` |
| Temporal gate (sustain + debounce + tracking) | `backend/app/services/detection/fall_detection.py::ModelFallDetector` |
| Strategy selection | `backend/app/services/detection/fall_pipeline.py` |

**A single frame never raises an alert.** At 0.846 precision, per-frame
triggering would alert on every one-frame false positive. A `Fall` box must
persist for the same tracked subject for `FALL_DETECTOR_MIN_SUSTAINED_SECONDS`
(default 0.6 s) before an event fires, and that subject is then debounced for
10 s. A real fall clears this trivially; isolated noise cannot.

The loader **verifies the checkpoint's class names are exactly `{0: 'Fall'}`**
and refuses to load anything else, so a swapped checkpoint fails loudly instead
of mapping some other model's class 0 to "fall".

### Configuration

| Setting | Default | Effect |
|---|---|---|
| `FALL_DETECTOR_MODEL_PATH` | `ml/exported/fall_detector_v1.pt` | Empty string forces the pose heuristic |
| `FALL_DETECTION_MODE` | `auto` | `auto` \| `model` \| `heuristic` \| `hybrid` |
| `FALL_DETECTOR_MIN_CONFIDENCE` | `0.4` | Per-box confidence floor |
| `FALL_DETECTOR_MIN_SUSTAINED_SECONDS` | `0.6` | How long a detection must persist |

`auto` uses the trained model when it loads and falls back to the heuristic when
it doesn't. Use `model` where a silent downgrade to the weaker heuristic would
be worse than a visible outage. `GET /api/system/status` reports both the
configured and the *active* mode, so a fallback is never silent to an operator.

### Inference cost

~120 ms per 640 px frame on CPU (Apple Silicon, batch 16). At the default
`VIDEO_ANALYSIS_FRAME_STRIDE=5`, a 60 s 25 fps upload runs 300 inferences ≈ 36 s
of model time. For live cameras this is a **third** model per processed frame
alongside pose and object detection — budget accordingly, or raise
`DETECTION_FRAME_STRIDE`.

---

## Reproducing

```bash
cd ml
python detector/download_dataset.py     # needs ROBOFLOW_API_KEY (see .env.example)
python detector/prepare_dataset.py      # single-class conversion + leakage cleanup
python detector/analyze_dataset.py      # the analysis this card cites
python detector/train.py                # 80 epochs; ~18.5 h on MPS
```

The training run directory (`ml/runs/`) and the raw dataset are gitignored —
large and regenerable. The **selected artifact**, its metadata sidecar, its
metrics curves and the full training results CSV are committed, so the numbers
in this card can be checked without retraining.

Do not overwrite `ml/exported/fall_detector_v1.pt`. Promote a retrained model as
`fall_detector_v2.pt` alongside it, with its own metadata sidecar and a row in
the results table above.
