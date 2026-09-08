"""
Trains the single-class 'Fall' YOLO detector on the prepared dataset from
prepare_dataset.py.

MODEL SIZE
----------
yolov8n, fine-tuned from COCO weights, at the dataset's native 640px.
Reasons, all grounded in the measured dataset + the actual deployment target:
  - Deployment is CPU (`MODEL_DEVICE=cpu` in backend/app/config.py) inside a
    per-camera loop that ALREADY runs two YOLO models per processed frame
    (yolov8n-pose + yolov8n). A third model has to be cheap or it halves the
    achievable frame rate on every camera simultaneously.
  - The detection problem is easy in the ways that normally justify a bigger
    backbone: ONE class, median ONE object per image, and 82% of objects are
    "large" by COCO convention at 640px (only 2.5% are small). Capacity is not
    the binding constraint here; data quality is.
  - COCO pretraining already encodes "person" strongly, which is most of the
    representation this task needs - we are teaching posture, not objectness.
yolov8s is trained as a controlled comparison by `--model yolov8s.pt` so the
size choice is backed by a measurement rather than an assumption (see
benchmark.py); nano is only kept if it is genuinely competitive.

AUGMENTATION
------------
Chosen from the dataset analysis, not from defaults. The governing constraint:
**this class is defined by orientation relative to gravity.** Any augmentation
that can rotate an upright person toward horizontal manufactures false
positives in the training signal, so the usual rotation/shear/perspective
knobs are deliberately near-zero here even though they are standard elsewhere.

  fliplr=0.5      YES. A fall is left-right symmetric; free 2x diversity.
  flipud=0.0      NEVER. Vertically flipping a person inverts gravity and would
                  teach the model that "head below hips" is an ordinary view -
                  destroying the exact cue that separates Fall from Person.
  degrees=5.0     SMALL, not the usual 10-15. Real CCTV cameras are sometimes
                  slightly tilted, so a few degrees is realistic; large
                  rotations would literally rotate a standing person into a
                  lying pose while keeping the "no fall" label.
  shear=0.0       NO. Unrealistic geometry, and mild shear mimics rotation's
                  orientation damage.
  perspective=0.0 NO. Held at zero for this controlled first run; elevated-camera
                  perspective is already present in the CCTV half of the data.
  scale=0.5       YES. Camera distance varies a lot across the fall sources.
  translate=0.1   YES. Cheap, safe, standard.
  hsv_h/s/v       YES (0.015/0.7/0.4). Measured brightness is strongly bimodal
                  (bright VOC photos vs dark, compressed CCTV frames), and the
                  product runs day and night, so colour/exposure invariance is
                  directly relevant rather than decorative.
  mosaic=1.0      YES, and it is the single most valuable augmentation here.
                  Finding A in the dataset analysis is that no image contains
                  both a fall and an upright person, so the model could cheat on
                  global image statistics. Mosaic composites 4 images into one,
                  so fall content and hard-negative upright-people content land
                  in the SAME image with the SAME global statistics - it
                  synthesizes the co-occurrence the dataset structurally lacks.
                  It also keeps the 62%-background split from starving batches:
                  a 4-tile mosaic contains at least one positive ~83% of the
                  time (1 - 0.62^4).
  close_mosaic=10 Standard: disable mosaic for the last 10 epochs so the model
                  finishes on undistorted, real-layout images.
  mixup=0.0       NO. Alpha-blending produces translucent ghost people that no
                  camera ever sees, and blending a fall image with a background
                  image yields an ambiguous target that would undercut the
                  hard-negative supervision this dataset is being used for.
  copy_paste=0.0  NO. Requires segmentation masks; this is a detection-only
                  export (verified: every label line has exactly 5 fields).

Note: albumentations is NOT installed in backend/venv, so ultralytics' optional
blur/CLAHE/grayscale pipeline does not silently run. Augmentation is exactly
what is listed above - worth stating, because those extras are applied
invisibly when the package happens to be present.

SMOKE TEST
----------
`--smoke` does NOT use ultralytics' `fraction` argument. `fraction` keeps the
first N entries of the *sorted* file list, and since the VOC background images
sort first (`2007_*`), it produced a 554-image train sample with zero Fall boxes
- box_loss and dfl_loss were identically 0 for the whole run because there was
nothing to regress. `--smoke` instead builds a stratified positives+backgrounds
subset via build_smoke_subset(); see that function.

Usage:
    python3 ml/detector/train.py                       # yolov8n, full run
    python3 ml/detector/train.py --smoke               # 2 epochs on a stratified subset
    python3 ml/detector/train.py --model yolov8s.pt --name fall_yolov8s
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
PREPARED = ML_DIR / "data" / "fall_detection_prepared"
DATA_YAML = PREPARED / "data.yaml"
SMOKE_DIR = ML_DIR / "data" / "fall_detection_smoke"
SMOKE_YAML = SMOKE_DIR / "data.yaml"
RUNS_DIR = ML_DIR / "runs"

AUGMENTATION = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=5.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
)

# Smoke-subset sizes. Small enough to run in a couple of minutes, large enough
# that a batch of 32 reliably contains positives.
SMOKE_SIZES = {"train": (220, 340), "valid": (120, 180)}  # (positives, backgrounds)


def build_smoke_subset(seed: int = 42) -> Path:
    """Materialise a small STRATIFIED sample of the prepared dataset.

    Why this exists instead of ultralytics' `fraction=0.05`: `fraction` slices
    the *sorted* file list (ultralytics/data/base.py:126,
    `im_files[: round(len(im_files) * fraction)]`). Our filenames start with the
    VOC-derived background images (`2007_*`), so a 5% slice of the train split
    returned 554 images with ZERO Fall boxes - the smoke run then trained on
    pure background and reported box_loss=0 / dfl_loss=0, which looks like a
    broken pipeline but is just an empty sample.

    This samples positives and backgrounds separately, at roughly the real
    split's 38% positive rate, so the smoke run exercises the same code paths as
    the full run with a guaranteed non-zero object count. The prepared dataset
    is only READ; images are symlinked to their real targets and labels copied,
    so nothing in fall_detection_prepared/ is moved, deleted, or rewritten.
    """
    rng = random.Random(seed)
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)

    summary = {}
    for split, (n_pos, n_bg) in SMOKE_SIZES.items():
        src_img = PREPARED / split / "images"
        src_lbl = PREPARED / split / "labels"
        (SMOKE_DIR / split / "images").mkdir(parents=True)
        (SMOKE_DIR / split / "labels").mkdir(parents=True)

        positives, backgrounds = [], []
        for lbl in sorted(src_lbl.glob("*.txt")):
            (positives if lbl.stat().st_size > 0 else backgrounds).append(lbl.stem)

        picked = (rng.sample(positives, min(n_pos, len(positives)))
                  + rng.sample(backgrounds, min(n_bg, len(backgrounds))))

        instances = 0
        for stem in picked:
            # resolve() so we link to the raw JPEG, not to prepared/'s own symlink
            (SMOKE_DIR / split / "images" / f"{stem}.jpg").symlink_to(
                (src_img / f"{stem}.jpg").resolve())
            text = (src_lbl / f"{stem}.txt").read_text()
            (SMOKE_DIR / split / "labels" / f"{stem}.txt").write_text(text)
            instances += len(text.split("\n")) - 1

        summary[split] = {"images": len(picked),
                          "positive_images": min(n_pos, len(positives)),
                          "background_images": min(n_bg, len(backgrounds)),
                          "instances": instances}

    SMOKE_YAML.write_text(
        "# Auto-generated by train.py --smoke. Stratified sample of\n"
        "# fall_detection_prepared/ - see build_smoke_subset(). Not for real training.\n"
        f"path: {SMOKE_DIR.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "\n"
        "nc: 1\n"
        "names: ['Fall']\n"
    )
    print("Smoke subset:", json.dumps(summary, indent=2))
    if summary["train"]["instances"] == 0:
        raise SystemExit("smoke subset has no Fall instances - refusing to run a blind smoke test")
    return SMOKE_YAML


# DATALOADER WORKERS ON MPS - measured, do not "fix" this back to workers>0.
#
# ultralytics/engine/trainer.py:127 unconditionally sets workers=0 for cpu and
# mps, commented "faster CPU training as time dominated by inference, not
# dataloading". With mosaic=1.0 every sample decodes 4 JPEGs, so that premise
# was worth testing on this box (M5, 10 cores, 16 GB unified, batch=32, 640px):
#
#     dataloader-only throughput      workers=0  125.9 ms/batch
#                                     workers=2   70.0 ms/batch
#                                     workers=4   44.1 ms/batch
#                                     workers=8   25.1 ms/batch
#     MPS fwd+bwd compute                        765.5 ms/batch
#
# So compute does dominate (ultralytics is directionally right), but workers=0
# leaves that 126 ms serial with the GPU step - about 14% of wall clock, ~1 h
# over an 80-epoch run - which looked worth reclaiming.
#
# It is not. Forcing workers=4 via a DetectionTrainer subclass HANGS, twice out
# of two attempts, partway through epoch 1: the process wedges at 0.2% CPU with
# the Metal stream stuck in MPSStream::copy_and_sync -> waitUntilCompleted and a
# 9.1 GB footprint. Cause is the 16 GB *unified* pool: training alone already
# peaks at ~8.95 GB of MPS allocation, and worker processes take their RSS from
# that same memory, so the Metal allocator deadlocks under the pressure. The
# same worker counts benchmark fine in isolation only because that measurement
# ran before MPS was initialised.
#
# Conclusion: keep ultralytics' workers=0 on MPS. The 14% is real but it is not
# worth a non-deterministic wedge on a multi-hour run. `workers` below is still
# passed for correctness on a CUDA box, where trainer.py:127 does not apply.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--name", default="fall_yolov8n")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 5% of the data, to validate the pipeline before a long run")
    args = ap.parse_args()

    from ultralytics import YOLO

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} missing - run prepare_dataset.py first.")

    # A smoke run trains on a stratified subset built here, NOT on `fraction`,
    # which would silently select an all-background alphabetical slice.
    data_yaml = build_smoke_subset(seed=42) if args.smoke else DATA_YAML

    kwargs = dict(
        data=str(data_yaml),
        epochs=2 if args.smoke else args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR),
        name=f"{args.name}_smoke" if args.smoke else args.name,
        exist_ok=False,          # never silently overwrite a previous experiment
        seed=42,
        deterministic=True,
        patience=args.patience,
        workers=8,  # forced to 0 on mps by ultralytics - see note above
        val=True,
        plots=True,
        **AUGMENTATION,
    )
    if args.smoke:
        kwargs["close_mosaic"] = 0

    print(f"Training {args.model} -> {RUNS_DIR / kwargs['name']}")
    print(json.dumps({k: v for k, v in kwargs.items() if k != "data"}, indent=2))

    model = YOLO(args.model)
    start = time.time()
    results = model.train(**kwargs)
    elapsed = time.time() - start

    save_dir = Path(results.save_dir)
    meta = {
        "base_model": args.model,
        "dataset": str(data_yaml),
        "epochs_requested": kwargs["epochs"],
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers_requested": 8,
        "elapsed_seconds": round(elapsed, 1),
        "augmentation": AUGMENTATION,
        "save_dir": str(save_dir),
        "final_metrics": {k: float(v) for k, v in (results.results_dict or {}).items()},
    }
    (save_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone in {elapsed / 60:.1f} min. Weights: {save_dir / 'weights' / 'best.pt'}")
    print(json.dumps(meta["final_metrics"], indent=2))


if __name__ == "__main__":
    main()
