"""
Builds the training-ready single-class 'Fall' dataset from the raw Roboflow
export, applying the three decisions justified in
reports/fall_detector_dataset_analysis.md:

1. DROP the `Person` class (nc 2 -> 1). The dataset's `Person` labels come
   entirely from PASCAL VOC while `Fall` comes entirely from separate CCTV/fall
   sources, with zero co-occurrence in any of the 15,439 images - so a 2-class
   model can separate them by image domain instead of by posture. Worse, 79.4%
   of confident person detections in Fall images have no ground-truth box at
   all (vs 17.8% on the VOC half): the fall sources annotate the fallen person
   and ignore the standing bystanders. SentinelCam already detects people with
   COCO-pretrained yolov8n/yolov8n-pose, which is strictly better annotated, so
   nothing is lost by not re-learning `Person` here.

2. KEEP the VOC images as explicit BACKGROUND (hard negatives). Once `Person`
   is gone they are 6,887 varied scenes full of upright, walking, sitting,
   riding people with no fall in them - exactly the supervision that suppresses
   false-positive falls, which is the production heuristic's known weak spot.
   An unannotated standing bystander in a Fall image is now *correctly*
   background too, which is what dissolves finding B rather than working
   around it.

3. DE-LEAK valid/test against train. 100% of the cross-split near-duplicates
   are Fall-domain video frames (the fall sources were split per-frame, not
   per-video), which is ~18.5% of the Fall val and test sets. Leaked images are
   MOVED INTO TRAIN rather than deleted, so no data is thrown away while the
   held-out splits stay honest.

Images are symlinked, not copied (the raw export is 861 MB); only the rewritten
label files are new bytes.

Usage:  python3 ml/detector/prepare_dataset.py [--dup-radius 6]
Writes: ml/data/fall_detection_prepared/{train,valid,test}/{images,labels}
        ml/data/fall_detection_prepared/data.yaml
        ml/reports/fall_detector_prepare_manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ML_DIR = Path(__file__).resolve().parent.parent
RAW = ML_DIR / "data" / "fall_detection"
OUT = ML_DIR / "data" / "fall_detection_prepared"
REPORT_DIR = ML_DIR / "reports"

FALL_CLASS_RAW = 0   # 'Fall' in the raw export
PERSON_CLASS_RAW = 1  # 'Person' - dropped, see docstring


def dhash(gray, size: int = 8) -> np.ndarray:
    resized = cv2.resize(gray, (size + 1, size))
    return np.packbits((resized[:, 1:] > resized[:, :-1]).flatten())


def read_rows(label_path: Path):
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        rows.append((int(float(parts[0])), *(float(v) for v in parts[1:])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dup-radius", type=int, default=6,
                    help="dHash Hamming radius treated as a near-duplicate (default 6)")
    args = ap.parse_args()

    if not (RAW / "data.yaml").exists():
        raise SystemExit(f"Raw dataset missing at {RAW} - run download_dataset.py first.")

    # ---- pass 1: index every raw image, its domain, and its perceptual hash ----
    print("Hashing raw images...")
    records = []  # (split, stem, img_path, is_fall, hash)
    for split in ("train", "valid", "test"):
        img_dir = RAW / split / "images"
        for img_path in sorted(img_dir.glob("*.jpg")):
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"  skipping unreadable {img_path.name}")
                continue
            rows = read_rows(RAW / split / "labels" / f"{img_path.stem}.txt")
            is_fall = any(r[0] == FALL_CLASS_RAW for r in rows)
            records.append({"split": split, "stem": img_path.stem, "path": img_path,
                            "is_fall": is_fall, "rows": rows, "hash": dhash(gray)})
    print(f"  indexed {len(records)} images")

    # ---- pass 2: find val/test images that near-duplicate a train image ----
    train_bits = np.unpackbits(
        np.stack([r["hash"] for r in records if r["split"] == "train"]), axis=1)
    leaked = set()
    for split in ("valid", "test"):
        idx = [i for i, r in enumerate(records) if r["split"] == split]
        bits = np.unpackbits(np.stack([records[i]["hash"] for i in idx]), axis=1)
        for start in range(0, len(idx), 256):
            block = bits[start:start + 256]
            dist = (block[:, None, :] != train_bits[None, :, :]).sum(axis=2)
            for off, is_dup in enumerate(dist.min(axis=1) <= args.dup_radius):
                if is_dup:
                    leaked.add(idx[start + off])
    print(f"  {len(leaked)} val/test images near-duplicate a train image "
          f"(radius<={args.dup_radius}) -> moved into train")

    # ---- pass 3: write the prepared dataset ----
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "valid", "test"):
        (OUT / split / "images").mkdir(parents=True)
        (OUT / split / "labels").mkdir(parents=True)

    stats = {s: Counter() for s in ("train", "valid", "test")}
    for i, rec in enumerate(records):
        split = "train" if i in leaked else rec["split"]
        stem = rec["stem"]

        link = OUT / split / "images" / f"{stem}.jpg"
        if not link.exists():
            link.symlink_to(rec["path"].resolve())

        # Keep only Fall boxes, remapped to class id 0. Person boxes are
        # dropped entirely; an image left with zero rows becomes an explicit
        # background/negative image (an empty .txt, which is how ultralytics
        # represents "this image genuinely contains nothing to detect").
        fall_rows = [r for r in rec["rows"] if r[0] == FALL_CLASS_RAW]
        text = "".join(f"0 {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n" for r in fall_rows)
        (OUT / split / "labels" / f"{stem}.txt").write_text(text)

        stats[split]["images"] += 1
        stats[split]["fall_instances"] += len(fall_rows)
        stats[split]["dropped_person_instances"] += sum(
            1 for r in rec["rows"] if r[0] == PERSON_CLASS_RAW)
        if fall_rows:
            stats[split]["positive_images"] += 1
        else:
            stats[split]["background_images"] += 1
        if i in leaked:
            stats[split]["moved_in_from_leak"] += 1

    (OUT / "data.yaml").write_text(
        "# Single-class Fall detector - see ml/reports/fall_detector_dataset_analysis.md\n"
        "# for why the raw export's 'Person' class was dropped and the VOC images\n"
        "# kept as background negatives.\n"
        f"path: {OUT.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "nc: 1\n"
        "names: ['Fall']\n"
    )

    manifest = {
        "source": str(RAW),
        "output": str(OUT),
        "dup_radius": args.dup_radius,
        "leaked_images_moved_to_train": len(leaked),
        "splits": {s: dict(c) for s, c in stats.items()},
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "fall_detector_prepare_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["splits"], indent=2))
    print(f"\nWrote {OUT}/data.yaml")


if __name__ == "__main__":
    main()
