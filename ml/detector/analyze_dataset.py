"""
Dataset research for the Roboflow fall-detection dataset, run BEFORE any
training decision is made (augmentation strategy, model size, image size
and the split/leakage sanity checks all depend on what's actually in
here, not on assumptions).

Measures, per split:
  - image/label file counts, orphans (image with no label, label with no image)
  - class distribution (instances and images per class)
  - boxes per image, empty-label (background) images
  - image resolutions and aspect ratios
  - bounding-box geometry: normalized w/h/area, COCO-style small/medium/large
    buckets (computed at the training image size, since that is what
    actually determines whether an object is "small" for the detector)
  - annotation validity: out-of-range coords, zero/negative area, bad class ids
  - corrupt/unreadable images
  - image quality proxies: mean brightness (lighting variation), variance of
    Laplacian (blur/sharpness)
  - exact + near-duplicate images via 64-bit dHash, INCLUDING across splits
    (train/val/test leakage is the single most common way a detection
    benchmark silently lies)

Writes a JSON summary (machine-readable, for later reference in the report)
and prints a human summary. Renders annotated sample grids separately via
--samples.

Usage:
    python3 ml/detector/analyze_dataset.py
    python3 ml/detector/analyze_dataset.py --samples   # also write sample grids
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ML_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ML_DIR / "data" / "fall_detection"
REPORT_DIR = ML_DIR / "reports"
SAMPLE_DIR = REPORT_DIR / "dataset_samples"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_IMGSZ = 640  # bucket small/medium/large at the size the model will see

# COCO object-size convention, in pixels^2 at TRAIN_IMGSZ.
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96


def find_splits(root: Path) -> dict:
    """Roboflow YOLOv8 exports lay out <split>/images + <split>/labels."""
    splits = {}
    for name in ("train", "valid", "val", "test"):
        d = root / name
        if (d / "images").is_dir():
            splits[name] = d
    return splits


def dhash(img_gray, hash_size=8) -> int:
    resized = cv2.resize(img_gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def parse_label(path: Path):
    """Returns (rows, errors). rows = [(cls, cx, cy, w, h)] normalized."""
    rows, errors = [], []
    try:
        text = path.read_text()
    except Exception as exc:
        return rows, [f"unreadable: {exc}"]
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # YOLO detection = 5 fields. More fields => segmentation polygon
        # or pose keypoints, which changes what task this dataset is for.
        if len(parts) != 5:
            errors.append(f"line {lineno}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"line {lineno}: non-numeric field")
            continue
        rows.append((cls, cx, cy, w, h))
    return rows, errors


def analyze(root: Path, quality_sample: int, seed: int = 42):
    with open(root / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get("names") or []
    if isinstance(class_names, dict):
        class_names = [class_names[k] for k in sorted(class_names)]
    nc = cfg.get("nc", len(class_names))

    splits = find_splits(root)
    rng = random.Random(seed)

    summary = {
        "dataset_root": str(root),
        "data_yaml": {"nc": nc, "names": class_names},
        "splits": {},
        "issues": defaultdict(list),
    }

    all_hashes = {}  # hash -> (split, filename)
    dupe_pairs = []
    cross_split_dupes = []

    for split_name, split_dir in splits.items():
        img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)

        cls_instances = Counter()
        cls_images = Counter()
        boxes_per_image = []
        empty_label_images = 0
        norm_w, norm_h, norm_area, box_ar = [], [], [], []
        size_buckets = Counter()
        per_class_area = defaultdict(list)
        ann_errors = []
        resolutions = Counter()
        corrupt = []
        brightness, sharpness = [], []
        multi_class_images = 0

        # Reading every image is the expensive part; resolution comes from
        # a cheap header read, quality metrics from a random subsample.
        quality_idx = set(rng.sample(range(len(images)), min(quality_sample, len(images)))) if images else set()

        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                corrupt.append(img_path.name)
                continue
            h_px, w_px = img.shape[:2]
            resolutions[(w_px, h_px)] += 1

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsh = dhash(gray)
            prev = all_hashes.get(hsh)
            if prev is not None:
                pair = (f"{prev[0]}/{prev[1]}", f"{split_name}/{img_path.name}")
                dupe_pairs.append(pair)
                if prev[0] != split_name:
                    cross_split_dupes.append(pair)
            else:
                all_hashes[hsh] = (split_name, img_path.name)

            if i in quality_idx:
                brightness.append(float(gray.mean()))
                sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            rows, errs = parse_label(lbl_path)
            for e in errs:
                ann_errors.append(f"{split_name}/{lbl_path.name}: {e}")

            if not rows:
                empty_label_images += 1
            boxes_per_image.append(len(rows))

            classes_here = set()
            for cls, cx, cy, bw, bh in rows:
                if cls < 0 or cls >= max(nc, 1):
                    ann_errors.append(f"{split_name}/{lbl_path.name}: class id {cls} out of range [0,{nc - 1}]")
                    continue
                if bw <= 0 or bh <= 0:
                    ann_errors.append(f"{split_name}/{lbl_path.name}: non-positive box {bw:.4f}x{bh:.4f}")
                    continue
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0) or bw > 1.001 or bh > 1.001:
                    ann_errors.append(
                        f"{split_name}/{lbl_path.name}: out-of-range box cx={cx:.3f} cy={cy:.3f} w={bw:.3f} h={bh:.3f}")

                cls_instances[cls] += 1
                classes_here.add(cls)
                norm_w.append(bw)
                norm_h.append(bh)
                area = bw * bh
                norm_area.append(area)
                box_ar.append(bw / bh if bh > 0 else 0.0)
                px_area = area * TRAIN_IMGSZ * TRAIN_IMGSZ
                per_class_area[cls].append(px_area)
                size_buckets["small" if px_area < SMALL_AREA
                              else "medium" if px_area < MEDIUM_AREA else "large"] += 1
            for c in classes_here:
                cls_images[c] += 1
            if len(classes_here) > 1:
                multi_class_images += 1

        def stats(vals):
            if not vals:
                return None
            a = np.array(vals, dtype=np.float64)
            return {
                "count": int(a.size), "min": round(float(a.min()), 4),
                "p25": round(float(np.percentile(a, 25)), 4),
                "median": round(float(np.median(a)), 4),
                "p75": round(float(np.percentile(a, 75)), 4),
                "max": round(float(a.max()), 4), "mean": round(float(a.mean()), 4),
            }

        summary["splits"][split_name] = {
            "images": len(images),
            "label_files": len(labels),
            "images_without_label_file": len(missing_labels),
            "label_files_without_image": len(orphan_labels),
            "corrupt_images": corrupt,
            "empty_label_images": empty_label_images,
            "total_instances": int(sum(cls_instances.values())),
            "instances_per_class": {
                (class_names[c] if c < len(class_names) else f"id{c}"): n
                for c, n in sorted(cls_instances.items())
            },
            "images_per_class": {
                (class_names[c] if c < len(class_names) else f"id{c}"): n
                for c, n in sorted(cls_images.items())
            },
            "images_with_multiple_classes": multi_class_images,
            "boxes_per_image": stats(boxes_per_image),
            "resolutions_top5": [
                {"wxh": f"{w}x{h}", "count": n} for (w, h), n in resolutions.most_common(5)
            ],
            "distinct_resolutions": len(resolutions),
            "box_norm_width": stats(norm_w),
            "box_norm_height": stats(norm_h),
            "box_norm_area": stats(norm_area),
            "box_aspect_ratio_w_over_h": stats(box_ar),
            f"size_buckets_at_{TRAIN_IMGSZ}px": dict(size_buckets),
            "per_class_median_px_area_at_640": {
                (class_names[c] if c < len(class_names) else f"id{c}"): round(float(np.median(v)), 1)
                for c, v in sorted(per_class_area.items())
            },
            "brightness_sampled": stats(brightness),
            "sharpness_var_laplacian_sampled": stats(sharpness),
            "annotation_errors": ann_errors[:50],
            "annotation_error_count": len(ann_errors),
            "sample_missing_labels": missing_labels[:10],
            "sample_orphan_labels": orphan_labels[:10],
        }

    summary["duplicates"] = {
        "exact_or_near_duplicate_pairs": len(dupe_pairs),
        "cross_split_duplicate_pairs": len(cross_split_dupes),
        "cross_split_examples": cross_split_dupes[:20],
        "within_split_examples": [p for p in dupe_pairs if p not in cross_split_dupes][:10],
        "method": "64-bit dHash, exact hash collision (identical or visually near-identical)",
    }
    summary["issues"] = dict(summary["issues"])
    return summary


def render_samples(root: Path, out_dir: Path, per_split: int = 12, seed: int = 42):
    """Writes annotated contact sheets so the annotations can actually be
    eyeballed - box coordinates looking valid numerically says nothing
    about whether they're on the right object."""
    with open(root / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names") or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    colors = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (200, 0, 200), (0, 200, 200)]

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = []

    for split_name, split_dir in find_splits(root).items():
        img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not images:
            continue
        picks = rng.sample(images, min(per_split, len(images)))

        tiles, cell = [], 320
        for p in picks:
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            rows, _ = parse_label(lbl_dir / f"{p.stem}.txt")
            for cls, cx, cy, bw, bh in rows:
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                color = colors[cls % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, w // 320))
                label = names[cls] if cls < len(names) else str(cls)
                cv2.putText(img, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.6, w / 1000), color, 2)
            scale = cell / max(h, w)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            canvas = np.full((cell, cell, 3), 30, dtype=np.uint8)
            oy, ox = (cell - resized.shape[0]) // 2, (cell - resized.shape[1]) // 2
            canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
            cv2.putText(canvas, p.name[:34], (4, cell - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            tiles.append(canvas)

        if not tiles:
            continue
        cols = 4
        while len(tiles) % cols:
            tiles.append(np.full((cell, cell, 3), 30, dtype=np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        out = out_dir / f"{split_name}_samples.jpg"
        cv2.imwrite(str(out), grid)
        written.append(out)
        print(f"  wrote {out}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA_DIR))
    ap.add_argument("--samples", action="store_true", help="also render annotated sample grids")
    ap.add_argument("--quality-sample", type=int, default=400,
                    help="images per split to compute brightness/sharpness on")
    args = ap.parse_args()

    root = Path(args.root)
    if not (root / "data.yaml").exists():
        raise SystemExit(f"No data.yaml under {root} - run download_dataset.py first.")

    summary = analyze(root, args.quality_sample)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORT_DIR / "fall_detector_dataset_analysis.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    if args.samples:
        print("\nRendering annotated samples...")
        render_samples(root, SAMPLE_DIR)


if __name__ == "__main__":
    main()
