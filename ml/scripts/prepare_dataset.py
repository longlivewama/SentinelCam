"""
Builds the fall/not_fall keypoint-feature dataset from two sources (see
ml/README.md for full provenance/license details):

  1. Simuletic CCTV Incident Dataset (laying/standing, YOLO-Pose format,
     CC-BY-4.0). Labels at ml/data/raw_simuletic/labels/*.txt. Provides
     both classes but is extremely imbalanced (111 "laying"/fall vs only
     10 "standing"/not_fall instances) - not enough negative examples
     alone for a meaningful held-out test set.
  2. COCO 2017 keypoints ("person") validation-split annotations,
     CC-BY-4.0, re-hosted as YOLO-Pose label files at
     ml/data/raw_coco_pose/datasets/labels/val/*.txt. COCO photos are
     everyday snapshots of people doing ordinary things (standing,
     walking, sitting, playing sports); a person lying flat on the
     ground is a vanishing minority and not something COCO's "person"
     category specifically curates for, so every instance here is
     treated as "not_fall". This supplies the bulk of negative examples.
     Only the numeric keypoint/bbox ANNOTATIONS are used - no COCO
     images are downloaded or touched, since training operates purely on
     keypoint-derived geometric features.

Output: ml/data/{train,val,test}.csv, one row per detected-person
instance: feature_0..feature_9, label (1=fall, 0=not_fall), source_image.

Split strategy: grouped by (dataset, image) so that if an image contains
multiple people, all of that image's instances land in the same split
(no leakage of near-duplicate same-scene crops across splits). Groups
are shuffled with a fixed seed and assigned to train/val/test by a
running-instance-count threshold rather than exact per-class
stratification - see the printed per-split class counts and
ml/reports/eval_report.md for the resulting (imperfect but disclosed)
class balance per split.
"""
from __future__ import annotations

import csv
import os
import random

from features import FEATURE_NAMES, extract_features, parse_yolo_pose_label_line

SIMULETIC_LABELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_simuletic", "labels")
COCO_VAL_LABELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_coco_pose", "datasets", "labels", "val")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Simuletic source class -> our label. class 0 = "laying" (fallen) -> 1
# (fall); class 1 = "standing" -> 0 (not_fall).
SIMULETIC_CLASS_TO_LABEL = {0: 1, 1: 0}

# Cap on how many COCO not_fall instances to include, so the dataset stays
# small/fast and not overwhelmingly negative-heavy relative to the 111
# fall instances (all of which come from Simuletic - COCO contributes 0
# fall examples by construction).
MAX_COCO_INSTANCES = 300

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder (0.15) -> test

SEED = 42


def load_simuletic():
    groups = {}  # group_key -> list of instance dicts
    for fname in sorted(os.listdir(SIMULETIC_LABELS_DIR)):
        if not fname.endswith(".txt"):
            continue
        group_key = ("simuletic", fname)
        with open(os.path.join(SIMULETIC_LABELS_DIR, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = parse_yolo_pose_label_line(line)
                if parsed is None:
                    continue
                class_id, bbox, keypoints = parsed
                if class_id not in SIMULETIC_CLASS_TO_LABEL:
                    continue
                features = extract_features(bbox, keypoints)
                if features is None:
                    continue
                groups.setdefault(group_key, []).append({
                    "features": features,
                    "label": SIMULETIC_CLASS_TO_LABEL[class_id],
                    "source_image": f"simuletic/{fname.replace('.txt', '.png')}",
                })
    return groups


def load_coco_negatives(rng: random.Random):
    all_files = [f for f in os.listdir(COCO_VAL_LABELS_DIR) if f.endswith(".txt")]
    rng.shuffle(all_files)

    groups = {}
    total = 0
    for fname in all_files:
        if total >= MAX_COCO_INSTANCES:
            break
        group_key = ("coco", fname)
        with open(os.path.join(COCO_VAL_LABELS_DIR, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = parse_yolo_pose_label_line(line)
                if parsed is None:
                    continue
                _class_id, bbox, keypoints = parsed  # COCO-pose has a single class (person)
                features = extract_features(bbox, keypoints)
                if features is None:
                    continue
                groups.setdefault(group_key, []).append({
                    "features": features,
                    "label": 0,  # not_fall - see module docstring
                    "source_image": f"coco_val2017/{fname.replace('.txt', '.jpg')}",
                })
                total += 1
    return groups


def grouped_split(groups: dict, rng: random.Random):
    group_items = list(groups.items())
    rng.shuffle(group_items)

    total_instances = sum(len(v) for _, v in group_items)
    train_target = total_instances * TRAIN_FRAC
    val_target = total_instances * (TRAIN_FRAC + VAL_FRAC)

    train, val, test = [], [], []
    running = 0
    for _key, rows in group_items:
        running += len(rows)
        if running <= train_target:
            train.extend(rows)
        elif running <= val_target:
            val.extend(rows)
        else:
            test.extend(rows)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["label", "source_image"])
        for row in rows:
            writer.writerow(row["features"] + [row["label"], row["source_image"]])


def main():
    rng = random.Random(SEED)

    simuletic_groups = load_simuletic()
    coco_groups = load_coco_negatives(rng)

    all_groups = {**simuletic_groups, **coco_groups}
    all_rows = [row for rows in all_groups.values() for row in rows]
    n_fall = sum(1 for r in all_rows if r["label"] == 1)
    n_not_fall = sum(1 for r in all_rows if r["label"] == 0)
    print(f"Total: {len(all_rows)} instances across {len(all_groups)} image groups "
          f"({n_fall} fall / {n_not_fall} not_fall)")
    print(f"  - Simuletic groups: {len(simuletic_groups)}")
    print(f"  - COCO groups: {len(coco_groups)}")

    train, val, test = grouped_split(all_groups, rng)
    os.makedirs(OUT_DIR, exist_ok=True)
    write_csv(os.path.join(OUT_DIR, "train.csv"), train)
    write_csv(os.path.join(OUT_DIR, "val.csv"), val)
    write_csv(os.path.join(OUT_DIR, "test.csv"), test)

    for name, split in (("train", train), ("val", val), ("test", test)):
        n_f = sum(1 for r in split if r["label"] == 1)
        n_nf = sum(1 for r in split if r["label"] == 0)
        print(f"{name}: {len(split)} rows ({n_f} fall / {n_nf} not_fall)")


if __name__ == "__main__":
    main()
