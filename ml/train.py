"""
Trains a small MLP fall/not_fall classifier on the keypoint-derived
features produced by scripts/prepare_dataset.py (10-dim feature vector
per detected person - see scripts/features.py for exactly what each
feature is).

A full YOLO image-classifier fine-tune (e.g. yolov8n-cls) was considered
but rejected for this iteration: the dataset is small (421 instances) and
scale-invariant keypoint geometry is already most of the signal a fall
detector needs (it's exactly what the existing production heuristic
uses), so a tiny MLP on engineered features trains in seconds on CPU,
is trivially exportable to ONNX, and avoids overfitting an image encoder
to a synthetic/COCO-mixed dataset with no camera-specific visual
statistics. See ml/README.md for the full rationale.

Usage: python train.py
Reads ml/data/{train,val}.csv, writes ml/exported/fall_classifier_v1.pt
(checkpoint) - export.py handles the ONNX conversion + versioned metadata.
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import torch
from torch import nn

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exported")

N_FEATURES = 10
HIDDEN_1 = 16
HIDDEN_2 = 8

EPOCHS = 200
LR = 0.01
SEED = 42


class FallClassifierMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, x):
        return self.net(x)  # logits - sigmoid applied by caller (BCEWithLogitsLoss / inference)


def load_csv(path):
    features, labels = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append([float(row[k]) for k in
                              ["aspect_ratio", "shoulder_hip_gap_norm", "hip_knee_gap_norm",
                               "knee_ankle_gap_norm", "shoulder_ankle_gap_norm", "head_hip_gap_norm",
                               "shoulder_width_norm", "hip_width_norm", "mean_keypoint_conf",
                               "visible_keypoint_frac"]])
            labels.append(float(row["label"]))
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train = load_csv(os.path.join(DATA_DIR, "train.csv"))
    X_val, y_val = load_csv(os.path.join(DATA_DIR, "val.csv"))

    # Standardize features using train-set statistics only (never fit on
    # val/test, to avoid any information leakage from held-out data).
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    # Class weighting: dataset is ~3.7:1 not_fall:fall (see prepare_dataset.py
    # output), so weight the minority (fall) class up in the loss rather
    # than resampling, to keep every real example in play.
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

    model = FallClassifierMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train_t = torch.from_numpy(X_train_norm)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    X_val_t = torch.from_numpy(X_val_norm)
    y_val_t = torch.from_numpy(y_val).unsqueeze(1)

    start = time.time()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()
            val_preds = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc = (val_preds == y_val_t).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s. Best val_loss={best_val_loss:.4f}")

    model.load_state_dict(best_state)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(EXPORT_DIR, "fall_classifier_v1.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "n_features": N_FEATURES,
        "hidden_1": HIDDEN_1,
        "hidden_2": HIDDEN_2,
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    with open(os.path.join(EXPORT_DIR, "train_run_metadata.json"), "w") as f:
        json.dump({
            "epochs": EPOCHS,
            "lr": LR,
            "train_instances": len(y_train),
            "val_instances": len(y_val),
            "train_fall": int(y_train.sum()),
            "train_not_fall": int(len(y_train) - y_train.sum()),
            "elapsed_seconds": round(elapsed, 2),
            "best_val_loss": round(best_val_loss, 4),
        }, f, indent=2)


if __name__ == "__main__":
    main()
