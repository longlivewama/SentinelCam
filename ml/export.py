"""
Exports the trained fall classifier checkpoint (ml/exported/
fall_classifier_v1.pt) to ONNX, and writes a metadata JSON describing
exactly how a consumer (backend/app/services/detection/fall_detection.py)
should call it. Also runs a quick equivalence check: same inputs through
the PyTorch model and the exported ONNX model must produce matching
outputs, otherwise the export is rejected.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
import torch

from train import FallClassifierMLP, N_FEATURES

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exported")
CHECKPOINT_PATH = os.path.join(EXPORT_DIR, "fall_classifier_v1.pt")
ONNX_PATH = os.path.join(EXPORT_DIR, "fall_classifier_v1.onnx")
METADATA_PATH = os.path.join(EXPORT_DIR, "fall_classifier_v1.metadata.json")


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model = FallClassifierMLP()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy_input = torch.randn(1, N_FEATURES, dtype=torch.float32)

    class WithSigmoid(torch.nn.Module):
        """Wrap the raw-logit model with a sigmoid so the ONNX graph
        outputs a fall probability directly - consumers shouldn't need to
        know the model was trained with BCEWithLogitsLoss internally."""

        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            return torch.sigmoid(self.base(x))

    export_model = WithSigmoid(model)
    export_model.eval()

    # dynamo=False: use torch's legacy TorchScript-based ONNX exporter.
    # The newer dynamo-based exporter (torch>=2.x default) requires the
    # `onnxscript` package, which pulls in numpy>=2.0 and would conflict
    # with backend/venv's ultralytics (numpy<2.0 pin) since this script
    # currently reuses that same venv - the legacy exporter needs no new
    # dependencies at all.
    torch.onnx.export(
        export_model,
        dummy_input,
        ONNX_PATH,
        input_names=["features"],
        output_names=["fall_probability"],
        dynamic_axes={"features": {0: "batch"}, "fall_probability": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX model to {ONNX_PATH}")

    # Equivalence check: PyTorch vs ONNX Runtime on the same random batch.
    torch.manual_seed(0)
    test_batch = torch.randn(8, N_FEATURES, dtype=torch.float32)
    with torch.no_grad():
        torch_out = export_model(test_batch).numpy()

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_out = session.run(["fall_probability"], {"features": test_batch.numpy()})[0]

    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"Max abs diff PyTorch vs ONNX Runtime: {max_diff:.2e}")
    if max_diff > 1e-4:
        raise RuntimeError(
            f"ONNX export failed equivalence check (max diff {max_diff:.2e} > 1e-4) - not writing metadata"
        )

    metadata = {
        "model_file": os.path.basename(ONNX_PATH),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "framework": "pytorch->onnx (opset 17)",
        "architecture": f"MLP: {N_FEATURES} -> {checkpoint['hidden_1']} (ReLU) -> {checkpoint['hidden_2']} (ReLU) -> 1 (sigmoid)",
        "input": {
            "name": "features",
            "shape": ["batch", N_FEATURES],
            "dtype": "float32",
            "description": (
                "10 keypoint-derived geometric features computed from one person's "
                "YOLOv8-Pose 17-keypoint output + bbox. See scripts/features.py::extract_features "
                "for the exact computation and scripts/features.py::FEATURE_NAMES for the order."
            ),
            "preprocessing": (
                "Standardize each feature with (x - feature_mean[i]) / feature_std[i] BEFORE "
                "calling the model. feature_mean/feature_std are the train-set statistics saved "
                "in fall_classifier_v1.pt (also duplicated below for convenience) - do NOT refit "
                "these at inference time."
            ),
            "feature_mean": checkpoint["feature_mean"],
            "feature_std": checkpoint["feature_std"],
            "feature_order": [
                "aspect_ratio", "shoulder_hip_gap_norm", "hip_knee_gap_norm",
                "knee_ankle_gap_norm", "shoulder_ankle_gap_norm", "head_hip_gap_norm",
                "shoulder_width_norm", "hip_width_norm", "mean_keypoint_conf",
                "visible_keypoint_frac",
            ],
        },
        "output": {
            "name": "fall_probability",
            "shape": ["batch", 1],
            "dtype": "float32",
            "description": "P(fall) in [0, 1]. Already through sigmoid - no further transform needed.",
        },
        "training": {
            "train_instances": None,  # filled in below from train_run_metadata.json if present
        },
        "onnx_pytorch_equivalence_max_abs_diff": max_diff,
    }

    train_meta_path = os.path.join(EXPORT_DIR, "train_run_metadata.json")
    if os.path.exists(train_meta_path):
        with open(train_meta_path) as f:
            metadata["training"] = json.load(f)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
