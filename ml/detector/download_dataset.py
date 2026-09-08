"""
Downloads the Roboflow fall-detection dataset (tfmbigdata2025 /
voc-fall-person_falls, version 1) in YOLOv8 format into
ml/data/fall_detection/.

Why not the `roboflow` pip package
----------------------------------
The documented snippet for this dataset is:

    from roboflow import Roboflow
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("tfmbigdata2025").project("voc-fall-person_falls")
    dataset = project.version(1).download("yolov8")

Installing `roboflow` into `backend/venv` (the venv this pipeline reuses,
per ml/README.md) resolves to `numpy>=2.3` and `opencv-python-headless
5.x`, which directly conflicts with the running application's pins
(`ultralytics==8.3.0` requires `numpy<2.0.0`; the backend uses
`opencv-python==4.10`). Installing it would break the live app, so this
script talks to the same public Roboflow export API the SDK itself calls
(`GET https://api.roboflow.com/{workspace}/{project}/{version}/{format}`)
using `requests`, which is already present. Identical dataset, identical
YOLOv8 layout, zero dependency risk to the application.

The API key is read from the environment (or ml/.env) and is never
logged, printed, or written to disk by this script.

Usage:
    ROBOFLOW_API_KEY=... python3 ml/detector/download_dataset.py
    # or put ROBOFLOW_API_KEY=... in ml/.env (gitignored) and just run it
"""
from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

WORKSPACE = "tfmbigdata2025"
PROJECT = "voc-fall-person_falls"
VERSION = 1
FMT = "yolov8"

ML_DIR = Path(__file__).resolve().parent.parent
DEST = ML_DIR / "data" / "fall_detection"
API_ROOT = "https://api.roboflow.com"

POLL_SECONDS = 5
POLL_ATTEMPTS = 60  # generation of a fresh export can take a few minutes


def _load_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        env_file = ML_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("ROBOFLOW_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        sys.exit(
            "ROBOFLOW_API_KEY is not set.\n"
            f"Set it in the environment or add it to {ML_DIR / '.env'} "
            "(gitignored). See ml/.env.example."
        )
    return key


def _redact(text: str, secret: str) -> str:
    """Roboflow echoes the api_key back in error payloads/URLs - never let
    it reach stdout or a log file."""
    return text.replace(secret, "<ROBOFLOW_API_KEY>") if secret else text


def request_export_link(api_key: str) -> str:
    url = f"{API_ROOT}/{WORKSPACE}/{PROJECT}/{VERSION}/{FMT}"
    for attempt in range(1, POLL_ATTEMPTS + 1):
        resp = requests.get(url, params={"api_key": api_key}, timeout=60)
        if resp.status_code == 401:
            sys.exit("Roboflow rejected the API key (401). Check ROBOFLOW_API_KEY.")
        if resp.status_code == 404:
            sys.exit(
                f"Roboflow returned 404 for {WORKSPACE}/{PROJECT} v{VERSION}. "
                "The dataset may be private to another account, renamed, or the "
                "version may not exist."
            )
        try:
            payload = resp.json()
        except ValueError:
            sys.exit(f"Unexpected non-JSON response ({resp.status_code}): "
                     f"{_redact(resp.text[:300], api_key)}")

        export = payload.get("export") or {}
        link = export.get("link")
        if link:
            return link

        # Export still being generated server-side.
        progress = payload.get("progress") or export.get("progress")
        print(f"  export not ready yet (attempt {attempt}/{POLL_ATTEMPTS}"
              + (f", progress={progress}" if progress is not None else "") + ") - waiting...")
        if attempt == POLL_ATTEMPTS:
            sys.exit(f"Export never became ready. Last payload: "
                     f"{_redact(str(payload)[:300], api_key)}")
        time.sleep(POLL_SECONDS)
    raise AssertionError("unreachable")


def download_and_extract(link: str, api_key: str):
    if DEST.exists() and any(DEST.iterdir()):
        print(f"{DEST} already exists and is non-empty - delete it to re-download.")
        return
    DEST.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset zip...")
    resp = requests.get(link, stream=True, timeout=600)
    resp.raise_for_status()
    buf = io.BytesIO()
    total = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
        total += len(chunk)
        print(f"\r  {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\n  downloaded {total / 1e6:.1f} MB")

    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        # Guard against path traversal in the archive before extracting.
        for name in zf.namelist():
            resolved = (DEST / name).resolve()
            if not str(resolved).startswith(str(DEST.resolve())):
                sys.exit(f"Refusing to extract unsafe archive member: {name}")
        zf.extractall(DEST)
    print(f"Extracted to {DEST}")


def main():
    api_key = _load_api_key()
    print(f"Requesting {FMT} export of {WORKSPACE}/{PROJECT} v{VERSION}...")
    link = request_export_link(api_key)
    download_and_extract(link, api_key)

    data_yaml = DEST / "data.yaml"
    if data_yaml.exists():
        print(f"\n--- {data_yaml} ---")
        print(data_yaml.read_text())
    else:
        print(f"\nWARNING: no data.yaml at {data_yaml}; layout:")
        for p in sorted(DEST.rglob("*"))[:40]:
            print("  ", p.relative_to(DEST))


if __name__ == "__main__":
    main()
