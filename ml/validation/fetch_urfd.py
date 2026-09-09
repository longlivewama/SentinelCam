#!/usr/bin/env python
"""
Builds a labelled validation corpus from the UR Fall Detection Dataset.

The framework in this directory measures the product's actual claim - a
fall raises an alert, ordinary activity does not - but it needed a
labelled corpus, and the repository had none. This assembles one from a
published dataset instead of from clips somebody has to film.

    python -m ml.validation.fetch_urfd            # 10 falls + 10 ADL
    python -m ml.validation.evaluate

Why URFD: it ships both halves of the question. 30 fall sequences and 40
activities-of-daily-living sequences - sitting, bending, lying down -
recorded in the same rooms by the same camera, so a detector cannot score
well by picking up on the setting rather than the posture.

Three things this script does that are worth knowing about:

* **The published mp4s are composites.** Each frame is the depth map and
  the RGB image side by side (640x240 = two 320x240 panels). Feeding that
  to a person detector measures nothing, so only the right-hand RGB panel
  is kept.

* **`fall-NN-cam0.mp4` is not linked from the dataset page** - only the
  ceiling camera (cam1) is - but the side-view files are served. Side
  view is what matters here: it is the view the ADL clips use, and
  comparing a ceiling-mounted positive against a side-view negative would
  confound viewpoint with class.

* **Incident times come from the dataset's own accelerometer**, not from
  someone eyeballing the video. `fall-NN-data.csv` gives, per frame, the
  milliseconds since the sequence started and the interpolated total
  acceleration; the impact is the peak, which sits 3-11x above the ~1g
  baseline. Guessing timings by hand would put a subjective label
  underneath every latency number downstream.

LICENSING - read before using the output for anything but local testing.
URFD is CC BY-NC-SA 4.0, for non-commercial academic use, and asks to be
cited:

    Bogdan Kwolek, Michal Kepski, "Human fall detection on embedded
    platform using depth maps and wireless accelerometer", Computer
    Methods and Programs in Biomedicine, 117(3), 2014, pp. 489-501.

The clips are downloaded into `ml/data/validation/`, which is gitignored
in its entirety. They are footage of identifiable people under a licence
that does not grant this repository redistribution rights, so they must
not be committed - which is why this is a fetch script rather than a
directory of files.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = REPO_ROOT / "ml" / "data" / "validation"
BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"

# How long after the impact peak to treat as part of the incident. The
# detector needs to see a sustained posture (see ModelFallDetector's
# min_sustained_seconds), so an incident window that ended at the impact
# frame itself would score a correct, slightly-late alert as a miss.
INCIDENT_TAIL_SECONDS = 2.0
# The impact is preceded by the loss of balance; starting the window a
# little early keeps latency measured from the fall, not from the floor.
INCIDENT_LEAD_SECONDS = 0.5

# ADL sequence -> the activity vocabulary in annotations.py. URFD does not
# publish a per-sequence activity label, so these stay at the generic
# "floor_activity"/"other" rather than inventing a specific one per clip:
# a wrong specific label would show up as confident nonsense in the
# false-positive breakdown.
ADL_ACTIVITY = "floor_activity"


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as fh:
            fh.write(response.read())
    except Exception as exc:  # noqa: BLE001 - any failure is just "skip this clip"
        print(f"    could not fetch {url}: {exc}")
        dest.unlink(missing_ok=True)
        return False
    return dest.stat().st_size > 0


def _extract_rgb_panel(source: Path, dest: Path) -> tuple:
    """Writes the RGB half of a depth|RGB composite. Returns
    `(frame_count, fps)`, or `(0, 0)` if the clip could not be read."""
    import cv2

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame[:, frame.shape[1] // 2:])
    capture.release()

    if not frames:
        return 0, 0.0

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return len(frames), fps


def _impact_seconds(sync_csv: Path) -> float | None:
    """Seconds into the clip of the largest total acceleration - the
    moment of impact. `None` if the file is unusable, in which case the
    caller drops the clip rather than annotating it with a guess."""
    try:
        with open(sync_csv, newline="") as fh:
            rows = [r for r in csv.reader(fh) if len(r) >= 3]
        peak = max(rows, key=lambda r: float(r[2]))
        return int(peak[1]) / 1000.0
    except (OSError, ValueError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--falls", type=int, default=10, help="Fall sequences to fetch (max 30)")
    parser.add_argument("--adl", type=int, default=10, help="ADL sequences to fetch (max 40)")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    args = parser.parse_args(argv)

    videos_dir = args.corpus / "videos"
    raw_dir = args.corpus / ".urfd_raw"
    videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"UR Fall Detection Dataset -> {args.corpus}")
    print("CC BY-NC-SA 4.0, non-commercial academic use. Not redistributable; do not commit.\n")

    entries = []

    print(f"falls ({args.falls}):")
    for n in range(1, min(args.falls, 30) + 1):
        stem = f"fall-{n:02d}-cam0"
        raw = raw_dir / f"{stem}.mp4"
        sync = raw_dir / f"fall-{n:02d}-data.csv"
        if not _download(f"{BASE_URL}/{stem}.mp4", raw):
            continue
        if not _download(f"{BASE_URL}/fall-{n:02d}-data.csv", sync):
            continue

        impact = _impact_seconds(sync)
        if impact is None:
            print(f"  {stem}: no usable sync data, skipped rather than guessed")
            continue

        out = videos_dir / f"{stem}.mp4"
        count, fps = _extract_rgb_panel(raw, out)
        if not count:
            print(f"  {stem}: unreadable, skipped")
            continue

        duration = round(count / fps, 2)
        entries.append({
            "filename": out.name,
            "duration_seconds": duration,
            "category": "fall",
            "incidents": [{
                "id": stem,
                "start_seconds": round(max(0.0, impact - INCIDENT_LEAD_SECONDS), 2),
                "end_seconds": round(min(duration, impact + INCIDENT_TAIL_SECONDS), 2),
                "notes": f"impact at {impact:.2f}s from the dataset's accelerometer peak",
            }],
            "notes": "URFD side-view camera (cam0), RGB panel of the published depth|RGB composite",
        })
        print(f"  {stem}: {duration}s, impact {impact:.2f}s")

    print(f"\nhard negatives ({args.adl}):")
    for n in range(1, min(args.adl, 40) + 1):
        stem = f"adl-{n:02d}-cam0"
        raw = raw_dir / f"{stem}.mp4"
        if not _download(f"{BASE_URL}/{stem}.mp4", raw):
            continue
        out = videos_dir / f"{stem}.mp4"
        count, fps = _extract_rgb_panel(raw, out)
        if not count:
            print(f"  {stem}: unreadable, skipped")
            continue
        duration = round(count / fps, 2)
        entries.append({
            "filename": out.name,
            "duration_seconds": duration,
            "category": "hard_negative",
            "activity": ADL_ACTIVITY,
            "incidents": [],
            "notes": "URFD activity of daily living - no fall occurs",
        })
        print(f"  {stem}: {duration}s")

    if not entries:
        print("\nNothing fetched; leaving any existing annotations alone.")
        return 1

    annotations = args.corpus / "annotations.json"
    annotations.write_text(json.dumps({"videos": entries}, indent=2) + "\n")

    falls = sum(1 for e in entries if e["category"] == "fall")
    print(f"\nWrote {annotations} - {falls} falls, {len(entries) - falls} hard negatives.")
    print("Next: python -m ml.validation.evaluate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
