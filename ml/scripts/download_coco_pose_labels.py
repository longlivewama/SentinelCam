"""Downloads and unpacks the COCO 2017 person-keypoints val split, YOLO-Pose
label format, re-hosted at huggingface.co/datasets/Mai0313/coco-pose-2017
(CC-BY-4.0, annotations only - re-hosts the official COCO keypoint
annotations in YOLO-Pose .txt format). Only `datasets/labels/val/` is kept
- the zip also contains a much larger `train` label set and empty image
placeholders, neither of which this pipeline uses (see ml/README.md)."""
import os
import shutil
import urllib.request
import zipfile

ZIP_URL = "https://huggingface.co/datasets/Mai0313/coco-pose-2017/resolve/main/coco2017labels-pose.zip"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_coco_pose")
ZIP_PATH = os.path.join(RAW_DIR, "coco2017labels-pose.zip")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading {ZIP_URL} ...")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        val_members = [m for m in zf.namelist() if m.startswith("datasets/labels/val/")]
        print(f"Extracting {len(val_members)} val label files ...")
        zf.extractall(RAW_DIR, members=val_members)

    # Trim anything else the zip may have created (train labels/images/
    # annotations are not used by this pipeline - see module docstring).
    datasets_dir = os.path.join(RAW_DIR, "datasets")
    for unused in ("labels/train", "images", "annotations", "train.txt"):
        path = os.path.join(datasets_dir, unused)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)

    print(f"COCO pose val labels ready at {os.path.join(datasets_dir, 'labels', 'val')}")


if __name__ == "__main__":
    main()
