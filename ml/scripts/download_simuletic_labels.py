"""Downloads the Simuletic CCTV Incident Dataset's label files (YOLO-Pose
format, CC-BY-4.0) via the HuggingFace Hub API - no auth required, this
dataset is public. Images are not downloaded here (see
download_images.py); training and evaluation only need the keypoint
annotations."""
import concurrent.futures
import json
import os
import urllib.parse
import urllib.request

API_URL = "https://huggingface.co/api/datasets/Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection"
BASE = "https://huggingface.co/datasets/Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection/resolve/main/"
DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_simuletic", "labels")


def fetch(rel_path):
    fname = rel_path.split("/")[-1]
    dest = os.path.join(DEST_DIR, fname)
    if os.path.exists(dest):
        return fname, True
    url = BASE + urllib.parse.quote(rel_path)
    try:
        urllib.request.urlretrieve(url, dest)
        return fname, True
    except Exception:
        return fname, False


if __name__ == "__main__":
    os.makedirs(DEST_DIR, exist_ok=True)
    with urllib.request.urlopen(API_URL, timeout=20) as resp:
        info = json.load(resp)
    labels = [
        s["rfilename"] for s in info.get("siblings", [])
        if s["rfilename"].startswith("laying_dataset/labels/") and s["rfilename"].endswith(".txt")
    ]
    print(f"Found {len(labels)} label files")

    ok, fail = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for _fname, success in pool.map(fetch, labels):
            ok += int(success)
            fail += int(not success)
    print(f"downloaded {ok} labels, {fail} failed")
