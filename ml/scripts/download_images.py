"""One-off downloader for the Simuletic CCTV laying/standing dataset's
images (labels are downloaded separately since they're tiny; images are
only needed for a handful of visual sanity-check crops in the eval
report, not for training - training runs entirely on the ground-truth
keypoints already present in the label files)."""
import concurrent.futures
import os
import urllib.parse
import urllib.request

BASE = "https://huggingface.co/datasets/Simuletic/CCTV_Incident_Dataset_Fall_Lying_Down_Detection/resolve/main/"
DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_simuletic", "images")


def fetch(name):
    url = BASE + urllib.parse.quote(f"laying_dataset/images/{name}")
    dest = os.path.join(DEST_DIR, name)
    if os.path.exists(dest):
        return name, True
    try:
        urllib.request.urlretrieve(url, dest)
        return name, True
    except Exception as e:
        return name, False


if __name__ == "__main__":
    os.makedirs(DEST_DIR, exist_ok=True)
    names = [f.replace(".txt", ".png") for f in os.listdir(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw_simuletic", "labels")
    )]
    ok, fail = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for name, success in pool.map(fetch, names):
            if success:
                ok += 1
            else:
                fail += 1
    print(f"downloaded {ok} images, {fail} failed")
