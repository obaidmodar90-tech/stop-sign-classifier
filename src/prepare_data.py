"""Download the stop-sign dataset and create a 90/10 train/val split.

The download is resumable: if the connection drops, it retries and continues
from where it stopped instead of starting over.

Output layout:
    dataset/train/{not_stop,stop}/*.jpg
    dataset/val/{not_stop,stop}/*.jpg
"""
import argparse
import json
import os
import random
import shutil
import time
import zipfile

import requests

URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "ExisQFol3hUHktTjm6a51w/final-project-stop-signs-1-2025-04-25-t-06-47-41-058-z.zip"
)
ZIP_PATH = "stop_signs.zip"
RAW_DIR = "not_stopandstop"
SOURCE_SUBDIR = "final-project-stop-signs-1-2025-04-25-t-06-47-41-058-z"


def download(url, dest, retries=15):
    """Download `url` to `dest`, resuming after connection errors."""
    total = None
    try:
        head = requests.head(url, allow_redirects=True, timeout=30)
        total = int(head.headers.get("Content-Length", 0)) or None
    except requests.RequestException:
        pass

    for attempt in range(1, retries + 1):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if total and have >= total:
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as r:
                if r.status_code == 416:  # already complete
                    return
                r.raise_for_status()
                mode = "ab" if r.status_code == 206 else "wb"
                with open(dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        print(f"\rDownloaded {f.tell() / 1e6:.1f} MB", end="", flush=True)
            print()
            if total is None or os.path.getsize(dest) >= total:
                return
        except requests.RequestException as e:
            print(f"\nConnection problem ({e.__class__.__name__}); retry {attempt}/{retries}...")
            time.sleep(min(2 * attempt, 15))
    raise RuntimeError("Download failed. Check your internet connection and run again.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    source_dir = os.path.join(RAW_DIR, SOURCE_SUBDIR)
    if not os.path.isdir(source_dir):
        download(URL, ZIP_PATH)
        if not zipfile.is_zipfile(ZIP_PATH):
            os.remove(ZIP_PATH)
            raise RuntimeError("Downloaded file is corrupted. Run the script again.")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW_DIR)
        print("Download and extraction complete.")
    else:
        print("Dataset already downloaded, skipping.")

    with open(os.path.join(source_dir, "_annotations.json")) as f:
        annotations = json.load(f)

    label_to_images = {}
    for filename, entry in annotations["annotations"].items():
        label_to_images.setdefault(entry[0]["label"], []).append(filename)

    shutil.rmtree(args.output_dir, ignore_errors=True)  # avoid mixing old splits
    for label, images in label_to_images.items():
        random.shuffle(images)
        cutoff = int(len(images) * args.train_ratio)
        splits = {"train": images[:cutoff], "val": images[cutoff:]}
        for split, names in splits.items():
            out_path = os.path.join(args.output_dir, split, label)
            os.makedirs(out_path, exist_ok=True)
            for name in names:
                shutil.copy2(os.path.join(source_dir, name), os.path.join(out_path, name))

    print("Train/Val split complete.")


if __name__ == "__main__":
    main()