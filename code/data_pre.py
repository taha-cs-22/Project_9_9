import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = "Fish_Dataset"
OUTPUT_DIR = "dataset_ready"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

random.seed(42)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    img_dir = None
    for sub in os.listdir(cls_path):
        if "GT" not in sub:
            img_dir = os.path.join(cls_path, sub)

    if img_dir is None:
        continue

    images = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ]

    random.shuffle(images)

    n = len(images)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in splits.items():
        dst_dir = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(dst_dir, exist_ok=True)

        for src in files:
            new_name = f"{Path(src).parent.parent.name}_{Path(src).name}"
            dst = os.path.join(dst_dir, new_name)
            shutil.copy(src, dst)

print("Clean split done. No leakage.")
