import os
from PIL import Image

DATASET_DIR = "dataset_ready"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

deleted = 0

for split in ["train", "val", "test"]:
    split_path = os.path.join(DATASET_DIR, split)

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue

        for img_name in os.listdir(cls_path):
            if not img_name.lower().endswith(IMAGE_EXTS):
                continue

            img_path = os.path.join(cls_path, img_name)

            try:
                with Image.open(img_path) as img:
                    img.verify()
            except:
                os.remove(img_path)
                deleted += 1

print(f"Deleted corrupted images: {deleted}")
