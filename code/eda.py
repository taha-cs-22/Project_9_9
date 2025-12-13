import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

DATASET_DIR = "dataset_ready/train"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

image_counts = {}
image_sizes = []
corrupted_images = []

for cls in os.listdir(DATASET_DIR):
    cls_path = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith(IMAGE_EXTS)
    ]

    image_counts[cls] = len(images)

    for img_name in images:
        img_path = os.path.join(cls_path, img_name)
        try:
            with Image.open(img_path) as img:
                image_sizes.append(img.size)
        except:
            corrupted_images.append(img_path)

# ====== PRINT SUMMARY ======
print("\nImage count per class:")
for cls, count in image_counts.items():
    print(f"{cls}: {count}")

print(f"\nTotal classes: {len(image_counts)}")
print(f"Total images: {sum(image_counts.values())}")
print(f"Corrupted images found: {len(corrupted_images)}")

# ====== BAR CHART ======
plt.figure(figsize=(10, 5))
plt.bar(image_counts.keys(), image_counts.values())
plt.xticks(rotation=45, ha="right")
plt.title("Images per Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()

# ====== IMAGE SIZE DISTRIBUTION ======
widths, heights = zip(*image_sizes)

plt.figure(figsize=(6, 4))
plt.scatter(widths, heights, alpha=0.4)
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Image Size Distribution")
plt.show()

# ====== SHOW & SAVE ONE SAMPLE IMAGE PER CLASS ======
plt.figure(figsize=(12, 8))

classes = list(image_counts.keys())

for i, cls in enumerate(classes):
    cls_path = os.path.join(DATASET_DIR, cls)

    img_name = next(
        (f for f in os.listdir(cls_path) if f.lower().endswith(IMAGE_EXTS)),
        None
    )

    if img_name is None:
        continue

    img_path = os.path.join(cls_path, img_name)

    img = Image.open(img_path).convert("RGB")

    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.suptitle("Sample Image from Each Class", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("results/sample_images_per_class.png", dpi=200)

plt.show()

plt.close()

