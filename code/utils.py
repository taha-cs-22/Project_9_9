import os

train_files = set()
for root, _, files in os.walk("dataset_ready/train"):
    for f in files:
        train_files.add(f)

test_files = set()
for root, _, files in os.walk("dataset_ready/test"):
    for f in files:
        test_files.add(f)

common = train_files.intersection(test_files)
print("Common files:", len(common))
