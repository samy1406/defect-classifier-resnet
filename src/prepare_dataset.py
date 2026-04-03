import os
import shutil
import random
from pathlib import Path

# Config
SOURCE_DIR = "raw_data/PCB_DATASET/images"   # fixed path
DEST_DIR = "data"
SPLIT = (0.7, 0.15, 0.15)
SEED = 42

random.seed(SEED)

classes = os.listdir(SOURCE_DIR)
print(f"Found classes: {classes}")

for cls in classes:
    cls_path = Path(SOURCE_DIR) / cls
    images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in splits.items():
        dest = Path(DEST_DIR) / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, dest / f.name)

    print(f"{cls}: {n} total → {n_train} train | {n_val} val | {n - n_train - n_val} test")