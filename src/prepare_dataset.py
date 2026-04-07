import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# Config
IMAGES_DIR = "raw_data/PCB_DATASET/images"
ANNOT_DIR  = "raw_data/PCB_DATASET/Annotations"
DEST_DIR   = "data"
SPLIT      = (0.7, 0.15, 0.15)
SEED       = 42
PADDING    = 20  # pixels of context around each defect

random.seed(SEED)

def parse_annotation(xml_path):
    """Extract bounding boxes from XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        boxes.append((name, xmin, ymin, xmax, ymax))
    return filename, boxes


def crop_and_save(image_path, boxes, dest_dir, split, cls, idx):
    """Crop each bounding box from image and save."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    saved = 0

    for i, (name, xmin, ymin, xmax, ymax) in enumerate(boxes):
        # Add padding around defect, clamp to image bounds
        xmin_p = max(0, xmin - PADDING)
        ymin_p = max(0, ymin - PADDING)
        xmax_p = min(w, xmax + PADDING)
        ymax_p = min(h, ymax + PADDING)

        crop = img.crop((xmin_p, ymin_p, xmax_p, ymax_p))

        dest = Path(dest_dir) / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        crop.save(dest / f"{cls}_{idx:04d}_crop{i}.jpg")
        saved += 1

    return saved


# Process each class
for cls in os.listdir(ANNOT_DIR):
    annot_cls_dir = Path(ANNOT_DIR) / cls
    xml_files = list(annot_cls_dir.glob("*.xml"))
    random.shuffle(xml_files)

    n = len(xml_files)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])

    splits = {
        "train": xml_files[:n_train],
        "val":   xml_files[n_train:n_train + n_val],
        "test":  xml_files[n_train + n_val:]
    }

    total_crops = {"train": 0, "val": 0, "test": 0}

    for split, files in splits.items():
        for idx, xml_path in enumerate(files):
            img_filename, boxes = parse_annotation(xml_path)
            img_path = Path(IMAGES_DIR) / cls / img_filename
            if not img_path.exists():
                continue
            n_saved = crop_and_save(img_path, boxes, DEST_DIR, split, cls, idx)
            total_crops[split] += n_saved

    print(f"{cls}: {total_crops['train']} train | {total_crops['val']} val | {total_crops['test']} test crops")