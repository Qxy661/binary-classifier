"""
Data pipeline for the binary classifier.

Builds a clean two-class image dataset:
  data/
    class_a/  img1.jpg img2.jpg ...
    class_b/  img1.jpg img2.jpg ...

Steps:
  1. Organize raw images into train/val folders (auto or manual)
  2. Load images, resize, convert to numpy arrays
  3. Split into train/val sets

Usage:
    python scripts/data_pipeline.py --data-dir data/ --source raw/
"""
import argparse
import os
import random
import shutil
from pathlib import Path

from PIL import Image
import numpy as np


def organize(source_dir, dest_dir, val_ratio=0.2):
    """Split source/<class>/*.jpg into dest/{train,val}/<class>/.

    Expected source structure:
        source/cat/*.jpg  source/dog/*.jpg
    Produces:
        dest/train/cat/*.jpg  dest/train/dog/*.jpg
        dest/val/cat/*.jpg    dest/val/dog/*.jpg
    """
    source = Path(source_dir)
    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        images = list(class_dir.glob("*.*"))
        random.shuffle(images)
        n_val = int(len(images) * val_ratio)

        for split, idxs in [("train", range(n_val, len(images))),
                            ("val", range(n_val))]:
            for i in idxs:
                dst = Path(dest_dir) / split / class_name
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(images[i], dst / images[i].name)

    print(f"Organized {source_dir} -> {dest_dir}")
    for split in ["train", "val"]:
        for cls in sorted((Path(dest_dir) / split).iterdir()):
            n = len(list(cls.glob("*.*")))
            print(f"  {split}/{cls.name}: {n} images")


def load_images(images_dir, size=(64, 64)):
    """Load all images in a class dir, resize, return (X, paths)."""
    images, paths = [], []
    for img_path in sorted(Path(images_dir).glob("*.*")):
        try:
            img = Image.open(img_path).convert("RGB").resize(size)
            images.append(np.array(img).astype(np.float32) / 255.0)
            paths.append(str(img_path))
        except Exception as e:
            print(f"  skip {img_path}: {e}")
    return np.stack(images) if images else np.array([]), paths


def build_dataset(data_dir, size=(64, 64)):
    """Load train/val datasets as numpy arrays.

    Returns: dict with 'train'/'val', each {images, labels, paths},
             and class_names list.
    """
    classes = sorted(p.name for p in (Path(data_dir) / "train").iterdir()
                     if p.is_dir())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes: {classes}")

    result = {}
    for split in ["train", "val"]:
        all_images, all_labels, all_paths = [], [], []
        for cls in classes:
            X, paths = load_images(Path(data_dir) / split / cls, size)
            all_images.append(X)
            all_labels.extend([class_to_idx[cls]] * len(X))
            all_paths.extend(paths)
        result[split] = {
            "images": np.concatenate(all_images) if all_images else np.array([]),
            "labels": np.array(all_labels),
            "paths": all_paths,
        }
        print(f"  {split}: {len(all_labels)} images")
    return result, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="raw/",
                        help="raw images organized by class folders")
    parser.add_argument("--data-dir", default="data/",
                        help="output organized data dir")
    parser.add_argument("--size", default=64, type=int)
    args = parser.parse_args()

    organize(args.source, args.data_dir)
    dataset, classes = build_dataset(args.data_dir, (args.size, args.size))
    print(f"\nDataset ready: {len(classes)} classes, "
          f"train {len(dataset['train']['images'])}, "
          f"val {len(dataset['val']['images'])}")


if __name__ == "__main__":
    main()
