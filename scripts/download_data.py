"""
Download the cat-vs-dog dataset (stage 1: public data).

Fetches a subset of HuggingFace's microsoft/cats_vs_dogs, saves images
into the expected raw/ structure:
    raw/cat/img_000.jpg ...
    raw/dog/img_000.jpg ...

Then organize with data_pipeline.py.

Usage:
    python scripts/download_data.py --n-per-class 500
"""
import argparse
import os
import io
from pathlib import Path

import numpy as np
from PIL import Image


def download_from_hf(n_per_class, out_dir):
    """Download cats_vs_dogs from HuggingFace, save as organized folders."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets first")

    print("Loading cats_vs_dogs from HuggingFace (this may take a while)...")
    ds = load_dataset("microsoft/cats_vs_dogs", split="train")

    out_dir = Path(out_dir)
    for label_name, target_label in [("cat", 0), ("dog", 1)]:
        cls_dir = out_dir / label_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for i, ex in enumerate(ds):
            if ex["labels"] == target_label and n < n_per_class:
                img = ex["image"]
                path = cls_dir / f"{label_name}_{n:03d}.jpg"
                img.convert("RGB").save(path, "JPEG")
                n += 1
        print(f"  Saved {n} {label_name} images to {cls_dir}")

    print(f"Done. Total: {n_per_class} per class")
    print("Organize with:")
    print(f"  python scripts/data_pipeline.py --source {out_dir} --data-dir data/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=500,
                        help="images per class (fewer = faster)")
    parser.add_argument("--out", default="raw/")
    args = parser.parse_args()

    download_from_hf(args.n_per_class, args.out)


if __name__ == "__main__":
    main()
