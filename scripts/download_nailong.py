"""
Download 奶龙 (Milk Dragon) images from the HuggingFace dataset
refoundd/NailongClassification. Filters label == "nailong" (896 images).

Saves to raw/奶龙/ for the binary-classifier pipeline.

Usage:
    python scripts/download_nailong.py --n 300 --out raw/奶龙
"""
import argparse
import os
from pathlib import Path


def download_nailong(n, out_dir, seed=0):
    """Download n nailong images from HuggingFace dataset."""
    from datasets import load_dataset

    print("Loading NailongClassification from HuggingFace...")
    ds = load_dataset("refoundd/NailongClassification", split="train")
    # Filter only actual nailong images
    nailong = ds.filter(lambda x: x["label"] == "nailong")
    print(f"Found {len(nailong)} nailong images")

    # Deterministic sample (for reproducibility)
    idx = list(range(len(nailong)))
    if n < len(idx):
        import random
        random.Random(seed).shuffle(idx)
        idx = idx[:n]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i in idx:
        try:
            img = nailong[i]["image"]
            path = out_dir / f"nailong_{saved:04d}.jpg"
            img.convert("RGB").save(path, "JPEG")
            saved += 1
        except Exception as e:
            print(f"  skip #{i}: {e}")
    print(f"Saved {saved} nailong images to {out_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300,
                        help="number of nailong images")
    parser.add_argument("--out", default="raw/奶龙")
    args = parser.parse_args()

    download_nailong(args.n, args.out)


if __name__ == "__main__":
    main()
