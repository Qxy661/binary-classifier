"""
Download 奶龙 (Milk Dragon) and 塔菲 (Taffy cat) images for the fun
binary classifier. Uses Bing image search via bing-image-downloader.

Downloads each with a few query variants to get variety, saves into
raw/ structure:
    raw/奶龙/*.jpg
    raw/塔菲/*.jpg

Usage:
    python scripts/download_milktaffy.py --n-per-class 150
"""
import argparse
import os
import shutil
from pathlib import Path

from bing_image_downloader import downloader

# Query variants for each class (improves variety & relevance)
CLASSES = {
    "奶龙": ["奶龙", "奶龙 表情包", "奶龙 卡通", "奶龙 恐龙"],
    "塔菲": ["塔菲猫", "塔菲 表情包", "塔菲 猫", "塔菲猫 卡通"],
}


def download_class(cls_name, queries, n_per_class, tmp_root):
    """Download images for one class across query variants."""
    tmp_dir = Path(tmp_root) / cls_name
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    per_query = max(n_per_class // len(queries), 10)
    total = 0
    for i, query in enumerate(queries):
        out = tmp_dir / f"q{i}"
        try:
            downloader.download(query, limit=per_query,
                                output_dir=str(out),
                                adult_filter_off=True,
                                force_replace=True)
            # Move downloaded images up one level, with index prefix
            src = out / query
            if src.exists():
                for j, img in enumerate(sorted(src.glob("*"))):
                    ext = img.suffix
                    dst = tmp_dir / f"{cls_name}_{total:03d}{ext}"
                    shutil.move(str(img), str(dst))
                    total += 1
        except Exception as e:
            print(f"  query '{query}' failed: {e}")
    print(f"  {cls_name}: downloaded {total} images")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=150)
    parser.add_argument("--out", default="raw/")
    parser.add_argument("--tmp", default="/tmp/milktaffy_dl")
    args = parser.parse_args()

    out_dir = Path(args.out)
    for cls_name, queries in CLASSES.items():
        download_class(cls_name, queries, args.n_per_class, args.tmp)
        # Copy to final raw/ location
        dst = out_dir / cls_name
        dst.mkdir(parents=True, exist_ok=True)
        for img in sorted((Path(args.tmp) / cls_name).glob("*")):
            if img.is_file():
                shutil.copy2(img, dst / img.name)

    # Report
    for cls_name in CLASSES:
        n = len(list((out_dir / cls_name).glob("*")))
        print(f"raw/{cls_name}: {n} images")

    print("\nOrganize with:")
    print(f"  python scripts/data_pipeline.py --source {args.out} --data-dir data/")


if __name__ == "__main__":
    main()
