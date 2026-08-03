"""
Download 永雏塔菲 (Yongchu Taffy, VTuber cat-girl) images.

Uses two sources:
  1. 萌娘百科 commons category page (quality seed images)
  2. icrawler Bing search (fill more, may include noise)

Saves to raw/塔菲/.

Usage:
    python scripts/download_taffy.py --n 200
"""
import argparse
import os
import re
from pathlib import Path

import requests


def download_moegirl(out_dir, max_n=50):
    """Download images from 萌娘百科 永雏塔菲 category page."""
    url = ("https://commons.moegirl.org.cn/index.php?"
           "title=Category:%E6%B0%B8%E9%9B%8F%E5%A1%94%E8%8F%B2")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            print(f"  moegirl page failed: {r.status_code}")
            return 0
        # Find image URLs (thumbnails and originals)
        pattern = re.compile(
            r"//commons\.moegirl\.org\.cn/images/[^\"\s]+\.(?:jpg|jpeg|png)",
            re.IGNORECASE)
        urls = list(dict.fromkeys(pattern.findall(r.text)))  # dedup, keep order
        print(f"  moegirl: found {len(urls)} image URLs")
        saved = 0
        for i, u in enumerate(urls[:max_n]):
            full = "https:" + u
            try:
                img = requests.get(full, headers=headers, timeout=15)
                if img.status_code == 200:
                    ext = Path(u).suffix or ".jpg"
                    path = out_dir / f"taffy_moe_{saved:03d}{ext}"
                    path.write_bytes(img.content)
                    saved += 1
            except Exception as e:
                pass
        print(f"  moegirl: saved {saved} images")
        return saved
    except Exception as e:
        print(f"  moegirl failed: {e}")
        return 0


def download_bing(out_dir, max_n=100):
    """Download via icrawler Bing search (fills more, may include noise)."""
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print("  icrawler not installed, skipping Bing")
        return 0
    saved = 0
    for query in ["永雏塔菲 表情包", "yongchu taffy", "永雏塔菲"]:
        crawler = BingImageCrawler(
            storage={"root_dir": str(out_dir / "_bing")})
        try:
            crawler.crawl(keyword=query, max_num=min(max_n // 2, 40))
        except Exception as e:
            print(f"  bing '{query}' failed: {e}")
    # Move downloaded images up and rename
    saved = 0
    bing_dir = out_dir / "_bing"
    if bing_dir.exists():
        for i, img in enumerate(sorted(bing_dir.rglob("*"))):
            if img.is_file() and img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                dst = out_dir / f"taffy_bing_{saved:03d}{img.suffix.lower()}"
                img.rename(dst)
                saved += 1
    print(f"  bing: saved {saved} images")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", default="raw/塔菲")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading 永雏塔菲 images...")
    n1 = download_moegirl(out_dir, max_n=50)
    n2 = download_bing(out_dir, max_n=args.n)

    total = n1 + n2
    print(f"\nTotal: {n1} (moegirl) + {n2} (bing) = {total} images")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
