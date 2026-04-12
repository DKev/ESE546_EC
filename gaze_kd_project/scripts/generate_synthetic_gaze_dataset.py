#!/usr/bin/env python3
"""
Generate a synthetic gaze dataset for quick experiments and report figures.

Each image is a simple \"face\" sketch (ellipse + pupil) where the pupil offset
is consistent with the (gaze_x, gaze_y) label in [-1, 1]. This is **not** a
replacement for MPIIGaze in the wild, but it yields reproducible training
curves and lets you debug the full pipeline on CPU/GPU without large downloads.

Usage (from ``gaze_kd_project``)::

    python scripts/generate_synthetic_gaze_dataset.py --out_root data/synthetic --seed 0
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw


def draw_sample(rng: random.Random, size: int) -> tuple[Image.Image, float, float]:
    gx = rng.uniform(-1.0, 1.0)
    gy = rng.uniform(-1.0, 1.0)

    img = Image.new("RGB", (size, size), color=(rng.randint(180, 230), rng.randint(160, 210), rng.randint(150, 200)))
    dr = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    rx, ry = int(size * 0.38), int(size * 0.45)
    dr.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=(40, 40, 40), width=3)

    # Pupil position follows label so a CNN can learn the mapping.
    k = 0.22 * size
    px = cx + gx * k + rng.uniform(-1.5, 1.5)
    py = cy + gy * k + rng.uniform(-1.5, 1.5)
    r = max(4, size // 28)
    dr.ellipse([px - r, py - r, px + r, py + r], fill=(20, 20, 20))
    dr.ellipse([px - r // 3, py - r // 3, px + r // 3, py + r // 3], fill=(240, 240, 240))

    # Light speckle noise
    for _ in range(size // 2):
        x, y = rng.randint(0, size - 1), rng.randint(0, size - 1)
        c = rng.randint(0, 255)
        img.putpixel((x, y), (c, c, c))

    return img, gx, gy


def write_split(
    rng: random.Random,
    image_dir: Path,
    csv_root: Path,
    split: str,
    n: int,
    size: int,
    rel_prefix: str,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, float, float]] = []
    for i in range(n):
        img, gx, gy = draw_sample(rng, size)
        fname = f"{split}_{i:06d}.png"
        img.save(image_dir / fname)
        rows.append((f"{rel_prefix}/{fname}", gx, gy))

    csv_path = csv_root / f"{split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "gaze_x", "gaze_y"])
        w.writerows(rows)
    print(f"Wrote {n} images and {csv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic gaze CSV + images")
    p.add_argument("--out_root", type=str, default="data/synthetic", help="root folder for images + CSVs")
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_val", type=int, default=800)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.out_root).resolve()
    train_img = root / "images" / "train"
    val_img = root / "images" / "val"
    root.mkdir(parents=True, exist_ok=True)

    rng_train = random.Random(args.seed)
    rng_val = random.Random(args.seed + 1)

    write_split(
        rng_train,
        train_img,
        root,
        "train",
        args.n_train,
        args.image_size,
        rel_prefix="images/train",
    )
    write_split(
        rng_val,
        val_img,
        root,
        "val",
        args.n_val,
        args.image_size,
        rel_prefix="images/val",
    )

    print("Done. Use:")
    print(f"  --data_root {root}")
    print(f"  --train_csv {root / 'train.csv'}")
    print(f"  --val_csv {root / 'val.csv'}")


if __name__ == "__main__":
    main()
