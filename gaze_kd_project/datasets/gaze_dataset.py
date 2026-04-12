"""
PyTorch Dataset for 2D gaze regression from a CSV file.

CSV format (header row required):
    image_path,gaze_x,gaze_y

Paths in image_path may be absolute or relative to ``root_dir``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GazeDataset(Dataset):
    """
    Load face images and 2D gaze targets from a CSV.

    Returns:
        image: FloatTensor [3, H, W], ImageNet-normalized.
        gaze: FloatTensor [2] with (gaze_x, gaze_y).
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str = ".",
        image_size: int = 224,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required = {"image_path", "gaze_x", "gaze_y"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must contain columns {required}, got {set(df.columns)}")

        self._paths = df["image_path"].astype(str).tolist()
        self._gx = df["gaze_x"].astype(float).tolist()
        self._gy = df["gaze_y"].astype(float).tolist()

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _resolve_path(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return (self.root_dir / p).resolve()

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self._resolve_path(self._paths[idx])
        if not path.is_file():
            raise FileNotFoundError(f"Image missing for index {idx}: {path}")

        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([self._gx[idx], self._gy[idx]], dtype=torch.float32)
        return x, y


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gaze_dataset.py <path_to.csv> [root_dir]")
        sys.exit(1)
    csv = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else "."
    ds = GazeDataset(csv, root_dir=root, image_size=224)
    print(f"Loaded {len(ds)} samples from {csv}")
    if len(ds) > 0:
        img, g = ds[0]
        print("Sample image shape:", tuple(img.shape), "gaze:", g.tolist())
