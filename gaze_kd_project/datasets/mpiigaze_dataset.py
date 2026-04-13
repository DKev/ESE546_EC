"""
Load MPIIGaze **Normalized** `.mat` files (``Data/Normalized/pXX/dayYY.mat``).

Each file contains ``filenames`` and ``data`` with ``left`` / ``right`` structs holding
``gaze`` (N, 3), ``image`` (N, 36, 60) uint8, ``pose`` (N, 3). We convert unit gaze
vectors to (gaze_x, gaze_y) in approximately [-1, 1] using yaw / pitch (see
``gaze_vector_to_xy``) so the same MSE heads as synthetic / CSV data apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def parse_val_person_ids(s: str) -> set[int]:
    """Parse ``\"14,15\"`` -> ``{14, 15}`` for ``p14``, ``p15``."""
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def gaze_vector_to_xy(g: np.ndarray) -> tuple[float, float]:
    """Map MPIIGaze normalized unit gaze (x, y, z) to (gaze_x, gaze_y) ~ [-1, 1]."""
    x, y, z = float(g[0]), float(g[1]), float(g[2])
    pitch = float(np.arcsin(np.clip(-y, -1.0, 1.0)))
    yaw = float(np.arctan2(-x, -z))
    return yaw / np.pi, pitch / (np.pi / 2)


def _list_participant_ids(normalized_root: Path) -> list[int]:
    ids: list[int] = []
    if not normalized_root.is_dir():
        return ids
    for child in sorted(normalized_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if len(name) == 3 and name[0] == "p" and name[1:].isdigit():
            ids.append(int(name[1:]))
    return ids


def _mat_sample_count(path: Path) -> int:
    m = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True, variable_names=["filenames"])
    fn = m["filenames"]
    if isinstance(fn, (str, bytes)):
        return 1
    arr = np.asarray(fn)
    if arr.ndim == 0:
        return 1
    return int(arr.size)


@dataclass(frozen=True)
class _SampleRef:
    mat_path: Path
    row: int
    side: str  # "left" or "right"


def _build_index_for_persons(
    normalized_root: Path,
    person_ids: Iterable[int],
    *,
    max_refs: int = 0,
) -> list[_SampleRef]:
    refs: list[_SampleRef] = []
    for pid in sorted(person_ids):
        pdir = normalized_root / f"p{pid:02d}"
        if not pdir.is_dir():
            continue
        for mat_path in sorted(pdir.glob("day*.mat")):
            try:
                n = _mat_sample_count(mat_path)
            except OSError:
                continue
            for i in range(n):
                refs.append(_SampleRef(mat_path, i, "left"))
                refs.append(_SampleRef(mat_path, i, "right"))
                if max_refs and len(refs) >= max_refs:
                    return refs
    return refs


class MPIIGazeNormalizedDataset(Dataset):
    """
    Train/val split by **participant id** (e.g. val on ``p14``, ``p15``).

    Images are grayscale crops expanded to RGB; same ImageNet normalization as
    :class:`GazeDataset`.
    """

    def __init__(
        self,
        mpi_root: str | Path,
        *,
        split: str,
        val_person_ids: set[int],
        image_size: int = 224,
        max_samples: int = 0,
        no_preload: bool = False,
        preload_max_unique_mats: int = 512,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        root = Path(mpi_root).resolve()
        norm = root / "Data" / "Normalized"
        if not norm.is_dir():
            raise FileNotFoundError(f"MPIIGaze Normalized folder not found: {norm}")

        all_ids = _list_participant_ids(norm)
        if not all_ids:
            raise FileNotFoundError(f"No pXX folders under {norm}")

        train_ids = [i for i in all_ids if i not in val_person_ids]
        val_ids = [i for i in all_ids if i in val_person_ids]
        if split == "train" and not train_ids:
            raise ValueError("No training participants left; check mpi_val_persons")
        if split == "val" and not val_ids:
            raise ValueError(
                "No validation participants matched mpi_val_persons; "
                f"available ids: {all_ids}, val filter: {sorted(val_person_ids)}"
            )

        use_ids = train_ids if split == "train" else val_ids
        cap = max_samples if max_samples > 0 else 0
        self._index = _build_index_for_persons(norm, use_ids, max_refs=cap)

        unique_paths = {ref.mat_path for ref in self._index}
        self._mat_cache: dict[Path, dict] | None = None
        if (
            not no_preload
            and unique_paths
            and len(unique_paths) <= max(1, preload_max_unique_mats)
        ):
            self._mat_cache = {}
            for p in sorted(unique_paths):
                self._mat_cache[p] = sio.loadmat(str(p), struct_as_record=False, squeeze_me=True)
            print(
                f"MPIIGaze [{split}]: preloaded {len(self._mat_cache)} .mat files "
                f"({len(self._index)} samples) into RAM"
            )
        elif not no_preload and len(unique_paths) > preload_max_unique_mats:
            print(
                f"MPIIGaze [{split}]: {len(unique_paths)} unique .mat files > "
                f"preload_max_unique={preload_max_unique_mats}; using lazy load "
                f"(raise limit or use --mpi_no_preload to silence)"
            )

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
        self._lazy_path: Path | None = None
        self._lazy_mat: dict | None = None

    def __len__(self) -> int:
        return len(self._index)

    def _ensure_mat(self, path: Path) -> dict:
        if self._mat_cache is not None:
            return self._mat_cache[path]
        if self._lazy_path != path:
            self._lazy_mat = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
            self._lazy_path = path
        assert self._lazy_mat is not None
        return self._lazy_mat

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ref = self._index[idx]
        mat = self._ensure_mat(ref.mat_path)
        data = mat["data"]
        side = getattr(data, ref.side)
        row = ref.row
        im = np.asarray(side.image[row], dtype=np.uint8)
        if im.ndim != 2:
            im = np.squeeze(im)
        g = np.asarray(side.gaze[row], dtype=np.float64).reshape(3)
        gx, gy = gaze_vector_to_xy(g)

        pil = Image.fromarray(im, mode="L").convert("RGB")
        x = self.transform(pil)
        y = torch.tensor([gx, gy], dtype=torch.float32)
        return x, y
