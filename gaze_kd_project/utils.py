"""
Shared utilities: seeding, train/val loops, checkpoints, latency, parameter count.
"""

from __future__ import annotations

import os
import random
import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def dataloader_common_kwargs(*, num_workers: int, pin_memory: bool) -> dict[str, Any]:
    """``persistent_workers`` avoids respawning workers each epoch when ``num_workers > 0``."""
    d: dict[str, Any] = {"num_workers": num_workers, "pin_memory": bool(pin_memory)}
    if num_workers > 0:
        d["persistent_workers"] = True
    return d


def grad_scaler_if_amp(*, use_amp: bool, device: torch.device) -> Any | None:
    if not use_amp or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_if_amp(*, use_amp: bool, device: torch.device):
    if not use_amp or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast()


def set_seed(seed: int) -> None:
    """Fix RNG seeds for reproducibility (slight CUDNN perf cost if deterministic)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Uncomment for stricter reproducibility on GPU (can slow training):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def configure_training_runtime(device: torch.device) -> None:
    """
    Enable common GPU throughput settings (safe defaults for training/inference).

    - cudnn.benchmark picks fast conv algorithms for fixed input sizes.
    - float32 matmul precision trade-off on Ampere+ (slightly faster, tiny numeric drift).
    """
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def append_metrics_csv(path: str, fieldnames: list[str], row: dict[str, Any]) -> None:
    """Append one row to a CSV (creates file with header if missing)."""
    import csv

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def count_parameters(model: nn.Module) -> int:
    """Total trainable + non-trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_metric: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save training state to ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load weights into ``model`` (and optionally optimizer). Returns checkpoint dict."""
    map_loc = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=map_loc, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    desc: str = "train",
    *,
    use_amp: bool = False,
    scaler: Any | None = None,
) -> float:
    """One supervised epoch; returns average MSE loss."""
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_if_amp(use_amp=use_amp, device=device):
            pred = model(x)
            loss = criterion(pred, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> float:
    """Validation average MSE loss."""
    model.eval()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_if_amp(use_amp=use_amp, device=device):
            pred = model(x)
            loss = criterion(pred, y)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def train_kd_one_epoch(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    mse: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    desc: str = "train_kd",
    *,
    use_amp: bool = False,
    scaler: Any | None = None,
) -> tuple[float, float, float]:
    """
    One KD epoch. Returns (total_loss_avg, loss_gt_avg, loss_kd_avg).
    total = mse(s, y) + alpha * mse(s, t).
    """
    teacher.eval()
    student.train()
    sum_total = sum_gt = sum_kd = 0.0
    n = 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_if_amp(use_amp=use_amp, device=device):
            with torch.no_grad():
                t_pred = teacher(x)
            s_pred = student(x)
            loss_gt = mse(s_pred, y)
            loss_kd = mse(s_pred, t_pred)
            loss = loss_gt + alpha * loss_kd
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = x.size(0)
        sum_total += loss.item() * bs
        sum_gt += loss_gt.item() * bs
        sum_kd += loss_kd.item() * bs
        n += bs
    denom = max(n, 1)
    return sum_total / denom, sum_gt / denom, sum_kd / denom


@torch.no_grad()
def validate_kd_epoch(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    mse: nn.Module,
    device: torch.device,
    alpha: float,
    *,
    use_amp: bool = False,
) -> tuple[float, float, float]:
    """KD validation: same losses as training. Returns (total, gt, kd) averages."""
    teacher.eval()
    student.eval()
    sum_total = sum_gt = sum_kd = 0.0
    n = 0
    for x, y in tqdm(loader, desc="val_kd", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_if_amp(use_amp=use_amp, device=device):
            t_pred = teacher(x)
            s_pred = student(x)
            loss_gt = mse(s_pred, y)
            loss_kd = mse(s_pred, t_pred)
            loss = loss_gt + alpha * loss_kd
        bs = x.size(0)
        sum_total += loss.item() * bs
        sum_gt += loss_gt.item() * bs
        sum_kd += loss_kd.item() * bs
        n += bs
    denom = max(n, 1)
    return sum_total / denom, sum_gt / denom, sum_kd / denom


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 1,
    n_warmup: int = 10,
    n_iter: int = 100,
) -> tuple[float, float]:
    """
    Mean inference time per image (ms) and approximate FPS for batch_size images per step.

    Uses a random dummy input of shape [batch_size, 3, H, W].
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)

    for _ in range(n_warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    ms_per_step = elapsed_ms / n_iter
    ms_per_image = ms_per_step / batch_size
    fps = 1000.0 / ms_per_image
    return ms_per_image, fps


@torch.no_grad()
def regression_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Dataset-level metrics: MSE (mean squared L2 per sample), MAE over all scalars,
    mean Euclidean distance between prediction and ground truth.
    """
    model.eval()
    sse = 0.0
    abs_sum = 0.0
    l2_sum = 0.0
    n_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        diff = pred - y
        # Per-sample sum of squared errors (2D)
        sse += diff.pow(2).sum(dim=1).sum().item()
        abs_sum += diff.abs().sum().item()
        l2_sum += torch.linalg.norm(diff, dim=1).sum().item()
        n_samples += x.size(0)

    n = max(n_samples, 1)
    mse = sse / n
    mae = abs_sum / (2 * n)
    mean_l2 = l2_sum / n
    return {"mse": mse, "mae": mae, "mean_l2": mean_l2}


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stacked predictions and ground-truth gaze tensors on CPU."""
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        preds.append(pred.detach().cpu())
        gts.append(y.detach().cpu())
    return torch.cat(preds, dim=0), torch.cat(gts, dim=0)
