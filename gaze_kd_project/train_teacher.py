"""
Train the ResNet18 teacher with supervised MSE regression on gaze (x, y).

Run from the ``gaze_kd_project`` directory:

    python train_teacher.py --train_csv data/train.csv --val_csv data/val.csv
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import default_train_config, ensure_parent_dir
from datasets.gaze_dataset import GazeDataset
from models.teacher_model import build_teacher
from utils import (
    append_metrics_csv,
    configure_training_runtime,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    validate_epoch,
)


def parse_args() -> argparse.Namespace:
    cfg = default_train_config()
    p = argparse.ArgumentParser(description="Train gaze teacher (ResNet18)")
    p.add_argument("--train_csv", type=str, default=cfg.train_csv)
    p.add_argument("--val_csv", type=str, default=cfg.val_csv)
    p.add_argument("--data_root", type=str, default=cfg.data_root)
    p.add_argument("--checkpoint", type=str, default=cfg.teacher_ckpt)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--image_size", type=int, default=cfg.image_size)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto")
    p.add_argument("--no_pretrained", action="store_true", help="train from scratch")
    p.add_argument(
        "--metrics_csv",
        type=str,
        default="",
        help="optional path to append per-epoch metrics (train/val MSE)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    configure_training_runtime(device)

    train_ds = GazeDataset(args.train_csv, root_dir=args.data_root, image_size=args.image_size)
    val_ds = GazeDataset(args.val_csv, root_dir=args.data_root, image_size=args.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_teacher(pretrained=not args.no_pretrained).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ensure_parent_dir(args.checkpoint)
    best_val = float("inf")
    metric_fields = ["epoch", "train_mse", "val_mse", "is_best"]

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, desc=f"epoch {epoch} train"
        )
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}  train_mse: {train_loss:.6f}  val_mse: {val_loss:.6f}")

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                epoch,
                best_val,
                extra={"args": vars(args)},
            )
            print(f"  saved best checkpoint -> {args.checkpoint} (val_mse={best_val:.6f})")

        if args.metrics_csv:
            append_metrics_csv(
                args.metrics_csv,
                metric_fields,
                {
                    "epoch": epoch,
                    "train_mse": train_loss,
                    "val_mse": val_loss,
                    "is_best": int(improved),
                },
            )

    print("Done. Best val MSE:", best_val)


if __name__ == "__main__":
    main()
