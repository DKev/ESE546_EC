"""
Train the teacher (ResNet18, MobileNetV2, or MobileNetV3-Small) with supervised MSE on gaze (x, y).

Run from the ``gaze_kd_project`` directory:

    python train_teacher.py --train_csv data/train.csv --val_csv data/val.csv
    python train_teacher.py --dataset mpiigaze ... --teacher_arch mobilenet_v3_small
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import default_train_config, ensure_parent_dir
from datasets.factory import add_gaze_data_args, build_train_val_datasets
from models.teacher_model import (
    TEACHER_ARCH_MOBILENET_V2,
    TEACHER_ARCH_MOBILENET_V3_SMALL,
    TEACHER_ARCH_RESNET18,
    build_teacher,
)
from utils import (
    append_metrics_csv,
    configure_training_runtime,
    dataloader_common_kwargs,
    grad_scaler_if_amp,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    validate_epoch,
)


def parse_args() -> argparse.Namespace:
    cfg = default_train_config()
    p = argparse.ArgumentParser(description="Train gaze teacher (ResNet18 / MobileNetV2 / MobileNetV3-Small)")
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
        "--teacher_arch",
        type=str,
        default=TEACHER_ARCH_RESNET18,
        choices=(
            TEACHER_ARCH_RESNET18,
            TEACHER_ARCH_MOBILENET_V2,
            TEACHER_ARCH_MOBILENET_V3_SMALL,
        ),
        help="Teacher: resnet18 | mobilenet_v2 | mobilenet_v3_small (use MV3-Small when it fits your data best)",
    )
    p.add_argument(
        "--metrics_csv",
        type=str,
        default="",
        help="optional path to append per-epoch metrics (train/val MSE)",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="CUDA automatic mixed precision (faster on GPU; no effect on CPU)",
    )
    add_gaze_data_args(p)
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

    train_ds, val_ds = build_train_val_datasets(args)
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    dl_kw = dataloader_common_kwargs(num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **dl_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **dl_kw,
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = grad_scaler_if_amp(use_amp=use_amp, device=device)
    if use_amp:
        print("AMP (mixed precision) enabled")

    print("Teacher backbone:", args.teacher_arch)
    model = build_teacher(pretrained=not args.no_pretrained, arch=args.teacher_arch).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ensure_parent_dir(args.checkpoint)
    best_val = float("inf")
    metric_fields = ["epoch", "train_mse", "val_mse", "is_best"]

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            desc=f"epoch {epoch} train",
            use_amp=use_amp,
            scaler=scaler,
        )
        val_loss = validate_epoch(model, val_loader, criterion, device, use_amp=use_amp)
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
