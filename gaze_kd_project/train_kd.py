"""
Knowledge distillation: train student to match labels and teacher predictions.

Loss: MSE(student, gt) + alpha * MSE(student, teacher_pred)

Teacher weights are loaded from a checkpoint and frozen.

Run from the ``gaze_kd_project`` directory:

    python train_kd.py --teacher_ckpt checkpoints/teacher_best.pt \\
        --train_csv data/train.csv --val_csv data/val.csv
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import default_train_config, ensure_parent_dir
from datasets.factory import add_gaze_data_args, build_train_val_datasets
from models.student_model import build_student
from models.teacher_model import build_teacher, resolve_teacher_arch
from utils import (
    append_metrics_csv,
    configure_training_runtime,
    dataloader_common_kwargs,
    grad_scaler_if_amp,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_kd_one_epoch,
    validate_kd_epoch,
)


def parse_args() -> argparse.Namespace:
    cfg = default_train_config()
    p = argparse.ArgumentParser(description="Train student with knowledge distillation")
    p.add_argument("--teacher_ckpt", type=str, default=cfg.teacher_ckpt)
    p.add_argument("--train_csv", type=str, default=cfg.train_csv)
    p.add_argument("--val_csv", type=str, default=cfg.val_csv)
    p.add_argument("--data_root", type=str, default=cfg.data_root)
    p.add_argument("--checkpoint", type=str, default=cfg.student_kd_ckpt)
    p.add_argument("--alpha", type=float, default=cfg.alpha, help="weight for KD MSE term")
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--image_size", type=int, default=cfg.image_size)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto")
    p.add_argument("--no_pretrained_student", action="store_true", help="student ImageNet init off")
    p.add_argument(
        "--teacher_arch",
        type=str,
        default="",
        help="Teacher backbone: resnet18 | mobilenet_v2; empty = read from teacher ckpt extra.args (else resnet18)",
    )
    p.add_argument(
        "--metrics_csv",
        type=str,
        default="",
        help="optional path to append per-epoch KD metrics",
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

    t_arch = resolve_teacher_arch(args.teacher_arch, args.teacher_ckpt)
    print("Teacher backbone:", t_arch)
    teacher = build_teacher(pretrained=False, arch=t_arch).to(device)
    load_checkpoint(args.teacher_ckpt, teacher, optimizer=None, device=device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    student = build_student(pretrained=not args.no_pretrained_student).to(device)
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    ensure_parent_dir(args.checkpoint)
    best_val_total = float("inf")
    metric_fields = [
        "epoch",
        "train_total",
        "train_mse_gt",
        "train_mse_kd",
        "val_total",
        "val_mse_gt",
        "val_mse_kd",
        "is_best",
    ]

    for epoch in range(1, args.epochs + 1):
        tr_total, tr_gt, tr_kd = train_kd_one_epoch(
            teacher,
            student,
            train_loader,
            mse,
            optimizer,
            device,
            args.alpha,
            desc=f"e{epoch} kd",
            use_amp=use_amp,
            scaler=scaler,
        )
        va_total, va_gt, va_kd = validate_kd_epoch(
            teacher, student, val_loader, mse, device, args.alpha, use_amp=use_amp
        )
        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train: total={tr_total:.6f} mse_gt={tr_gt:.6f} mse_kd={tr_kd:.6f} | "
            f"val: total={va_total:.6f} mse_gt={va_gt:.6f} mse_kd={va_kd:.6f}"
        )

        improved = va_total < best_val_total
        if improved:
            best_val_total = va_total
            save_checkpoint(
                args.checkpoint,
                student,
                optimizer,
                epoch,
                best_val_total,
                extra={"args": vars(args), "val_mse_gt": va_gt, "val_mse_kd": va_kd},
            )
            print(f"  saved best student KD checkpoint -> {args.checkpoint}")

        if args.metrics_csv:
            append_metrics_csv(
                args.metrics_csv,
                metric_fields,
                {
                    "epoch": epoch,
                    "train_total": tr_total,
                    "train_mse_gt": tr_gt,
                    "train_mse_kd": tr_kd,
                    "val_total": va_total,
                    "val_mse_gt": va_gt,
                    "val_mse_kd": va_kd,
                    "is_best": int(improved),
                },
            )

    print("Done. Best val total loss:", best_val_total)


if __name__ == "__main__":
    main()
