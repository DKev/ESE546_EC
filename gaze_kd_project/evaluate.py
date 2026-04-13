"""
Evaluate a trained gaze model: regression metrics, size, speed.

Run from the ``gaze_kd_project`` directory:

    python evaluate.py --model teacher --checkpoint checkpoints/teacher_best.pt --csv data/val.csv
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import default_train_config
from datasets.factory import add_gaze_data_args, build_eval_dataset
from models.student_model import build_student
from models.teacher_model import build_teacher
from utils import (
    collect_predictions,
    configure_training_runtime,
    count_parameters,
    load_checkpoint,
    measure_latency,
    regression_metrics,
)


def parse_args() -> argparse.Namespace:
    cfg = default_train_config()
    p = argparse.ArgumentParser(description="Evaluate gaze model")
    p.add_argument("--model", type=str, choices=("teacher", "student"), required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="validation/test CSV (required when --dataset csv)",
    )
    p.add_argument("--data_root", type=str, default=cfg.data_root)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--image_size", type=int, default=cfg.image_size)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers)
    p.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto")
    p.add_argument("--latency_batch_size", type=int, default=1, help="batch size for timing")
    p.add_argument("--latency_iters", type=int, default=100)
    p.add_argument("--no_pretrained", action="store_true", help="build backbone without ImageNet weights")
    p.add_argument(
        "--export_json",
        type=str,
        default="",
        help="optional path to write all reported metrics as JSON",
    )
    p.add_argument(
        "--save_predictions",
        type=str,
        default="",
        help="optional path to save .npz with pred, gt arrays for scatter plots",
    )
    add_gaze_data_args(p)
    args = p.parse_args()
    if args.dataset == "csv" and not args.csv:
        p.error("--csv is required when --dataset csv")
    return args


def main() -> None:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    configure_training_runtime(device)

    if args.model == "teacher":
        model = build_teacher(pretrained=not args.no_pretrained).to(device)
    else:
        model = build_student(pretrained=not args.no_pretrained).to(device)

    load_checkpoint(args.checkpoint, model, optimizer=None, device=device)

    ds = build_eval_dataset(args)
    print(f"Eval samples: {len(ds)}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    metrics = regression_metrics(model, loader, device)
    n_params = count_parameters(model)
    ckpt_bytes = os.path.getsize(args.checkpoint)
    ckpt_mb = ckpt_bytes / (1024 * 1024)
    ms_per_img, fps = measure_latency(
        model,
        device,
        image_size=args.image_size,
        batch_size=args.latency_batch_size,
        n_iter=args.latency_iters,
    )

    print("--- Regression metrics ---")
    print(f"MSE (mean squared L2 per sample): {metrics['mse']:.6f}")
    print(f"MAE (mean abs error per scalar x,y): {metrics['mae']:.6f}")
    print(f"Mean Euclidean distance (x,y):       {metrics['mean_l2']:.6f}")
    print("--- Model footprint ---")
    print(f"Parameters:           {n_params:,}")
    print(f"Checkpoint file size: {ckpt_mb:.3f} MB ({ckpt_bytes} bytes)")
    print("--- Inference (dummy input, same image size) ---")
    print(f"Latency: ~{ms_per_img:.3f} ms/image (batch_size={args.latency_batch_size})")
    print(f"Approx FPS: ~{fps:.1f}")

    if args.export_json:
        payload = {
            "model": args.model,
            "checkpoint": os.path.abspath(args.checkpoint),
            "dataset": args.dataset,
            "csv": os.path.abspath(args.csv) if args.csv else "",
            "mpi_root": os.path.abspath(args.mpi_root) if args.dataset == "mpiigaze" else "",
            "mpi_eval_split": args.mpi_eval_split if args.dataset == "mpiigaze" else "",
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "mean_l2": metrics["mean_l2"],
            "params": n_params,
            "ckpt_mb": ckpt_mb,
            "ckpt_bytes": ckpt_bytes,
            "ms_per_image": ms_per_img,
            "fps": fps,
            "latency_batch_size": args.latency_batch_size,
            "image_size": args.image_size,
        }
        os.makedirs(os.path.dirname(args.export_json) or ".", exist_ok=True)
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote metrics JSON -> {args.export_json}")

    if args.save_predictions:
        pred, gt = collect_predictions(model, loader, device)
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        np.savez(
            args.save_predictions,
            pred=pred.numpy(),
            gt=gt.numpy(),
        )
        print(f"Wrote predictions -> {args.save_predictions}")


if __name__ == "__main__":
    main()
