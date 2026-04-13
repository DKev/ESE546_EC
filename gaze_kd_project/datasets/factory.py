"""Build train/val :class:`torch.utils.data.Dataset` from CLI-style args (CSV or MPIIGaze)."""

from __future__ import annotations

import argparse

from torch.utils.data import Dataset

from datasets.gaze_dataset import GazeDataset
from datasets.mpiigaze_dataset import MPIIGazeNormalizedDataset, parse_val_person_ids


def _mpi_dataset_kwargs(args: argparse.Namespace) -> dict:
    lim = args.mpi_preload_max_unique
    if lim <= 0:
        return {"no_preload": True, "preload_max_unique_mats": 512}
    return {
        "no_preload": args.mpi_no_preload,
        "preload_max_unique_mats": lim,
    }


def add_gaze_data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument(
        "--dataset",
        type=str,
        default="csv",
        choices=("csv", "mpiigaze"),
        help="csv: image_path,gaze_x,gaze_y CSVs; mpiigaze: Data/Normalized .mat",
    )
    g.add_argument(
        "--mpi_root",
        type=str,
        default="../MPIIGaze",
        help="Root folder containing Data/Normalized (ignored for dataset=csv)",
    )
    g.add_argument(
        "--mpi_val_persons",
        type=str,
        default="14,15",
        help="Comma-separated participant indices for validation, e.g. 14,15 -> p14,p15",
    )
    g.add_argument(
        "--mpi_max_samples",
        type=int,
        default=0,
        help="If >0, cap train and val at this many samples each (ignored if train/val-specific caps are set)",
    )
    g.add_argument(
        "--mpi_max_train_samples",
        type=int,
        default=0,
        help="If >0, cap training split only (e.g. 10000 for fast epochs; val unchanged unless mpi_max_val_samples set)",
    )
    g.add_argument(
        "--mpi_max_val_samples",
        type=int,
        default=0,
        help="If >0, cap validation split only",
    )
    g.add_argument(
        "--mpi_eval_split",
        type=str,
        default="val",
        choices=("train", "val"),
        help="For evaluate.py with dataset=mpiigaze: which participant split to score",
    )
    g.add_argument(
        "--mpi_no_preload",
        action="store_true",
        help="MPIIGaze: do not load .mat files into RAM (slower per sample; use if RAM is tight or many DataLoader workers duplicate memory on Windows)",
    )
    g.add_argument(
        "--mpi_preload_max_unique",
        type=int,
        default=512,
        help="MPIIGaze: preload at most this many distinct dayYY.mat files per split (0 disables preload same as --mpi_no_preload)",
    )


def build_train_val_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.dataset == "csv":
        train = GazeDataset(args.train_csv, root_dir=args.data_root, image_size=args.image_size)
        val = GazeDataset(args.val_csv, root_dir=args.data_root, image_size=args.image_size)
        return train, val

    val_ids = parse_val_person_ids(args.mpi_val_persons)
    train_cap = (
        args.mpi_max_train_samples
        if args.mpi_max_train_samples > 0
        else (args.mpi_max_samples if args.mpi_max_samples > 0 else 0)
    )
    val_cap = (
        args.mpi_max_val_samples
        if args.mpi_max_val_samples > 0
        else (args.mpi_max_samples if args.mpi_max_samples > 0 else 0)
    )
    mk = _mpi_dataset_kwargs(args)
    train = MPIIGazeNormalizedDataset(
        args.mpi_root,
        split="train",
        val_person_ids=val_ids,
        image_size=args.image_size,
        max_samples=train_cap,
        **mk,
    )
    val = MPIIGazeNormalizedDataset(
        args.mpi_root,
        split="val",
        val_person_ids=val_ids,
        image_size=args.image_size,
        max_samples=val_cap,
        **mk,
    )
    return train, val


def build_eval_dataset(args: argparse.Namespace) -> Dataset:
    if args.dataset == "csv":
        return GazeDataset(args.csv, root_dir=args.data_root, image_size=args.image_size)

    val_ids = parse_val_person_ids(args.mpi_val_persons)
    split = args.mpi_eval_split
    if split not in ("train", "val"):
        raise ValueError("mpi_eval_split must be train or val")
    cap = (
        args.mpi_max_train_samples
        if split == "train" and args.mpi_max_train_samples > 0
        else args.mpi_max_val_samples
        if split == "val" and args.mpi_max_val_samples > 0
        else (args.mpi_max_samples if args.mpi_max_samples > 0 else 0)
    )
    return MPIIGazeNormalizedDataset(
        args.mpi_root,
        split=split,
        val_person_ids=val_ids,
        image_size=args.image_size,
        max_samples=cap,
        **_mpi_dataset_kwargs(args),
    )
