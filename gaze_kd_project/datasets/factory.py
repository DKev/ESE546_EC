"""Build train/val :class:`torch.utils.data.Dataset` from CLI-style args (CSV or MPIIGaze)."""

from __future__ import annotations

import argparse

from torch.utils.data import Dataset

from datasets.gaze_dataset import GazeDataset
from datasets.mpiigaze_dataset import MPIIGazeNormalizedDataset, parse_val_person_ids


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
        help="If >0, cap each split at this many samples (debug / quick runs)",
    )
    g.add_argument(
        "--mpi_eval_split",
        type=str,
        default="val",
        choices=("train", "val"),
        help="For evaluate.py with dataset=mpiigaze: which participant split to score",
    )


def build_train_val_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.dataset == "csv":
        train = GazeDataset(args.train_csv, root_dir=args.data_root, image_size=args.image_size)
        val = GazeDataset(args.val_csv, root_dir=args.data_root, image_size=args.image_size)
        return train, val

    val_ids = parse_val_person_ids(args.mpi_val_persons)
    train = MPIIGazeNormalizedDataset(
        args.mpi_root,
        split="train",
        val_person_ids=val_ids,
        image_size=args.image_size,
        max_samples=args.mpi_max_samples,
    )
    val = MPIIGazeNormalizedDataset(
        args.mpi_root,
        split="val",
        val_person_ids=val_ids,
        image_size=args.image_size,
        max_samples=args.mpi_max_samples,
    )
    return train, val


def build_eval_dataset(args: argparse.Namespace) -> Dataset:
    if args.dataset == "csv":
        return GazeDataset(args.csv, root_dir=args.data_root, image_size=args.image_size)

    val_ids = parse_val_person_ids(args.mpi_val_persons)
    split = args.mpi_eval_split
    if split not in ("train", "val"):
        raise ValueError("mpi_eval_split must be train or val")
    return MPIIGazeNormalizedDataset(
        args.mpi_root,
        split=split,
        val_person_ids=val_ids,
        image_size=args.image_size,
        max_samples=args.mpi_max_samples,
    )
