"""
Central configuration defaults for gaze estimation training.

Override via CLI arguments in each training / evaluation script.
Tuned for a single-GPU setup (e.g. RTX 3060 Ti).
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # Optimization
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 20
    num_workers: int = 4

    # Data
    image_size: int = 224
    data_root: str = "."

    # Knowledge distillation
    alpha: float = 0.5

    # Paths (placeholders — set via CLI or edit for your machine)
    train_csv: str = "data/train.csv"
    val_csv: str = "data/val.csv"
    teacher_ckpt: str = "checkpoints/teacher_best.pt"
    student_baseline_ckpt: str = "checkpoints/student_baseline_best.pt"
    student_kd_ckpt: str = "checkpoints/student_kd_best.pt"

    # Reproducibility
    seed: int = 42


def default_train_config() -> TrainConfig:
    return TrainConfig()


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file path if needed."""
    p = Path(path)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)
