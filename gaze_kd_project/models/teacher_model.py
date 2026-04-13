"""Teacher backbones with a 2D gaze regression head (ResNet18 or MobileNetV2)."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    resnet18,
)

# Canonical names for CLI / checkpoints
TEACHER_ARCH_RESNET18 = "resnet18"
TEACHER_ARCH_MOBILENET_V2 = "mobilenet_v2"


def normalize_teacher_arch(name: str) -> str:
    """Map aliases to ``resnet18`` or ``mobilenet_v2``."""
    n = name.strip().lower().replace("-", "_")
    aliases = {
        "r18": TEACHER_ARCH_RESNET18,
        "mv2": TEACHER_ARCH_MOBILENET_V2,
        "mobilenetv2": TEACHER_ARCH_MOBILENET_V2,
    }
    n = aliases.get(n, n)
    if n not in (TEACHER_ARCH_RESNET18, TEACHER_ARCH_MOBILENET_V2):
        raise ValueError(
            f"Unknown teacher arch {name!r}; use {TEACHER_ARCH_RESNET18} or {TEACHER_ARCH_MOBILENET_V2}"
        )
    return n


def teacher_arch_from_checkpoint(path: str, default: str = TEACHER_ARCH_RESNET18) -> str:
    """Read ``extra.args.teacher_arch`` from a training checkpoint, if present."""
    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return default
    extra = ckpt.get("extra")
    if not isinstance(extra, dict):
        return default
    args_blob = extra.get("args")
    if not isinstance(args_blob, dict):
        return default
    raw = args_blob.get("teacher_arch")
    if not raw:
        return default
    try:
        return normalize_teacher_arch(str(raw))
    except ValueError:
        return default


def resolve_teacher_arch(cli_value: str, checkpoint_path: str) -> str:
    """Use explicit CLI arch if set; otherwise infer from ``checkpoint_path``."""
    if cli_value and str(cli_value).strip():
        return normalize_teacher_arch(str(cli_value))
    return teacher_arch_from_checkpoint(checkpoint_path)


def build_teacher(pretrained: bool = True, arch: str = TEACHER_ARCH_RESNET18) -> nn.Module:
    """
    ImageNet-pretrained backbone with output dimension 2.

    Args:
        pretrained: If True, load torchvision ImageNet weights (except the new head).
        arch: ``resnet18`` (default) or ``mobilenet_v2`` (lighter; often less prone to
            overfitting on small gaze splits than ResNet18).
    """
    arch = normalize_teacher_arch(arch)
    if arch == TEACHER_ARCH_RESNET18:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        return model

    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model
