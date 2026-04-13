"""Teacher backbones with a 2D gaze regression head."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    mobilenet_v3_small,
    resnet18,
)

TEACHER_ARCH_RESNET18 = "resnet18"
TEACHER_ARCH_MOBILENET_V2 = "mobilenet_v2"
TEACHER_ARCH_MOBILENET_V3_SMALL = "mobilenet_v3_small"


def normalize_teacher_arch(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    aliases = {
        "r18": TEACHER_ARCH_RESNET18,
        "mv2": TEACHER_ARCH_MOBILENET_V2,
        "mobilenetv2": TEACHER_ARCH_MOBILENET_V2,
        "mv3": TEACHER_ARCH_MOBILENET_V3_SMALL,
        "mv3_small": TEACHER_ARCH_MOBILENET_V3_SMALL,
        "mobilenetv3_small": TEACHER_ARCH_MOBILENET_V3_SMALL,
    }
    n = aliases.get(n, n)
    allowed = (
        TEACHER_ARCH_RESNET18,
        TEACHER_ARCH_MOBILENET_V2,
        TEACHER_ARCH_MOBILENET_V3_SMALL,
    )
    if n not in allowed:
        raise ValueError(f"Unknown teacher arch {name!r}; use one of {allowed}")
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
    if cli_value and str(cli_value).strip():
        return normalize_teacher_arch(str(cli_value))
    return teacher_arch_from_checkpoint(checkpoint_path)


def build_teacher(pretrained: bool = True, arch: str = TEACHER_ARCH_RESNET18) -> nn.Module:
    """
    ImageNet-pretrained backbone with output dimension 2.

    Args:
        pretrained: If True, load torchvision ImageNet weights (except the new head).
        arch: ``resnet18``, ``mobilenet_v2``, or ``mobilenet_v3_small`` (same trunk as the
            default student; use when MV3-Small is the strongest backbone on your data).
    """
    arch = normalize_teacher_arch(arch)
    if arch == TEACHER_ARCH_RESNET18:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        return model
    if arch == TEACHER_ARCH_MOBILENET_V2:
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        return model

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model
