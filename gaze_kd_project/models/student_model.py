"""Student backbones with a 2D gaze regression head (torchvision or custom micro-CNN)."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ShuffleNet_V2_X0_5_Weights,
    mobilenet_v3_small,
    shufflenet_v2_x0_5,
)

STUDENT_ARCH_MOBILENET_V3_SMALL = "mobilenet_v3_small"
STUDENT_ARCH_SHUFFLENET_V2_X0_5 = "shufflenet_v2_x0_5"
STUDENT_ARCH_GAZE_MICRO = "gaze_micro"

_STUDENT_ARCHS = (
    STUDENT_ARCH_MOBILENET_V3_SMALL,
    STUDENT_ARCH_SHUFFLENET_V2_X0_5,
    STUDENT_ARCH_GAZE_MICRO,
)


class GazeMicroNet(nn.Module):
    """
    Depthwise-separable CNN for 224x224 RGB -> 2D gaze (~101k parameters).

    No ImageNet weights; ``pretrained`` is ignored for this arch.
    """

    def __init__(self) -> None:
        super().__init__()

        def ds_block(in_c: int, out_c: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 3, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            ds_block(48, 80, 2),
            ds_block(80, 120, 2),
            ds_block(120, 160, 2),
            ds_block(160, 176, 2),
            ds_block(176, 176, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(176, 2)

    def forward(self, x):  # type: ignore[no-untyped-def]
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def normalize_student_arch(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    aliases = {
        "mv3": STUDENT_ARCH_MOBILENET_V3_SMALL,
        "mv3_small": STUDENT_ARCH_MOBILENET_V3_SMALL,
        "mobilenetv3_small": STUDENT_ARCH_MOBILENET_V3_SMALL,
        "shuffle": STUDENT_ARCH_SHUFFLENET_V2_X0_5,
        "shufflenet": STUDENT_ARCH_SHUFFLENET_V2_X0_5,
        "shufflenet_v2_x0_5": STUDENT_ARCH_SHUFFLENET_V2_X0_5,
        "tiny": STUDENT_ARCH_SHUFFLENET_V2_X0_5,
        "micro": STUDENT_ARCH_GAZE_MICRO,
        "gaze_micro": STUDENT_ARCH_GAZE_MICRO,
        "ultra_tiny": STUDENT_ARCH_GAZE_MICRO,
    }
    n = aliases.get(n, n)
    if n not in _STUDENT_ARCHS:
        raise ValueError(
            f"Unknown student arch {name!r}; use one of: {', '.join(_STUDENT_ARCHS)}"
        )
    return n


def student_arch_from_checkpoint(path: str, default: str = STUDENT_ARCH_MOBILENET_V3_SMALL) -> str:
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
    raw = args_blob.get("student_arch")
    if not raw:
        return default
    try:
        return normalize_student_arch(str(raw))
    except ValueError:
        return default


def resolve_student_arch(cli_value: str, checkpoint_path: str) -> str:
    if cli_value and str(cli_value).strip():
        return normalize_student_arch(str(cli_value))
    return student_arch_from_checkpoint(checkpoint_path)


def build_student(pretrained: bool = True, arch: str = STUDENT_ARCH_MOBILENET_V3_SMALL) -> nn.Module:
    """
    ImageNet-pretrained backbone with two gaze outputs (except ``gaze_micro``, trained from scratch).

    Args:
        pretrained: If True, load torchvision ImageNet weights for torchvision backbones
            (ignored for ``gaze_micro``).
        arch: ``mobilenet_v3_small`` (~1.5M), ``shufflenet_v2_x0_5`` (~0.34M),
            or ``gaze_micro`` (~0.10M custom CNN).
    """
    arch = normalize_student_arch(arch)
    if arch == STUDENT_ARCH_GAZE_MICRO:
        return GazeMicroNet()

    if arch == STUDENT_ARCH_MOBILENET_V3_SMALL:
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        return model

    weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
    model = shufflenet_v2_x0_5(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model
