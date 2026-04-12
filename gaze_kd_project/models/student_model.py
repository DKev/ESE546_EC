"""Student: MobileNetV3-Small with a 2D regression head (lightweight)."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_student(pretrained: bool = True) -> nn.Module:
    """
    MobileNetV3-Small with ImageNet pretrained weights and output dimension 2.

    Args:
        pretrained: If True, load torchvision ImageNet weights.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model
