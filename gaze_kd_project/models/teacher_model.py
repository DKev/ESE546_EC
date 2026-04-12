"""Teacher: ResNet18 backbone with a 2D regression head."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_teacher(pretrained: bool = True) -> nn.Module:
    """
    ResNet18 with ImageNet pretrained weights and output dimension 2.

    Args:
        pretrained: If True, load torchvision ImageNet weights.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model
