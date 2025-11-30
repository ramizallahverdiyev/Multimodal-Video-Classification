from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class VisionEncoder(nn.Module):
    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.proj = nn.Linear(backbone.fc.in_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        feats = feats.flatten(1)
        return self.proj(feats)


__all__ = ["VisionEncoder"]
