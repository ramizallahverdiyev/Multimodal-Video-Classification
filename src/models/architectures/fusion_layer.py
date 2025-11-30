from __future__ import annotations

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, vision_dim: int, audio_dim: int, text_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(vision_dim + audio_dim + text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, vision_feat: torch.Tensor, audio_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([vision_feat, audio_feat, text_feat], dim=1)
        return self.classifier(fused)


__all__ = ["FusionHead"]
