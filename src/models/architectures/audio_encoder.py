from __future__ import annotations

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(self, input_channels: int = 1, output_dim: int = 256) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) -> reshape to (batch, 1, seq_len)
        x = x.unsqueeze(1)
        feats = self.model(x).squeeze(-1)
        return self.proj(feats)


__all__ = ["AudioEncoder"]
