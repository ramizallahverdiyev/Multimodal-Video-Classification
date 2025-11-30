from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", output_dim: int = 256) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = torch.mean(hidden, dim=1)
        return self.proj(pooled)


__all__ = ["TextEncoder"]
