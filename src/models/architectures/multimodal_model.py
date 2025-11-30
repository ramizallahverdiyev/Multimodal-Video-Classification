from __future__ import annotations

import torch
import torch.nn as nn

from .vision_encoder import VisionEncoder
from .audio_encoder import AudioEncoder
from .text_encoder import TextEncoder
from .fusion_layer import FusionHead


class TubeSenseModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        text_model: str = "distilbert-base-uncased",
        vision_dim: int = 512,
        audio_dim: int = 256,
        text_dim: int = 256,
    ) -> None:
        super().__init__()
        self.vision = VisionEncoder(output_dim=vision_dim)
        self.audio = AudioEncoder(output_dim=audio_dim)
        self.text = TextEncoder(model_name=text_model, output_dim=text_dim)
        self.fusion = FusionHead(vision_dim, audio_dim, text_dim, num_classes)

    def forward(self, image: torch.Tensor, audio: torch.Tensor, text: dict[str, torch.Tensor]) -> torch.Tensor:
        vision_feat = self.vision(image)
        audio_feat = self.audio(audio)
        text_feat = self.text(input_ids=text["input_ids"], attention_mask=text["attention_mask"])
        return self.fusion(vision_feat, audio_feat, text_feat)


__all__ = ["TubeSenseModel"]
