"""PyTorch Dataset for TubeSense multimodal samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence
import random

import librosa
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


DEFAULT_VISION_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@dataclass
class TubeSenseSample:
    image: torch.Tensor
    audio: torch.Tensor
    text: Dict[str, torch.Tensor]
    label: torch.Tensor


class TubeSenseDataset(torch.utils.data.Dataset[TubeSenseSample]):
    def __init__(
        self,
        csv_path: Path,
        label_to_idx: Dict[str, int],
        tokenizer_name: str = "distilbert-base-uncased",
        vision_transform: Callable | None = None,
        max_length: int = 64,
        sample_rate: int = 16000,
        segment_seconds: int = 20,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.label_to_idx = label_to_idx
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vision_transform = vision_transform or DEFAULT_VISION_TRANSFORM
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> TubeSenseSample:
        row = self.df.iloc[idx]
        image = self._load_image(Path(row["frame_path"]))
        audio = self._load_audio(Path(row["audio_path"]))
        text_inputs = self._encode_text(str(row.get("text", row.get("description", ""))))
        label_str = str(row.get("label", "unknown")).lower()
        label_idx = self.label_to_idx.get(label_str, -1)
        return TubeSenseSample(
            image=image,
            audio=audio,
            text=text_inputs,
            label=torch.tensor(label_idx, dtype=torch.long),
        )

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.vision_transform(img)

    def _load_audio(self, path: Path) -> torch.Tensor:
        total_duration = librosa.get_duration(path=path)
        max_offset = max(total_duration - self.segment_seconds, 0)
        offset = random.uniform(0, max_offset) if max_offset > 0 else 0
        waveform, _ = librosa.load(
            path,
            sr=self.sample_rate,
            mono=True,
            offset=offset,
            duration=self.segment_seconds,
        )
        return torch.tensor(waveform, dtype=torch.float32)

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}


def tubesense_collate(batch: Sequence[TubeSenseSample]) -> Dict[str, torch.Tensor]:
    images = torch.stack([item.image for item in batch])
    labels = torch.stack([item.label for item in batch])

    audios = [item.audio for item in batch]
    padded_audio = pad_sequence(audios, batch_first=True)

    text_keys = batch[0].text.keys()
    text = {k: torch.stack([item.text[k] for item in batch]) for k in text_keys}

    return {"image": images, "audio": padded_audio, "text": text, "label": labels}


__all__ = ["TubeSenseDataset", "tubesense_collate"]
