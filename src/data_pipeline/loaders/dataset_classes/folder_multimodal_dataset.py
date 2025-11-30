"""Dataset that loads multimodal samples from folder structure without metadata CSV.

Assumes:
  data/interim/frames/<label>/<video_id>/*.jpg
  data/interim/audio/<label>/<video_id>.wav
  data/interim/text/<label>/<video_id>.txt (optional)
Labels are derived from folder names.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set, Tuple, Optional
import random

import librosa
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
class FolderSample:
    image: torch.Tensor
    audio: torch.Tensor
    text: Dict[str, torch.Tensor]
    label: torch.Tensor


class FolderMultimodalDataset(torch.utils.data.Dataset[FolderSample]):
    def __init__(
        self,
        frames_root: Path,
        audio_root: Path,
        text_root: Path | None,
        label_to_idx: Dict[str, int],
        split_file: Path | None = None,
        split: str = "train",
        tokenizer_name: str = "distilbert-base-uncased",
        vision_transform: Callable | None = None,
        max_length: int = 64,
        sample_rate: int = 16000,
        segment_seconds: int = 20,
    ) -> None:
        self.frames_root = frames_root
        self.audio_root = audio_root
        self.text_root = text_root
        self.label_to_idx = label_to_idx
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.vision_transform = vision_transform or DEFAULT_VISION_TRANSFORM
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.allowed: Optional[Set[Tuple[str, str]]] = None
        if split_file:
            self.allowed = self._load_allowed(split_file, split)

        self.items: List[tuple[str, str]] = []  # (label, video_id)
        for label_dir in frames_root.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            if label not in label_to_idx:
                continue
            for vid_dir in label_dir.iterdir():
                if vid_dir.is_dir():
                    vid_id = vid_dir.name
                    if self.allowed and (label, vid_id) not in self.allowed:
                        continue
                    self.items.append((label, vid_id))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> FolderSample:
        label, vid_id = self.items[idx]
        frame_dir = self.frames_root / label / vid_id
        image = self._load_image(frame_dir)

        audio_path = self.audio_root / label / f"{vid_id}.wav"
        audio = self._load_audio(audio_path)

        text = self._load_text(label, vid_id)

        label_idx = self.label_to_idx[label]
        return FolderSample(image=image, audio=audio, text=text, label=torch.tensor(label_idx, dtype=torch.long))

    def _load_image(self, frame_dir: Path) -> torch.Tensor:
        frames = sorted(frame_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"No frames found in {frame_dir}")
        frame_path = random.choice(frames)
        with Image.open(frame_path) as img:
            img = img.convert("RGB")
            return self.vision_transform(img)

    def _load_audio(self, path: Path) -> torch.Tensor:
        waveform, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        total_duration = len(waveform) / self.sample_rate
        max_offset = max(total_duration - self.segment_seconds, 0)
        offset_sec = random.uniform(0, max_offset) if max_offset > 0 else 0
        start = int(offset_sec * self.sample_rate)
        end = start + int(self.segment_seconds * self.sample_rate)
        segment = waveform[start:end]
        return torch.tensor(segment, dtype=torch.float32)

    def _load_text(self, label: str, vid_id: str) -> Dict[str, torch.Tensor]:
        if not self.text_root:
            return {k: torch.zeros(self.max_length, dtype=torch.long) for k in ["input_ids", "attention_mask"]}
        text_path = self.text_root / label / f"{vid_id}.txt"
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8")
        else:
            text = ""
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    @staticmethod
    def _load_allowed(split_file: Path, split: str) -> Set[Tuple[str, str]]:
        allowed: Set[Tuple[str, str]] = set()
        with split_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not {"label", "movie_id", "split"}.issubset(reader.fieldnames or set()):
                raise ValueError("Split CSV must have columns: label,movie_id,split")
            for row in reader:
                if row["split"] == split:
                    allowed.add((row["label"], row["movie_id"]))
        if not allowed:
            raise ValueError(f"No entries found for split '{split}' in {split_file}")
        return allowed


def folder_collate(batch: Sequence[FolderSample]) -> Dict[str, torch.Tensor]:
    images = torch.stack([b.image for b in batch])
    labels = torch.stack([b.label for b in batch])
    audios = [b.audio for b in batch]
    padded_audio = pad_sequence(audios, batch_first=True)
    text_keys = batch[0].text.keys()
    text = {k: torch.stack([b.text[k] for b in batch]) for k in text_keys}
    return {"image": images, "audio": padded_audio, "text": text, "label": labels}


__all__ = ["FolderMultimodalDataset", "folder_collate"]
