"""Image augmentation utilities."""

from __future__ import annotations

from typing import Callable
from PIL import Image
import torchvision.transforms as T


def build_transform() -> Callable[[Image.Image], Image.Image]:
    return T.Compose(
        [
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomHorizontalFlip(),
        ]
    )


__all__ = ["build_transform"]
