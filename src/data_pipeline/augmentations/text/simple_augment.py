"""Lightweight text augmentation helpers."""

from __future__ import annotations

import random
from typing import Iterable, List


def random_swap(tokens: List[str]) -> List[str]:
    if len(tokens) < 2:
        return tokens
    idx1, idx2 = random.sample(range(len(tokens)), 2)
    tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
    return tokens


def dropout(tokens: Iterable[str], p: float = 0.1) -> List[str]:
    return [t for t in tokens if random.random() > p]


def augment_text(text: str) -> str:
    tokens = text.split()
    tokens = random_swap(tokens)
    tokens = dropout(tokens)
    return " ".join(tokens)


__all__ = ["augment_text", "random_swap", "dropout"]
