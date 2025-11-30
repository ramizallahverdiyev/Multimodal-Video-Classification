"""Simple text cleaner for YouTube titles/descriptions."""

from __future__ import annotations

import re


URL_PATTERN = re.compile(r"https?://\S+")
EMOJI_PATTERN = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)


def clean_text(text: str) -> str:
    text = URL_PATTERN.sub("", text)
    text = EMOJI_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


__all__ = ["clean_text"]
