"""Rename downloaded raw videos to clean film titles per label folder.

Rules:
 - Works under data/raw/<label>/*.mp4
 - Removes trailing YouTube IDs of pattern *_XXXXXXXXXXX (11 chars)
 - Slugifies to ASCII and underscores.
 - Skips rename if target exists.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"


def slugify(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def clean_stem(stem: str) -> str:
    parts = stem.split("_")
    if parts and re.fullmatch(r"[A-Za-z0-9_-]{11}", parts[-1]):
        parts = parts[:-1]
    cleaned = "_".join(parts)
    return slugify(cleaned)


def rename_all() -> None:
    for label_dir in RAW_ROOT.iterdir():
        if not label_dir.is_dir():
            continue
        for mp4 in label_dir.glob("*.mp4"):
            stem = clean_stem(mp4.stem)
            new_path = mp4.with_name(f"{stem}.mp4")
            if new_path == mp4:
                continue
            if new_path.exists():
                print(f"[!] Skip, exists: {new_path.name}")
                continue
            mp4.rename(new_path)
            old = mp4.name.encode("ascii", "ignore").decode("ascii")
            new = new_path.name.encode("ascii", "ignore").decode("ascii")
            print(f"[+] {old} -> {new}")


if __name__ == "__main__":
    rename_all()
