"""
Utility to build a train/val/test split CSV from the folder-based multimodal data.

It expects the standard layout:
  data/interim/frames/<label>/<movie_id>/*.jpg
  data/interim/audio/<label>/<movie_id>.wav
  data/interim/text/<label>/<movie_id>.txt

Only samples that exist in all three modalities are included. Output CSV columns:
label,movie_id,split
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def collect_ids(root: Path, labels: Iterable[str] | None, require_text: bool = True) -> Dict[str, Set[str]]:
    frames_root = root / "frames"
    audio_root = root / "audio"
    text_root = root / "text"

    label_filter = set(labels) if labels else None
    collected: Dict[str, Set[str]] = {}

    for label_dir in frames_root.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label_filter and label not in label_filter:
            continue

        frame_ids = {d.name for d in label_dir.iterdir() if d.is_dir()}
        audio_dir = audio_root / label
        text_dir = text_root / label
        if not audio_dir.exists():
            continue
        audio_ids = {p.stem for p in audio_dir.glob("*.wav")}
        ids = frame_ids & audio_ids
        if require_text:
            if not text_dir.exists():
                continue
            text_ids = {p.stem for p in text_dir.glob("*.txt")}
            ids &= text_ids
        if ids:
            collected[label] = ids
    return collected


def write_split(
    items: List[Tuple[str, str]],
    out_path: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, int]:
    random.seed(seed)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    splits = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
    rows = [(label, movie_id, split) for (label, movie_id), split in zip(items, splits, strict=True)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "movie_id", "split"])
        writer.writerows(rows)

    return {"total": n, "train": n_train, "val": n_val, "test": n_test}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val/test split CSV from folder multimodal data.")
    parser.add_argument("--root", type=Path, default=Path("data/interim"), help="Root that contains frames/audio/text")
    parser.add_argument("--out", type=Path, default=Path("data/processed/split.csv"), help="Output CSV path")
    parser.add_argument("--labels", nargs="*", help="Optional subset of labels to include")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--no-text", action="store_true", help="Do not require text modality")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_ratio + args.val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio must be < 1.0 to leave room for test split")

    available = collect_ids(args.root, args.labels, require_text=not args.no_text)
    items: List[Tuple[str, str]] = []
    for label, ids in available.items():
        for movie_id in ids:
            items.append((label, movie_id))

    if not items:
        raise SystemExit("No samples found with all modalities present.")

    summary = write_split(
        items=items,
        out_path=args.out,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"[+] Wrote split to {args.out} -> {summary}")


if __name__ == "__main__":
    main()
