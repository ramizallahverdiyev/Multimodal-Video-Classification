"""Rename raw videos per category to sequential names: movie_1.mp4, movie_2.mp4, ...

Assumes structure: data/raw/<label>/*.mp4
Processing order is sorted by current filename to keep determinism.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"


def rename_label(label_dir: Path) -> None:
    files = sorted(label_dir.glob("*.mp4"))
    if not files:
        return
    # First pass: rename to temp to avoid collisions
    temp_files = []
    for idx, f in enumerate(files, start=1):
        temp = f.with_name(f"__tmp_{idx}__.mp4")
        f.rename(temp)
        temp_files.append(temp)
    # Second pass: final names
    for idx, temp in enumerate(temp_files, start=1):
        new_name = temp.with_name(f"movie_{idx}.mp4")
        temp.rename(new_name)
    print(f"[+] {label_dir.name}: renamed {len(files)} files to movie_1..movie_{len(files)}")


def main() -> None:
    if not RAW_ROOT.exists():
        print(f"[!] raw folder not found: {RAW_ROOT}")
        return
    for label_dir in RAW_ROOT.iterdir():
        if label_dir.is_dir():
            rename_label(label_dir)
    print("Done.")


if __name__ == "__main__":
    main()
