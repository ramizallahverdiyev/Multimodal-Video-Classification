"""Download YouTube trailers for titles listed in imdb_top_*.txt files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[4]
LIST_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

import sys

sys.path.append(str(ROOT))

from src.data_pipeline.sourcing.scrapers.youtube_scraper import search_and_download  # noqa: E402


def load_titles(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def download_for_genre(genre: str, limit: int, resolution: int) -> None:
    list_path = LIST_DIR / f"imdb_top_{genre}.txt"
    if not list_path.exists():
        print(f"[!] Missing list file for {genre}: {list_path}")
        return
    titles = load_titles(list_path)[:limit]
    target_dir = RAW_DIR / genre
    for idx, title in enumerate(titles, start=1):
        query = f"{title} official trailer"
        print(f"[{genre}] {idx}/{len(titles)} -> {query}")
        try:
            search_and_download(
                query=query,
                max_results=1,
                label=genre,
                target_dir=target_dir,
                resolution_height=resolution,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    [!] Failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download trailers for IMDb genre lists.")
    parser.add_argument("--limit", type=int, default=25, help="Titles per genre to download")
    parser.add_argument("--resolution", type=int, default=1080, help="Max video height")
    parser.add_argument(
        "--genres",
        nargs="*",
        default=None,
        help="Optional subset of genres (default: all imdb_top_*.txt present)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.genres:
        genres = args.genres
    else:
        genres = [p.stem.replace("imdb_top_", "") for p in LIST_DIR.glob("imdb_top_*.txt")]
    for genre in genres:
        download_for_genre(genre, limit=args.limit, resolution=args.resolution)
    print("Done.")


if __name__ == "__main__":
    main()
