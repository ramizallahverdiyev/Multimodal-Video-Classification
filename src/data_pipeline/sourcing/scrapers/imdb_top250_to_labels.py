"""Fetch IMDb Top 250 list and export top titles per target genre."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import json
import unicodedata

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "data" / "processed"
IMDB_TOP250_URL = "https://www.imdb.com/chart/top"

# Target genres to extract (case-insensitive, normalized)
TARGET_GENRES = [
    "drama",
    "adventure",
    "thriller",
    "crime",
    "action",
    "comedy",
    "mystery",
    "war",
    "fantasy",
    "sci-fi",
    "family",
    "romance",
]


def normalize_title(title: str) -> str:
    """Normalize unicode titles to ASCII-friendly text."""
    return unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")


def parse_top250() -> List[Dict[str, List[str]]]:
    """Return list of dicts with title and genres (lowercase)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(IMDB_TOP250_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    html = resp.text

    # Extract JSON block containing chartTitles
    key = '"chartTitles"'
    idx = html.find(key)
    if idx == -1:
        raise RuntimeError("chartTitles not found in IMDb page.")
    pos = idx + len(key)
    while pos < len(html) and html[pos] not in "{[":
        pos += 1
    start = pos
    brace = 0
    end = None
    for i in range(start, len(html)):
        ch = html[i]
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                end = i + 1
                break
    if end is None:
        raise RuntimeError("Failed to parse chartTitles JSON block.")
    chunk = html[idx:end]
    data = json.loads("{" + chunk + "}")
    edges = data["chartTitles"]["edges"]

    items: List[Dict[str, List[str]]] = []
    for edge in edges:
        node = edge.get("node", {})
        raw_title = node.get("titleText", {}).get("text", "")
        title = normalize_title(raw_title)
        genres_block = node.get("titleGenres", {}).get("genres", []) or []
        genres = []
        for g in genres_block:
            genre_text = g.get("genre", {}).get("text", "")
            if genre_text:
                genres.append(genre_text.strip().lower())
        if title:
            items.append({"title": title, "genres": genres})
    return items


def export_top_per_genre(limit: int = 25) -> Dict[str, List[str]]:
    data = parse_top250()
    buckets: Dict[str, List[str]] = {g: [] for g in TARGET_GENRES}
    for item in data:
        title = item["title"]
        genres = item["genres"]
        for g in genres:
            if g in buckets and len(buckets[g]) < limit:
                buckets[g].append(title)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for genre, titles in buckets.items():
        out_path = OUT_DIR / f"imdb_top_{genre}.txt"
        out_path.write_text("\n".join(titles), encoding="utf-8")
        print(f"[+] {genre}: saved {len(titles)} titles -> {out_path}")
    return buckets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export IMDb Top 250 titles per genre.")
    parser.add_argument("--limit", type=int, default=25, help="Titles per genre (default: 25)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_top_per_genre(limit=args.limit)
    print("Done.")


if __name__ == "__main__":
    main()
