"""IMDB/OMDb metadata fetcher to enrich downloaded trailers with labels."""

from __future__ import annotations

import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List

import requests


ROOT = Path(__file__).resolve().parents[4]
METADATA_CSV = ROOT / "data" / "processed" / "metadata.csv"


def fetch_imdb_label(title: str, api_key: str) -> Dict[str, str]:
    url = "https://www.omdbapi.com/"
    params = {"t": title, "apikey": api_key, "type": "movie"}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    genre = payload.get("Genre", "")
    return {"imdb_genre": genre, "imdb_title": payload.get("Title", ""), "imdb_year": payload.get("Year", "")}


def enrich_metadata(api_key: str) -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {METADATA_CSV}")

    rows: List[Dict[str, str]] = []
    with METADATA_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    for row in rows:
        title = row.get("title", "")
        if not title:
            continue
        try:
            extra = fetch_imdb_label(title, api_key)
            row.update(extra)
            if extra.get("imdb_genre"):
                row["label"] = extra["imdb_genre"].split(",")[0].strip().lower()
            print(f"[+] Enriched {title} -> {row.get('label')}")
        except Exception as exc:  # noqa: BLE001
            print(f"[!] Failed to enrich {title}: {exc}")

    with METADATA_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[+] Saved enriched metadata to {METADATA_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch IMDB/OMDb metadata for TubeSense videos.")
    parser.add_argument("--api-key", default=os.getenv("OMDB_API_KEY"), help="OMDb API key (env: OMDB_API_KEY)")
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("OMDb API key is required (set --api-key or OMDB_API_KEY).")
    enrich_metadata(args.api_key)


if __name__ == "__main__":
    main()
