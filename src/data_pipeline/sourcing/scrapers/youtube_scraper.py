"""YouTube scraper for TubeSense using yt-dlp for reliability."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
from typing import List

import yt_dlp
import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_CSV = PROCESSED_DIR / "metadata.csv"


@dataclass
class VideoRecord:
    video_id: str
    title: str
    description: str
    filepath: str
    channel: str
    length_seconds: int
    publish_date: str
    label: str | None = None


def _slugify(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    return safe.strip("_").lower() or "video"


def search_and_download(
    query: str,
    max_results: int,
    label: str | None = None,
    target_dir: Path | None = None,
    resolution_height: int = 1080,
) -> List[VideoRecord]:
    target_dir = target_dir or RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    search_url = f"ytsearch{max_results}:{query}"
    fmt = f"bestvideo[ext=mp4][height<={resolution_height}]+bestaudio/best[ext=mp4][height<={resolution_height}]"
    ffmpeg_bin = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    outtmpl = str(target_dir / "%(title)s.%(ext)s")
    ydl_opts = {
        "format": fmt,
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "ffmpeg_location": ffmpeg_bin,
        "quiet": True,
        "nocheckcertificate": True,
    }

    videos: List[VideoRecord] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_url, download=True)
        entries = info.get("entries", [])
        for entry in entries:
            title = entry.get("title", "") or ""
            slug = _slugify(title)
            filename = target_dir / f"{slug}.mp4"
            record = VideoRecord(
                video_id=entry.get("id", ""),
                title=title,
                description=entry.get("description", "") or "",
                filepath=str(filename),
                channel=entry.get("uploader", ""),
                length_seconds=int(entry.get("duration") or 0),
                publish_date=str(entry.get("upload_date") or ""),
                label=label,
            )
            videos.append(record)

    if videos:
        write_metadata(videos)
    return videos


def write_metadata(videos: List[VideoRecord]) -> None:
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(videos[0]).keys())
    write_header = not METADATA_CSV.exists()
    with METADATA_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for video in videos:
            writer.writerow(asdict(video))
    print(f"[+] Wrote metadata to {METADATA_CSV}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search and download YouTube videos for TubeSense.")
    parser.add_argument("--query", required=True, help='Search query, e.g. "Official Trailer 2024"')
    parser.add_argument("--max-results", type=int, default=5, help="How many videos to download")
    parser.add_argument("--label", help="Optional label to attach to all downloaded videos")
    parser.add_argument("--resolution", type=int, default=1080, help="Max video height in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_and_download(
        args.query,
        args.max_results,
        label=args.label,
        resolution_height=args.resolution,
    )
    print("Done.")


if __name__ == "__main__":
    main()
