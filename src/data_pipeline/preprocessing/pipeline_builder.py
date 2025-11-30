"""End-to-end preprocessing runner for TubeSense.

Reads metadata.csv (from the YouTube scraper), extracts frames/audio,
cleans text, and writes back enriched metadata with paths ready for training.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.data_pipeline.preprocessing.cleaners.text_cleaner import clean_text  # noqa: E402
from src.data_pipeline.preprocessing.transformers.video_to_frames import extract_frames  # noqa: E402
from src.data_pipeline.preprocessing.transformers.audio_extractor import extract_audio  # noqa: E402
RAW_DIR = ROOT / "data" / "raw"
INTERIM_FRAMES = ROOT / "data" / "interim" / "frames"
INTERIM_AUDIO = ROOT / "data" / "interim" / "audio"
METADATA_CSV = ROOT / "data" / "processed" / "metadata.csv"


def process_row(row: Dict[str, str], fps: int, sample_rate: int) -> Dict[str, str]:
    video_path = Path(row["filepath"])
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frame_out_dir = INTERIM_FRAMES / video_path.stem
    audio_out_path = INTERIM_AUDIO / f"{video_path.stem}.wav"

    frame_out_dir.mkdir(parents=True, exist_ok=True)
    INTERIM_AUDIO.mkdir(parents=True, exist_ok=True)

    saved_frames = extract_frames(video_path, frame_out_dir, fps=fps, frame_size=(224, 224))
    if saved_frames == 0:
        raise RuntimeError(f"No frames extracted for {video_path}")
    first_frame = sorted(frame_out_dir.glob("*.jpg"))[0]

    extract_audio(video_path, audio_out_path, sample_rate=sample_rate)

    row["frame_path"] = str(first_frame)
    row["audio_path"] = str(audio_out_path)
    row["text"] = clean_text(row.get("description", ""))
    return row


def build_dataset(fps: int, sample_rate: int) -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {METADATA_CSV}. Run youtube_scraper.py first.")

    with METADATA_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = list(reader)
 
    updated: List[Dict[str, str]] = []
    for row in rows:
        try:
            updated.append(process_row(row, fps=fps, sample_rate=sample_rate))
            print(f"[+] Processed {row.get('title', row.get('filepath'))}")
        except Exception as exc:  # noqa: BLE001
            print(f"[!] Skipped {row.get('filepath')}: {exc}")

    if not updated:
        raise SystemExit("No rows processed.")

    fieldnames = list(updated[0].keys())
    with METADATA_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated)
    print(f"[+] Updated metadata saved to {METADATA_CSV}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TubeSense dataset from raw videos.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to sample from video")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(fps=args.fps, sample_rate=args.sample_rate)


if __name__ == "__main__":
    main()
