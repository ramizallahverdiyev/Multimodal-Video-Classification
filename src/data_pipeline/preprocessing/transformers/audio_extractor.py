"""Extract mono WAV audio from a video using ffmpeg."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import shutil

import imageio_ffmpeg


def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 16000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract WAV audio from a video file.")
    parser.add_argument("--video", required=True, type=Path, help="Input video path")
    parser.add_argument("--out", required=True, type=Path, help="Output WAV path")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_audio(args.video, args.out, sample_rate=args.sample_rate)
    print(f"[+] Saved audio to {args.out}")


if __name__ == "__main__":
    main()
