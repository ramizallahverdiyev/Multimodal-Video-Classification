"""Extract frames, audio, and optional ASR transcripts from raw videos by folder label.

Assumes structure: data/raw/<label>/<video>.mp4
Outputs:
  frames: data/interim/frames/<label>/<video_stem>/*.jpg
  audio:  data/interim/audio/<label>/<video_stem>.wav
  text:   data/interim/text/<label>/<video_stem>.txt  (if --asr enabled)
"""

from __future__ import annotations

import argparse
import subprocess
import shutil
import os
from pathlib import Path
from typing import List

import imageio_ffmpeg
import librosa

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "raw"
FRAMES_ROOT = ROOT / "data" / "interim" / "frames"
AUDIO_ROOT = ROOT / "data" / "interim" / "audio"
TEXT_ROOT = ROOT / "data" / "interim" / "text"


def find_videos(labels_filter: List[str] | None) -> List[tuple[str, Path]]:
    videos: List[tuple[str, Path]] = []
    for label_dir in RAW_ROOT.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if labels_filter and label not in labels_filter:
            continue
        for video in label_dir.glob("*.mp4"):
            videos.append((label, video))
    return videos


def extract_audio(video_path: Path, audio_path: Path, sample_rate: int = 16000) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
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
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_frames(video_path: Path, frames_dir: Path, fps: int = 1, size: int = 224) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    pattern = str(frames_dir / f"{video_path.stem}_%05d.jpg")
    vf = f"fps={fps},scale={size}:{size}"
    cmd = [ffmpeg_bin, "-y", "-i", str(video_path), "-vf", vf, pattern]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_asr(audio_path: Path, text_path: Path, model_name: str = "base") -> None:
    try:
        import whisper
    except ImportError:
        print("[!] Whisper not installed; skipping ASR.")
        return
    text_path.parent.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(model_name)
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    result = model.transcribe(audio, fp16=False, language="en")
    text_path.write_text(result.get("text", "").strip(), encoding="utf-8")


def process_video(
    label: str,
    video_path: Path,
    fps: int,
    sample_rate: int,
    asr: bool,
    asr_model: str,
    only_frames: bool,
    only_audio: bool,
    only_text: bool,
    skip_existing: bool,
) -> None:
    frames_dir = FRAMES_ROOT / label / video_path.stem
    audio_path = AUDIO_ROOT / label / f"{video_path.stem}.wav"
    text_path = TEXT_ROOT / label / f"{video_path.stem}.txt"

    try:
        if not only_audio and not only_text:
            if skip_existing and frames_dir.exists() and any(frames_dir.glob("*.jpg")):
                pass
            else:
                extract_frames(video_path, frames_dir, fps=fps)
        if not only_frames and not only_text:
            if skip_existing and audio_path.exists():
                pass
            else:
                extract_audio(video_path, audio_path, sample_rate=sample_rate)
        if (asr or only_text) and not only_frames:
            if audio_path.exists():
                if skip_existing and text_path.exists():
                    pass
                else:
                    run_asr(audio_path, text_path, model_name=asr_model)
        print(f"[+] Processed {video_path.name} -> label={label}")
    except Exception as exc:  # noqa: BLE001
        # Avoid encoding errors on Windows terminals
        safe_name = video_path.name.encode("ascii", "ignore").decode("ascii")
        print(f"[!] Failed {safe_name}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames/audio/text from raw videos by label folder.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to sample")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--asr", action="store_true", help="Run Whisper ASR to produce text")
    parser.add_argument("--asr-model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--labels", nargs="*", help="Optional subset of labels to process (folder names under data/raw)")
    parser.add_argument("--only-frames", action="store_true", help="Extract only frames")
    parser.add_argument("--only-audio", action="store_true", help="Extract only audio")
    parser.add_argument("--only-text", action="store_true", help="Extract only text (requires ASR)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-extract even if outputs exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = find_videos(labels_filter=args.labels)
    if not videos:
        print(f"[!] No videos found under {RAW_ROOT}")
        return
    for label, video in videos:
        process_video(
            label,
            video,
            fps=args.fps,
            sample_rate=args.sample_rate,
            asr=args.asr,
            asr_model=args.asr_model,
            only_frames=args.only_frames,
            only_audio=args.only_audio,
            only_text=args.only_text,
            skip_existing=not args.no_skip_existing,
        )
    print("Done.")


if __name__ == "__main__":
    main()
