"""Extract frames from videos at a fixed FPS."""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path, fps: int = 1, frame_size: tuple[int, int] | None = (224, 224)) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(int(native_fps // fps), 1)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    saved = 0
    for idx in tqdm(range(total_frames), desc=f"Frames from {video_path.name}"):
        success, frame = capture.read()
        if not success:
            break
        if idx % frame_interval != 0:
            continue
        if frame_size:
            frame = cv2.resize(frame, frame_size)
        frame_path = output_dir / f"{video_path.stem}_frame_{saved:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        saved += 1

    capture.release()
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("--video", required=True, type=Path, help="Input video path")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for frames")
    parser.add_argument("--fps", type=int, default=1, help="Frames to sample per second")
    parser.add_argument("--no-resize", action="store_true", help="Keep original frame size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_size = None if args.no_resize else (224, 224)
    saved = extract_frames(args.video, args.out, fps=args.fps, frame_size=frame_size)
    print(f"[+] Saved {saved} frames to {args.out}")


if __name__ == "__main__":
    main()
