"""Generate the two tiny, redistributable L2R data-preparation examples.

The clips are made entirely from NumPy shapes and color fields. They contain no
third-party footage and are only intended to check the preparation command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "videos"


def _moving_shapes(frame_count: int, height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    frames: list[np.ndarray] = []
    for frame_index in range(frame_count):
        phase = frame_index / max(frame_count - 1, 1)
        background = np.empty((height, width, 3), dtype=np.float32)
        background[..., 0] = 30 + 45 * xx / max(width - 1, 1)
        background[..., 1] = 55 + 55 * yy / max(height - 1, 1)
        background[..., 2] = 105 + 30 * np.sin(2 * np.pi * phase)

        circle_x = width * (0.15 + 0.70 * phase)
        circle_y = height * (0.5 + 0.2 * np.sin(2 * np.pi * phase))
        circle = (xx - circle_x) ** 2 + (yy - circle_y) ** 2 <= (height * 0.09) ** 2
        background[circle] = (244, 188, 66)

        square_size = max(12, height // 8)
        square_x = round(width * (0.78 - 0.55 * phase))
        square_y = round(height * (0.18 + 0.50 * phase))
        x0 = max(0, square_x - square_size // 2)
        x1 = min(width, x0 + square_size)
        y0 = max(0, square_y - square_size // 2)
        y1 = min(height, y0 + square_size)
        background[y0:y1, x0:x1] = (87, 214, 170)
        frames.append(np.clip(background, 0, 255).astype(np.uint8))
    return np.stack(frames)


def _color_waves(frame_count: int, height: int, width: int) -> np.ndarray:
    y = np.linspace(0, 2 * np.pi, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 2 * np.pi, width, dtype=np.float32)[None, :]
    frames: list[np.ndarray] = []
    for frame_index in range(frame_count):
        phase = 2 * np.pi * frame_index / frame_count
        red = 127.5 + 100 * np.sin(x + phase)
        green = 127.5 + 100 * np.sin(y * 1.5 - phase)
        blue = 127.5 + 100 * np.sin(x * 0.6 + y * 0.8 + phase * 0.7)
        frame = np.stack(
            (
                np.broadcast_to(red, (height, width)),
                np.broadcast_to(green, (height, width)),
                blue,
            ),
            axis=-1,
        )
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return np.stack(frames)


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    with imageio.get_writer(
        str(path),
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        quality=None,
        output_params=["-crf", "18"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(args.num_frames, args.height, args.width, args.fps) < 1:
        raise ValueError("frames, dimensions, and fps must be positive")
    if args.height % 2 or args.width % 2:
        raise ValueError("H.264 sample dimensions must be even")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = {
        "moving_shapes.mp4": _moving_shapes(args.num_frames, args.height, args.width),
        "color_waves.mp4": _color_waves(args.num_frames, args.height, args.width),
    }
    for filename, frames in clips.items():
        path = output_dir / filename
        _write_video(path, frames, args.fps)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
