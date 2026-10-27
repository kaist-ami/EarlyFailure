from __future__ import annotations

import argparse
import json
import re
from contextlib import nullcontext
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from ..model_specs import (
    DEFAULT_MODELS,
    resolve_base_model_spec,
    safe_model_reference,
)
from .data import PAPER_FRAME_INDICES, VIDEO_EXTENSIONS, load_video_frames


def _center_crop_resize(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return np.asarray(resized.crop((left, top, left + width, top + height)))


def _standardize_video(
    path: Path, *, num_frames: int, height: int, width: int
) -> np.ndarray:
    decoded = load_video_frames(path).numpy()
    if decoded.shape[0] < num_frames:
        raise ValueError(
            f"{path} has {decoded.shape[0]} frames; at least {num_frames} are required"
        )
    start = (decoded.shape[0] - num_frames) // 2
    selected = decoded[start : start + num_frames]
    return np.stack([_center_crop_resize(frame, height, width) for frame in selected])


def _write_video(path: Path, frames: np.ndarray, fps: int, crf: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path),
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        quality=None,
        output_params=["-crf", str(crf)],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def _load_vae(args: argparse.Namespace):
    common = {
        "torch_dtype": torch.float32,
        "local_files_only": args.local_files_only,
    }
    if args.cache_dir is not None:
        common["cache_dir"] = str(args.cache_dir.expanduser())
    model_id, model_revision = resolve_base_model_spec(
        args.backend,
        model_id=args.model_id,
        revision=args.model_revision,
    )
    if model_revision is not None:
        common["revision"] = model_revision
    if args.backend == "cogvideox":
        from diffusers import AutoencoderKLCogVideoX

        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_id, subfolder="vae", **common
        )
    else:
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", **common)
    if args.vae_tiling:
        vae.enable_tiling()
    else:
        vae.disable_tiling()
    return vae.eval().to(args.device), model_id, model_revision


@torch.inference_mode()
def _encode_video(
    vae,
    video: torch.Tensor,
    *,
    backend: str,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    context = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )
    with context:
        latent = vae.encode(video).latent_dist.mode()
    if backend == "cogvideox":
        latent = latent * float(vae.config.scaling_factor)
    else:
        mean = torch.tensor(
            vae.config.latents_mean, device=latent.device, dtype=latent.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        std = torch.tensor(
            vae.config.latents_std, device=latent.device, dtype=latent.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latent = (latent - mean) / std
    return latent.float().cpu()


def _frame_indices(source_frames: int, latent_frames: int) -> list[int]:
    if source_frames == 49 and latent_frames == len(PAPER_FRAME_INDICES):
        return list(PAPER_FRAME_INDICES)
    return torch.linspace(0, source_frames - 1, latent_frames).round().long().tolist()


def _collect_videos(args: argparse.Namespace) -> list[Path]:
    paths = [path.expanduser().resolve() for path in args.video]
    if args.input_dir is not None:
        directory = args.input_dir.expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"input video directory not found: {directory}")
        paths.extend(
            sorted(
                path.resolve()
                for path in directory.iterdir()
                if path.suffix.lower() in VIDEO_EXTENSIONS
            )
        )
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise ValueError("provide at least one --video or --input-dir")
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input video not found: {missing[0]}")
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create paired RGB/latent files and an L2R training manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "This command does not download a dataset. Supply videos you own or are "
            "permitted to process."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=sorted(DEFAULT_MODELS),
        required=True,
        help="VAE/latent convention to use",
    )
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=[],
        help="Input clip; repeat this option for multiple clips",
    )
    parser.add_argument(
        "--input-dir", type=Path, help="Add supported video files in this directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Prepared dataset directory"
    )
    parser.add_argument("--model-id", help="Override the Hugging Face VAE source")
    parser.add_argument(
        "--model-revision",
        help="Hugging Face revision (built-in model IDs use the tested commit)",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda", help="CUDA device for VAE encoding")
    parser.add_argument(
        "--amp",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
        help="VAE autocast precision",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=49,
        help="Contiguous center frames to keep (paper-compatible setting: 49)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Prepared RGB height (paper-compatible setting: 480)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="Prepared RGB width (paper-compatible setting: 720)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help=(
            "Playback FPS written to target.mp4; training aligns decoded frame "
            "indices and does not use timestamps"
        ),
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality for paired target clips (paper data used CRF 18)",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Reduce VAE memory use, with a possible speed tradeoff",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing prepared output"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("L2R data preparation requires an available CUDA device")
    if args.num_frames < 1 or args.height % 8 or args.width % 8:
        raise ValueError("num-frames must be positive and height/width divisible by 8")
    if args.fps < 1:
        raise ValueError("fps must be positive")
    if not 0 <= args.crf <= 51:
        raise ValueError("crf must be between 0 and 51")

    videos = _collect_videos(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; choose another output or pass --overwrite"
        )

    vae, model_id, model_revision = _load_vae(args)
    safe_model_id = safe_model_reference(model_id)
    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    records = []
    for index, source in enumerate(videos):
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", source.stem).strip("-")
        sample_dir = output_dir / f"{index:04d}_{safe_stem or 'video'}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        target_path = sample_dir / "target.mp4"
        latent_path = sample_dir / "latent.safetensors"
        if not args.overwrite and (target_path.exists() or latent_path.exists()):
            raise FileExistsError(
                f"prepared sample already exists in {sample_dir}; pass --overwrite"
            )

        print(f"[{index + 1}/{len(videos)}] Reading {source.name}")
        frames = _standardize_video(
            source,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
        )
        _write_video(target_path, frames, args.fps, args.crf)
        # Encode the same compressed frames that the training loader will see.
        # This mirrors the paper's CRF-18 video/latent pairing procedure.
        paired_frames = load_video_frames(target_path).numpy()
        if paired_frames.shape[0] != args.num_frames:
            raise RuntimeError(
                f"encoded target has {paired_frames.shape[0]} frames; expected {args.num_frames}"
            )
        video_tensor = (
            torch.from_numpy(paired_frames.copy())
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
            .div(127.5)
            .sub(1.0)
        )
        print(f"[{index + 1}/{len(videos)}] Encoding with {safe_model_id} VAE")
        latent = _encode_video(
            vae,
            video_tensor,
            backend=args.backend,
            amp_dtype=amp_dtype,
        )[0].contiguous()
        save_file({"latent": latent}, str(latent_path))
        indices = _frame_indices(paired_frames.shape[0], latent.shape[1])
        records.append(
            {
                "backend": args.backend,
                "latent": latent_path.relative_to(output_dir).as_posix(),
                "latent_key": "latent",
                "latent_layout": "cthw",
                "target": target_path.relative_to(output_dir).as_posix(),
                "target_layout": "thwc",
                "frame_indices": indices,
            }
        )
        metadata = {
            "source": source.name,
            "backend": args.backend,
            "model_id": safe_model_id,
            "model_revision": model_revision,
            "target_fps": args.fps,
            "target_crf": args.crf,
            "target_shape": list(paired_frames.shape),
            "latent_shape": list(latent.shape),
            "frame_indices": indices,
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        del video_tensor

    manifest_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    print(f"Prepared {len(records)} paired samples")
    print(f"Training manifest: {manifest_path}")
    print("Next: early-failure-init-l2r, then early-failure-train-l2r")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
