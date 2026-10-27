from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch

from .assets import VICLIP_REVISION, VICLIP_SHA256
from .l2r import LatentToRGB
from .model_specs import (
    DEFAULT_MODELS,
    resolve_base_model_spec,
    safe_model_reference,
)
from .prompt_refiner import (
    DEFAULT_QWEN_MODEL,
    OpenAIPromptRefiner,
    QwenPromptRefiner,
)
from .runner import HierarchicalInterventionRunner, RunnerConfig
from .scoring import ViCLIPScorer
from .test_sets import TEST_SET_OUTPUT_DIRECTORIES, TEST_SET_SEED, inference_examples


def _safe_model_reference(value: str) -> str:
    return safe_model_reference(value)


def _base_model_spec(args: argparse.Namespace) -> tuple[str, str | None]:
    return resolve_base_model_spec(
        args.backend,
        model_id=args.model_id,
        revision=args.model_revision,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Early failure detection and hierarchical intervention for video diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Bundled L2R weights are used automatically. ViCLIP and the selected "
            "Hugging Face base model are downloaded when they are not cached."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=sorted(DEFAULT_MODELS),
        required=True,
        help="Base video model and matching bundled L2R checkpoint",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt", help="Text-to-video prompt")
    input_group.add_argument(
        "--test-set",
        action="store_true",
        help="Run the two released examples for the selected backend",
    )
    parser.add_argument(
        "--l2r-checkpoint",
        type=Path,
        help="Override the bundled checkpoint for the selected backend",
    )
    parser.add_argument(
        "--viclip-checkpoint",
        type=Path,
        help="Manual ViCLIP checkpoint path (downloaded automatically when omitted)",
    )
    parser.add_argument(
        "--viclip-vocab",
        type=Path,
        help="Manual ViCLIP BPE vocabulary path (downloaded automatically when omitted)",
    )
    parser.add_argument(
        "--viclip-cache-dir",
        type=Path,
        help="ViCLIP asset directory (default: ~/.cache/vbench/ViCLIP)",
    )
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", type=Path, help="Single output video path")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help="Root directory for paired test-set outputs",
    )
    parser.add_argument(
        "--model-id", help="Override the backend's Hugging Face model ID"
    )
    parser.add_argument(
        "--model-revision",
        help="Hugging Face revision (built-in model IDs use the tested commit)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Optional Hugging Face cache directory"
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require all Hugging Face and ViCLIP assets to be cached/manual",
    )
    parser.add_argument(
        "--prompt-refiner",
        choices=("openai", "qwen"),
        default="openai",
        help="Multimodal model used only when prompt refinement is needed",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-5-mini",
        help="Vision-capable Responses API model for failure prompt refinement",
    )
    parser.add_argument(
        "--qwen-model",
        default=DEFAULT_QWEN_MODEL,
        help="Hugging Face model loaded only during the refinement handoff",
    )
    parser.add_argument(
        "--qwen-revision",
        help="Qwen Hugging Face revision (the built-in model uses the tested commit)",
    )
    parser.add_argument(
        "--qwen-python",
        type=Path,
        default=Path(os.environ.get("QWEN_PYTHON", ".venv-qwen/bin/python")),
        help="Python executable in the isolated Qwen Transformers environment",
    )
    parser.add_argument(
        "--refiner-max-images",
        type=int,
        default=12,
        help="Uniformly sampled failed-preview frames sent to the refiner",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Defaults to enabled for CogVideoX and for either backend when the "
            "Qwen refiner is selected"
        ),
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Generation seed; released test-set mode is fixed to 42",
    )
    parser.add_argument("--fps", type=int, default=8)
    return parser


def _validate_output_mode(args: argparse.Namespace) -> None:
    if args.test_set and args.output_dir is None:
        raise ValueError("--test-set requires --output-dir")
    if args.test_set and args.seed != TEST_SET_SEED:
        raise ValueError(f"--test-set uses the fixed released seed {TEST_SET_SEED}")
    if not args.test_set and args.output is None:
        raise ValueError("--prompt requires --output")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _export_video(frames: Any, path: Path, fps: int) -> None:
    from diffusers.utils import export_to_video

    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(path), fps=fps)


def _runtime_metadata(
    args: argparse.Namespace,
    *,
    l2r_source: str,
    viclip_source: str,
    refiner_model: str,
    refiner_revision: str | None,
) -> dict[str, Any]:
    model_id, model_revision = _base_model_spec(args)
    return {
        "backend": args.backend,
        "model_id": _safe_model_reference(model_id),
        "model_revision": model_revision,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "fps": args.fps,
        "l2r_checkpoint": l2r_source,
        "viclip_checkpoint": viclip_source,
        "viclip_revision": (
            VICLIP_REVISION if viclip_source.startswith("official:") else None
        ),
        "viclip_sha256": (
            VICLIP_SHA256 if viclip_source.startswith("official:") else None
        ),
        "prompt_refiner": args.prompt_refiner,
        "prompt_refiner_model": _safe_model_reference(refiner_model),
        "prompt_refiner_revision": refiner_revision,
        "prompt_refiner_runtime": (
            "openai_responses_api"
            if args.prompt_refiner == "openai"
            else "one_shot_gpu_handoff"
        ),
    }


def _run_single(
    args: argparse.Namespace,
    runner: HierarchicalInterventionRunner,
    runtime_metadata: dict[str, Any],
) -> None:
    assert args.prompt is not None
    assert args.output is not None
    result = runner.run(args.prompt)
    output_path = args.output.expanduser()
    _export_video(result.frames, output_path, args.fps)

    metadata_path = output_path.with_suffix(".json")
    metadata = result.metadata()
    metadata.update(runtime_metadata)
    _write_json(metadata_path, metadata)
    print(f"video: {output_path}")
    print(f"metadata: {metadata_path}")
    print(f"decision_path: {result.decision_path}")


def _run_test_set(
    args: argparse.Namespace,
    runner: HierarchicalInterventionRunner,
    runtime_metadata: dict[str, Any],
) -> None:
    assert args.output_dir is not None
    backend_directory = TEST_SET_OUTPUT_DIRECTORIES[args.backend]
    output_root = args.output_dir.expanduser() / backend_directory / args.prompt_refiner
    manifest_examples: list[dict[str, Any]] = []

    for example in inference_examples(args.backend):
        example_directory = output_root / example.slug
        initial_path = example_directory / "initial.mp4"
        output_path = example_directory / "output.mp4"
        print(f"test example: {example.slug}")
        print(f"prompt: {example.prompt}")

        baseline = runner.generate_baseline(example.prompt)
        _export_video(baseline.frames, initial_path, args.fps)
        initial_metadata = baseline.trace("baseline")
        initial_metadata.update(
            {
                "generation_mode": "baseline",
                "prompt": example.prompt,
                "test_example": example.slug,
                **runtime_metadata,
            }
        )
        _write_json(example_directory / "initial.json", initial_metadata)

        result = runner.run(example.prompt)
        _export_video(result.frames, output_path, args.fps)
        output_metadata = result.metadata()
        output_metadata.update(
            {
                "generation_mode": "intervention",
                "test_example": example.slug,
                "baseline_video": "initial.mp4",
                **runtime_metadata,
            }
        )
        _write_json(example_directory / "output.json", output_metadata)

        manifest_examples.append(
            {
                "id": example.slug,
                "prompt": example.prompt,
                "initial_video": f"{example.slug}/initial.mp4",
                "output_video": f"{example.slug}/output.mp4",
                "decision_path": result.decision_path,
            }
        )
        print(f"initial video: {initial_path}")
        print(f"output video: {output_path}")
        print(f"decision_path: {result.decision_path}")

    manifest = {
        **runtime_metadata,
        "generation_mode": "paired_test_set",
        "examples": manifest_examples,
    }
    _write_json(output_root / "manifest.json", manifest)
    print(f"test-set manifest: {output_root / 'manifest.json'}")


def _load_pipeline(args: argparse.Namespace, dtype: torch.dtype):
    model_id, model_revision = _base_model_spec(args)
    pretrained_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    if model_revision is not None:
        pretrained_kwargs["revision"] = model_revision
    if args.cache_dir is not None:
        pretrained_kwargs["cache_dir"] = str(args.cache_dir.expanduser())

    if args.backend == "cogvideox":
        from .pipelines import EarlyFailureCogVideoXPipeline

        pipeline = EarlyFailureCogVideoXPipeline.from_pretrained(
            model_id, **pretrained_kwargs
        )
    else:
        from diffusers import AutoencoderKLWan
        from diffusers.schedulers import UniPCMultistepScheduler

        from .pipelines import EarlyFailureWanPipeline

        vae_kwargs = dict(pretrained_kwargs)
        vae_kwargs["torch_dtype"] = torch.float32
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", **vae_kwargs)
        pipeline = EarlyFailureWanPipeline.from_pretrained(
            model_id, vae=vae, **pretrained_kwargs
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=3.0,
        )

    if hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    use_offload = (
        args.cpu_offload
        if args.cpu_offload is not None
        else args.backend == "cogvideox" or args.prompt_refiner == "qwen"
    )
    if use_offload:
        pipeline.enable_model_cpu_offload(device=args.device)
    else:
        pipeline.to(args.device)
    return pipeline


def _load_prompt_refiner(
    args: argparse.Namespace,
    *,
    before_qwen=None,
    after_qwen=None,
):
    if args.prompt_refiner == "openai":
        return OpenAIPromptRefiner(
            model=args.openai_model,
            max_images=args.refiner_max_images,
        )
    return QwenPromptRefiner(
        model=args.qwen_model,
        revision=args.qwen_revision,
        python_executable=args.qwen_python,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        max_images=args.refiner_max_images,
        seed=args.seed,
        before_inference=before_qwen,
        after_inference=after_qwen,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_output_mode(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "the released generation and ViCLIP inspection path requires CUDA"
        )
    if args.num_frames < 1 or args.num_inference_steps < 1:
        raise ValueError("num-frames and num-inference-steps must be positive")
    if args.height < 1 or args.width < 1 or args.height % 8 or args.width % 8:
        raise ValueError("height and width must be positive and divisible by 8")
    if args.fps < 1:
        raise ValueError("fps must be positive")
    if args.refiner_max_images < 1:
        raise ValueError("refiner-max-images must be positive")
    if args.prompt_refiner == "qwen" and args.cpu_offload is False:
        raise ValueError(
            "the one-GPU Qwen handoff requires CPU offload; remove --no-cpu-offload"
        )

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.l2r_checkpoint is None:
        l2r = LatentToRGB.from_pretrained(args.backend)
        l2r_source = f"bundled:{args.backend}"
    else:
        l2r = LatentToRGB.from_checkpoint(args.l2r_checkpoint)
        l2r_source = "custom"
    l2r = l2r.eval().to(device)

    scorer = ViCLIPScorer(
        args.viclip_checkpoint,
        vocab_path=args.viclip_vocab,
        cache_dir=args.viclip_cache_dir,
        local_files_only=args.local_files_only,
        device=device,
    )
    print(f"L2R checkpoint: {l2r_source}")
    viclip_source = (
        "official:ViClip-InternVid-10M-FLT.pth"
        if args.viclip_checkpoint is None
        else "custom"
    )
    print(f"ViCLIP checkpoint: {viclip_source}")
    pipeline = _load_pipeline(args, dtype)
    pipeline.set_inspector(l2r, scorer)

    def before_qwen() -> None:
        print("Qwen handoff: parking diffusion, L2R, and ViCLIP on CPU...")
        pipeline.maybe_free_model_hooks()
        l2r.to("cpu")
        scorer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def after_qwen() -> None:
        # The short-lived worker has exited, so its CUDA context and weights are
        # gone before the inspector returns to the GPU. Diffusers stays behind
        # CPU-offload hooks and moves each component back only when it is used.
        torch.cuda.empty_cache()
        l2r.to(device)
        scorer.to(device)
        print("Qwen handoff: Qwen released; video pipeline is ready to resume.")

    refiner = _load_prompt_refiner(
        args,
        before_qwen=before_qwen if args.prompt_refiner == "qwen" else None,
        after_qwen=after_qwen if args.prompt_refiner == "qwen" else None,
    )
    refiner_model = (
        args.openai_model if args.prompt_refiner == "openai" else args.qwen_model
    )
    refiner_revision = getattr(refiner, "revision", None)
    print(
        f"Prompt refiner: {args.prompt_refiner}:"
        f"{_safe_model_reference(refiner_model)}"
    )
    config = RunnerConfig(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    def make_generator() -> torch.Generator:
        return torch.Generator(device=device).manual_seed(args.seed)

    runner = HierarchicalInterventionRunner(
        pipeline,
        refiner,
        config=config,
        generator_factory=make_generator,
    )
    runtime_metadata = _runtime_metadata(
        args,
        l2r_source=l2r_source,
        viclip_source=viclip_source,
        refiner_model=refiner_model,
        refiner_revision=refiner_revision,
    )
    if args.test_set:
        _run_test_set(args, runner, runtime_metadata)
    else:
        _run_single(args, runner, runtime_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
