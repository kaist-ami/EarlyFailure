from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .checkpoint import load_factorized_checkpoint, save_factorized_checkpoint
from .data import L2RManifestDataset


def _learning_rate_schedule(total_steps: int, warmup_ratio: float):
    warmup_steps = round(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return multiplier


def _make_perceptual_loss(weight: float, device: torch.device, network: str):
    if weight <= 0:
        return None
    try:
        import lpips
    except ImportError as error:
        raise ImportError(
            "perceptual loss requires the training extra: "
            '`pip install -e ".[training]"`'
        ) from error
    loss_model = lpips.LPIPS(net=network).to(device).eval()
    loss_model.requires_grad_(False)
    return loss_model


def _save_training_state(
    *,
    model,
    backend: str,
    step: int,
    optimizer: AdamW,
    scheduler: LambdaLR,
    destination: Path,
    manifest: Path,
) -> Path:
    return save_factorized_checkpoint(
        model,
        destination,
        backend=backend,
        step=step,
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        extra={"manifest": str(manifest), "stage": "trained"},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train factorized L2R weights from a prepared paired manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "No training data is bundled. --manifest must describe your own paired "
            "RGB/latent samples."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Initialized or previously trained factorized checkpoint (new optimizer)",
    )
    source.add_argument(
        "--resume",
        type=Path,
        help="Periodic training checkpoint including optimizer/scheduler state",
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Paired JSONL training manifest"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Logs and checkpoints directory"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200_000,
        help="Target global optimization step (not additional steps when resuming)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--perceptual-weight", type=float, default=0.0)
    parser.add_argument("--lpips-network", choices=("alex", "vgg"), default="vgg")
    parser.add_argument("--save-every", type=int, default=40_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp",
        choices=("none", "bf16"),
        default="none",
        help="The paper training used float32; bf16 is available to reduce memory",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    if args.save_every < 1 or args.log_every < 1:
        raise ValueError("save-every and log-every must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.perceptual_weight < 0:
        raise ValueError("perceptual-weight cannot be negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if args.gradient_clip <= 0:
        raise ValueError("gradient-clip must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    checkpoint_path = args.resume or args.init_checkpoint
    model, payload = load_factorized_checkpoint(checkpoint_path)
    backend = str(payload["backend"])
    start_step = int(payload.get("step", 0)) if args.resume else 0
    if start_step >= args.steps:
        raise ValueError(
            f"checkpoint is already at step {start_step}, but --steps is {args.steps}"
        )
    model = model.to(device).train()

    manifest = args.manifest.expanduser().resolve()
    dataset = L2RManifestDataset(manifest)
    if len(dataset.declared_backends) > 1:
        raise ValueError(
            f"manifest mixes backend labels: {sorted(dataset.declared_backends)}"
        )
    if dataset.declared_backends and dataset.declared_backends != {backend}:
        declared = next(iter(dataset.declared_backends))
        raise ValueError(
            f"manifest backend is {declared!r}, but checkpoint backend is {backend!r}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    scheduler = LambdaLR(
        optimizer, _learning_rate_schedule(args.steps, args.warmup_ratio)
    )
    if args.resume:
        if "optimizer" not in payload or "scheduler" not in payload:
            raise ValueError("resume checkpoint has no optimizer/scheduler state")
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])

    perceptual_loss = _make_perceptual_loss(
        args.perceptual_weight, device, args.lpips_network
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.jsonl"
    autocast_enabled = args.amp == "bf16"
    data_iterator = iter(loader)
    running_loss = 0.0
    running_count = 0

    print(f"Backend: {backend}")
    print(f"Paired samples: {len(dataset)}")
    print(f"Training steps: {start_step} -> {args.steps}")
    print(f"Output directory: {output_dir}")

    for step in range(start_step + 1, args.steps + 1):
        try:
            latents, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            latents, target = next(data_iterator)
        latents = latents.to(device, non_blocking=True, dtype=torch.float32)
        target = target.to(device, non_blocking=True, dtype=torch.float32)
        target = target.permute(0, 2, 1, 3, 4).contiguous()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            prediction = model(latents)
            pixel_loss = F.l1_loss(prediction, target)
            loss = pixel_loss
        perceptual_value = torch.zeros((), device=device)
        if perceptual_loss is not None:
            batch, _, frames, height, width = prediction.shape
            prediction_2d = prediction.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, 3, height, width
            )
            target_2d = target.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, 3, height, width
            )
            perceptual_value = perceptual_loss(
                prediction_2d.float(), target_2d.float()
            ).mean()
            loss = loss + args.perceptual_weight * perceptual_value

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        scheduler.step()
        running_loss += float(loss.detach())
        running_count += 1

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            metrics = {
                "step": step,
                "loss": running_loss / running_count,
                "pixel_l1": float(pixel_loss.detach()),
                "perceptual": float(perceptual_value.detach()),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(
                f"step {step}/{args.steps} | loss {metrics['loss']:.6f} | "
                f"lr {metrics['learning_rate']:.3e}"
            )
            running_loss = 0.0
            running_count = 0

        if step % args.save_every == 0 and step < args.steps:
            checkpoint = _save_training_state(
                model=model,
                backend=backend,
                step=step,
                optimizer=optimizer,
                scheduler=scheduler,
                destination=output_dir / f"factorized_step_{step:07d}.pt",
                manifest=manifest,
            )
            print(f"Saved resumable checkpoint: {checkpoint}")

    final_checkpoint = _save_training_state(
        model=model,
        backend=backend,
        step=args.steps,
        optimizer=optimizer,
        scheduler=scheduler,
        destination=output_dir / "factorized_final.pt",
        manifest=manifest,
    )
    print(f"Training complete: {final_checkpoint}")
    print("Next: early-failure-fuse-l2r --checkpoint <factorized_final.pt> ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
