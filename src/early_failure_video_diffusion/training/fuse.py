from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from ..l2r import LatentToRGB
from .checkpoint import atomic_torch_save, load_factorized_checkpoint
from .model import FactorizedLatentToRGB


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def verify_deploy_equivalence(
    factorized: FactorizedLatentToRGB,
    deploy: LatentToRGB,
    *,
    frames: int = 4,
    latent_height: int = 3,
    latent_width: int = 4,
    seed: int = 2026,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> float:
    """Verify that factorized and fused models produce the same RGB tensor."""

    device = next(factorized.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        1,
        16,
        frames,
        latent_height,
        latent_width,
        generator=generator,
        device=device,
        dtype=next(factorized.parameters()).dtype,
    )
    expected = factorized.eval()(latents)
    actual = deploy.eval()(latents)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return float((actual - expected).abs().max())


def export_deploy_checkpoint(
    factorized: FactorizedLatentToRGB,
    destination: str | Path,
    *,
    verify: bool = True,
) -> tuple[Path, float | None]:
    """Fuse the basis axes, strictly validate, and save a plain 29-key state dict."""

    factorized.eval()
    deploy = factorized.to_deploy()
    if len(deploy.state_dict()) != 29:
        raise RuntimeError(
            f"expected a 29-key deploy state dict, got {len(deploy.state_dict())}"
        )
    max_error = verify_deploy_equivalence(factorized, deploy) if verify else None
    cpu_state = {
        key: value.detach().float().cpu() for key, value in deploy.state_dict().items()
    }

    # This strict load is the same contract used by the main inference pipeline.
    strict_model = LatentToRGB()
    strict_model.load_state_dict(cpu_state, strict=True)
    output = atomic_torch_save(cpu_state, destination)
    LatentToRGB.from_checkpoint(output, strict=True)
    return output, max_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse a trained rank-basis L2R checkpoint for inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "The output is a plain 29-key state dict accepted by "
            "early-failure-video --l2r-checkpoint."
        ),
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Trained factorized checkpoint"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="29-key deploy checkpoint path"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Fusion verification device; CPU works without a GPU",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip numerical output comparison (not recommended for release artifacts)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint.expanduser().resolve() == args.output.expanduser().resolve():
        raise ValueError("factorized input and fused output must be different files")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    factorized, payload = load_factorized_checkpoint(args.checkpoint)
    factorized = factorized.to(device).eval()
    output, max_error = export_deploy_checkpoint(
        factorized,
        args.output,
        verify=not args.skip_verification,
    )
    print(f"Fused deploy checkpoint: {output}")
    print(f"Backend: {payload['backend']}")
    print("State-dict keys: 29 (strict load passed)")
    if max_error is not None:
        print(f"Factorized/fused maximum absolute error: {max_error:.9g}")
    print(f"SHA-256: {_sha256(output)}")
    print("Use it with: early-failure-video --l2r-checkpoint <this-file> ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
