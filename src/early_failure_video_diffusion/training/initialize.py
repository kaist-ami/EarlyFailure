from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import save_factorized_checkpoint
from .model import FactorizedL2RConfig, FactorizedLatentToRGB


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize the train-time factorized L2R basis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Next, pass the output to early-failure-train-l2r --init-checkpoint.",
    )
    parser.add_argument(
        "--backend",
        choices=("cogvideox", "wan"),
        required=True,
        help="Latent convention recorded in the checkpoint",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Initialized checkpoint path"
    )
    parser.add_argument(
        "--basis-rank", type=int, default=4096, help="Rank of each train-time basis"
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        default=0.02,
        help="Normal initialization standard deviation",
    )
    parser.add_argument(
        "--no-tanh",
        action="store_true",
        help="Disable tanh parameterization (the paper checkpoints use tanh)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Initialization seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    config = FactorizedL2RConfig(
        basis_rank=args.basis_rank,
        init_scale=args.init_scale,
        use_tanh=not args.no_tanh,
    )
    model = FactorizedLatentToRGB(config)
    output = save_factorized_checkpoint(
        model,
        args.output,
        backend=args.backend,
        step=0,
        extra={"seed": args.seed, "stage": "initialized"},
    )
    trainable = sum(parameter.numel() for parameter in model.parameters())
    print(f"Initialized factorized L2R: {output}")
    print(f"Backend: {args.backend}")
    print(f"Basis rank: {config.basis_rank}")
    print(f"Trainable parameters: {trainable:,}")
    print("Next: early-failure-train-l2r --init-checkpoint ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
