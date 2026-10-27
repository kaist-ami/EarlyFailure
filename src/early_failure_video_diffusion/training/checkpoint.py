from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from .model import FactorizedL2RConfig, FactorizedLatentToRGB


FACTORIZED_CHECKPOINT_FORMAT = "early-failure-factorized-l2r-v1"


def atomic_torch_save(payload: Any, destination: str | Path) -> Path:
    """Write a PyTorch file atomically so interrupted saves do not look complete."""

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def make_factorized_payload(
    model: FactorizedLatentToRGB,
    *,
    backend: str,
    step: int,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    if backend not in {"cogvideox", "wan"}:
        raise ValueError("backend must be 'cogvideox' or 'wan'")
    payload: dict[str, Any] = {
        "format": FACTORIZED_CHECKPOINT_FORMAT,
        "backend": backend,
        "step": int(step),
        "model_config": model.config.to_dict(),
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    if scheduler_state is not None:
        payload["scheduler"] = scheduler_state
    if extra:
        payload["extra"] = dict(extra)
    return payload


def save_factorized_checkpoint(
    model: FactorizedLatentToRGB,
    destination: str | Path,
    *,
    backend: str,
    step: int,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    extra: dict | None = None,
) -> Path:
    payload = make_factorized_payload(
        model,
        backend=backend,
        step=step,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        extra=extra,
    )
    return atomic_torch_save(payload, destination)


def load_factorized_checkpoint(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[FactorizedLatentToRGB, dict[str, Any]]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"factorized L2R checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != FACTORIZED_CHECKPOINT_FORMAT
    ):
        raise ValueError(
            f"{checkpoint} is not a {FACTORIZED_CHECKPOINT_FORMAT} checkpoint"
        )
    config_values = payload.get("model_config")
    state_dict = payload.get("state_dict")
    if not isinstance(config_values, dict) or not isinstance(state_dict, dict):
        raise ValueError("factorized checkpoint is missing model_config or state_dict")

    model = FactorizedLatentToRGB(FactorizedL2RConfig.from_dict(config_values))
    model.load_state_dict(state_dict, strict=True)
    return model, payload
