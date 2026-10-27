from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from .stopping import AdaptiveBalancedStopper, StopDecision
from .types import GenerationResult


class VideoTextScorer(Protocol):
    def score(self, prompt: str, rgb: torch.Tensor) -> torch.Tensor:
        """Return one score per RGB candidate in ``[B, 3, T, H, W]``."""


@dataclass
class _Snapshot:
    step: int
    score: float
    x0: torch.Tensor


class InspectionHistory:
    """Retain only last/best latent previews instead of all large RGB videos."""

    def __init__(self) -> None:
        self.best: _Snapshot | None = None
        self.last: _Snapshot | None = None

    def record(self, step: int, score: float, x0: torch.Tensor) -> None:
        snapshot = _Snapshot(int(step), float(score), x0.detach().clone())
        self.last = snapshot
        if self.best is None or score > self.best.score:
            self.best = snapshot

    def select(self, decision: StopDecision) -> _Snapshot:
        if self.best is not None and decision.selected_step == self.best.step:
            return self.best
        if self.last is not None and decision.selected_step == self.last.step:
            return self.last
        raise RuntimeError(
            "selected inspection preview is not available in retained history"
        )


class InspectionPipelineMixin:
    """Shared L2R/scorer plumbing for the model-specific denoising loops."""

    _l2r: torch.nn.Module | None = None
    _video_text_scorer: VideoTextScorer | None = None

    def set_inspector(
        self,
        l2r: torch.nn.Module,
        scorer: VideoTextScorer,
    ) -> "InspectionPipelineMixin":
        l2r.eval()
        l2r.requires_grad_(False)
        self._l2r = l2r
        self._video_text_scorer = scorer
        return self

    def _require_inspector(self) -> None:
        if self._l2r is None or self._video_text_scorer is None:
            raise RuntimeError(
                "call set_inspector(l2r, scorer) before enabling inspection"
            )

    @torch.inference_mode()
    def _inspect_candidates(
        self,
        x0_candidates: torch.Tensor,
        prompt: str,
        *,
        layout: Literal["btchw", "bcthw"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_inspector()
        assert self._l2r is not None
        assert self._video_text_scorer is not None

        if layout == "btchw":
            l2r_input = x0_candidates.permute(0, 2, 1, 3, 4).contiguous()
        elif layout == "bcthw":
            l2r_input = x0_candidates
        else:
            raise ValueError(f"unsupported latent layout: {layout}")

        try:
            l2r_device = next(self._l2r.parameters()).device
        except StopIteration:
            l2r_device = next(self._l2r.buffers()).device
        l2r_input = l2r_input.to(device=l2r_device, dtype=torch.float32)
        rgb = self._l2r(l2r_input).clamp(-1.0, 1.0)
        scores = self._video_text_scorer.score(prompt, rgb)
        scores = torch.as_tensor(
            scores, device=rgb.device, dtype=torch.float32
        ).reshape(-1)
        if scores.numel() != rgb.shape[0]:
            raise ValueError(
                f"scorer returned {scores.numel()} scores for {rgb.shape[0]} candidates"
            )
        return rgb, scores

    @torch.inference_mode()
    def _preview_single_x0(
        self,
        x0: torch.Tensor,
        *,
        layout: Literal["btchw", "bcthw"],
    ) -> torch.Tensor:
        self._require_inspector()
        assert self._l2r is not None
        candidate = x0.unsqueeze(0)
        if layout == "btchw":
            candidate = candidate.permute(0, 2, 1, 3, 4).contiguous()
        try:
            device = next(self._l2r.parameters()).device
        except StopIteration:
            device = next(self._l2r.buffers()).device
        return self._l2r(candidate.to(device=device, dtype=torch.float32)).clamp(
            -1.0, 1.0
        )

    def _early_result(
        self,
        *,
        status: Literal["failed", "probe"],
        decision: StopDecision,
        snapshot: _Snapshot,
        layout: Literal["btchw", "bcthw"],
        steps_executed: int,
        total_steps: int,
    ) -> GenerationResult:
        preview = self._preview_single_x0(snapshot.x0, layout=layout).detach().cpu()
        anchor = snapshot.x0.detach().float().cpu()
        return GenerationResult(
            status=status,
            preview=preview,
            anchor_x0=anchor,
            score=float(decision.score),
            stop_step=int(decision.stop_step),
            selected_step=int(decision.selected_step),
            stop_reason=decision.reason,
            steps_executed=int(steps_executed),
            total_steps=int(total_steps),
        )


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int,
    device: str | torch.device,
    timesteps: list[int] | None = None,
) -> tuple[torch.Tensor, int]:
    if timesteps is None:
        scheduler.set_timesteps(num_inference_steps, device=device)
        return scheduler.timesteps, num_inference_steps

    if "timesteps" not in inspect.signature(scheduler.set_timesteps).parameters:
        raise ValueError(
            f"{type(scheduler).__name__} does not support custom timesteps"
        )
    scheduler.set_timesteps(timesteps=timesteps, device=device)
    return scheduler.timesteps, len(scheduler.timesteps)


def cogvideox_predicted_x0(
    scheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    timestep_index = int(timestep.detach().cpu().item())
    alpha = scheduler.alphas_cumprod[timestep_index].to(
        device=sample.device, dtype=sample.dtype
    )
    beta = 1.0 - alpha
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return (sample - beta.sqrt() * model_output) / alpha.sqrt()
    if prediction_type == "sample":
        return model_output
    if prediction_type == "v_prediction":
        return alpha.sqrt() * sample - beta.sqrt() * model_output
    raise ValueError(f"unsupported CogVideoX prediction type: {prediction_type}")


def _sigma_for_timestep(
    scheduler, timestep: torch.Tensor, sample: torch.Tensor
) -> torch.Tensor:
    schedule = scheduler.timesteps.to(sample.device)
    index = scheduler.index_for_timestep(timestep.to(sample.device), schedule)
    sigma = scheduler.sigmas[index].to(device=sample.device, dtype=sample.dtype)
    while sigma.ndim < sample.ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def flow_predicted_x0(
    scheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    if scheduler.config.prediction_type != "flow_prediction":
        raise ValueError(
            "Wan inspection requires scheduler prediction_type='flow_prediction'"
        )
    sigma = _sigma_for_timestep(scheduler, timestep, sample)
    return sample - sigma * model_output


def add_flow_noise(
    scheduler,
    original: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    sigma = _sigma_for_timestep(scheduler, timestep, noise)
    if scheduler.config.use_flow_sigmas:
        alpha = 1.0 - sigma
        sigma_t = sigma
    else:
        alpha = 1.0 / torch.sqrt(sigma.square() + 1.0)
        sigma_t = sigma * alpha
    return alpha * original + sigma_t * noise


def progressive_temporal_mask(
    frames: int,
    step: int,
    max_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if frames < 1 or max_steps < 1:
        raise ValueError("frames and max_steps must be positive")
    ratio = 1.0 - step / max_steps
    active = max(1, min(frames, int(ratio * (frames - 1)) + 1))
    weights = torch.zeros(frames, device=device, dtype=dtype)
    if active == 1:
        weights[0] = 1.0
    else:
        positions = torch.linspace(0.0, 1.0, active, device=device, dtype=dtype)
        weights[:active] = 0.5 * (1.0 + torch.cos(math.pi * positions))
    return weights


def first_video(video):
    """Strip Diffusers' leading batch dimension from postprocessed video."""

    return video[0]


def choose_candidate_scales(probe_only: bool, guidance_scale: float) -> list[float]:
    return [1.0, 3.0, 6.0] if probe_only else [float(guidance_scale)]


def inspect_step(
    *,
    stopper: AdaptiveBalancedStopper,
    history: InspectionHistory,
    step: int,
    scores: torch.Tensor,
    x0_candidates: torch.Tensor,
) -> tuple[int, StopDecision | None]:
    candidate_index = int(torch.argmax(scores).item())
    score = float(scores[candidate_index].item())
    history.record(step, score, x0_candidates[candidate_index])
    return candidate_index, stopper.update(step, score)
