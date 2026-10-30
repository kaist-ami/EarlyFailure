from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from diffusers.schedulers import CogVideoXDDIMScheduler, UniPCMultistepScheduler

from early_failure_video_diffusion.pipelines import (
    EarlyFailureCogVideoXPipeline,
    EarlyFailureWanPipeline,
)
from early_failure_video_diffusion.stopping import AdaptiveBalancedStopper


class DummyTransformer(nn.Module):
    def __init__(self, backend: str) -> None:
        super().__init__()
        self.scalar = nn.Parameter(torch.zeros(()))
        if backend == "cogvideox":
            self.config = SimpleNamespace(
                sample_height=8,
                sample_width=8,
                patch_size_t=None,
                in_channels=16,
                use_rotary_positional_embeddings=False,
            )
        else:
            self.config = SimpleNamespace(patch_size=(1, 1, 1), in_channels=16)

    @property
    def dtype(self) -> torch.dtype:
        return self.scalar.dtype

    def cache_context(self, *_args):
        return nullcontext()

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        return (torch.zeros_like(hidden_states),)


class DummyVAE(nn.Module):
    def __init__(self, backend: str) -> None:
        super().__init__()
        self.scalar = nn.Parameter(torch.zeros(()))
        if backend == "cogvideox":
            self.config = SimpleNamespace(
                block_out_channels=[1],
                temporal_compression_ratio=1,
                scaling_factor=1.0,
            )
        else:
            self.config = SimpleNamespace(
                scale_factor_temporal=4,
                scale_factor_spatial=1,
                z_dim=16,
                latents_mean=[0.0] * 16,
                latents_std=[1.0] * 16,
            )

    @property
    def dtype(self) -> torch.dtype:
        return self.scalar.dtype


class DummyL2R(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scalar = nn.Parameter(torch.zeros(()))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            latents.shape[0],
            3,
            latents.shape[2],
            latents.shape[3],
            latents.shape[4],
            device=latents.device,
        )


class DummyScorer:
    def score(self, _prompt: str, rgb: torch.Tensor) -> torch.Tensor:
        return torch.full((rgb.shape[0],), 0.1, device=rgb.device)


class SmokeCogVideoXPipeline(EarlyFailureCogVideoXPipeline):
    @property
    def _execution_device(self) -> torch.device:
        return torch.device("cpu")

    def encode_prompt(self, *_args, **_kwargs):
        embeddings = torch.zeros(1, 2, 4)
        return embeddings, embeddings.clone()


class SmokeWanPipeline(EarlyFailureWanPipeline):
    @property
    def _execution_device(self) -> torch.device:
        return torch.device("cpu")

    def encode_prompt(self, *_args, **_kwargs):
        embeddings = torch.zeros(1, 2, 4)
        return embeddings, embeddings.clone()


def immediate_stopper() -> AdaptiveBalancedStopper:
    return AdaptiveBalancedStopper(min_step=0, max_step=0)


def test_cogvideox_probe_and_anchor_shapes() -> None:
    pipeline = SmokeCogVideoXPipeline(
        None,
        None,
        DummyVAE("cogvideox"),
        DummyTransformer("cogvideox"),
        CogVideoXDDIMScheduler(num_train_timesteps=20, clip_sample=False),
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.set_inspector(DummyL2R(), DummyScorer())

    probe = pipeline.generate_with_inspection(
        "prompt",
        stopper=immediate_stopper(),
        probe_only=True,
        num_frames=1,
        num_inference_steps=2,
        height=8,
        width=8,
        output_type="latent",
    )
    assert probe.anchor_x0 is not None
    assert probe.anchor_x0.shape == (1, 16, 8, 8)
    assert probe.stop_step == 0
    assert probe.selected_step == 0
    assert probe.steps_executed == 1

    output = pipeline.generate_with_inspection(
        "prompt",
        stopper=None,
        inspect_generation=False,
        anchor_x0=probe.anchor_x0,
        num_frames=5,
        num_inference_steps=2,
        height=8,
        width=8,
        output_type="latent",
    )
    assert output.frames.shape == (1, 5, 16, 8, 8)

    paired_initial = pipeline.generate_with_inspection(
        "prompt",
        stopper=immediate_stopper(),
        failure_threshold=0.22,
        continue_on_failure=True,
        num_frames=5,
        num_inference_steps=2,
        height=8,
        width=8,
        output_type="latent",
    )
    assert paired_initial.failed
    assert paired_initial.frames.shape == (1, 5, 16, 8, 8)
    assert paired_initial.preview is not None
    assert paired_initial.stop_step == 0
    assert paired_initial.steps_executed == 1
    assert paired_initial.total_steps == 2


def test_wan_probe_and_anchor_shapes() -> None:
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=20,
        flow_shift=3.0,
    )
    pipeline = SmokeWanPipeline(
        None,
        None,
        DummyVAE("wan"),
        scheduler,
        DummyTransformer("wan"),
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.set_inspector(DummyL2R(), DummyScorer())

    probe = pipeline.generate_with_inspection(
        "prompt",
        stopper=immediate_stopper(),
        probe_only=True,
        num_frames=1,
        num_inference_steps=2,
        height=16,
        width=16,
        output_type="latent",
    )
    assert probe.anchor_x0 is not None
    assert probe.anchor_x0.shape == (16, 1, 16, 16)
    assert probe.stop_step == 0
    assert probe.selected_step == 0
    assert probe.steps_executed == 1

    output = pipeline.generate_with_inspection(
        "prompt",
        stopper=None,
        inspect_generation=False,
        anchor_x0=probe.anchor_x0,
        num_frames=5,
        num_inference_steps=2,
        height=16,
        width=16,
        output_type="latent",
    )
    assert output.frames.shape == (1, 16, 2, 16, 16)

    paired_initial = pipeline.generate_with_inspection(
        "prompt",
        stopper=immediate_stopper(),
        failure_threshold=0.22,
        continue_on_failure=True,
        num_frames=5,
        num_inference_steps=2,
        height=16,
        width=16,
        output_type="latent",
    )
    assert paired_initial.failed
    assert paired_initial.frames.shape == (1, 16, 2, 16, 16)
    assert paired_initial.preview is not None
    assert paired_initial.stop_step == 0
    assert paired_initial.steps_executed == 1
    assert paired_initial.total_steps == 2
