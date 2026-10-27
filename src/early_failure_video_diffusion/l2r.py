from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


_BUNDLED_CHECKPOINTS = {
    "cogvideox": "deploy_fused_crf18_cogvideox.pt",
    "wan": "deploy_fused_crf18_wan1.5B.pt",
}


class CausalConv3d(nn.Module):
    """Small causal 3D convolution with first-frame temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        time_kernel, height_kernel, width_kernel = kernel_size
        self.time_kernel = int(time_kernel)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(0, (height_kernel - 1) // 2, (width_kernel - 1) // 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.time_kernel > 1:
            prefix = inputs[:, :, :1].repeat(1, 1, self.time_kernel - 1, 1, 1)
            inputs = torch.cat((prefix, inputs), dim=2)
        return self.conv(inputs)


class MemoryBlock3d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_blocks: int, time_kernel: int
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least one")

        layers: list[nn.Module] = [
            CausalConv3d(in_channels, out_channels, (time_kernel, 3, 3)),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks - 1):
            layers.extend(
                [
                    CausalConv3d(out_channels, out_channels, (time_kernel, 3, 3)),
                    nn.ReLU(inplace=True),
                ]
            )

        # Attribute names intentionally match the released deployment checkpoints.
        self.blocks = nn.ModuleList(layers)
        self.conv_out = CausalConv3d(out_channels, out_channels, (time_kernel, 3, 3))
        self.skip = CausalConv3d(in_channels, out_channels, (1, 1, 1))
        self.act_out = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.blocks:
            hidden = layer(hidden)
        return self.act_out(self.conv_out(hidden) + self.skip(inputs))


class FusedTemporalUpsample(nn.Module):
    """Inference-only fused temporal/spatial rank-basis upsampler."""

    def __init__(
        self,
        channels: int = 16,
        heads: int = 1,
        scale_height: int = 2,
        scale_width: int = 2,
        time_kernel: int = 2,
        height_kernel: int = 5,
        width_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.heads = int(heads)
        self.scale_height = int(scale_height)
        self.scale_width = int(scale_width)
        self.time_kernel = int(time_kernel)
        self.height_kernel = int(height_kernel)
        self.width_kernel = int(width_kernel)
        self.subpixels = self.scale_height * self.scale_width
        self.pad_height = self.height_kernel // 2
        self.pad_width = self.width_kernel // 2

        self.register_buffer(
            "fused_weight",
            torch.zeros(
                self.channels * self.heads * self.subpixels,
                1,
                self.time_kernel,
                self.height_kernel,
                self.width_kernel,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"expected [B, {self.channels}, T, H, W], received {tuple(inputs.shape)}"
            )

        batch, _, frames, height, width = inputs.shape
        if self.time_kernel > 1:
            prefix = inputs[:, :, :1].repeat(1, 1, self.time_kernel - 1, 1, 1)
            inputs = torch.cat((prefix, inputs), dim=2)
        inputs = F.pad(
            inputs,
            (self.pad_width, self.pad_width, self.pad_height, self.pad_height, 0, 0),
        )
        hidden = F.conv3d(inputs, self.fused_weight, groups=self.channels)
        hidden = hidden.permute(0, 2, 1, 3, 4).reshape(
            batch * frames,
            self.channels * self.heads * self.subpixels,
            height,
            width,
        )

        if self.scale_height == self.scale_width:
            hidden = F.pixel_shuffle(hidden, self.scale_height)
        else:
            hidden = hidden.view(
                batch * frames,
                self.heads,
                self.channels,
                self.scale_height,
                self.scale_width,
                height,
                width,
            )
            hidden = hidden.permute(0, 1, 2, 5, 3, 6, 4).contiguous()
            hidden = hidden.view(
                batch * frames,
                self.heads * self.channels,
                height * self.scale_height,
                width * self.scale_width,
            )

        out_height, out_width = hidden.shape[-2:]
        return (
            hidden.view(
                batch, frames, self.heads * self.channels, out_height, out_width
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )


class LatentToRGB(nn.Module):
    """Deployment architecture for the paper's L2R/RI module.

    The parameter hierarchy is kept byte-for-byte compatible with the fused
    CogVideoX and Wan checkpoints used by the research code.
    """

    def __init__(self) -> None:
        super().__init__()
        channels = 16
        self.proj_in = MemoryBlock3d(channels, channels, num_blocks=2, time_kernel=2)
        self.rank_basis_up1 = FusedTemporalUpsample()
        self.proj2 = MemoryBlock3d(channels, channels, num_blocks=2, time_kernel=2)
        self.rank_basis_up2 = FusedTemporalUpsample()
        self.proj3 = MemoryBlock3d(channels, channels, num_blocks=1, time_kernel=2)
        self.rank_basis_up3 = FusedTemporalUpsample()
        self.proj4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
        )
        self.nonlinearity = nn.SiLU()

    @classmethod
    def from_pretrained(
        cls,
        backend: str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "LatentToRGB":
        """Load the bundled deployment checkpoint for a supported backend."""

        normalized_backend = backend.strip().lower()
        try:
            filename = _BUNDLED_CHECKPOINTS[normalized_backend]
        except KeyError as error:
            choices = ", ".join(sorted(_BUNDLED_CHECKPOINTS))
            raise ValueError(
                f"unknown L2R backend {backend!r}; choose one of: {choices}"
            ) from error

        checkpoint = resources.files("early_failure_video_diffusion").joinpath(
            "weights", filename
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"bundled L2R checkpoint is missing from the installation: {filename}"
            )
        with resources.as_file(checkpoint) as checkpoint_path:
            return cls.from_checkpoint(checkpoint_path, map_location=map_location)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "LatentToRGB":
        payload: Any = torch.load(
            checkpoint, map_location=map_location, weights_only=True
        )
        if isinstance(payload, Mapping) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, Mapping):
            raise TypeError("L2R checkpoint must contain a state dictionary")

        state_dict = dict(payload)
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {
                key.removeprefix("module."): value for key, value in state_dict.items()
            }

        model = cls()
        model.load_state_dict(state_dict, strict=strict)
        return model

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5 or latents.shape[1] != 16:
            raise ValueError(
                f"expected latent tensor [B, 16, T, H, W], got {tuple(latents.shape)}"
            )

        hidden = torch.tanh(latents / 3.0) * 3.0
        hidden = self.proj_in(hidden)
        hidden = self.nonlinearity(self.rank_basis_up1(hidden))
        hidden = self.proj2(hidden)
        hidden = self.nonlinearity(self.rank_basis_up2(hidden))
        hidden = self.proj3(hidden)
        hidden = self.nonlinearity(self.rank_basis_up3(hidden))

        batch, channels, frames, height, width = hidden.shape
        hidden = hidden.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        hidden = self.proj4(hidden)
        return (
            hidden.view(batch, frames, 3, height, width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
