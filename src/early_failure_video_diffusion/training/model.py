from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..l2r import LatentToRGB, MemoryBlock3d


@dataclass(frozen=True)
class FactorizedL2RConfig:
    """Configuration for the train-time rank-basis representation.

    The deployment architecture is fixed to the configuration used by the
    paper. Only the factorization rank and its initialization are train-time
    choices; neither remains in the fused checkpoint.
    """

    basis_rank: int = 4096
    init_scale: float = 0.02
    use_tanh: bool = True

    def __post_init__(self) -> None:
        if self.basis_rank < 1:
            raise ValueError("basis_rank must be positive")
        if self.init_scale < 0:
            raise ValueError("init_scale must be non-negative")

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "FactorizedL2RConfig":
        return cls(
            basis_rank=int(values["basis_rank"]),
            init_scale=float(values["init_scale"]),
            use_tanh=bool(values["use_tanh"]),
        )


class FactorizedTemporalUpsample(nn.Module):
    """Train-time temporal upsampler represented by basis and expansion weights."""

    def __init__(
        self,
        *,
        channels: int = 16,
        heads: int = 1,
        scale_height: int = 2,
        scale_width: int = 2,
        time_kernel: int = 2,
        height_kernel: int = 5,
        width_kernel: int = 5,
        basis_rank: int = 4096,
        use_tanh: bool = True,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if basis_rank < 1:
            raise ValueError("basis_rank must be positive")
        self.channels = int(channels)
        self.heads = int(heads)
        self.scale_height = int(scale_height)
        self.scale_width = int(scale_width)
        self.time_kernel = int(time_kernel)
        self.height_kernel = int(height_kernel)
        self.width_kernel = int(width_kernel)
        self.basis_rank = int(basis_rank)
        self.use_tanh = bool(use_tanh)
        self.subpixels = self.scale_height * self.scale_width
        self.pad_height = self.height_kernel // 2
        self.pad_width = self.width_kernel // 2

        self.dw_weight = nn.Parameter(
            torch.empty(
                self.heads * self.channels * self.basis_rank,
                1,
                self.time_kernel,
                self.height_kernel,
                self.width_kernel,
            )
        )
        self.expand_weight = nn.Parameter(
            torch.empty(
                self.heads * self.channels * self.subpixels,
                self.basis_rank,
                1,
                1,
                1,
            )
        )
        nn.init.normal_(self.dw_weight, mean=0.0, std=float(init_scale))
        nn.init.normal_(self.expand_weight, mean=0.0, std=float(init_scale))

    def _activated(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.tanh(weight) if self.use_tanh else weight

    def fused_kernel(self) -> torch.Tensor:
        """Contract the basis axis into one grouped-convolution kernel."""

        depthwise = self._activated(self.dw_weight).view(
            self.heads,
            self.channels,
            self.basis_rank,
            self.time_kernel,
            self.height_kernel,
            self.width_kernel,
        )
        expansion = self._activated(self.expand_weight).view(
            self.heads,
            self.channels,
            self.subpixels,
            self.basis_rank,
        )
        fused = (
            expansion[..., None, None, None] * depthwise[:, :, None, :, :, :, :]
        ).sum(dim=3)
        return (
            fused.permute(1, 0, 2, 3, 4, 5)
            .contiguous()
            .view(
                self.channels * self.heads * self.subpixels,
                1,
                self.time_kernel,
                self.height_kernel,
                self.width_kernel,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"expected [B, {self.channels}, T, H, W], got {tuple(inputs.shape)}"
            )

        batch, _, frames, height, width = inputs.shape
        if self.time_kernel > 1:
            prefix = inputs[:, :, :1].repeat(1, 1, self.time_kernel - 1, 1, 1)
            inputs = torch.cat((prefix, inputs), dim=2)
        inputs = F.pad(
            inputs,
            (self.pad_width, self.pad_width, self.pad_height, self.pad_height, 0, 0),
        )
        hidden = F.conv3d(inputs, self.fused_kernel(), groups=self.channels)
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


class FactorizedLatentToRGB(nn.Module):
    """Train-time L2R model whose three upsamplers retain their rank basis."""

    def __init__(self, config: FactorizedL2RConfig | None = None) -> None:
        super().__init__()
        self.config = config or FactorizedL2RConfig()
        channels = 16

        def upsample() -> FactorizedTemporalUpsample:
            return FactorizedTemporalUpsample(
                channels=channels,
                basis_rank=self.config.basis_rank,
                use_tanh=self.config.use_tanh,
                init_scale=self.config.init_scale,
            )

        self.proj_in = MemoryBlock3d(channels, channels, num_blocks=2, time_kernel=2)
        self.rank_basis_up1 = upsample()
        self.proj2 = MemoryBlock3d(channels, channels, num_blocks=2, time_kernel=2)
        self.rank_basis_up2 = upsample()
        self.proj3 = MemoryBlock3d(channels, channels, num_blocks=1, time_kernel=2)
        self.rank_basis_up3 = upsample()
        self.proj4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=9, padding=4),
            nn.SiLU(),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
        )
        self.nonlinearity = nn.SiLU()

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

    @torch.no_grad()
    def to_deploy(self) -> LatentToRGB:
        """Fuse all basis weights and return the 29-key inference model."""

        parameter = next(self.parameters())
        deploy = LatentToRGB().to(device=parameter.device, dtype=parameter.dtype)
        factorized_state = self.state_dict()
        fused_weights = {
            "rank_basis_up1.fused_weight": self.rank_basis_up1.fused_kernel(),
            "rank_basis_up2.fused_weight": self.rank_basis_up2.fused_kernel(),
            "rank_basis_up3.fused_weight": self.rank_basis_up3.fused_kernel(),
        }
        deploy_state = {
            key: fused_weights.get(key, factorized_state.get(key))
            for key in deploy.state_dict()
        }
        if any(value is None for value in deploy_state.values()):
            missing = [key for key, value in deploy_state.items() if value is None]
            raise RuntimeError(f"could not construct deploy state: {missing}")
        deploy.load_state_dict(deploy_state, strict=True)
        return deploy.eval()
