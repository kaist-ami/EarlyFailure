from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .assets import resolve_viclip_assets


class ViCLIPScorer:
    """In-memory VBench ViCLIP overall-consistency scorer."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str | torch.device = "cuda",
        vocab_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        image_size: int = 224,
        num_frames: int = 8,
        amp_dtype: str = "bf16",
        fast_single_frame: bool = True,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError(
                "the released ViCLIP inspection path requires an available CUDA device"
            )

        try:
            from ._vendor.viclip import SimpleTokenizer, ViCLIP
        except (ImportError, RuntimeError) as error:
            raise ImportError(
                "ViCLIP dependencies are unavailable. Install this project with "
                '`pip install -e ".[pipeline]"`.'
            ) from error

        checkpoint_path, resolved_vocab_path = resolve_viclip_assets(
            checkpoint=checkpoint,
            vocab_path=vocab_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.checkpoint_path = checkpoint_path
        self.vocab_path = resolved_vocab_path

        self.image_size = int(image_size)
        self.num_frames = int(num_frames)
        self.fast_single_frame = bool(fast_single_frame)
        self.amp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "none": None,
        }.get(amp_dtype)
        if amp_dtype not in {"bf16", "fp16", "none"}:
            raise ValueError("amp_dtype must be 'bf16', 'fp16', or 'none'")

        tokenizer = SimpleTokenizer(str(resolved_vocab_path))
        self.model = ViCLIP(tokenizer=tokenizer, pretrain=str(checkpoint_path)).to(
            self.device
        )
        self.model.eval().requires_grad_(False)

        self._mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=self.device
        ).view(1, 1, 3, 1, 1)
        self._std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=self.device
        ).view(1, 1, 3, 1, 1)
        self._cached_prompt: str | None = None
        self._cached_text_features: torch.Tensor | None = None

    def to(self, device: str | torch.device) -> "ViCLIPScorer":
        """Move ViCLIP and its cached tensors for a temporary GPU handoff."""

        self.device = torch.device(device)
        self.model.to(self.device)
        self._mean = self._mean.to(self.device)
        self._std = self._std.to(self.device)
        if self._cached_text_features is not None:
            self._cached_text_features = self._cached_text_features.to(self.device)
        return self

    @staticmethod
    def _middle_indices(
        count: int, video_length: int, device: torch.device
    ) -> torch.Tensor:
        if video_length < 1:
            raise ValueError("RGB preview contains no frames")
        count = min(count, video_length)
        boundaries = torch.linspace(0, video_length, count + 1, device=device)
        starts = boundaries[:-1].floor().long()
        ends = boundaries[1:].floor().long() - 1
        indices = (starts + ends.clamp_min(starts)) // 2
        return indices.clamp(0, video_length - 1)

    def _preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[1] != 3:
            raise ValueError(f"expected RGB [B, 3, T, H, W], got {tuple(rgb.shape)}")
        rgb = rgb.to(self.device)
        frame_count = (
            1 if self.fast_single_frame and rgb.shape[2] == 1 else self.num_frames
        )
        indices = self._middle_indices(frame_count, rgb.shape[2], rgb.device)
        frames = rgb[:, :, indices].permute(0, 2, 1, 3, 4).contiguous()
        frames = (frames.clamp(-1.0, 1.0) + 1.0) * 0.5
        batch, count, channels, height, width = frames.shape
        frames = F.interpolate(
            frames.reshape(batch * count, channels, height, width),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        ).view(batch, count, channels, self.image_size, self.image_size)
        return (frames - self._mean) / self._std

    def _text_features(self, prompt: str) -> torch.Tensor:
        prompt = str(prompt)
        if prompt != self._cached_prompt or self._cached_text_features is None:
            features = self.model.encode_text(prompt).float()
            self._cached_text_features = F.normalize(features, dim=-1).to(self.device)
            self._cached_prompt = prompt
        return self._cached_text_features

    @torch.inference_mode()
    def score(self, prompt: str, rgb: torch.Tensor) -> torch.Tensor:
        images = self._preprocess(rgb)
        if self.amp_dtype is None:
            video_features = self.model.encode_vision(images, test=True).float()
        else:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                video_features = self.model.encode_vision(images, test=True).float()
        video_features = F.normalize(video_features, dim=-1)
        text_features = self._text_features(prompt)
        return (video_features @ text_features.T).squeeze(-1)
