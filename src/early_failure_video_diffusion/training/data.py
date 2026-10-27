from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import Dataset


# RGB targets used by the released L2R training recipe for a 49-frame clip.
# The final research loader starts from [0, 1, 5, ..., 45] and adds two to every
# index except the first, producing the alignment below for both backends.
PAPER_FRAME_INDICES = (0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_video_frames(path: str | Path) -> torch.Tensor:
    """Load a video as uint8 ``[T, H, W, 3]`` RGB frames."""

    path = Path(path)
    try:
        frames = [np.asarray(frame) for frame in iio.imiter(path)]
    except Exception as error:
        raise RuntimeError(
            f"could not decode video {path}; install imageio[ffmpeg] and verify the file"
        ) from error
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    array = np.stack(frames)
    if array.ndim != 4 or array.shape[-1] not in {3, 4}:
        raise ValueError(
            f"expected RGB/RGBA video frames, got {array.shape} from {path}"
        )
    return torch.from_numpy(array[..., :3].copy())


def _load_tensor_file(path: Path, key: str | None) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return torch.from_numpy(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            selected = key or (archive.files[0] if len(archive.files) == 1 else None)
            if selected is None or selected not in archive:
                raise ValueError(f"specify a key for multi-array NPZ file: {path}")
            return torch.from_numpy(archive[selected])
    if suffix == ".safetensors":
        tensors = load_safetensors(str(path))
        selected = key or (next(iter(tensors)) if len(tensors) == 1 else None)
        if selected is None or selected not in tensors:
            raise ValueError(f"specify a key for multi-tensor file: {path}")
        return tensors[selected]
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, torch.Tensor):
            return payload
        if isinstance(payload, Mapping):
            selected = key or (next(iter(payload)) if len(payload) == 1 else None)
            if selected is not None and isinstance(payload.get(selected), torch.Tensor):
                return payload[selected]
        raise ValueError(f"could not select one tensor from {path}")
    raise ValueError(f"unsupported tensor file: {path}")


def _load_target(path: Path, key: str | None) -> torch.Tensor:
    if path.is_dir():
        image_paths = sorted(
            item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"target frame directory is empty: {path}")
        arrays = [np.asarray(Image.open(item).convert("RGB")) for item in image_paths]
        return torch.from_numpy(np.stack(arrays))
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return load_video_frames(path)
    return _load_tensor_file(path, key)


def _to_cthw(tensor: torch.Tensor, layout: str | None, channels: int) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"expected a 4D tensor, got {tuple(tensor.shape)}")
    if layout is not None:
        layout = layout.lower()
        if layout == "cthw":
            result = tensor
        elif layout == "tchw":
            result = tensor.permute(1, 0, 2, 3)
        elif layout == "thwc":
            result = tensor.permute(3, 0, 1, 2)
        else:
            raise ValueError(f"unsupported tensor layout: {layout}")
    elif tensor.shape[0] == channels:
        result = tensor
    elif tensor.shape[1] == channels:
        result = tensor.permute(1, 0, 2, 3)
    elif tensor.shape[-1] == channels:
        result = tensor.permute(3, 0, 1, 2)
    else:
        raise ValueError(
            f"could not infer the {channels}-channel axis for shape {tuple(tensor.shape)}"
        )
    return result.contiguous()


def _normalize_rgb(target: torch.Tensor) -> torch.Tensor:
    if target.dtype == torch.uint8:
        return target.float().div(127.5).sub(1.0)
    target = target.float()
    minimum = float(target.min())
    maximum = float(target.max())
    if minimum >= -1.001 and maximum <= 1.001:
        if minimum >= 0.0:
            return target.mul(2.0).sub(1.0)
        return target
    if minimum >= 0.0 and maximum <= 255.0:
        return target.div(127.5).sub(1.0)
    raise ValueError(
        f"RGB target must be uint8, [0, 1], or [-1, 1]; observed [{minimum}, {maximum}]"
    )


class L2RManifestDataset(Dataset):
    """Paired latent/RGB dataset described by a small JSONL manifest.

    Each line needs ``latent`` and ``target`` paths. Paths are resolved relative
    to the manifest. Optional fields are ``latent_key``, ``target_key``,
    ``latent_layout``, ``target_layout``, and ``frame_indices``.
    """

    def __init__(self, manifest: str | Path, *, spatial_scale: int = 8) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(f"training manifest not found: {self.manifest}")
        self.spatial_scale = int(spatial_scale)
        self.records: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(
            self.manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {self.manifest}:{line_number}"
                ) from error
            if (
                not isinstance(record, dict)
                or not {"latent", "target"} <= record.keys()
            ):
                raise ValueError(
                    f"{self.manifest}:{line_number} needs latent and target fields"
                )
            record["_line"] = line_number
            self.records.append(record)
        if not self.records:
            raise ValueError(f"training manifest is empty: {self.manifest}")

    @property
    def declared_backends(self) -> set[str]:
        """Return backend labels recorded by the preparation command, if any."""

        return {
            str(record["backend"])
            for record in self.records
            if record.get("backend") is not None
        }

    def __len__(self) -> int:
        return len(self.records)

    def _path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.manifest.parent / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"manifest asset not found: {path}")
        return path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        latent_path = self._path(str(record["latent"]))
        target_path = self._path(str(record["target"]))

        latent = _load_tensor_file(latent_path, record.get("latent_key"))
        latent = _to_cthw(latent, record.get("latent_layout"), channels=16).float()
        target = _load_target(target_path, record.get("target_key"))
        target = _to_cthw(target, record.get("target_layout"), channels=3)
        target = _normalize_rgb(target).permute(1, 0, 2, 3).contiguous()

        frame_indices = record.get("frame_indices")
        if frame_indices is not None:
            indices = torch.tensor(
                [int(item) for item in frame_indices], dtype=torch.long
            )
            if indices.numel() != latent.shape[1]:
                raise ValueError(
                    f"manifest line {record['_line']} has {indices.numel()} frame indices "
                    f"for {latent.shape[1]} latent frames"
                )
            if int(indices.min()) < 0 or int(indices.max()) >= target.shape[0]:
                raise IndexError(
                    f"frame_indices exceed target length at manifest line {record['_line']}"
                )
            target = target.index_select(0, indices)
        elif target.shape[0] != latent.shape[1]:
            if latent.shape[1] == len(PAPER_FRAME_INDICES) and target.shape[0] > max(
                PAPER_FRAME_INDICES
            ):
                target = target[list(PAPER_FRAME_INDICES)]
            else:
                raise ValueError(
                    f"temporal mismatch at manifest line {record['_line']}: latent has "
                    f"{latent.shape[1]} frames and target has {target.shape[0]}; add frame_indices"
                )

        expected_size = (
            latent.shape[-2] * self.spatial_scale,
            latent.shape[-1] * self.spatial_scale,
        )
        if tuple(target.shape[-2:]) != expected_size:
            raise ValueError(
                f"spatial mismatch at manifest line {record['_line']}: expected RGB "
                f"{expected_size}, got {tuple(target.shape[-2:])}"
            )
        return latent, target
