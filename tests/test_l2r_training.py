from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from early_failure_video_diffusion.l2r import LatentToRGB
from early_failure_video_diffusion.training.checkpoint import (
    load_factorized_checkpoint,
    save_factorized_checkpoint,
)
from early_failure_video_diffusion.training.data import (
    L2RManifestDataset,
    PAPER_FRAME_INDICES,
    load_video_frames,
)
from early_failure_video_diffusion.training.fuse import (
    export_deploy_checkpoint,
    main as fuse_main,
)
from early_failure_video_diffusion.training.initialize import main as initialize_main
from early_failure_video_diffusion.training.model import (
    FactorizedL2RConfig,
    FactorizedLatentToRGB,
)
from early_failure_video_diffusion.training.train import main as train_main


def _write_tiny_manifest(directory: Path) -> Path:
    latent = torch.linspace(-1.0, 1.0, 16 * 2 * 2 * 3).reshape(16, 2, 2, 3)
    target = np.arange(2 * 16 * 24 * 3, dtype=np.uint8).reshape(2, 16, 24, 3)
    save_file({"latent": latent}, str(directory / "latent.safetensors"))
    np.save(directory / "target.npy", target)
    record = {
        "backend": "cogvideox",
        "latent": "latent.safetensors",
        "latent_key": "latent",
        "latent_layout": "cthw",
        "target": "target.npy",
        "target_layout": "thwc",
    }
    manifest = directory / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest


def test_factorized_model_fuses_to_strict_29_key_deploy_model() -> None:
    torch.manual_seed(7)
    model = FactorizedLatentToRGB(
        FactorizedL2RConfig(basis_rank=2, init_scale=0.01)
    ).eval()
    deploy = model.to_deploy()
    latents = torch.randn(1, 16, 3, 2, 3)

    with torch.inference_mode():
        factorized_output = model(latents)
        deploy_output = deploy(latents)

    assert len(model.state_dict()) == 32
    assert len(deploy.state_dict()) == 29
    torch.testing.assert_close(deploy_output, factorized_output)
    LatentToRGB().load_state_dict(deploy.state_dict(), strict=True)


def test_factorized_checkpoint_roundtrip_and_export(tmp_path: Path) -> None:
    model = FactorizedLatentToRGB(FactorizedL2RConfig(basis_rank=2))
    factorized_path = tmp_path / "factorized.pt"
    save_factorized_checkpoint(
        model,
        factorized_path,
        backend="wan",
        step=12,
        extra={"purpose": "test"},
    )

    restored, payload = load_factorized_checkpoint(factorized_path)
    assert payload["backend"] == "wan"
    assert payload["step"] == 12
    assert restored.config.basis_rank == 2
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])

    deploy_path, max_error = export_deploy_checkpoint(restored, tmp_path / "deploy.pt")
    assert max_error is not None and max_error < 1e-5
    assert len(LatentToRGB.from_checkpoint(deploy_path).state_dict()) == 29


def test_manifest_dataset_loads_relative_paired_assets(tmp_path: Path) -> None:
    manifest = _write_tiny_manifest(tmp_path)
    dataset = L2RManifestDataset(manifest)
    latent, target = dataset[0]

    assert dataset.declared_backends == {"cogvideox"}
    assert latent.shape == (16, 2, 2, 3)
    assert target.shape == (2, 3, 16, 24)
    assert target.dtype == torch.float32
    assert -1.0 <= float(target.min()) <= float(target.max()) <= 1.0


def test_49_frame_manifest_uses_released_temporal_alignment(tmp_path: Path) -> None:
    latent = torch.zeros(16, 13, 1, 1)
    target = np.broadcast_to(
        np.arange(49, dtype=np.uint8)[:, None, None, None], (49, 8, 8, 3)
    ).copy()
    save_file({"latent": latent}, str(tmp_path / "latent.safetensors"))
    np.save(tmp_path / "target.npy", target)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "latent": "latent.safetensors",
                "latent_key": "latent",
                "latent_layout": "cthw",
                "target": "target.npy",
                "target_layout": "thwc",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _, selected = L2RManifestDataset(manifest)[0]

    assert PAPER_FRAME_INDICES == (0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
    expected = torch.tensor(PAPER_FRAME_INDICES, dtype=torch.float32) / 127.5 - 1.0
    torch.testing.assert_close(selected[:, 0, 0, 0], expected)


def test_checked_in_sample_videos_are_valid_preparation_inputs() -> None:
    sample_dir = Path(__file__).parents[1] / "examples" / "l2r_training" / "videos"
    samples = sorted(sample_dir.glob("*.mp4"))

    assert [sample.name for sample in samples] == [
        "color_waves.mp4",
        "moving_shapes.mp4",
    ]
    for sample in samples:
        frames = load_video_frames(sample)
        assert frames.shape == (49, 480, 720, 3)
        assert frames.dtype == torch.uint8


def test_one_step_cli_path_produces_strict_deploy_checkpoint(tmp_path: Path) -> None:
    latent = torch.zeros(16, 1, 1, 1)
    target = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    save_file({"latent": latent}, str(tmp_path / "latent.safetensors"))
    np.save(tmp_path / "target.npy", target)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "backend": "cogvideox",
                "latent": "latent.safetensors",
                "latent_key": "latent",
                "latent_layout": "cthw",
                "target": "target.npy",
                "target_layout": "thwc",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    initialized = tmp_path / "factorized_init.pt"
    assert (
        initialize_main(
            [
                "--backend",
                "cogvideox",
                "--basis-rank",
                "1",
                "--output",
                str(initialized),
            ]
        )
        == 0
    )

    run_dir = tmp_path / "run"
    assert (
        train_main(
            [
                "--init-checkpoint",
                str(initialized),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(run_dir),
                "--steps",
                "1",
                "--batch-size",
                "1",
                "--num-workers",
                "0",
                "--save-every",
                "1",
                "--log-every",
                "1",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    deploy_path = tmp_path / "deploy.pt"
    assert (
        fuse_main(
            [
                "--checkpoint",
                str(run_dir / "factorized_final.pt"),
                "--output",
                str(deploy_path),
            ]
        )
        == 0
    )
    assert len(LatentToRGB.from_checkpoint(deploy_path).state_dict()) == 29
