from pathlib import Path

import pytest
import torch

from early_failure_video_diffusion.l2r import LatentToRGB


def test_checkpoint_layout_and_output_shape() -> None:
    model = LatentToRGB().eval()
    state = model.state_dict()

    assert len(state) == 29
    assert state["proj_in.blocks.0.conv.weight"].shape == (16, 16, 2, 3, 3)
    assert state["rank_basis_up1.fused_weight"].shape == (64, 1, 2, 5, 5)
    assert state["proj4.2.weight"].shape == (3, 16, 3, 3)

    with torch.inference_mode():
        output = model(torch.randn(1, 16, 2, 3, 4))
    assert output.shape == (1, 3, 2, 24, 32)


def test_strict_checkpoint_roundtrip(tmp_path: Path) -> None:
    original = LatentToRGB().eval()
    checkpoint = tmp_path / "l2r.pt"
    torch.save(original.state_dict(), checkpoint)

    restored = LatentToRGB.from_checkpoint(checkpoint)
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


@pytest.mark.parametrize("backend", ["cogvideox", "wan"])
def test_bundled_checkpoint_loads_strictly(backend: str) -> None:
    model = LatentToRGB.from_pretrained(backend)

    assert len(model.state_dict()) == 29


def test_unknown_bundled_backend_is_clear() -> None:
    with pytest.raises(ValueError, match="choose one of: cogvideox, wan"):
        LatentToRGB.from_pretrained("unknown")
