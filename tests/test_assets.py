from pathlib import Path

import early_failure_video_diffusion.assets as assets
from early_failure_video_diffusion.assets import resolve_viclip_assets


def test_manual_viclip_paths_work_offline(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ViClip-InternVid-10M-FLT.pth"
    vocabulary = tmp_path / "bpe_simple_vocab_16e6.txt.gz"
    checkpoint.write_bytes(b"manual checkpoint")
    vocabulary.write_bytes(b"manual vocabulary")

    resolved_checkpoint, resolved_vocabulary = resolve_viclip_assets(
        checkpoint=checkpoint,
        vocab_path=vocabulary,
        cache_dir=tmp_path / "unused-cache",
        local_files_only=True,
    )

    assert resolved_checkpoint == checkpoint.resolve()
    assert resolved_vocabulary == vocabulary.resolve()
    assert not (tmp_path / "unused-cache").exists()


def test_missing_viclip_assets_are_downloaded_to_the_selected_cache(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "viclip"
    hub_call = {}

    def fake_hub_download(**kwargs):
        hub_call.update(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.write_bytes(b"checkpoint")
        return str(destination)

    def fake_bpe_download(destination: Path) -> Path:
        destination.write_bytes(b"vocabulary")
        return destination

    monkeypatch.setattr(assets, "hf_hub_download", fake_hub_download)
    monkeypatch.setattr(assets, "_download_bpe", fake_bpe_download)
    monkeypatch.setattr(assets, "_sha256", lambda _: assets.VICLIP_SHA256)

    checkpoint, vocabulary = resolve_viclip_assets(cache_dir=cache_dir)

    assert checkpoint == (cache_dir / assets.VICLIP_FILENAME).resolve()
    assert vocabulary == (cache_dir / assets.BPE_FILENAME).resolve()
    assert hub_call["repo_id"] == "OpenGVLab/VBench_Used_Models"
    assert hub_call["revision"] == assets.VICLIP_REVISION
    assert hub_call["local_dir"] == cache_dir.resolve()


def test_official_viclip_checkpoint_checksum_is_enforced(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "viclip"
    cache_dir.mkdir()
    (cache_dir / assets.VICLIP_FILENAME).write_bytes(b"corrupt checkpoint")
    (cache_dir / assets.BPE_FILENAME).write_bytes(b"vocabulary")
    monkeypatch.setattr(assets, "_sha256", lambda _: "incorrect")

    try:
        resolve_viclip_assets(cache_dir=cache_dir, local_files_only=True)
    except RuntimeError as error:
        assert "checkpoint checksum mismatch" in str(error)
    else:
        raise AssertionError("corrupt official checkpoint was accepted")
