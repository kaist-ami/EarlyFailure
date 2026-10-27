from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download


VICLIP_REPO_ID = "OpenGVLab/VBench_Used_Models"
VICLIP_FILENAME = "ViClip-InternVid-10M-FLT.pth"
VICLIP_REVISION = "4a9f73381ee3c9753c0258b50479b388f043e53a"
VICLIP_SHA256 = "7a4d6ad6eac6632db3693f4b97f9f8f6445b65b1e139a6fc6686150522238c56"
VICLIP_DOWNLOAD_URL = (
    f"https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/{VICLIP_REVISION}/"
    "ViClip-InternVid-10M-FLT.pth"
)
BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
BPE_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/openai/CLIP/main/clip/"
    "bpe_simple_vocab_16e6.txt.gz"
)
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"


def default_viclip_cache_dir() -> Path:
    root = Path(os.environ.get("VBENCH_CACHE_DIR", Path.home() / ".cache" / "vbench"))
    return root.expanduser() / "ViCLIP"


def _require_file(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_bpe(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        BPE_DOWNLOAD_URL,
        headers={"User-Agent": "early-failure-video-diffusion/0.1"},
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{BPE_FILENAME}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, temporary)
        actual_hash = _sha256(temporary_path)
        if actual_hash != BPE_SHA256:
            raise RuntimeError(
                f"downloaded ViCLIP vocabulary checksum mismatch: {actual_hash}"
            )
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def resolve_viclip_assets(
    *,
    checkpoint: str | Path | None = None,
    vocab_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> tuple[Path, Path]:
    """Resolve manual paths or download the official VBench ViCLIP assets."""

    asset_dir = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else default_viclip_cache_dir()
    )
    if checkpoint is None or vocab_path is None:
        asset_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint is not None:
        checkpoint_path = _require_file(checkpoint, "ViCLIP checkpoint")
    else:
        checkpoint_path = asset_dir / VICLIP_FILENAME
        if not checkpoint_path.is_file():
            try:
                checkpoint_path = Path(
                    hf_hub_download(
                        repo_id=VICLIP_REPO_ID,
                        filename=VICLIP_FILENAME,
                        revision=VICLIP_REVISION,
                        local_dir=asset_dir,
                        local_files_only=local_files_only,
                    )
                )
            except Exception as error:
                raise RuntimeError(
                    "ViCLIP checkpoint is not cached. Connect to the internet for automatic "
                    "download or pass --viclip-checkpoint /path/to/checkpoint. "
                    f"Official file: {VICLIP_DOWNLOAD_URL}"
                ) from error
        actual_hash = _sha256(checkpoint_path)
        if actual_hash != VICLIP_SHA256:
            raise RuntimeError(
                "official ViCLIP checkpoint checksum mismatch: "
                f"expected {VICLIP_SHA256}, received {actual_hash}"
            )

    if vocab_path is not None:
        resolved_vocab = _require_file(vocab_path, "ViCLIP BPE vocabulary")
    else:
        resolved_vocab = asset_dir / BPE_FILENAME
        if not resolved_vocab.is_file():
            if local_files_only:
                raise FileNotFoundError(
                    "ViCLIP BPE vocabulary is not cached; pass --viclip-vocab or disable "
                    "--local-files-only for automatic download."
                )
            resolved_vocab = _download_bpe(resolved_vocab)
        elif _sha256(resolved_vocab) != BPE_SHA256:
            raise RuntimeError(
                f"cached ViCLIP BPE vocabulary has an unexpected checksum: {resolved_vocab}"
            )

    return checkpoint_path.resolve(), resolved_vocab.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download official VBench ViCLIP assets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Asset directory (uses ~/.cache/vbench/ViCLIP when omitted)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Check the selected cache without downloading missing files",
    )
    args = parser.parse_args(argv)
    checkpoint, vocabulary = resolve_viclip_assets(
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    print(f"ViCLIP checkpoint: {checkpoint}")
    print(f"ViCLIP vocabulary: {vocabulary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
