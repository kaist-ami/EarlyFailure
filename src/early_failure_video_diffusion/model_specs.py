from __future__ import annotations

from pathlib import Path


DEFAULT_MODELS = {
    "cogvideox": "zai-org/CogVideoX-5b",
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}

DEFAULT_MODEL_REVISIONS = {
    "cogvideox": "8fc5b281006c82b82d34fd2543d2f0ebb4e7e321",
    "wan": "0fad780a534b6463e45facd96134c9f345acfa5b",
}


def is_local_model_reference(value: str) -> bool:
    expanded = Path(value).expanduser()
    return (
        expanded.exists()
        or expanded.is_absolute()
        or value.startswith((".", "~"))
        or (len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"})
    )


def safe_model_reference(value: str) -> str:
    """Return a metadata-safe model identifier without local path disclosure."""

    return "local" if is_local_model_reference(value) else value


def resolve_base_model_spec(
    backend: str,
    *,
    model_id: str | None = None,
    revision: str | None = None,
) -> tuple[str, str | None]:
    try:
        default_model = DEFAULT_MODELS[backend]
        default_revision = DEFAULT_MODEL_REVISIONS[backend]
    except KeyError as error:
        choices = ", ".join(sorted(DEFAULT_MODELS))
        raise ValueError(
            f"unknown backend {backend!r}; choose one of: {choices}"
        ) from error

    resolved_model = model_id or default_model
    if is_local_model_reference(resolved_model):
        return resolved_model, None
    if revision is None and resolved_model == default_model:
        revision = default_revision
    return resolved_model, revision
