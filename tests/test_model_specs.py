from pathlib import Path

from early_failure_video_diffusion.model_specs import (
    DEFAULT_MODEL_REVISIONS,
    resolve_base_model_spec,
    safe_model_reference,
)


def test_default_model_revisions_are_pinned() -> None:
    for backend, expected_revision in DEFAULT_MODEL_REVISIONS.items():
        _, revision = resolve_base_model_spec(backend)
        assert revision == expected_revision


def test_custom_remote_model_does_not_inherit_an_unrelated_revision() -> None:
    model, revision = resolve_base_model_spec(
        "wan", model_id="organization/custom-video-model"
    )

    assert model == "organization/custom-video-model"
    assert revision is None


def test_local_model_paths_are_redacted_and_ignore_revision(tmp_path: Path) -> None:
    local_model = tmp_path / "private-model"
    local_model.mkdir()
    model, revision = resolve_base_model_spec(
        "cogvideox",
        model_id=str(local_model),
        revision="main",
    )

    assert model == str(local_model)
    assert revision is None
    assert safe_model_reference(model) == "local"
    assert safe_model_reference(r"C:\\private\\model") == "local"
