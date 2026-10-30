from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from early_failure_video_diffusion.cli import (
    _base_model_spec,
    _prepare_video_frames,
    _runtime_metadata,
    _validate_output_mode,
    build_parser,
)
from early_failure_video_diffusion.model_specs import DEFAULT_MODEL_REVISIONS
from early_failure_video_diffusion.prompt_refiner import DEFAULT_QWEN_REVISION
from early_failure_video_diffusion.test_sets import (
    TEST_SET_OUTPUT_DIRECTORIES,
    TEST_SET_SEED,
    inference_examples,
)


def test_checkpoint_arguments_are_optional() -> None:
    args = build_parser().parse_args(
        [
            "--backend",
            "wan",
            "--prompt",
            "a paper crane",
            "--output",
            "output.mp4",
        ]
    )

    assert args.l2r_checkpoint is None
    assert args.viclip_checkpoint is None
    assert args.viclip_vocab is None
    assert args.prompt_refiner == "openai"


def test_uint8_video_frames_are_not_scaled_twice() -> None:
    frames = np.array(
        [
            [[[0, 1, 255], [17, 128, 254]]],
            [[[255, 0, 127], [3, 64, 192]]],
        ],
        dtype=np.uint8,
    )

    prepared = _prepare_video_frames(frames)

    assert all(isinstance(frame, Image.Image) for frame in prepared)
    np.testing.assert_array_equal(
        np.stack([np.asarray(frame) for frame in prepared]), frames
    )


def test_qwen_refiner_arguments_are_explicit_and_friendly() -> None:
    args = build_parser().parse_args(
        [
            "--backend",
            "cogvideox",
            "--prompt",
            "a paper crane",
            "--output",
            "output.mp4",
            "--prompt-refiner",
            "qwen",
        ]
    )

    assert args.qwen_model == "Qwen/Qwen3.8-27B"
    assert args.qwen_revision is None
    assert args.qwen_python.as_posix() == ".venv-qwen/bin/python"
    assert args.refiner_max_images == 12


def test_built_in_models_resolve_to_tested_revisions() -> None:
    for backend in ("cogvideox", "wan"):
        args = build_parser().parse_args(
            [
                "--backend",
                backend,
                "--prompt",
                "a paper crane",
                "--output",
                "output.mp4",
            ]
        )
        _, revision = _base_model_spec(args)
        assert revision == DEFAULT_MODEL_REVISIONS[backend]


def test_runtime_metadata_redacts_local_paths(tmp_path: Path) -> None:
    local_model = tmp_path / "private-model-name"
    local_model.mkdir()
    args = build_parser().parse_args(
        [
            "--backend",
            "wan",
            "--model-id",
            str(local_model),
            "--prompt-refiner",
            "qwen",
            "--qwen-model",
            str(local_model),
            "--prompt",
            "a paper crane",
            "--output",
            "output.mp4",
        ]
    )

    metadata = _runtime_metadata(
        args,
        l2r_source="custom",
        viclip_source="custom",
        refiner_model=str(local_model),
        refiner_revision=None,
    )

    assert metadata["model_id"] == "local"
    assert metadata["model_revision"] is None
    assert metadata["prompt_refiner_model"] == "local"
    assert metadata["prompt_refiner_revision"] is None
    assert metadata["viclip_revision"] is None
    assert metadata["viclip_sha256"] is None
    assert str(tmp_path) not in str(metadata)


def test_default_qwen_revision_is_the_tested_commit() -> None:
    from early_failure_video_diffusion.prompt_refiner import QwenPromptRefiner

    assert QwenPromptRefiner().revision == DEFAULT_QWEN_REVISION


def test_released_test_sets_and_output_directories_are_exact() -> None:
    cogvideox = inference_examples("cogvideox")
    wan = inference_examples("wan")

    assert [example.prompt for example in cogvideox] == [
        "Iron Man flying in the sky",
        "A confused panda in calculus class",
    ]
    assert [example.prompt for example in wan] == [
        "A corgi's head depicted as an explosion of a nebula",
        (
            "A couple in formal evening wear going home get caught in a heavy "
            "downpour with umbrellas by Hokusai, in the style of Ukiyoss"
        ),
    ]
    assert TEST_SET_OUTPUT_DIRECTORIES == {
        "cogvideox": "cogvideox-5b",
        "wan": "wan2.1-1.3b",
    }
    assert TEST_SET_SEED == 42


def test_test_set_mode_requires_output_directory() -> None:
    valid = build_parser().parse_args(
        [
            "--backend",
            "wan",
            "--test-set",
            "--output-dir",
            "outputs/test-set",
        ]
    )
    _validate_output_mode(valid)

    invalid = build_parser().parse_args(
        ["--backend", "wan", "--test-set", "--output", "output.mp4"]
    )
    with pytest.raises(ValueError, match="--test-set requires --output-dir"):
        _validate_output_mode(invalid)


def test_test_set_mode_rejects_seed_override() -> None:
    args = build_parser().parse_args(
        [
            "--backend",
            "cogvideox",
            "--test-set",
            "--output-dir",
            "outputs/test-set",
            "--seed",
            "7",
        ]
    )

    with pytest.raises(ValueError, match="fixed released seed 42"):
        _validate_output_mode(args)
