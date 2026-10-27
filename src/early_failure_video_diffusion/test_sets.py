from __future__ import annotations

from dataclasses import dataclass


TEST_SET_SEED = 42


@dataclass(frozen=True)
class InferenceExample:
    slug: str
    prompt: str


INFERENCE_TEST_SETS: dict[str, tuple[InferenceExample, ...]] = {
    "cogvideox": (
        InferenceExample(
            slug="01-iron-man-flying-in-the-sky",
            prompt="Iron Man flying in the sky",
        ),
        InferenceExample(
            slug="02-confused-panda-in-calculus-class",
            prompt="A confused panda in calculus class",
        ),
    ),
    "wan": (
        InferenceExample(
            slug="01-corgi-nebula-head",
            prompt="A corgi's head depicted as an explosion of a nebula",
        ),
        InferenceExample(
            slug="02-couple-in-heavy-downpour-by-hokusai",
            prompt=(
                "A couple in formal evening wear going home get caught in a heavy "
                "downpour with umbrellas by Hokusai, in the style of Ukiyoss"
            ),
        ),
    ),
}


TEST_SET_OUTPUT_DIRECTORIES = {
    "cogvideox": "cogvideox-5b",
    "wan": "wan2.1-1.3b",
}


def inference_examples(backend: str) -> tuple[InferenceExample, ...]:
    try:
        return INFERENCE_TEST_SETS[backend]
    except KeyError as error:
        supported = ", ".join(sorted(INFERENCE_TEST_SETS))
        raise ValueError(
            f"unsupported test-set backend {backend!r}; choose one of: {supported}"
        ) from error
