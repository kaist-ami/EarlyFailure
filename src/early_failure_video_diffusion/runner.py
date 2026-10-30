from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from .prompt_refiner import PromptRefiner
from .stopping import make_frame_stopper, make_video_stopper
from .types import GenerationResult, InterventionResult


class InspectionBackend(Protocol):
    def generate_with_inspection(
        self, prompt: str, **kwargs: Any
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class RunnerConfig:
    height: int = 480
    width: int = 720
    num_frames: int = 49
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    failure_threshold: float = 0.22
    anchor_min_score: float = 0.10
    anchor_margin: float = 0.05
    output_type: str = "pil"


class HierarchicalInterventionRunner:
    """Model-independent implementation of the paper's intervention hierarchy."""

    def __init__(
        self,
        pipeline: InspectionBackend,
        prompt_refiner: PromptRefiner,
        *,
        config: RunnerConfig | None = None,
        generator_factory: Callable[[], torch.Generator] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.prompt_refiner = prompt_refiner
        self.config = config or RunnerConfig()
        self.generator_factory = generator_factory

    def _generator(self) -> torch.Generator | None:
        return self.generator_factory() if self.generator_factory is not None else None

    def _generate(
        self,
        generation_prompt: str,
        *,
        inspection_prompt: str,
        negative_prompt: str | None = None,
        probe_only: bool = False,
        failure_threshold: float | None = None,
        anchor_x0: torch.Tensor | None = None,
        inspect_generation: bool = True,
        continue_on_failure: bool = False,
    ) -> GenerationResult:
        kwargs: dict[str, Any] = {
            "negative_prompt": negative_prompt,
            "inspection_prompt": inspection_prompt,
            "stopper": (make_frame_stopper() if probe_only else make_video_stopper())
            if inspect_generation
            else None,
            "failure_threshold": (
                self.config.failure_threshold
                if failure_threshold is None
                else float(failure_threshold)
            ),
            "probe_only": probe_only,
            "inspect_generation": inspect_generation,
            "anchor_x0": anchor_x0,
            "height": self.config.height,
            "width": self.config.width,
            "num_frames": 1 if probe_only else self.config.num_frames,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "output_type": self.config.output_type,
            "generator": self._generator(),
        }
        if continue_on_failure:
            kwargs["continue_on_failure"] = True
        return self.pipeline.generate_with_inspection(generation_prompt, **kwargs)

    @staticmethod
    def _require_probe_payload(result: GenerationResult, stage: str) -> None:
        if result.score is None or result.anchor_x0 is None:
            raise RuntimeError(f"{stage} did not return a score and anchor x0")

    @staticmethod
    def _require_failure_payload(result: GenerationResult, stage: str) -> None:
        if result.score is None or result.preview is None:
            raise RuntimeError(f"{stage} did not return a score and RGB preview")

    def generate_baseline(self, prompt: str) -> GenerationResult:
        """Run ordinary full denoising without inspection or intervention."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        prompt = prompt.strip()
        result = self._generate(
            prompt,
            inspection_prompt=prompt,
            inspect_generation=False,
        )
        if not result.completed:
            raise RuntimeError("baseline generation did not complete")
        return result

    def _run_from_initial(
        self,
        prompt: str,
        initial: GenerationResult,
    ) -> InterventionResult:
        stages: list[dict[str, Any]] = []
        stages.append(initial.trace("initial_video"))
        if initial.completed:
            return InterventionResult(
                frames=initial.frames,
                decision_path="initial_video",
                original_prompt=prompt,
                stages=stages,
            )
        self._require_failure_payload(initial, "initial_video")

        initial_probe = self._generate(
            prompt, inspection_prompt=prompt, probe_only=True
        )
        stages.append(initial_probe.trace("initial_frame_probe"))
        self._require_probe_payload(initial_probe, "initial_frame_probe")
        assert initial.score is not None and initial_probe.score is not None

        if (
            initial_probe.score > self.config.anchor_min_score
            and initial_probe.score - initial.score > self.config.anchor_margin
        ):
            anchored = self._generate(
                prompt,
                inspection_prompt=prompt,
                anchor_x0=initial_probe.anchor_x0,
                inspect_generation=False,
            )
            stages.append(anchored.trace("initial_anchor_generation"))
            return InterventionResult(
                frames=anchored.frames,
                decision_path="initial_frame_anchor",
                original_prompt=prompt,
                stages=stages,
            )

        assert initial.preview is not None
        revision = self.prompt_refiner.refine(
            original_prompt=prompt,
            preview=initial.preview,
            score=initial.score,
        )

        refined_probe = self._generate(
            revision.refined_prompt,
            negative_prompt=revision.negative_prompt,
            inspection_prompt=prompt,
            probe_only=True,
        )
        stages.append(refined_probe.trace("refined_frame_probe"))
        self._require_probe_payload(refined_probe, "refined_frame_probe")
        assert refined_probe.score is not None

        refined = self._generate(
            revision.refined_prompt,
            negative_prompt=revision.negative_prompt,
            inspection_prompt=prompt,
            failure_threshold=refined_probe.score,
        )
        stages.append(refined.trace("refined_video"))
        if refined.completed:
            return InterventionResult(
                frames=refined.frames,
                decision_path="refined_video",
                original_prompt=prompt,
                refined_prompt=revision.refined_prompt,
                negative_prompt=revision.negative_prompt,
                stages=stages,
            )

        refined_anchored = self._generate(
            revision.refined_prompt,
            negative_prompt=revision.negative_prompt,
            inspection_prompt=prompt,
            anchor_x0=refined_probe.anchor_x0,
            inspect_generation=False,
        )
        stages.append(refined_anchored.trace("refined_anchor_generation"))
        return InterventionResult(
            frames=refined_anchored.frames,
            decision_path="refined_frame_anchor",
            original_prompt=prompt,
            refined_prompt=revision.refined_prompt,
            negative_prompt=revision.negative_prompt,
            stages=stages,
        )

    def run(self, prompt: str) -> InterventionResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        prompt = prompt.strip()
        initial = self._generate(prompt, inspection_prompt=prompt)
        return self._run_from_initial(prompt, initial)

    def run_paired(
        self,
        prompt: str,
        *,
        on_baseline: Callable[[GenerationResult], None] | None = None,
    ) -> tuple[GenerationResult, InterventionResult]:
        """Generate the baseline and intervention from one inspected first pass.

        A failed early inspection is retained while denoising continues to the
        ordinary final video.  That final video is the baseline; the retained
        RGB/x0 intermediate then drives the intervention hierarchy without
        repeating the initial trajectory.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        prompt = prompt.strip()
        initial = self._generate(
            prompt,
            inspection_prompt=prompt,
            continue_on_failure=True,
        )
        if initial.frames is None:
            raise RuntimeError("paired initial pass did not return baseline frames")
        baseline = GenerationResult(
            status="completed",
            frames=initial.frames,
            score=initial.score,
            stop_step=initial.stop_step,
            selected_step=initial.selected_step,
            stop_reason=initial.stop_reason,
            steps_executed=initial.total_steps,
            total_steps=initial.total_steps,
        )
        if on_baseline is not None:
            on_baseline(baseline)
        intervention = self._run_from_initial(prompt, initial)
        return baseline, intervention
