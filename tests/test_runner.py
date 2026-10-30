from __future__ import annotations

from dataclasses import dataclass

import torch

from early_failure_video_diffusion.runner import HierarchicalInterventionRunner
from early_failure_video_diffusion.types import GenerationResult, PromptRevision


class FakePipeline:
    def __init__(self, results: list[GenerationResult]) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    def generate_with_inspection(self, prompt: str, **kwargs) -> GenerationResult:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.results.pop(0)


@dataclass
class FakeRefiner:
    calls: int = 0

    def refine(self, *, original_prompt: str, **kwargs) -> PromptRevision:
        self.calls += 1
        return PromptRevision(original_prompt, "refined prompt", "bad anatomy")


def failed(score: float) -> GenerationResult:
    return GenerationResult(
        status="failed",
        preview=torch.zeros(1, 3, 2, 8, 8),
        score=score,
        stop_step=14,
        selected_step=3,
        stop_reason="budget_best",
        steps_executed=15,
    )


def probe(score: float) -> GenerationResult:
    return GenerationResult(
        status="probe",
        preview=torch.zeros(1, 3, 1, 8, 8),
        anchor_x0=torch.zeros(16, 1, 1, 1),
        score=score,
        stop_step=30,
        selected_step=7,
        stop_reason="budget_best",
        steps_executed=31,
    )


def completed(frames="frames") -> GenerationResult:
    return GenerationResult(status="completed", frames=frames)


def test_baseline_disables_inspection_and_returns_completed_result() -> None:
    pipeline = FakePipeline([completed("baseline")])
    runner = HierarchicalInterventionRunner(pipeline, FakeRefiner())

    result = runner.generate_baseline(" prompt ")

    assert result.frames == "baseline"
    assert pipeline.calls == [
        {
            "prompt": "prompt",
            "negative_prompt": None,
            "inspection_prompt": "prompt",
            "stopper": None,
            "failure_threshold": 0.22,
            "probe_only": False,
            "inspect_generation": False,
            "anchor_x0": None,
            "height": 480,
            "width": 720,
            "num_frames": 49,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "output_type": "pil",
            "generator": None,
        }
    ]


def test_paired_run_reuses_the_inspected_initial_pass() -> None:
    initial = failed(0.08)
    initial.frames = "baseline"
    initial.total_steps = 50
    pipeline = FakePipeline([initial, probe(0.20), completed("anchored")])
    refiner = FakeRefiner()
    callback_observations = []

    baseline, result = HierarchicalInterventionRunner(pipeline, refiner).run_paired(
        " prompt ",
        on_baseline=lambda value: callback_observations.append(
            (value.frames, len(pipeline.calls))
        ),
    )

    assert baseline.completed
    assert baseline.frames == "baseline"
    assert baseline.steps_executed == 50
    assert callback_observations == [("baseline", 1)]
    assert result.decision_path == "initial_frame_anchor"
    assert result.frames == "anchored"
    assert pipeline.calls[0]["continue_on_failure"] is True
    assert len(pipeline.calls) == 3
    assert refiner.calls == 0


def test_paired_initial_success_uses_one_generation_for_both_videos() -> None:
    pipeline = FakePipeline([completed("shared frames")])
    runner = HierarchicalInterventionRunner(pipeline, FakeRefiner())

    baseline, result = runner.run_paired("prompt")

    assert baseline.frames == "shared frames"
    assert result.frames == "shared frames"
    assert result.decision_path == "initial_video"
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["continue_on_failure"] is True


def test_initial_success_exits_immediately() -> None:
    pipeline = FakePipeline([completed()])
    refiner = FakeRefiner()
    result = HierarchicalInterventionRunner(pipeline, refiner).run("prompt")

    assert result.decision_path == "initial_video"
    assert len(pipeline.calls) == 1
    assert refiner.calls == 0


def test_strong_initial_probe_uses_anchor_without_refiner() -> None:
    pipeline = FakePipeline([failed(0.08), probe(0.20), completed("anchored")])
    refiner = FakeRefiner()
    result = HierarchicalInterventionRunner(pipeline, refiner).run("prompt")

    assert result.decision_path == "initial_frame_anchor"
    assert result.frames == "anchored"
    assert pipeline.calls[-1]["inspect_generation"] is False
    assert refiner.calls == 0


def test_refined_video_path() -> None:
    pipeline = FakePipeline(
        [failed(0.10), probe(0.12), probe(0.27), completed("refined")]
    )
    refiner = FakeRefiner()
    result = HierarchicalInterventionRunner(pipeline, refiner).run("prompt")

    assert result.decision_path == "refined_video"
    assert result.refined_prompt == "refined prompt"
    assert pipeline.calls[-1]["failure_threshold"] == 0.27
    assert refiner.calls == 1
    assert result.stages[0]["stop_step"] == 14
    assert result.stages[0]["selected_step"] == 3


def test_failed_refined_video_uses_refined_anchor() -> None:
    pipeline = FakePipeline(
        [failed(0.10), probe(0.12), probe(0.24), failed(0.18), completed("final")]
    )
    result = HierarchicalInterventionRunner(pipeline, FakeRefiner()).run("prompt")

    assert result.decision_path == "refined_frame_anchor"
    assert result.frames == "final"
    assert pipeline.calls[-1]["anchor_x0"] is not None
    assert pipeline.calls[-1]["inspect_generation"] is False
