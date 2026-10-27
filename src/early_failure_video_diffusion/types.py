from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


GenerationStatus = Literal["completed", "failed", "probe"]


@dataclass
class GenerationResult:
    """Backend-independent result returned by an inspection-aware pipeline."""

    status: GenerationStatus
    frames: Any | None = None
    preview: torch.Tensor | None = field(default=None, repr=False)
    anchor_x0: torch.Tensor | None = field(default=None, repr=False)
    score: float | None = None
    # Zero-based step where the inspection policy made its stop decision.
    stop_step: int | None = None
    # Zero-based step supplying ``score``, ``preview``, and ``anchor_x0``.
    selected_step: int | None = None
    stop_reason: str | None = None
    steps_executed: int = 0
    total_steps: int = 0

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    def trace(self, stage: str) -> dict[str, Any]:
        """Return JSON-safe stage metadata without video/tensor payloads."""

        return {
            "stage": stage,
            "status": self.status,
            "score": self.score,
            "stop_step": self.stop_step,
            "selected_step": self.selected_step,
            "stop_reason": self.stop_reason,
            "steps_executed": self.steps_executed,
            "total_steps": self.total_steps,
        }


@dataclass(frozen=True)
class PromptRevision:
    original_prompt: str
    refined_prompt: str
    negative_prompt: str = ""


@dataclass
class InterventionResult:
    """Final output and auditable decision path for the hierarchy."""

    frames: Any
    decision_path: str
    original_prompt: str
    refined_prompt: str | None = None
    negative_prompt: str | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {
            "decision_path": self.decision_path,
            "original_prompt": self.original_prompt,
            "refined_prompt": self.refined_prompt,
            "negative_prompt": self.negative_prompt,
            "stages": list(self.stages),
        }
