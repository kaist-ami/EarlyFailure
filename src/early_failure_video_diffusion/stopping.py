from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class StopDecision:
    """Inspection decision and the preview selected by that decision.

    ``stop_step`` is the zero-based step where inspection stopped. The score
    and preview belong to ``selected_step``, which can be earlier when
    ``reason`` is ``"budget_best"``.
    """

    stop_step: int
    selected_step: int
    reason: str
    score: float


class AdaptiveBalancedStopper:
    """Adaptive early-stop policy used by the paper's inference pipeline."""

    def __init__(
        self,
        *,
        eps: float = 0.004,
        patience: int = 2,
        min_step: int = 10,
        window: int = 3,
        tau_up: float = 0.003,
        s_bad: float = 0.10,
        tau_down: float = -0.0005,
        s_reject: float = 0.10,
        eps_reject: float = 0.02,
        patience_reject: int = 3,
        s_good: float = 0.24,
        eps_good: float = 0.015,
        max_step: int = 15,
        pick: str = "best",
    ) -> None:
        if patience < 1 or patience_reject < 1 or window < 1:
            raise ValueError("patience and window values must be positive")
        if min_step < 0 or max_step < 0:
            raise ValueError("step limits must be non-negative")
        if pick not in {"best", "last"}:
            raise ValueError("pick must be 'best' or 'last'")

        self.eps = float(eps)
        self.patience = int(patience)
        self.min_step = int(min_step)
        self.window = int(window)
        self.tau_up = float(tau_up)
        self.s_bad = float(s_bad)
        self.tau_down = float(tau_down)
        self.s_reject = float(s_reject)
        self.eps_reject = float(eps_reject)
        self.patience_reject = int(patience_reject)
        self.s_good = float(s_good)
        self.eps_good = float(eps_good)
        self.max_step = int(max_step)
        self.pick = pick

        self.previous_score: float | None = None
        self.absolute_deltas: deque[float] = deque(
            maxlen=max(self.patience, self.patience_reject)
        )
        self.signed_deltas: deque[float] = deque(maxlen=self.window)
        self.last_step: int | None = None
        self.last_score: float | None = None
        self.best_step: int | None = None
        self.best_score = -float("inf")

    def update(self, step: int, score: float) -> StopDecision | None:
        score = float(score)
        if math.isnan(score):
            return None

        step = int(step)
        self.last_step = step
        self.last_score = score

        if step <= self.max_step and score > self.best_score:
            self.best_step = step
            self.best_score = score

        if self.previous_score is not None:
            delta = score - self.previous_score
            self.signed_deltas.append(delta)
            self.absolute_deltas.append(abs(delta))
        self.previous_score = score

        if step < self.min_step:
            return self._budget_decision(step) if step >= self.max_step else None

        trend = (
            sum(self.signed_deltas) / len(self.signed_deltas)
            if self.signed_deltas
            else 0.0
        )
        stable = len(self.absolute_deltas) >= self.patience and all(
            delta < self.eps for delta in list(self.absolute_deltas)[-self.patience :]
        )

        if (
            score <= self.s_reject
            and len(self.absolute_deltas) >= self.patience_reject
            and all(
                delta < self.eps_reject
                for delta in list(self.absolute_deltas)[-self.patience_reject :]
            )
        ):
            return StopDecision(
                stop_step=step,
                selected_step=step,
                reason="early_reject",
                score=score,
            )

        if score < self.s_bad and trend < self.tau_down:
            return StopDecision(
                stop_step=step,
                selected_step=step,
                reason="give_up",
                score=score,
            )

        if (
            score >= self.s_good
            and len(self.absolute_deltas) >= self.patience
            and all(
                delta < self.eps_good
                for delta in list(self.absolute_deltas)[-self.patience :]
            )
        ):
            return StopDecision(
                stop_step=step,
                selected_step=step,
                reason="early_accept",
                score=score,
            )

        if stable and trend < self.tau_up:
            return StopDecision(
                stop_step=step,
                selected_step=step,
                reason="stable_stop",
                score=score,
            )

        if step >= self.max_step:
            return self._budget_decision(step)

        return None

    def _budget_decision(self, step: int) -> StopDecision:
        if self.pick == "last" or self.best_step is None:
            assert self.last_score is not None
            return StopDecision(
                stop_step=step,
                selected_step=step,
                reason="budget_last",
                score=self.last_score,
            )
        return StopDecision(
            stop_step=step,
            selected_step=self.best_step,
            reason="budget_best",
            score=self.best_score,
        )

    def finalize(self) -> StopDecision:
        if self.last_step is None or self.last_score is None:
            return StopDecision(
                stop_step=0,
                selected_step=0,
                reason="no_valid_score",
                score=float("nan"),
            )
        return self._budget_decision(self.last_step)


def make_video_stopper() -> AdaptiveBalancedStopper:
    return AdaptiveBalancedStopper(
        max_step=14,
        eps=0.002,
        patience=2,
        min_step=8,
        window=3,
        tau_up=0.003,
        s_bad=0.10,
        tau_down=-0.001,
        s_good=0.22,
        eps_good=0.01,
    )


def make_frame_stopper() -> AdaptiveBalancedStopper:
    return AdaptiveBalancedStopper(
        max_step=30,
        eps=0.003,
        patience=3,
        min_step=20,
        window=5,
        tau_up=0.0015,
        s_bad=0.08,
        tau_down=-0.0005,
        s_good=0.24,
        eps_good=0.01,
    )
