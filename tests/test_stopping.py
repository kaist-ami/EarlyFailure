import math

import pytest
import torch

from early_failure_video_diffusion.inspection import InspectionHistory, inspect_step
from early_failure_video_diffusion.stopping import (
    AdaptiveBalancedStopper,
    make_video_stopper,
)


def test_video_stopper_rejects_low_flat_scores() -> None:
    stopper = make_video_stopper()
    decision = None
    for step in range(15):
        decision = stopper.update(step, 0.05)
        if decision is not None:
            break

    assert decision is not None
    assert decision.stop_step == 8
    assert decision.selected_step == 8
    assert decision.reason == "early_reject"
    assert decision.score == 0.05


def test_budget_can_select_best_previous_step() -> None:
    stopper = AdaptiveBalancedStopper(min_step=99, max_step=3, pick="best")
    decision = None
    for step, score in enumerate((0.1, 0.3, 0.2, 0.1)):
        decision = stopper.update(step, score)

    assert decision is not None
    assert decision.stop_step == 3
    assert decision.selected_step == 1
    assert decision.reason == "budget_best"
    assert decision.score == 0.3


def test_nan_score_is_ignored() -> None:
    stopper = AdaptiveBalancedStopper(max_step=1, min_step=0)
    assert stopper.update(0, math.nan) is None
    assert stopper.last_step is None


def test_candidate_selection_ignores_non_finite_scores() -> None:
    stopper = AdaptiveBalancedStopper(max_step=3, min_step=0)
    history = InspectionHistory()

    index, decision = inspect_step(
        stopper=stopper,
        history=history,
        step=0,
        scores=torch.tensor([math.nan, 0.25, math.inf]),
        x0_candidates=torch.arange(3, dtype=torch.float32).view(3, 1),
    )

    assert index == 1
    assert decision is None
    assert history.best is not None
    assert history.best.score == 0.25


def test_candidate_selection_rejects_all_non_finite_scores() -> None:
    with pytest.raises(RuntimeError, match="no finite candidate scores"):
        inspect_step(
            stopper=AdaptiveBalancedStopper(max_step=3, min_step=0),
            history=InspectionHistory(),
            step=0,
            scores=torch.tensor([math.nan, math.inf]),
            x0_candidates=torch.zeros(2, 1),
        )
