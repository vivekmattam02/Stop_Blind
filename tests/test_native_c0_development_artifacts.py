from __future__ import annotations

import math

import pytest

from scripts.run_spark_native_c0_development import summarize_decisions
from spatialvlm.training.native_c0_development import evaluate_existence_gate


def _rows(split: str, count: int, *, ce: float, stop_correct: int = 64):
    actions = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
    rows = []
    seen_stop = 0
    for index in range(count):
        target = actions[index % 4]
        predicted = target
        if target == "STOP":
            predicted = "STOP" if seen_stop < stop_correct else "MOVE_FORWARD"
            seen_stop += 1
        rows.append(
            {
                "split": split,
                "sample_index": index,
                "episode_id": str(index),
                "trajectory_id": f"trajectory-{index}",
                "scene_id": f"scene-{index % 8}",
                "target_action": target,
                "predicted_action": predicted,
                "correct": predicted == target,
                "target_token_nll": ce,
                "target_restricted_probability": math.exp(-ce),
                "restricted_action_ce": ce,
                "supervised_token_count": 1,
                "supervised_token_id": index % 4,
            }
        )
    return rows


def test_decision_summary_preserves_natural_missing_action_surface() -> None:
    rows = _rows("selection", 8, ce=0.5)
    rows = [row for row in rows if row["target_action"] != "STOP"]
    summary = summarize_decisions(rows)
    assert summary["decision_count"] == len(rows)
    assert summary["per_target_action"]["STOP"] == {
        "count": 0,
        "mean_restricted_action_ce": None,
        "recall": None,
    }


def test_completed_not_demonstrated_is_distinct_from_execution_failure() -> None:
    primary = _rows("primary_evaluation", 640, ce=1.0)
    diagnostic = _rows("balanced_diagnostic", 256, ce=1.36, stop_correct=64)
    outcome = evaluate_existence_gate(
        primary_decisions=primary,
        diagnostic_decisions=diagnostic,
    )
    assert outcome["status"] == "not_demonstrated_at_frozen_budget"
    assert outcome["conditions"]["natural_upper_95_below_blind_prior_minus_margin"] is False


def test_decision_summary_rejects_nonfinite_metrics() -> None:
    rows = _rows("selection", 4, ce=0.5)
    rows[0]["restricted_action_ce"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        summarize_decisions(rows)
