"""Pure protocol tests for native-Qwen optimizer admission."""

from __future__ import annotations

from copy import deepcopy

import pytest

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.training.native_optimizer_admission import (
    NATIVE_OPTIMIZER_PHASE_A_RATES,
    NativeOptimizerAdmissionProtocol,
    learning_rate_multiplier,
    select_admitted_rate,
    summarize_calibration,
    training_sample_order,
    validate_protocol_record,
)


def test_main_protocol_reconciles_exact_units() -> None:
    protocol = NativeOptimizerAdmissionProtocol()
    assert protocol.unique_train_decisions == 64
    assert protocol.unique_calibration_decisions == 64
    assert protocol.effective_batch == 16
    assert protocol.epochs == 16
    assert protocol.planned_optimizer_updates == 64
    assert protocol.unit_ledger.total_sample_exposures == 1_024
    assert protocol.unit_ledger.supports_learning_quality_claim is False
    assert protocol.to_record()["paper_claim_boundary"] == "numerical_range_finder_only"


def test_stress_protocol_is_one_update_per_arm() -> None:
    protocol = NativeOptimizerAdmissionProtocol.stress()
    assert protocol.unique_train_decisions == 16
    assert protocol.unique_calibration_decisions == 16
    assert protocol.planned_optimizer_updates == 1
    assert protocol.calibration_updates == (0, 1)


def test_protocol_record_round_trip_and_tamper_detection() -> None:
    record = NativeOptimizerAdmissionProtocol().to_record()
    assert validate_protocol_record(record) == []
    tampered = deepcopy(record)
    tampered["unit_ledger"]["derived"]["reconciled_optimizer_updates"] = 4
    assert validate_protocol_record(tampered)


def test_learning_rate_schedule_is_bounded_and_keeps_every_update_nonzero() -> None:
    protocol = NativeOptimizerAdmissionProtocol()
    values = [
        learning_rate_multiplier(step, protocol)
        for step in range(1, protocol.planned_optimizer_updates + 1)
    ]
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values[0] < values[1]
    assert values[1] == 1.0
    assert values[-1] == protocol.minimum_lr_ratio


def test_training_order_covers_every_sample_once_per_epoch() -> None:
    protocol = NativeOptimizerAdmissionProtocol()
    samples = [{"sample_index": value} for value in range(protocol.unique_train_decisions)]
    order = training_sample_order(samples, protocol)
    assert len(order) == 1_024
    for epoch in range(protocol.epochs):
        start = epoch * protocol.unique_train_decisions
        assert set(order[start : start + protocol.unique_train_decisions]) == set(range(64))
    assert order[:64] != order[64:128]


def _calibration_rows(predictions: tuple[str, ...] | None = None) -> list[dict]:
    actions = tuple(action.name for action in R2RCEAction)
    predictions = predictions or actions
    rows = []
    for index, action in enumerate(actions):
        predicted = predictions[index]
        rows.append(
            {
                "target_action": action,
                "target_token_nll": 2.0 + index,
                "restricted_action_ce": 1.0 + index / 10,
                "target_restricted_probability": 0.3,
                "predicted_action": predicted,
                "correct": predicted == action,
            }
        )
    return rows


def test_calibration_summary_is_action_balanced() -> None:
    summary = summarize_calibration(_calibration_rows())
    assert summary["decision_count"] == 4
    assert summary["distinct_predicted_action_count"] == 4
    assert set(summary["per_target_action"]) == {action.name for action in R2RCEAction}


def _arm_result(rate: float, *, improved: bool = True, distinct: int = 4) -> dict:
    baseline = summarize_calibration(_calibration_rows())
    selected = deepcopy(baseline)
    selected["mean_restricted_action_ce"] -= 0.1 if improved else 0.0
    selected["distinct_predicted_action_count"] = distinct
    calibrations = []
    for step in (0, 8, 16, 32, 64):
        calibrations.append(
            {"update_step": step, "summary": baseline if step == 0 else selected}
        )
    return {
        "status": "passed",
        "learning_rate": rate,
        "all_updates_finite": True,
        "changed_update_count": 64,
        "calibrations": calibrations,
    }


def test_selection_chooses_lowest_admitted_rate() -> None:
    protocol = NativeOptimizerAdmissionProtocol()
    results = [_arm_result(rate) for rate in NATIVE_OPTIMIZER_PHASE_A_RATES]
    selection = select_admitted_rate(results, protocol)
    assert selection["status"] == "rate_admitted"
    assert selection["selected_learning_rate"] == NATIVE_OPTIMIZER_PHASE_A_RATES[0]


def test_selection_rejects_unchanged_or_collapsed_arms() -> None:
    protocol = NativeOptimizerAdmissionProtocol()
    results = [
        _arm_result(rate, improved=False, distinct=1)
        for rate in NATIVE_OPTIMIZER_PHASE_A_RATES
    ]
    selection = select_admitted_rate(results, protocol)
    assert selection["status"] == "no_rate_admitted"
    assert selection["selected_learning_rate"] is None


def test_learning_rates_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        NativeOptimizerAdmissionProtocol(learning_rates=(2e-6, 2e-7))
