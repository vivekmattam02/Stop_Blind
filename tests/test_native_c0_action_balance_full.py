from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from spatialvlm.training.native_c0_action_balance import (
    NATIVE_C0_ACTION_BALANCE_GATE,
    NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
    reconstruct_class_weights,
    weighted_restricted_ce,
)
from spatialvlm.training.native_c0_action_balance_full import (
    NativeC0ActionBalanceProtocol,
    validate_action_balance_protocol_record,
    verify_action_balance_training_records,
    weighted_contribution_summary,
)
from spatialvlm.training.native_c0_development import (
    NativeC0DevelopmentProtocol,
    training_sample_order,
)


def test_protocol_changes_only_the_declared_training_objective_and_boundary() -> None:
    original = NativeC0DevelopmentProtocol().to_record()
    treatment = NativeC0ActionBalanceProtocol().to_record()
    assert treatment["gate"] == NATIVE_C0_ACTION_BALANCE_GATE
    assert treatment["objective"] == NATIVE_C0_ACTION_BALANCE_OBJECTIVE
    assert treatment["claim_boundary"] == "adaptive_train_release_action_balance_comparison_only"
    assert (
        "Only the train-time target-action loss weight differs"
        in treatment["unit_ledger"]["declared"]["notes"]
    )
    ignored = {"schema_version", "gate", "objective", "claim_boundary", "unit_ledger"}
    assert {key: value for key, value in treatment.items() if key not in ignored} == {
        key: value for key, value in original.items() if key not in ignored
    }
    assert validate_action_balance_protocol_record(treatment) == []


def _real_samples() -> list[dict]:
    path = (
        Path(__file__).parents[1]
        / "artifacts/spark/native_c0_action_balance_stress_v1/manifest.json"
    )
    if not path.is_file():
        pytest.skip("mirrored action-balance stress manifest is unavailable")
    return json.loads(path.read_text(encoding="utf-8"))["samples"]


def _sha(step: int) -> str:
    return hashlib.sha256(f"state-{step}".encode()).hexdigest()


def _events(samples: list[dict]) -> tuple[dict, list[dict]]:
    protocol = NativeC0ActionBalanceProtocol()
    weights = reconstruct_class_weights(samples)
    by_index = {int(row["sample_index"]): row for row in samples if row["split"] == "train"}
    order = training_sample_order(samples, protocol)
    events = []
    for step in range(1, protocol.planned_optimizer_updates + 1):
        start = (step - 1) * protocol.accumulation_steps
        microbatches = []
        for offset, sample_index in enumerate(order[start : start + protocol.accumulation_steps]):
            sample = by_index[sample_index]
            target = sample["target_action"]
            unweighted = 0.5 + offset / 100.0
            microbatches.append(
                {
                    "microbatch_index": offset,
                    "sample_index": sample_index,
                    "episode_id": sample["episode_id"],
                    "trajectory_id": sample["trajectory_id"],
                    "scene_id": sample["scene_id"],
                    "target_action": target,
                    "target_action_weight": weights["weights"][target],
                    "unweighted_restricted_action_ce": unweighted,
                    "weighted_restricted_action_ce": weighted_restricted_ce(
                        unweighted,
                        target_action=target,
                        weight_record=weights,
                    ),
                    "supervised_token_count": 1,
                    "supervised_token_id": 10 + offset,
                }
            )
        events.append(
            {
                "schema_version": 1,
                "gate": NATIVE_C0_ACTION_BALANCE_GATE,
                "history_contract": "H4_CAUSAL",
                "objective": NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
                "update_step": step,
                "sample_exposure_start": start,
                "sample_exposure_end_exclusive": start + 16,
                "microbatch_count": 16,
                "accumulation_steps": 16,
                "learning_rate": 2e-5,
                "mean_unweighted_restricted_action_ce": sum(
                    row["unweighted_restricted_action_ce"] for row in microbatches
                )
                / 16,
                "mean_weighted_restricted_action_ce": sum(
                    row["weighted_restricted_action_ce"] for row in microbatches
                )
                / 16,
                "per_action_weighted_contributions": weighted_contribution_summary(microbatches),
                "gradient_report_before_clip": {
                    "gradient_tensor_count": 288,
                    "trainable_tensor_count": 288,
                    "nonzero_gradient_tensor_count": 144,
                    "all_gradients_finite": True,
                    "gradient_l2_norm": 2.0,
                },
                "global_gradient_norm_before_clip": 2.0,
                "gradient_report_after_clip": {
                    "all_gradients_finite": True,
                    "gradient_l2_norm": 0.99995,
                },
                "clip_max_norm": 1.0,
                "update_report": {"changed_tensor_count": 17, "all_updates_finite": True},
                "optimizer_state": {
                    "exp_avg": {"all_finite": True},
                    "exp_avg_sq": {"all_finite": True},
                    "step": {"all_finite": True},
                },
                "before_trainable_state_sha256": _sha(step - 1),
                "after_trainable_state_sha256": _sha(step),
                "microbatches": microbatches,
                "elapsed_seconds": 1.0,
            }
        )
    return {"weights": weights}, events


def test_full_event_verifier_reconstructs_all_weighted_updates_and_detects_tamper() -> None:
    samples = _real_samples()
    contract, events = _events(samples)
    protocol = NativeC0ActionBalanceProtocol()
    assert (
        verify_action_balance_training_records(
            contract=contract,
            events=events,
            samples=samples,
            protocol=protocol,
            initial_trainable_sha256=_sha(0),
            final_trainable_sha256=_sha(192),
        )
        == []
    )
    broken = copy.deepcopy(events)
    broken[-1]["microbatches"][-1]["weighted_restricted_action_ce"] += 1.0
    errors = verify_action_balance_training_records(
        contract=contract,
        events=broken,
        samples=samples,
        protocol=protocol,
        initial_trainable_sha256=_sha(0),
        final_trainable_sha256=_sha(192),
    )
    assert any("weighted CE does not reconstruct" in error for error in errors)


def test_protocol_validation_rejects_objective_drift() -> None:
    record = NativeC0ActionBalanceProtocol().to_record()
    record["objective"] = "unweighted"
    assert "differs from deterministic reconstruction" in " ".join(
        validate_action_balance_protocol_record(record)
    )
