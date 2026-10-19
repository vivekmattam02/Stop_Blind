"""Frozen full-run contracts for the adaptive native-C0 action-balance comparison."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from spatialvlm.training.native_c0_action_balance import (
    NATIVE_C0_ACTION_BALANCE_GATE,
    NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
    NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
    OFFICIAL_ACTIONS,
    reconstruct_class_weights,
    weighted_restricted_ce,
)
from spatialvlm.training.native_c0_development import (
    NATIVE_C0_HISTORY,
    NativeC0DevelopmentProtocol,
    training_sample_order,
)
from spatialvlm.training.unit_ledger import verify_unit_ledger_record

NATIVE_C0_ACTION_BALANCE_VERIFIER_GATE = "independent_native_qwen_c0_action_balance_verification_v1"


class NativeC0ActionBalanceProtocol(NativeC0DevelopmentProtocol):
    """The original C0 protocol with only its train-time loss objective changed."""

    def to_record(self) -> dict[str, Any]:
        record = super().to_record()
        record.update(
            {
                "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
                "gate": NATIVE_C0_ACTION_BALANCE_GATE,
                "objective": NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
                "claim_boundary": "adaptive_train_release_action_balance_comparison_only",
            }
        )
        ledger = dict(record["unit_ledger"])
        declared = dict(ledger["declared"])
        declared["notes"] = (
            "Adaptive native-Qwen C0 action-balance comparison. Only the train-time target-action "
            "loss weight differs from the original C0 development run; selection and outcomes "
            "retain their original trajectory-disjoint surfaces and rules."
        )
        ledger["declared"] = declared
        record["unit_ledger"] = ledger
        return record


def validate_action_balance_protocol_record(record: Mapping[str, Any]) -> list[str]:
    """Independently rebuild the action-balanced protocol and its optimizer unit ledger."""
    errors: list[str] = []
    try:
        protocol = NativeC0ActionBalanceProtocol(
            train_decisions=int(record["train_decisions"]),
            train_trajectories=int(record["train_trajectories"]),
            selection_decisions=int(record["selection_decisions"]),
            primary_evaluation_decisions=int(record["primary_evaluation_decisions"]),
            diagnostic_decisions_per_action=int(record["diagnostic_decisions_per_action"]),
            microbatch_size=int(record["microbatch_size"]),
            accumulation_steps=int(record["accumulation_steps"]),
            epochs=int(record["epochs"]),
            learning_rate=float(record["learning_rate"]),
            weight_decay=float(record["weight_decay"]),
            max_grad_norm=float(record["max_grad_norm"]),
            hard_max_gradient_norm=float(record["hard_max_gradient_norm"]),
            hard_max_restricted_ce=float(record["hard_max_restricted_ce"]),
            calibration_updates=tuple(int(value) for value in record["calibration_updates"]),
            checkpoint_updates=tuple(int(value) for value in record["checkpoint_updates"]),
            natural_blind_prior_ce_nats=float(record["natural_blind_prior_ce_nats"]),
            balanced_uniform_ce_nats=float(record["balanced_uniform_ce_nats"]),
            required_margin_nats=float(record["required_margin_nats"]),
            stop_recall_floor=float(record["stop_recall_floor"]),
            bootstrap_samples=int(record["bootstrap_samples"]),
            bootstrap_seed=int(record["bootstrap_seed"]),
            sampler_seed=int(record["sampler_seed"]),
            model_seed=int(record["model_seed"]),
            min_decision_step=int(record["min_decision_step"]),
            max_replay_prefix_actions=int(record["max_replay_prefix_actions"]),
            hard_total_hours=float(record["hard_total_hours"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [f"Action-balance protocol cannot reconstruct: {exc}"]
    if dict(record) != protocol.to_record():
        errors.append("Action-balance protocol differs from deterministic reconstruction")
    ledger = record.get("unit_ledger")
    if not isinstance(ledger, dict):
        errors.append("Action-balance unit ledger is missing")
    else:
        errors.extend(verify_unit_ledger_record(ledger))
    return errors


def weighted_contribution_summary(
    microbatches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the exact weighted objective contribution by target action."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in microbatches:
        grouped[str(row["target_action"])].append(float(row["weighted_restricted_action_ce"]))
    return {
        action: {
            "count": len(grouped[action]),
            "sum_weighted_restricted_action_ce": sum(grouped[action]),
            "mean_weighted_restricted_action_ce": (
                sum(grouped[action]) / len(grouped[action]) if grouped[action] else None
            ),
        }
        for action in OFFICIAL_ACTIONS
    }


def verify_action_balance_training_records(
    *,
    contract: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    protocol: NativeC0ActionBalanceProtocol,
    initial_trainable_sha256: str,
    final_trainable_sha256: str,
) -> list[str]:
    """Reconstruct all 192 weighted updates without trusting the producing runner."""
    errors: list[str] = []
    weights_record = contract.get("weights")
    if not isinstance(weights_record, Mapping):
        return ["action-balance contract lacks its weight record"]
    try:
        reconstructed_weights = reconstruct_class_weights(samples)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"action-balance weights cannot reconstruct: {exc}"]
    if dict(weights_record) != reconstructed_weights:
        errors.append("action-balance weight record differs from frozen training samples")
    if len(events) != protocol.planned_optimizer_updates:
        return ["action-balance update-event count differs from the unit ledger"]

    train = [row for row in samples if row.get("split") == "train"]
    by_index = {int(row["sample_index"]): row for row in train}
    order = training_sample_order(samples, protocol)
    previous = initial_trainable_sha256
    observed_actions: Counter[str] = Counter()
    for expected_step, event in enumerate(events, start=1):
        label = f"action-balance update {expected_step}"
        start = (expected_step - 1) * protocol.accumulation_steps
        end = start + protocol.accumulation_steps
        if (
            event.get("schema_version") != NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION
            or event.get("gate") != NATIVE_C0_ACTION_BALANCE_GATE
            or event.get("history_contract") != NATIVE_C0_HISTORY
            or event.get("objective") != NATIVE_C0_ACTION_BALANCE_OBJECTIVE
            or event.get("update_step") != expected_step
        ):
            errors.append(f"{label} identity mismatch")
        if (
            event.get("sample_exposure_start") != start
            or event.get("sample_exposure_end_exclusive") != end
            or event.get("microbatch_count") != protocol.accumulation_steps
            or event.get("accumulation_steps") != protocol.accumulation_steps
        ):
            errors.append(f"{label} exposure units mismatch")
        if event.get("before_trainable_state_sha256") != previous:
            errors.append(f"{label} trainable hash chain broke")
        after = event.get("after_trainable_state_sha256")
        if not isinstance(after, str) or len(after) != 64 or after == previous:
            errors.append(f"{label} after-update trainable hash is invalid")
        else:
            previous = after
        if float(event.get("learning_rate", math.nan)) != protocol.learning_rate:
            errors.append(f"{label} learning-rate mismatch")
        if not math.isfinite(float(event.get("elapsed_seconds", math.nan))):
            errors.append(f"{label} elapsed time is invalid")

        gradients_before = event.get("gradient_report_before_clip")
        gradients_after = event.get("gradient_report_after_clip")
        preclip = float(event.get("global_gradient_norm_before_clip", math.nan))
        clip_max = float(event.get("clip_max_norm", math.nan))
        if not isinstance(gradients_before, Mapping) or not isinstance(gradients_after, Mapping):
            errors.append(f"{label} gradient reports are missing")
        else:
            before_norm = float(gradients_before.get("gradient_l2_norm", math.nan))
            after_norm = float(gradients_after.get("gradient_l2_norm", math.nan))
            if (
                gradients_before.get("all_gradients_finite") is not True
                or gradients_before.get("gradient_tensor_count")
                != gradients_before.get("trainable_tensor_count")
                or gradients_before.get("nonzero_gradient_tensor_count", 0) <= 0
                or gradients_after.get("all_gradients_finite") is not True
                or not math.isfinite(preclip)
                or preclip <= 0.0
                or not math.isclose(preclip, before_norm, rel_tol=2e-5, abs_tol=2e-5)
                or not math.isfinite(after_norm)
                or after_norm <= 0.0
                or clip_max != protocol.max_grad_norm
                or after_norm > clip_max + 2e-4
            ):
                errors.append(f"{label} gradient/clipping contract mismatch")
        updates = event.get("update_report")
        if not isinstance(updates, Mapping) or (
            updates.get("changed_tensor_count", 0) <= 0
            or updates.get("all_updates_finite") is not True
        ):
            errors.append(f"{label} update report is invalid")
        optimizer_state = event.get("optimizer_state")
        if not isinstance(optimizer_state, Mapping) or any(
            not isinstance(value, Mapping) or value.get("all_finite") is not True
            for value in optimizer_state.values()
        ):
            errors.append(f"{label} optimizer state is missing or non-finite")

        microbatches = event.get("microbatches")
        if not isinstance(microbatches, list) or len(microbatches) != protocol.accumulation_steps:
            errors.append(f"{label} microbatch ledger mismatch")
            continue
        if [row.get("sample_index") for row in microbatches] != list(order[start:end]):
            errors.append(f"{label} sample order mismatch")
        unweighted_values: list[float] = []
        weighted_values: list[float] = []
        for offset, row in enumerate(microbatches):
            sample = by_index.get(row.get("sample_index"))
            if sample is None:
                errors.append(f"{label} references unknown training sample")
                continue
            target = str(sample["target_action"])
            if row.get("microbatch_index") != offset or any(
                row.get(name) != sample[name]
                for name in ("episode_id", "trajectory_id", "scene_id", "target_action")
            ):
                errors.append(f"{label} sample identity/order mismatch")
            if row.get("supervised_token_count") != 1 or not isinstance(
                row.get("supervised_token_id"), int
            ):
                errors.append(f"{label} supervision contract mismatch")
            try:
                unweighted = float(row["unweighted_restricted_action_ce"])
                weighted = float(row["weighted_restricted_action_ce"])
                observed_weight = float(row["target_action_weight"])
                expected_weight = float(weights_record["weights"][target])
                reconstructed = weighted_restricted_ce(
                    unweighted,
                    target_action=target,
                    weight_record=weights_record,
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{label} invalid weighted-loss record: {exc}")
                continue
            if unweighted > protocol.hard_max_restricted_ce:
                errors.append(f"{label} unweighted CE exceeds the frozen ceiling")
            if not math.isclose(observed_weight, expected_weight, rel_tol=0.0, abs_tol=0.0):
                errors.append(f"{label} target-action weight mismatch")
            if not math.isclose(weighted, reconstructed, rel_tol=2e-6, abs_tol=2e-6):
                errors.append(f"{label} weighted CE does not reconstruct")
            unweighted_values.append(unweighted)
            weighted_values.append(weighted)
            observed_actions[target] += 1
        if unweighted_values and not math.isclose(
            float(event.get("mean_unweighted_restricted_action_ce", math.nan)),
            sum(unweighted_values) / len(unweighted_values),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(f"{label} mean unweighted CE mismatch")
        if weighted_values and not math.isclose(
            float(event.get("mean_weighted_restricted_action_ce", math.nan)),
            sum(weighted_values) / len(weighted_values),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(f"{label} mean weighted CE mismatch")
        if event.get("per_action_weighted_contributions") != weighted_contribution_summary(
            microbatches
        ):
            errors.append(f"{label} per-action weighted contributions do not reconstruct")

    if previous != final_trainable_sha256:
        errors.append("action-balance final training hash does not close the update chain")
    expected_histogram = reconstructed_weights["target_action_histogram"]
    if {action: observed_actions[action] for action in OFFICIAL_ACTIONS} != expected_histogram:
        errors.append("action-balance run did not execute the complete frozen training surface")
    return errors
