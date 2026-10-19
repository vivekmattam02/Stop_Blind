"""Pure contracts for the adaptive native-C0 action-balanced loss comparison.

The module deliberately contains no model or simulator code. It reconstructs the proposed
train-only inverse-frequency weights from the immutable development manifest and freezes the
small numerical-stress prefix that must pass before a second full development run is allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION = 1
NATIVE_C0_ACTION_BALANCE_GATE = "native_qwen_c0_action_balance_v1"
NATIVE_C0_ACTION_BALANCE_STRESS_GATE = "native_qwen_c0_action_balance_stress_v1"
NATIVE_C0_ACTION_BALANCE_OBJECTIVE = "class_balanced_restricted_four_action_cross_entropy_v1"
OFFICIAL_ACTIONS = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
TRAIN_DECISIONS = 3_072
ACCUMULATION_STEPS = 16
STRESS_UPDATES = 4
STRESS_EXPOSURES = STRESS_UPDATES * ACCUMULATION_STEPS
SAMPLER_SEED = 20260825
EXPECTED_STRESS_HISTOGRAM = {
    "STOP": 1,
    "MOVE_FORWARD": 44,
    "TURN_LEFT": 10,
    "TURN_RIGHT": 9,
}
EXPECTED_FIRST_STOP_EXPOSURE = 61


def canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value with deterministic serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reconstruct_class_weights(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reconstruct expectation-one inverse-frequency weights from frozen train samples."""
    train = [row for row in samples if row.get("split") == "train"]
    if len(train) != TRAIN_DECISIONS:
        raise ValueError("Action-balance weights require exactly 3,072 frozen train decisions.")
    indices = [int(row["sample_index"]) for row in train]
    if len(indices) != len(set(indices)):
        raise ValueError("Action-balance train sample indices are not unique.")
    histogram = Counter(str(row.get("target_action")) for row in train)
    if set(histogram) != set(OFFICIAL_ACTIONS) or any(
        histogram[action] <= 0 for action in OFFICIAL_ACTIONS
    ):
        raise ValueError("Action-balance training data must contain every official action.")
    weights = {
        action: len(train) / (len(OFFICIAL_ACTIONS) * histogram[action])
        for action in OFFICIAL_ACTIONS
    }
    expected_weight = sum(
        histogram[action] * weights[action] for action in OFFICIAL_ACTIONS
    ) / len(train)
    if not math.isclose(expected_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Action-balance weights do not have empirical expectation one.")
    if any(not math.isfinite(value) or value <= 0.0 for value in weights.values()):
        raise ValueError("Action-balance weights must be finite and positive.")
    record = {
        "formula": "N / (4 * n_a)",
        "normalization": "empirical_expected_sample_weight_equals_one",
        "train_decision_count": len(train),
        "target_action_histogram": {
            action: histogram[action] for action in OFFICIAL_ACTIONS
        },
        "weights": weights,
        "empirical_expected_weight": expected_weight,
    }
    return {**record, "record_sha256": canonical_sha256(record)}


def weighted_restricted_ce(
    unweighted_ce: float,
    *,
    target_action: str,
    weight_record: Mapping[str, Any],
) -> float:
    """Apply the frozen scalar target-action weight to one restricted CE value."""
    value = float(unweighted_ce)
    weights = weight_record.get("weights")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Unweighted restricted CE must be finite and non-negative.")
    if not isinstance(weights, Mapping) or set(weights) != set(OFFICIAL_ACTIONS):
        raise ValueError("Action-balance weight record is incomplete.")
    if target_action not in OFFICIAL_ACTIONS:
        raise ValueError("Target action is not an official action.")
    weight = float(weights[target_action])
    result = value * weight
    if not math.isfinite(weight) or weight <= 0.0 or not math.isfinite(result):
        raise ValueError("Weighted restricted CE is invalid.")
    return result


def stress_prefix(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze the unchanged first four optimizer updates of the original exposure order."""
    train = [row for row in samples if row.get("split") == "train"]
    if len(train) != TRAIN_DECISIONS:
        raise ValueError("Stress prefix requires the exact 3,072-decision training partition.")
    by_index = {int(row["sample_index"]): row for row in train}
    if len(by_index) != len(train):
        raise ValueError("Stress prefix train sample indices are not unique.")
    order = list(by_index)
    random.Random(SAMPLER_SEED).shuffle(order)
    prefix = order[:STRESS_EXPOSURES]
    actions = [str(by_index[index]["target_action"]) for index in prefix]
    histogram = Counter(actions)
    observed_histogram = {action: histogram[action] for action in OFFICIAL_ACTIONS}
    if observed_histogram != EXPECTED_STRESS_HISTOGRAM:
        raise ValueError("Stress prefix action histogram differs from the frozen development run.")
    first_stop = actions.index("STOP")
    if first_stop != EXPECTED_FIRST_STOP_EXPOSURE:
        raise ValueError(
            "Stress prefix first-STOP exposure differs from the frozen development run."
        )
    record = {
        "sampler_seed": SAMPLER_SEED,
        "optimizer_updates": STRESS_UPDATES,
        "accumulation_steps": ACCUMULATION_STEPS,
        "exposure_count": STRESS_EXPOSURES,
        "sample_indices": prefix,
        "target_action_histogram": observed_histogram,
        "first_stop_exposure_zero_based": first_stop,
        "first_stop_optimizer_update_one_based": first_stop // ACCUMULATION_STEPS + 1,
        "first_stop_microbatch_zero_based": first_stop % ACCUMULATION_STEPS,
    }
    return {**record, "record_sha256": canonical_sha256(record)}


def build_action_balance_contract(
    *,
    development_manifest: Mapping[str, Any],
    development_result_sha256: str,
    development_verification_sha256: str,
    failure_analysis: Mapping[str, Any],
    failure_analysis_sha256: str,
    failure_verification: Mapping[str, Any],
    failure_verification_sha256: str,
) -> dict[str, Any]:
    """Bind the one-variable treatment and numerical stress to the verified negative result."""
    if (
        development_manifest.get("gate") != "native_qwen_c0_development_v1"
        or development_manifest.get("status") != "frozen"
    ):
        raise ValueError("Action-balance contract requires the frozen C0 development manifest.")
    if (
        failure_analysis.get("gate") != "native_qwen_c0_failure_analysis_v1"
        or failure_analysis.get("status") != "passed"
    ):
        raise ValueError("Action-balance contract requires the passed failure analysis.")
    if (
        failure_verification.get("gate")
        != "independent_native_qwen_c0_failure_analysis_verification_v1"
        or failure_verification.get("status") != "passed"
        or failure_verification.get("error_count") != 0
        or failure_verification.get("analysis_sha256") != failure_analysis_sha256
    ):
        raise ValueError(
            "Action-balance contract requires linked independent failure verification."
        )
    observations = failure_analysis.get("confirmed_descriptive_observations")
    if not isinstance(observations, Mapping) or (
        observations.get("balanced_missing_predicted_stop") is not True
        and observations.get("balanced_stop_never_argmax") is not True
    ):
        raise ValueError("Observed failure does not activate the action-balance intervention.")
    weights = reconstruct_class_weights(development_manifest.get("samples", []))
    prefix = stress_prefix(development_manifest.get("samples", []))
    return {
        "schema_version": NATIVE_C0_ACTION_BALANCE_SCHEMA_VERSION,
        "gate": NATIVE_C0_ACTION_BALANCE_GATE,
        "status": "blocked_pending_numerical_stress",
        "objective": NATIVE_C0_ACTION_BALANCE_OBJECTIVE,
        "changed_variable": "train_only_target_action_loss_weight",
        "unchanged_contract": {
            "samples_and_exposure_order": True,
            "fresh_model_initialization_and_seed": True,
            "history_and_cache_objects": True,
            "optimizer_learning_rate_and_gradient_clip": True,
            "update_horizon_and_checkpoint_cadence": True,
            "selection_rule_and_evaluation_partitions": True,
            "outcome_thresholds": True,
        },
        "weights": weights,
        "numerical_stress": {
            **prefix,
            "fresh_model_initialization": True,
            "scientific_result_emitted": False,
            "full_run_authorized_if_passed": True,
        },
        "input_evidence": {
            "development_manifest_records_sha256": development_manifest.get(
                "sample_records_sha256"
            ),
            "development_result_sha256": development_result_sha256,
            "development_verification_sha256": development_verification_sha256,
            "failure_analysis_sha256": failure_analysis_sha256,
            "failure_verification_sha256": failure_verification_sha256,
        },
        "evidence_boundary": {
            "stress_pass_establishes": (
                "The exact weighted objective executes four fresh-model updates through the first "
                "high-weight STOP exposure with finite gradients, clipping, optimizer state, and "
                "trainable changes."
            ),
            "stress_pass_does_not_establish": (
                "Learning quality, superiority of class weighting, action support, navigation "
                "competence, or any C1-C4 claim."
            ),
        },
    }


def verify_stress_records(
    *,
    contract: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    initial_trainable_sha256: str,
    final_trainable_sha256: str,
) -> list[str]:
    """Independently check the four-update numerical-stress event chain."""
    errors: list[str] = []
    stress = contract.get("numerical_stress")
    weights_record = contract.get("weights")
    if not isinstance(stress, Mapping) or not isinstance(weights_record, Mapping):
        return ["action-balance contract lacks stress or weight records"]
    expected_indices = [int(value) for value in stress.get("sample_indices", [])]
    if len(expected_indices) != STRESS_EXPOSURES:
        return ["action-balance stress prefix has the wrong exposure count"]
    train_by_index = {
        int(row["sample_index"]): row for row in samples if row.get("split") == "train"
    }
    if len(events) != STRESS_UPDATES:
        return ["action-balance stress update-event count mismatch"]
    previous_sha = initial_trainable_sha256
    observed_actions: list[str] = []
    for expected_step, event in enumerate(events, start=1):
        label = f"stress update {expected_step}"
        start = (expected_step - 1) * ACCUMULATION_STEPS
        end = start + ACCUMULATION_STEPS
        if (
            event.get("gate") != NATIVE_C0_ACTION_BALANCE_STRESS_GATE
            or event.get("objective") != NATIVE_C0_ACTION_BALANCE_OBJECTIVE
            or event.get("update_step") != expected_step
            or event.get("sample_exposure_start") != start
            or event.get("sample_exposure_end_exclusive") != end
            or event.get("before_trainable_state_sha256") != previous_sha
        ):
            errors.append(f"{label} identity/exposure/hash-chain mismatch")
        microbatches = event.get("microbatches")
        if not isinstance(microbatches, list) or len(microbatches) != ACCUMULATION_STEPS:
            errors.append(f"{label} microbatch count mismatch")
            continue
        expected_slice = expected_indices[start:end]
        unweighted: list[float] = []
        weighted: list[float] = []
        for offset, (row, sample_index) in enumerate(zip(microbatches, expected_slice)):
            expected = train_by_index.get(sample_index)
            if expected is None:
                errors.append(f"{label} references an unknown training sample")
                continue
            target = str(expected["target_action"])
            if (
                row.get("microbatch_index") != offset
                or row.get("sample_index") != sample_index
                or row.get("target_action") != target
            ):
                errors.append(f"{label} microbatch identity/order mismatch")
            try:
                observed_unweighted = float(row["unweighted_restricted_action_ce"])
                observed_weighted = float(row["weighted_restricted_action_ce"])
                observed_weight = float(row["target_action_weight"])
                expected_weight = float(weights_record["weights"][target])
                reconstructed = weighted_restricted_ce(
                    observed_unweighted,
                    target_action=target,
                    weight_record=weights_record,
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{label} invalid weighted-loss record: {exc}")
                continue
            if not math.isclose(observed_weight, expected_weight, rel_tol=0.0, abs_tol=0.0):
                errors.append(f"{label} target-action weight mismatch")
            if not math.isclose(
                observed_weighted, reconstructed, rel_tol=2e-6, abs_tol=2e-6
            ):
                errors.append(f"{label} weighted CE does not reconstruct")
            unweighted.append(observed_unweighted)
            weighted.append(observed_weighted)
            observed_actions.append(target)
        if unweighted and not math.isclose(
            float(event.get("mean_unweighted_restricted_action_ce", math.nan)),
            sum(unweighted) / len(unweighted),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(f"{label} mean unweighted CE mismatch")
        if weighted and not math.isclose(
            float(event.get("mean_weighted_restricted_action_ce", math.nan)),
            sum(weighted) / len(weighted),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(f"{label} mean weighted CE mismatch")
        before = event.get("gradient_report_before_clip")
        after = event.get("gradient_report_after_clip")
        clip_max = float(event.get("clip_max_norm", math.nan))
        preclip = float(event.get("global_gradient_norm_before_clip", math.nan))
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            errors.append(f"{label} gradient reports are missing")
        else:
            before_norm = float(before.get("gradient_l2_norm", math.nan))
            after_norm = float(after.get("gradient_l2_norm", math.nan))
            if (
                before.get("all_gradients_finite") is not True
                or after.get("all_gradients_finite") is not True
                or not math.isfinite(preclip)
                or preclip <= 0.0
                or not math.isclose(preclip, before_norm, rel_tol=2e-5, abs_tol=2e-5)
                or not math.isfinite(after_norm)
                or after_norm <= 0.0
                or not math.isfinite(clip_max)
                or clip_max != 1.0
                or after_norm > clip_max + 2e-4
            ):
                errors.append(f"{label} gradient/clipping contract mismatch")
        optimizer_state = event.get("optimizer_state")
        if not isinstance(optimizer_state, Mapping) or any(
            not isinstance(value, Mapping) or value.get("all_finite") is not True
            for value in optimizer_state.values()
        ):
            errors.append(f"{label} optimizer state is missing or non-finite")
        after_sha = event.get("after_trainable_state_sha256")
        if not isinstance(after_sha, str) or after_sha == previous_sha:
            errors.append(f"{label} did not change the trainable-state hash")
        else:
            previous_sha = after_sha
    if previous_sha != final_trainable_sha256:
        errors.append("action-balance stress final trainable hash does not close the chain")
    histogram = Counter(observed_actions)
    if {action: histogram[action] for action in OFFICIAL_ACTIONS} != EXPECTED_STRESS_HISTOGRAM:
        errors.append("action-balance stress did not execute the frozen four-action prefix")
    return errors
