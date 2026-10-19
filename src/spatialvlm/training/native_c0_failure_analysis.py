"""Pure, deterministic analysis of a verified native-C0 development failure.

The analysis consumes only immutable JSON evidence produced by the development runner.  It does
not load a model, reopen an evaluation split, or infer a causal failure mechanism.  The output is
therefore a descriptive bridge between the frozen existence-test result and a later, separately
preregistered intervention.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

NATIVE_C0_FAILURE_ANALYSIS_SCHEMA_VERSION = 1
NATIVE_C0_FAILURE_ANALYSIS_GATE = "native_qwen_c0_failure_analysis_v1"
NATIVE_C0_DEVELOPMENT_GATE = "native_qwen_c0_development_v1"
NATIVE_C0_DEVELOPMENT_VERIFIER_GATE = (
    "independent_native_qwen_c0_development_verification_v1"
)
OFFICIAL_ACTIONS = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute a mean from an empty sequence.")
    return sum(values) / len(values)


def _probabilities(row: Mapping[str, Any], label: str) -> dict[str, float]:
    decision = row.get("action_decision")
    if not isinstance(decision, Mapping):
        raise ValueError(f"{label} has no action-decision record")
    scores = decision.get("scores")
    if not isinstance(scores, Mapping) or set(scores) != set(OFFICIAL_ACTIONS):
        raise ValueError(f"{label} does not contain exactly four official-action scores")
    probabilities: dict[str, float] = {}
    for action in OFFICIAL_ACTIONS:
        score = scores[action]
        if not isinstance(score, Mapping):
            raise ValueError(f"{label} {action} score is not a mapping")
        probability = _finite_float(
            score.get("restricted_probability"), f"{label} {action} probability"
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{label} {action} probability lies outside [0, 1]")
        probabilities[action] = probability
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=2e-6):
        raise ValueError(f"{label} restricted probabilities do not sum to one")
    return probabilities


def analyze_partition(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Reconstruct per-action descriptive metrics for one frozen evaluation partition."""
    rows = record.get("per_decision")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} per-decision evidence is missing")

    confusion = {
        target: {predicted: 0 for predicted in OFFICIAL_ACTIONS} for target in OFFICIAL_ACTIONS
    }
    target_rows: dict[str, list[dict[str, Any]]] = {action: [] for action in OFFICIAL_ACTIONS}
    predicted = Counter()
    entropies: list[float] = []
    margins: list[float] = []
    all_ce: list[float] = []
    all_target_probability: list[float] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label} decision {index} is not an object")
        target = str(row.get("target_action"))
        prediction = str(row.get("predicted_action"))
        if target not in OFFICIAL_ACTIONS or prediction not in OFFICIAL_ACTIONS:
            raise ValueError(f"{label} decision {index} has an unofficial action")
        probabilities = _probabilities(row, f"{label} decision {index}")
        target_probability = _finite_float(
            row.get("target_restricted_probability"),
            f"{label} decision {index} target probability",
        )
        ce = _finite_float(row.get("restricted_action_ce"), f"{label} decision {index} CE")
        if target_probability <= 0.0 or not math.isclose(
            ce, -math.log(target_probability), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{label} decision {index} CE/probability linkage is invalid")
        if not math.isclose(
            target_probability, probabilities[target], rel_tol=0.0, abs_tol=2e-6
        ):
            raise ValueError(f"{label} decision {index} target probability is inconsistent")
        correct = prediction == target
        if row.get("correct") is not correct:
            raise ValueError(f"{label} decision {index} correctness flag is inconsistent")
        decision = row["action_decision"]
        entropy = _finite_float(
            decision.get("restricted_entropy_nats"), f"{label} decision {index} entropy"
        )
        margin = _finite_float(
            decision.get("top_two_margin_nats"), f"{label} decision {index} margin"
        )
        row_with_probabilities = {**row, "_probabilities": probabilities}
        target_rows[target].append(row_with_probabilities)
        confusion[target][prediction] += 1
        predicted[prediction] += 1
        entropies.append(entropy)
        margins.append(margin)
        all_ce.append(ce)
        all_target_probability.append(target_probability)

    per_target: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    for action in OFFICIAL_ACTIONS:
        action_rows = target_rows[action]
        if not action_rows:
            per_target[action] = {
                "count": 0,
                "correct": 0,
                "recall": None,
                "mean_restricted_action_ce": None,
                "median_restricted_action_ce": None,
                "mean_target_restricted_probability": None,
                "mean_predicted_distribution": None,
            }
            continue
        correct_count = confusion[action][action]
        recall = correct_count / len(action_rows)
        recalls.append(recall)
        action_ce = [float(row["restricted_action_ce"]) for row in action_rows]
        action_target_probability = [
            float(row["target_restricted_probability"]) for row in action_rows
        ]
        per_target[action] = {
            "count": len(action_rows),
            "correct": correct_count,
            "recall": recall,
            "mean_restricted_action_ce": _mean(action_ce),
            "median_restricted_action_ce": statistics.median(action_ce),
            "mean_target_restricted_probability": _mean(action_target_probability),
            "mean_predicted_distribution": {
                candidate: _mean(
                    [row["_probabilities"][candidate] for row in action_rows]
                )
                for candidate in OFFICIAL_ACTIONS
            },
        }

    correct_count = sum(confusion[action][action] for action in OFFICIAL_ACTIONS)
    return {
        "decision_count": len(rows),
        "scene_count": len({str(row["scene_id"]) for row in rows}),
        "target_action_histogram": {
            action: len(target_rows[action]) for action in OFFICIAL_ACTIONS
        },
        "predicted_action_histogram": {
            action: predicted.get(action, 0) for action in OFFICIAL_ACTIONS
        },
        "confusion_matrix_by_target": confusion,
        "descriptive_accuracy": correct_count / len(rows),
        "macro_recall_over_present_actions": _mean(recalls),
        "distinct_predicted_action_count": sum(
            predicted.get(action, 0) > 0 for action in OFFICIAL_ACTIONS
        ),
        "mean_restricted_action_ce": _mean(all_ce),
        "mean_target_restricted_probability": _mean(all_target_probability),
        "mean_restricted_entropy_nats": _mean(entropies),
        "mean_top_two_margin_nats": _mean(margins),
        "per_target_action": per_target,
    }


def _training_inventory(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in samples if row.get("split") == "train"]
    if not rows:
        raise ValueError("Development manifest contains no training samples.")
    histogram = Counter(str(row.get("target_action")) for row in rows)
    if set(histogram) != set(OFFICIAL_ACTIONS) or any(
        histogram[action] <= 0 for action in OFFICIAL_ACTIONS
    ):
        raise ValueError("Training samples do not contain every official action.")
    count = len(rows)
    weights = {
        action: count / (len(OFFICIAL_ACTIONS) * histogram[action])
        for action in OFFICIAL_ACTIONS
    }
    expected_weight = (
        sum(histogram[action] * weights[action] for action in OFFICIAL_ACTIONS) / count
    )
    if not math.isclose(expected_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Inverse-frequency training weights do not normalize to expectation one.")
    return {
        "decision_count": count,
        "physical_trajectory_count": len({str(row["trajectory_id"]) for row in rows}),
        "target_action_histogram": {action: histogram[action] for action in OFFICIAL_ACTIONS},
        "inverse_frequency_weights_normalized_to_expected_one": weights,
        "weighting_status": "descriptive_candidate_only_not_an_authorized_intervention",
    }


def build_failure_analysis(
    *,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    verification: Mapping[str, Any],
    manifest_sha256: str,
    result_sha256: str,
    updates_sha256: str,
    verification_sha256: str,
    selected_checkpoint: Mapping[str, Any],
    source_locks: Mapping[str, str],
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Build the exact JSON-safe diagnostic from already-opened immutable evidence."""
    if manifest.get("gate") != NATIVE_C0_DEVELOPMENT_GATE or manifest.get("status") != "frozen":
        raise ValueError("C0 development manifest gate/status is invalid.")
    if (
        result.get("gate") != NATIVE_C0_DEVELOPMENT_GATE
        or result.get("status") != "not_demonstrated"
        or result.get("manifest_sha256") != manifest_sha256
        or result.get("updates_sha256") != updates_sha256
    ):
        raise ValueError("C0 result is not the linked completed not-demonstrated result.")
    if (
        verification.get("gate") != NATIVE_C0_DEVELOPMENT_VERIFIER_GATE
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("errors") != []
        or verification.get("manifest_sha256") != manifest_sha256
        or verification.get("result_sha256") != result_sha256
        or verification.get("updates_sha256") != updates_sha256
        or verification.get("dataset_reconstruction_performed") is not True
        or verification.get("outcome_reconstruction_performed") is not True
        or verification.get("all_checkpoint_files_fully_loaded_and_hashed") is not True
    ):
        raise ValueError("C0 independent verification is missing, failed, or unlinked.")
    selection = result.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(selection.get("checkpoint"), Mapping):
        raise ValueError("C0 selected-checkpoint record is missing.")
    checkpoint_record = selection["checkpoint"]
    if (
        selected_checkpoint.get("sha256") != checkpoint_record.get("sha256")
        or selected_checkpoint.get("bytes") != checkpoint_record.get("bytes")
        or selected_checkpoint.get("update_step") != selection.get("selected_update")
    ):
        raise ValueError("Mirrored selected checkpoint does not match the result record.")

    primary = analyze_partition(result.get("primary_evaluation", {}), "primary evaluation")
    balanced = analyze_partition(result.get("balanced_diagnostic", {}), "balanced diagnostic")
    calibrations = result.get("calibrations")
    if not isinstance(calibrations, list) or not calibrations:
        raise ValueError("C0 selection calibrations are missing.")
    calibration_trace = []
    for calibration in calibrations:
        if not isinstance(calibration, Mapping):
            raise ValueError("C0 selection calibration is not an object.")
        summary = analyze_partition(calibration, "selection calibration")
        calibration_trace.append(
            {
                "update_step": int(calibration["update_step"]),
                "mean_restricted_action_ce": summary["mean_restricted_action_ce"],
                "descriptive_accuracy": summary["descriptive_accuracy"],
                "macro_recall_over_present_actions": summary[
                    "macro_recall_over_present_actions"
                ],
                "predicted_action_histogram": summary["predicted_action_histogram"],
                "per_target_action": summary["per_target_action"],
            }
        )

    gate = result.get("existence_gate")
    if not isinstance(gate, Mapping) or gate.get("status") != "not_demonstrated_at_frozen_budget":
        raise ValueError("C0 existence-gate failure record is missing.")
    conditions = gate.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("C0 existence-gate conditions are missing.")
    stop = balanced["per_target_action"]["STOP"]
    observations = {
        "natural_gate_passed": conditions.get(
            "natural_upper_95_below_blind_prior_minus_margin"
        )
        is True,
        "balanced_gate_passed": conditions.get(
            "balanced_upper_95_below_uniform_minus_margin"
        )
        is True,
        "stop_non_mute_gate_passed": conditions.get(
            "stop_recall_at_or_above_non_mute_floor"
        )
        is True,
        "balanced_stop_never_argmax": stop["count"] > 0 and stop["correct"] == 0,
        "balanced_missing_predicted_stop": balanced["predicted_action_histogram"]["STOP"] == 0,
        "balanced_mean_ce_above_uniform_reference": (
            balanced["mean_restricted_action_ce"] > math.log(len(OFFICIAL_ACTIONS))
        ),
    }
    return {
        "schema_version": NATIVE_C0_FAILURE_ANALYSIS_SCHEMA_VERSION,
        "gate": NATIVE_C0_FAILURE_ANALYSIS_GATE,
        "status": "passed",
        "recorded_at_utc": recorded_at_utc,
        "input_evidence": {
            "development_run_id": result.get("run_id"),
            "manifest_sha256": manifest_sha256,
            "result_sha256": result_sha256,
            "updates_sha256": updates_sha256,
            "verification_sha256": verification_sha256,
            "scientific_result": result.get("status"),
            "independent_verification_status": verification.get("status"),
        },
        "selected_checkpoint_mirror": dict(selected_checkpoint),
        "training_inventory": _training_inventory(manifest.get("samples", [])),
        "selected_update": int(selection["selected_update"]),
        "selection_calibration_trace": calibration_trace,
        "primary_natural": primary,
        "balanced_diagnostic": balanced,
        "confirmed_descriptive_observations": observations,
        "source_locks": dict(source_locks),
        "evidence_boundary": {
            "established_if_verified": (
                "Per-action confusion, recall, probability, CE, prediction-support, and training-"
                "distribution summaries reconstruct from the already-opened verified C0 result."
            ),
            "not_established": (
                "A causal mechanism for the failure, action-prior collapse, closed-loop navigation "
                "competence, superiority of a balancing intervention, or any C1-C4 claim."
            ),
        },
    }
