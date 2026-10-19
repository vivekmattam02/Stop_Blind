"""Pure comparison of verified unweighted and action-balanced native-C0 runs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = 1
COMPARISON_GATE = "native_qwen_c0_action_balance_comparison_v1"
VERIFIER_GATES = {
    "unweighted": "independent_native_qwen_c0_development_verification_v1",
    "action_balanced": "independent_native_qwen_c0_action_balance_verification_v1",
}
RUN_GATES = {
    "unweighted": "native_qwen_c0_development_v1",
    "action_balanced": "native_qwen_c0_action_balance_v1",
}
OFFICIAL_ACTIONS = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
SAMPLE_IDENTITY_FIELDS = (
    "split",
    "sample_index",
    "episode_id",
    "trajectory_id",
    "scene_id",
    "target_action",
)


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _checkpoint_record(result: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = result.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(selection.get("checkpoint"), Mapping):
        raise ValueError("selected checkpoint record is missing")
    return selection["checkpoint"]


def _validate_run(bundle: Mapping[str, Any], label: str) -> None:
    manifest = bundle.get("manifest")
    result = bundle.get("result")
    verification = bundle.get("verification")
    digests = bundle.get("digests")
    checkpoint = bundle.get("selected_checkpoint")
    if not all(
        isinstance(value, Mapping)
        for value in (manifest, result, verification, digests, checkpoint)
    ):
        raise ValueError(f"{label} evidence bundle is incomplete")
    if manifest.get("gate") != RUN_GATES[label] or manifest.get("status") != "frozen":
        raise ValueError(f"{label} manifest identity is invalid")
    if (
        result.get("gate") != RUN_GATES[label]
        or result.get("status") != "not_demonstrated"
        or result.get("completed_updates") != 192
        or result.get("manifest_sha256") != digests.get("manifest_sha256")
        or result.get("updates_sha256") != digests.get("updates_sha256")
    ):
        raise ValueError(f"{label} result identity or linkage is invalid")
    required_verifier_flags = {
        "dataset_reconstruction_performed",
        "outcome_reconstruction_performed",
        "all_checkpoint_files_fully_loaded_and_hashed",
    }
    if label == "action_balanced":
        required_verifier_flags.update(
            {
                "weighted_event_chain_reconstruction_performed",
                "stress_authorization_reconstruction_performed",
            }
        )
    if (
        verification.get("gate") != VERIFIER_GATES[label]
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("errors") != []
        or verification.get("manifest_sha256") != digests.get("manifest_sha256")
        or verification.get("result_sha256") != digests.get("result_sha256")
        or verification.get("updates_sha256") != digests.get("updates_sha256")
        or any(verification.get(flag) is not True for flag in required_verifier_flags)
    ):
        raise ValueError(f"{label} independent verification is missing or unlinked")
    selected = _checkpoint_record(result)
    if (
        checkpoint.get("sha256") != selected.get("sha256")
        or checkpoint.get("bytes") != selected.get("bytes")
        or checkpoint.get("update_step") != result["selection"].get("selected_update")
    ):
        raise ValueError(f"{label} selected-checkpoint mirror is missing or unlinked")


def _normalized_protocol(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    for key in ("schema_version", "gate", "objective", "claim_boundary"):
        normalized.pop(key, None)
    ledger = dict(normalized.get("unit_ledger", {}))
    declared = dict(ledger.get("declared", {}))
    declared.pop("notes", None)
    ledger["declared"] = declared
    normalized["unit_ledger"] = ledger
    return normalized


def _validate_controlled_change(
    unweighted: Mapping[str, Any], action_balanced: Mapping[str, Any]
) -> dict[str, Any]:
    control_manifest = unweighted["manifest"]
    treatment_manifest = action_balanced["manifest"]
    exact_fields = ("dataset", "samples", "sample_records_sha256", "partition_summary", "cache")
    for field in exact_fields:
        if control_manifest.get(field) != treatment_manifest.get(field):
            raise ValueError(f"controlled comparison changed manifest field: {field}")
    control_model = dict(control_manifest.get("model", {}))
    treatment_model = dict(treatment_manifest.get("model", {}))
    treatment_model.pop("weight_locks", None)
    if control_model != treatment_model:
        raise ValueError("controlled comparison changed the declared model contract")
    if unweighted["result"].get("model_weight_locks") != action_balanced["result"].get(
        "model_weight_locks"
    ):
        raise ValueError("controlled comparison changed base-model weight identities")
    if _normalized_protocol(control_manifest["protocol"]) != _normalized_protocol(
        treatment_manifest["protocol"]
    ):
        raise ValueError("controlled comparison changed more than the declared objective")
    if treatment_manifest["protocol"].get("objective") != (
        "class_balanced_restricted_four_action_cross_entropy_v1"
    ):
        raise ValueError("treatment objective is not the authorized class-balanced loss")
    development = treatment_manifest.get("development_evidence", {})
    if (
        development.get("manifest_sha256") != unweighted["digests"].get("manifest_sha256")
        or development.get("result_sha256") != unweighted["digests"].get("result_sha256")
        or development.get("verification_sha256")
        != unweighted["digests"].get("verification_sha256")
    ):
        raise ValueError("treatment does not link the exact unweighted development evidence")
    if action_balanced["result"].get("adaptive_development_comparison") is not True:
        raise ValueError("treatment lost its adaptive-development interpretation boundary")
    return {
        "same_dataset": True,
        "same_samples_and_order": True,
        "same_partitions": True,
        "same_content_cache": True,
        "same_model_and_initialization_contract": True,
        "same_optimizer_rate_horizon_selection_and_outcomes": True,
        "only_declared_change": "train_time_target_action_loss_weighting",
        "adaptive_development_comparison": True,
    }


def _partition(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    rows = record.get("per_decision")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} per-decision evidence is missing")
    confusion = {
        target: {prediction: 0 for prediction in OFFICIAL_ACTIONS} for target in OFFICIAL_ACTIONS
    }
    predicted: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    ce_values: list[float] = []
    identities: list[list[Any]] = []
    for index, row in enumerate(rows):
        target = str(row.get("target_action"))
        prediction = str(row.get("predicted_action"))
        if target not in OFFICIAL_ACTIONS or prediction not in OFFICIAL_ACTIONS:
            raise ValueError(f"{label} decision {index} has an unofficial action")
        ce = _finite(row.get("restricted_action_ce"), f"{label} decision {index} CE")
        probability = _finite(
            row.get("target_restricted_probability"),
            f"{label} decision {index} target probability",
        )
        if probability <= 0.0 or not math.isclose(
            ce, -math.log(probability), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{label} decision {index} CE/probability linkage is invalid")
        if row.get("correct") is not (target == prediction):
            raise ValueError(f"{label} decision {index} correctness is inconsistent")
        confusion[target][prediction] += 1
        predicted[prediction] += 1
        target_counts[target] += 1
        ce_values.append(ce)
        identities.append([row.get(field) for field in SAMPLE_IDENTITY_FIELDS])
    correct = sum(confusion[action][action] for action in OFFICIAL_ACTIONS)
    recalls = {
        action: confusion[action][action] / target_counts[action]
        for action in OFFICIAL_ACTIONS
        if target_counts[action]
    }
    return {
        "decision_count": len(rows),
        "sample_identities": identities,
        "mean_restricted_action_ce": sum(ce_values) / len(ce_values),
        "descriptive_accuracy": correct / len(rows),
        "macro_recall": sum(recalls.values()) / len(recalls),
        "target_action_histogram": {action: target_counts[action] for action in OFFICIAL_ACTIONS},
        "predicted_action_histogram": {action: predicted[action] for action in OFFICIAL_ACTIONS},
        "confusion_matrix_by_target": confusion,
        "per_action_recall": recalls,
    }


def _surface(
    control_result: Mapping[str, Any],
    treatment_result: Mapping[str, Any],
    surface_name: str,
    gate_key: str,
) -> dict[str, Any]:
    control = _partition(control_result[surface_name], f"unweighted {surface_name}")
    treatment = _partition(treatment_result[surface_name], f"action-balanced {surface_name}")
    if control.pop("sample_identities") != treatment.pop("sample_identities"):
        raise ValueError(f"{surface_name} sample identities/order differ between runs")
    control_gate = control_result["existence_gate"][gate_key]
    treatment_gate = treatment_result["existence_gate"][gate_key]
    for partition, observed in ((control_gate, control), (treatment_gate, treatment)):
        if not math.isclose(
            _finite(partition["observed_mean_restricted_action_ce"], "gate mean CE"),
            observed["mean_restricted_action_ce"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{surface_name} gate mean does not reconstruct from decisions")
    return {
        "unweighted": {**control, "gate": dict(control_gate)},
        "action_balanced": {**treatment, "gate": dict(treatment_gate)},
        "action_balanced_minus_unweighted": {
            "mean_restricted_action_ce": (
                treatment["mean_restricted_action_ce"] - control["mean_restricted_action_ce"]
            ),
            "upper_95_restricted_action_ce": (
                _finite(treatment_gate["upper_95_restricted_action_ce"], "treatment upper CE")
                - _finite(control_gate["upper_95_restricted_action_ce"], "control upper CE")
            ),
            "descriptive_accuracy": (
                treatment["descriptive_accuracy"] - control["descriptive_accuracy"]
            ),
            "macro_recall": treatment["macro_recall"] - control["macro_recall"],
            "per_action_recall": {
                action: treatment["per_action_recall"][action]
                - control["per_action_recall"][action]
                for action in OFFICIAL_ACTIONS
            },
            "predicted_action_histogram": {
                action: treatment["predicted_action_histogram"][action]
                - control["predicted_action_histogram"][action]
                for action in OFFICIAL_ACTIONS
            },
        },
    }


def build_comparison(
    *,
    unweighted: Mapping[str, Any],
    action_balanced: Mapping[str, Any],
    source_locks: Mapping[str, str],
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Reconstruct the compact comparison from already-opened verified evidence."""
    _validate_run(unweighted, "unweighted")
    _validate_run(action_balanced, "action_balanced")
    controlled_change = _validate_controlled_change(unweighted, action_balanced)
    control_result = unweighted["result"]
    treatment_result = action_balanced["result"]
    natural = _surface(
        control_result,
        treatment_result,
        "primary_evaluation",
        "primary_natural",
    )
    balanced = _surface(
        control_result,
        treatment_result,
        "balanced_diagnostic",
        "balanced_diagnostic",
    )
    control_stop = balanced["unweighted"]["per_action_recall"]["STOP"]
    treatment_stop = balanced["action_balanced"]["per_action_recall"]["STOP"]
    treatment_gate = treatment_result["existence_gate"]
    if treatment_gate.get("status") != "not_demonstrated_at_frozen_budget":
        raise ValueError("action-balanced run is not the observed frozen negative result")
    thresholds = {
        "natural_upper_95_gap_to_required_below": (
            treatment_gate["primary_natural"]["upper_95_restricted_action_ce"]
            - treatment_gate["primary_natural"]["threshold_ce_nats"]
        ),
        "balanced_upper_95_gap_to_required_below": (
            treatment_gate["balanced_diagnostic"]["upper_95_restricted_action_ce"]
            - treatment_gate["balanced_diagnostic"]["threshold_ce_nats"]
        ),
        "stop_recall_gap_to_required_minimum": (
            treatment_stop - treatment_gate["balanced_diagnostic"]["stop_recall_floor"]
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": COMPARISON_GATE,
        "recorded_at_utc": recorded_at_utc,
        "status": "passed",
        "comparison_classification": (
            "balanced_support_improved_natural_fit_degraded_no_gate_rescue"
        ),
        "controlled_change_audit": controlled_change,
        "upstream_evidence": {
            "unweighted": {
                **dict(unweighted["digests"]),
                "selected_checkpoint": dict(unweighted["selected_checkpoint"]),
            },
            "action_balanced": {
                **dict(action_balanced["digests"]),
                "selected_checkpoint": dict(action_balanced["selected_checkpoint"]),
            },
        },
        "selection": {
            "unweighted_selected_update": control_result["selection"]["selected_update"],
            "action_balanced_selected_update": treatment_result["selection"]["selected_update"],
            "unweighted_selected_mean_restricted_action_ce": control_result["selection"][
                "selected_mean_restricted_action_ce"
            ],
            "action_balanced_selected_mean_restricted_action_ce": treatment_result["selection"][
                "selected_mean_restricted_action_ce"
            ],
        },
        "primary_natural": natural,
        "balanced_diagnostic": balanced,
        "action_balanced_threshold_gaps": thresholds,
        "condition_transitions": {
            key: {
                "unweighted": control_result["existence_gate"]["conditions"][key],
                "action_balanced": treatment_result["existence_gate"]["conditions"][key],
            }
            for key in sorted(treatment_result["existence_gate"]["conditions"])
        },
        "confirmed_descriptive_observations": {
            "balanced_mean_ce_improved": (
                balanced["action_balanced_minus_unweighted"]["mean_restricted_action_ce"] < 0.0
            ),
            "balanced_upper_95_ce_improved": (
                balanced["action_balanced_minus_unweighted"]["upper_95_restricted_action_ce"] < 0.0
            ),
            "balanced_stop_recall_improved": treatment_stop > control_stop,
            "natural_mean_ce_degraded": (
                natural["action_balanced_minus_unweighted"]["mean_restricted_action_ce"] > 0.0
            ),
            "natural_upper_95_ce_degraded": (
                natural["action_balanced_minus_unweighted"]["upper_95_restricted_action_ce"] > 0.0
            ),
            "all_three_frozen_treatment_conditions_failed": not any(
                treatment_result["existence_gate"]["conditions"].values()
            ),
        },
        "interpretation_boundary": {
            "established": (
                "On identical opened development surfaces, inverse-frequency train-loss weighting "
                "improved balanced action support and STOP recall but degraded "
                "natural-distribution CE and did not rescue the conjunctive native-C0 "
                "existence gate."
            ),
            "not_established": (
                "A causal mechanism, an optimal weighting rule, independent confirmation, "
                "closed-loop navigation competence, val_unseen performance, or any C1-C4 claim."
            ),
            "reuse_rule": (
                "Both outcome surfaces have now informed two adaptive development results and are "
                "closed to further tuning or confirmatory claims."
            ),
        },
        "source_locks": dict(source_locks),
    }
