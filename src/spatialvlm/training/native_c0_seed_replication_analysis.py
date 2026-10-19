"""Pure aggregation for the frozen three-seed native-C0 paired comparison."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = 1
GATE = "native_qwen_c0_seed_replication_analysis_v1"
VERIFIER_GATE = "independent_native_qwen_c0_seed_replication_analysis_verification_v1"
SEEDS = (20260825, 20260826, 20260827)
TREATMENTS = ("unweighted", "inverse_frequency")
ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")


def _macro_recall(summary: Mapping[str, Any]) -> float:
    return statistics.fmean(
        float(summary["per_target_action"][action]["recall"]) for action in ACTIONS
    )


def _selected_calibration(result: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = int(result["selection"]["selected_update"])
    rows = [row for row in result["calibrations"] if int(row["update_step"]) == selected]
    if len(rows) != 1:
        raise ValueError("Selected calibration does not resolve to exactly one row.")
    return rows[0]


def run_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the same frozen metrics from one independently verified run."""
    if result.get("status") not in {"passed", "not_demonstrated"}:
        raise ValueError("Seed-replication input is not a completed scientific result.")
    if int(result.get("completed_updates", -1)) != 192:
        raise ValueError("Seed-replication input did not complete all 192 updates.")
    selected = _selected_calibration(result)
    natural = result["primary_evaluation"]["summary"]
    balanced = result["balanced_diagnostic"]["summary"]
    gate = result["existence_gate"]
    conditions = {name: bool(value) for name, value in gate["conditions"].items()}
    return {
        "scientific_status": str(result["status"]),
        "existence_gate_status": str(gate["status"]),
        "selected_update": int(result["selection"]["selected_update"]),
        "selection_mean_ce": float(result["selection"]["selected_mean_restricted_action_ce"]),
        "selection_stop_predictions": int(
            selected["summary"]["predicted_action_histogram"]["STOP"]
        ),
        "selection_decisions": int(selected["summary"]["decision_count"]),
        "natural_mean_ce": float(natural["mean_restricted_action_ce"]),
        "natural_upper_95_ce": float(gate["primary_natural"]["upper_95_restricted_action_ce"]),
        "natural_accuracy": float(natural["descriptive_accuracy"]),
        "natural_macro_recall": _macro_recall(natural),
        "natural_stop_predictions": int(natural["predicted_action_histogram"]["STOP"]),
        "natural_stop_recall": float(natural["per_target_action"]["STOP"]["recall"]),
        "balanced_mean_ce": float(balanced["mean_restricted_action_ce"]),
        "balanced_upper_95_ce": float(gate["balanced_diagnostic"]["upper_95_restricted_action_ce"]),
        "balanced_accuracy": float(balanced["descriptive_accuracy"]),
        "balanced_macro_recall": _macro_recall(balanced),
        "balanced_stop_predictions": int(balanced["predicted_action_histogram"]["STOP"]),
        "balanced_stop_correct": int(gate["balanced_diagnostic"]["stop_correct"]),
        "balanced_stop_count": int(gate["balanced_diagnostic"]["stop_count"]),
        "balanced_stop_recall": float(gate["balanced_diagnostic"]["stop_recall"]),
        "balanced_per_action_recall": {
            action: float(balanced["per_target_action"][action]["recall"]) for action in ACTIONS
        },
        "conditions": conditions,
    }


def _summary(values_by_seed: Mapping[int, float]) -> dict[str, Any]:
    values = [float(values_by_seed[seed]) for seed in SEEDS]
    return {
        "values_by_seed": {str(seed): float(values_by_seed[seed]) for seed in SEEDS},
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def build_analysis(
    *,
    runs: Mapping[int, Mapping[str, Mapping[str, Any]]],
    evidence: Mapping[str, Any],
    source_locks: Mapping[str, str],
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Build the complete paired analysis; never select or discard a seed."""
    if tuple(sorted(runs)) != SEEDS:
        raise ValueError("Analysis requires exactly the three preregistered seeds.")
    extracted: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        if set(runs[seed]) != set(TREATMENTS):
            raise ValueError(f"Seed {seed} does not contain both frozen treatments.")
        extracted[seed] = {
            treatment: run_metrics(runs[seed][treatment]) for treatment in TREATMENTS
        }

    paired_initialization = {
        str(seed): len(
            {
                str(runs[seed][treatment]["initial_trainable_state_sha256"])
                for treatment in TREATMENTS
            }
        )
        == 1
        for seed in SEEDS
    }
    if not all(paired_initialization.values()):
        raise ValueError("At least one seed pair does not share its initial trainable state.")

    numeric_metrics = (
        "selection_mean_ce",
        "selection_stop_predictions",
        "natural_mean_ce",
        "natural_upper_95_ce",
        "natural_accuracy",
        "natural_macro_recall",
        "natural_stop_predictions",
        "balanced_mean_ce",
        "balanced_upper_95_ce",
        "balanced_accuracy",
        "balanced_macro_recall",
        "balanced_stop_predictions",
        "balanced_stop_recall",
    )
    treatment_summaries: dict[str, Any] = {}
    for treatment in TREATMENTS:
        treatment_summaries[treatment] = {
            metric: _summary({seed: float(extracted[seed][treatment][metric]) for seed in SEEDS})
            for metric in numeric_metrics
        }
        treatment_summaries[treatment]["selected_updates_by_seed"] = {
            str(seed): extracted[seed][treatment]["selected_update"] for seed in SEEDS
        }

    paired_metrics = (
        "natural_mean_ce",
        "natural_accuracy",
        "natural_macro_recall",
        "balanced_mean_ce",
        "balanced_accuracy",
        "balanced_macro_recall",
        "balanced_stop_recall",
        "balanced_stop_predictions",
    )
    paired_deltas = {
        metric: _summary(
            {
                seed: float(extracted[seed]["inverse_frequency"][metric])
                - float(extracted[seed]["unweighted"][metric])
                for seed in SEEDS
            }
        )
        for metric in paired_metrics
    }

    unweighted = {seed: extracted[seed]["unweighted"] for seed in SEEDS}
    weighted = {seed: extracted[seed]["inverse_frequency"] for seed in SEEDS}
    consistency = {
        "unweighted_selected_update_160_seed_count": sum(
            row["selected_update"] == 160 for row in unweighted.values()
        ),
        "unweighted_zero_selection_stop_prediction_seed_count": sum(
            row["selection_stop_predictions"] == 0 for row in unweighted.values()
        ),
        "unweighted_zero_balanced_stop_recall_seed_count": sum(
            row["balanced_stop_correct"] == 0 for row in unweighted.values()
        ),
        "unweighted_natural_condition_pass_seed_count": sum(
            row["conditions"]["natural_upper_95_below_blind_prior_minus_margin"]
            for row in unweighted.values()
        ),
        "inverse_frequency_natural_ce_worse_seed_count": sum(
            weighted[seed]["natural_mean_ce"] > unweighted[seed]["natural_mean_ce"]
            for seed in SEEDS
        ),
        "inverse_frequency_balanced_ce_better_seed_count": sum(
            weighted[seed]["balanced_mean_ce"] < unweighted[seed]["balanced_mean_ce"]
            for seed in SEEDS
        ),
        "inverse_frequency_balanced_accuracy_better_seed_count": sum(
            weighted[seed]["balanced_accuracy"] > unweighted[seed]["balanced_accuracy"]
            for seed in SEEDS
        ),
        "inverse_frequency_stop_recall_better_seed_count": sum(
            weighted[seed]["balanced_stop_recall"] > unweighted[seed]["balanced_stop_recall"]
            for seed in SEEDS
        ),
        "existence_gate_pass_run_count": sum(
            row["scientific_status"] == "passed"
            for seed in SEEDS
            for row in extracted[seed].values()
        ),
    }
    if any(extracted[seed][t]["balanced_stop_count"] != 64 for seed in SEEDS for t in TREATMENTS):
        raise ValueError("Balanced STOP denominator changed across replication runs.")
    if any(not math.isfinite(value["mean"]) for value in paired_deltas.values()):
        raise ValueError("A paired aggregate is non-finite.")

    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "status": "passed",
        "recorded_at_utc": recorded_at_utc,
        "seeds": list(SEEDS),
        "treatments": list(TREATMENTS),
        "reporting_rule": "all_preregistered_seeds_reported_never_best_seed",
        "paired_initialization_confirmed": paired_initialization,
        "evidence": dict(evidence),
        "runs": {str(seed): extracted[seed] for seed in SEEDS},
        "treatment_summaries": treatment_summaries,
        "paired_deltas_inverse_frequency_minus_unweighted": paired_deltas,
        "consistency_counts_out_of_three": consistency,
        "primary_findings": {
            "checkpoint_selection_failure_reproduced": (
                consistency["unweighted_zero_selection_stop_prediction_seed_count"] == 3
            ),
            "qualitative_weighting_tradeoff_reproduced": (
                consistency["inverse_frequency_natural_ce_worse_seed_count"] == 3
                and consistency["inverse_frequency_balanced_ce_better_seed_count"] == 3
                and consistency["inverse_frequency_stop_recall_better_seed_count"] == 3
            ),
            "frozen_existence_gate_ever_passed": (consistency["existence_gate_pass_run_count"] > 0),
        },
        "interpretation_boundary": {
            "established": (
                "Training-randomness robustness of the paired adaptive train-release comparison "
                "across exactly three preregistered seeds."
            ),
            "not_established": (
                "Independent-data confirmation, causal mechanism, learned closed-loop success, "
                "val_seen or val_unseen generalization, or any C1-C4 geometry claim."
            ),
        },
        "source_locks": dict(source_locks),
    }
