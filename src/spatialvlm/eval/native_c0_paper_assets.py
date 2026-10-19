"""Artifact-locked values and LaTeX fragments for the frozen native-C0 paper.

This module deliberately contains no model or simulator code.  It accepts only
the independently verified post-hoc analyses and the verified evaluator-control
summary, validates their hash links, and turns them into a small paper-facing
record.  Rendering code consumes that record; paper numbers are never typed into
LaTeX or figure scripts by hand.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
GATE = "native_qwen_c0_paper_assets_v2"
FROZEN_CLAIM = (
    "Natural-distribution action-prediction cross-entropy used for checkpoint selection "
    "during native VLM navigation fine-tuning can prefer a STOP-muted policy. Across three "
    "paired seeds, inverse-frequency weighting partially restores minority-action support "
    "but consistently worsens natural cross-entropy and never satisfies the preregistered gate."
)
ACTION_ORDER = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
DISPLAY_ACTIONS = {
    "MOVE_FORWARD": "Forward",
    "TURN_LEFT": "Left",
    "TURN_RIGHT": "Right",
    "STOP": "STOP",
}


def sha256_file(path: Path) -> str:
    """Return the full SHA-256 of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject non-object top-level values."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_linked_analysis(
    analysis_path: Path,
    verification_path: Path,
    *,
    expected_analysis_gate: str,
    expected_verifier_gate: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load an analysis only when its independent verifier is passed and linked."""

    analysis = load_json(analysis_path)
    verification = load_json(verification_path)
    actual_sha256 = sha256_file(analysis_path)
    if analysis.get("gate") != expected_analysis_gate or analysis.get("status") != "passed":
        raise ValueError(f"Analysis is not the expected passed gate: {analysis_path}")
    if (
        verification.get("gate") != expected_verifier_gate
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("errors") != []
        or verification.get("analysis_gate") != expected_analysis_gate
        or verification.get("analysis_sha256") != actual_sha256
        or verification.get("analysis_reconstruction_performed") is not True
    ):
        raise ValueError(f"Analysis verification is missing or hash-unlinked: {analysis_path}")
    return analysis, verification


def validate_policy_controls(path: Path, repo: Path) -> dict[str, Any]:
    """Validate the locally reconstructed non-learning evaluator-control summary."""

    record = load_json(path)
    checks = record.get("checks")
    preregistered = record.get("recomputed_pre_registered_analysis")
    if not isinstance(checks, Mapping) or not isinstance(preregistered, Mapping):
        raise ValueError("Policy-control verification is structurally incomplete")
    if (
        record.get("status") != "passed"
        or checks.get("errors") != []
        or preregistered.get("status") != "passed"
        or preregistered.get("all_required_checks_pass") is not True
    ):
        raise ValueError("Policy-control verification did not pass")
    source_root = repo / str(record.get("source_artifact_root"))
    expected_hashes = record.get("source_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise ValueError("Policy-control source hashes are missing")
    for filename in ("suite_manifest.json", "suite.json", "progress.json"):
        expected = expected_hashes.get(filename)
        source = source_root / filename
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"Policy-control source is missing or changed: {source}")
    return record


def _require_close(observed: Any, expected: float, label: str) -> None:
    if not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Frozen invariant changed for {label}: {observed!r}")


def _metric_record(surface: Mapping[str, Any]) -> dict[str, Any]:
    gate = surface["gate"]
    return {
        "decision_count": int(surface["decision_count"]),
        "mean_ce": float(surface["mean_restricted_action_ce"]),
        "upper_95_ce": float(gate["upper_95_restricted_action_ce"]),
        "accuracy": float(surface["descriptive_accuracy"]),
        "accuracy_correct": int(round(surface["descriptive_accuracy"] * surface["decision_count"])),
        "macro_recall": float(surface["macro_recall"]),
        "per_action_recall": {
            action: float(surface["per_action_recall"][action]) for action in ACTION_ORDER
        },
        "per_action_correct": {
            action: int(surface["confusion_matrix_by_target"][action][action])
            for action in ACTION_ORDER
        },
        "per_action_count": {
            action: int(surface["target_action_histogram"][action]) for action in ACTION_ORDER
        },
        "predicted_action_histogram": {
            action: int(surface["predicted_action_histogram"][action]) for action in ACTION_ORDER
        },
    }


def build_paper_values(
    *,
    comparison: Mapping[str, Any],
    failure_analysis: Mapping[str, Any],
    policy_controls: Mapping[str, Any],
    c0_manifest: Mapping[str, Any],
    seed_replication: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate the complete frozen paper-facing value record."""

    if comparison.get("comparison_classification") != (
        "balanced_support_improved_natural_fit_degraded_no_gate_rescue"
    ):
        raise ValueError("Unexpected C0 comparison classification")
    boundary = comparison.get("interpretation_boundary", {})
    if "A causal mechanism" not in str(boundary.get("not_established")):
        raise ValueError("Comparison lost its non-causal interpretation boundary")
    if (
        seed_replication.get("gate") != "native_qwen_c0_seed_replication_analysis_v1"
        or seed_replication.get("status") != "passed"
        or seed_replication.get("seeds") != [20260825, 20260826, 20260827]
        or seed_replication.get("reporting_rule")
        != "all_preregistered_seeds_reported_never_best_seed"
        or seed_replication.get("primary_findings", {}).get(
            "checkpoint_selection_failure_reproduced"
        )
        is not True
        or seed_replication.get("primary_findings", {}).get(
            "qualitative_weighting_tradeoff_reproduced"
        )
        is not True
        or seed_replication.get("primary_findings", {}).get("frozen_existence_gate_ever_passed")
        is not False
    ):
        raise ValueError("Three-seed replication evidence is incomplete or changed")

    natural = comparison["primary_natural"]
    balanced = comparison["balanced_diagnostic"]
    unweighted_natural = _metric_record(natural["unweighted"])
    weighted_natural = _metric_record(natural["action_balanced"])
    unweighted_balanced = _metric_record(balanced["unweighted"])
    weighted_balanced = _metric_record(balanced["action_balanced"])

    uniform_ce = float(balanced["unweighted"]["gate"]["uniform_reference_ce_nats"])
    balanced_threshold = float(balanced["unweighted"]["gate"]["threshold_ce_nats"])
    natural_prior_ce = float(natural["unweighted"]["gate"]["blind_reference_ce_nats"])
    natural_threshold = float(natural["unweighted"]["gate"]["threshold_ce_nats"])
    stop_floor = float(balanced["unweighted"]["gate"]["stop_recall_floor"])
    _require_close(uniform_ce, math.log(4.0), "uniform four-action CE")
    _require_close(stop_floor, 0.25, "STOP recall floor")
    _require_close(
        weighted_natural["mean_ce"] - unweighted_natural["mean_ce"],
        0.22287712471258314,
        "natural CE trade-off",
    )
    _require_close(unweighted_balanced["accuracy"], 84 / 256, "unweighted balanced accuracy")
    _require_close(weighted_balanced["accuracy"], 113 / 256, "weighted balanced accuracy")
    _require_close(unweighted_balanced["mean_ce"], 1.7671635118916424, "unweighted balanced CE")

    trace: list[dict[str, Any]] = []
    for row in failure_analysis["selection_calibration_trace"]:
        stop = row["per_target_action"]["STOP"]
        trace.append(
            {
                "update": int(row["update_step"]),
                "stop_predictions": int(row["predicted_action_histogram"]["STOP"]),
                "selection_decisions": 256,
                "stop_target_correct": int(stop["correct"]),
                "stop_target_count": int(stop["count"]),
            }
        )
    if not trace or trace[0]["stop_predictions"] != 28 or trace[1]["stop_predictions"] != 0:
        raise ValueError("The verified STOP-suppression signature changed")
    if any(row["stop_target_count"] != 3 for row in trace):
        raise ValueError("Selection STOP denominator changed")

    train_histogram = {
        action: int(failure_analysis["training_inventory"]["target_action_histogram"][action])
        for action in ACTION_ORDER
    }
    partition_summary = c0_manifest.get("partition_summary")
    if not isinstance(partition_summary, Mapping):
        raise ValueError("C0 manifest partition summary is missing")
    trajectory_counts = partition_summary.get("trajectory_counts")
    if not isinstance(trajectory_counts, Mapping):
        raise ValueError("C0 trajectory allocation is missing")
    allocated = sum(int(trajectory_counts[key]) for key in trajectory_counts)
    if allocated != int(partition_summary.get("unique_trajectory_count", -1)) or allocated != 3603:
        raise ValueError("Frozen trajectory allocation no longer sums to 3,603")

    families = policy_controls["recomputed_summary"]["families"]
    controls = []
    control_labels = (
        ("released_ground_truth_replay", "Released trajectory replay"),
        ("immediate_stop", "Immediate STOP"),
        ("always_forward", "Always forward"),
        ("invalid_action", "Invalid action"),
        ("official_random", "Official random"),
        ("official_handcrafted", "Official handcrafted"),
    )
    for key, label in control_labels:
        family = families[key]
        controls.append(
            {
                "id": key,
                "label": label,
                "run_count": int(family["run_count"]),
                "episode_count_per_run": 61,
                "success": float(family["success"]["mean_across_seed_runs"]),
                "spl": float(family["spl"]["mean_across_seed_runs"]),
                "ndtw": float(family["ndtw"]["mean_across_seed_runs"]),
                "seeds": [int(seed) for seed in family["seeds"]],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE,
        "status": "passed",
        "frozen_claim": FROZEN_CLAIM,
        "references": {
            "balanced_uniform_ce_nats": uniform_ce,
            "balanced_uniform_accuracy": 0.25,
            "balanced_required_upper_95_ce_below": balanced_threshold,
            "natural_blind_prior_ce_nats": natural_prior_ce,
            "natural_required_upper_95_ce_below": natural_threshold,
            "stop_recall_floor": stop_floor,
        },
        "c0": {
            "unweighted": {
                "natural": unweighted_natural,
                "balanced": unweighted_balanced,
                "selected_update": int(comparison["selection"]["unweighted_selected_update"]),
                "existence_gate": "not_demonstrated_at_frozen_budget",
            },
            "action_balanced": {
                "natural": weighted_natural,
                "balanced": weighted_balanced,
                "selected_update": int(comparison["selection"]["action_balanced_selected_update"]),
                "existence_gate": "not_demonstrated_at_frozen_budget",
            },
            "deltas_action_balanced_minus_unweighted": {
                "natural_mean_ce": float(
                    natural["action_balanced_minus_unweighted"]["mean_restricted_action_ce"]
                ),
                "natural_accuracy": float(
                    natural["action_balanced_minus_unweighted"]["descriptive_accuracy"]
                ),
                "natural_macro_recall": float(
                    natural["action_balanced_minus_unweighted"]["macro_recall"]
                ),
                "balanced_mean_ce": float(
                    balanced["action_balanced_minus_unweighted"]["mean_restricted_action_ce"]
                ),
                "balanced_accuracy": float(
                    balanced["action_balanced_minus_unweighted"]["descriptive_accuracy"]
                ),
                "balanced_macro_recall": float(
                    balanced["action_balanced_minus_unweighted"]["macro_recall"]
                ),
            },
        },
        "training_action_histogram": train_histogram,
        "selection_stop_trace": trace,
        "seed_replication": {
            "seeds": list(seed_replication["seeds"]),
            "runs": seed_replication["runs"],
            "treatment_summaries": seed_replication["treatment_summaries"],
            "paired_deltas_inverse_frequency_minus_unweighted": seed_replication[
                "paired_deltas_inverse_frequency_minus_unweighted"
            ],
            "consistency_counts_out_of_three": seed_replication["consistency_counts_out_of_three"],
            "primary_findings": seed_replication["primary_findings"],
            "dispersion_note": (
                "mean and sample standard deviation over exactly three preregistered seeds; "
                "descriptive dispersion, not a population confidence interval"
            ),
        },
        "trajectory_allocation": {
            "partition_unit": "physical_trajectory_id",
            "counts": {key: int(value) for key, value in trajectory_counts.items()},
            "unique_trajectory_count": allocated,
            "unused_trajectory_count": 0,
        },
        "evaluator_controls": controls,
        "interpretation_boundary": {
            "frequency_result": (
                "descriptive association consistent with training imbalance; "
                "not a proven causal mechanism"
            ),
            "balanced_ce_result": (
                "unweighted CE exceeds ln(4); accuracy remains above 0.25 and "
                "must be reported separately"
            ),
            "selection_stop_result": (
                "STOP predictions fall from 28/256 to 0/256 by update 32; the 1/3 "
                "target-specific selection recall is not a headline rate"
            ),
            "replication_scope": (
                "three paired seeds estimate training-randomness robustness on the already "
                "adaptive train-release surfaces; they are not independent-data confirmation"
            ),
        },
    }


def render_seed_replication_table(values: Mapping[str, Any]) -> str:
    """Return the primary three-seed paired comparison as a LaTeX fragment."""

    replication = values["seed_replication"]
    unweighted = replication["treatment_summaries"]["unweighted"]
    weighted = replication["treatment_summaries"]["inverse_frequency"]
    deltas = replication["paired_deltas_inverse_frequency_minus_unweighted"]

    def mean_sd(record: Mapping[str, Any]) -> str:
        return f"{record['mean']:.3f} $\\pm$ {record['sample_standard_deviation']:.3f}"

    rows = [
        (
            "Natural mean CE $\\downarrow$",
            mean_sd(unweighted["natural_mean_ce"]),
            mean_sd(weighted["natural_mean_ce"]),
            mean_sd(deltas["natural_mean_ce"]),
        ),
        (
            "Balanced mean CE $\\downarrow$",
            mean_sd(unweighted["balanced_mean_ce"]),
            mean_sd(weighted["balanced_mean_ce"]),
            mean_sd(deltas["balanced_mean_ce"]),
        ),
        (
            "Balanced accuracy $\\uparrow$",
            mean_sd(unweighted["balanced_accuracy"]),
            mean_sd(weighted["balanced_accuracy"]),
            mean_sd(deltas["balanced_accuracy"]),
        ),
        (
            "Balanced STOP recall $\\uparrow$",
            "0/64 in 3/3 seeds",
            "12/64, 1/64, 6/64",
            mean_sd(deltas["balanced_stop_recall"]),
        ),
        (
            "Existence gate",
            "0/3 passed",
            "0/3 passed",
            "---",
        ),
    ]
    body = "\n".join("{} & {} & {} & {} \\\\".format(*row) for row in rows)
    return (
        "% Generated from independently verified JSON; do not edit by hand.\n"
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Three-seed paired replication. Values are mean $\\pm$ sample standard "
        "deviation; deltas are inverse-frequency minus unweighted. All seeds are reported.}\n"
        "\\label{tab:seed-replication}\n"
        "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n"
        "Metric & Unweighted & Inverse-frequency & Paired delta "
        + r"\\"
        + "\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def _recall_with_denominator(record: Mapping[str, Any], action: str) -> str:
    correct = record["per_action_correct"][action]
    count = record["per_action_count"][action]
    recall = record["per_action_recall"][action]
    return f"{correct}/{count} ({recall:.3f})"


def render_c0_gate_table(values: Mapping[str, Any]) -> str:
    """Return the main C0 table as a LaTeX fragment."""

    refs = values["references"]
    unweighted = values["c0"]["unweighted"]
    weighted = values["c0"]["action_balanced"]
    un = unweighted["natural"]
    wn = weighted["natural"]
    ub = unweighted["balanced"]
    wb = weighted["balanced"]
    rows = [
        (
            "Natural mean CE $\\downarrow$",
            f"blind prior {refs['natural_blind_prior_ce_nats']:.4f}",
            f"{un['mean_ce']:.4f}",
            f"{wn['mean_ce']:.4f}",
        ),
        (
            "Natural upper 95\\% CE $\\downarrow$",
            f"required $<{refs['natural_required_upper_95_ce_below']:.4f}$",
            f"{un['upper_95_ce']:.4f} (pass)",
            f"{wn['upper_95_ce']:.4f} (fail)",
        ),
        (
            "Natural accuracy $\\uparrow$",
            "descriptive; no chance claim",
            f"{un['accuracy_correct']}/{un['decision_count']} ({un['accuracy']:.3f})",
            f"{wn['accuracy_correct']}/{wn['decision_count']} ({wn['accuracy']:.3f})",
        ),
        (
            "Balanced mean CE $\\downarrow$",
            f"$\\ln 4={refs['balanced_uniform_ce_nats']:.4f}$",
            f"{ub['mean_ce']:.4f}",
            f"{wb['mean_ce']:.4f}",
        ),
        (
            "Balanced upper 95\\% CE $\\downarrow$",
            f"required $<{refs['balanced_required_upper_95_ce_below']:.4f}$",
            f"{ub['upper_95_ce']:.4f} (fail)",
            f"{wb['upper_95_ce']:.4f} (fail)",
        ),
        (
            "Balanced accuracy $\\uparrow$",
            f"uniform $={refs['balanced_uniform_accuracy']:.2f}$",
            f"{ub['accuracy_correct']}/{ub['decision_count']} ({ub['accuracy']:.3f})",
            f"{wb['accuracy_correct']}/{wb['decision_count']} ({wb['accuracy']:.3f})",
        ),
        (
            "Balanced macro recall $\\uparrow$",
            "four-action macro",
            f"{ub['macro_recall']:.3f}",
            f"{wb['macro_recall']:.3f}",
        ),
        (
            "Balanced STOP recall $\\uparrow$",
            f"floor $={int(refs['stop_recall_floor'] * 64)}/64$",
            _recall_with_denominator(ub, "STOP"),
            _recall_with_denominator(wb, "STOP"),
        ),
        ("Existence gate", "all three conditions", "not demonstrated", "not demonstrated"),
    ]
    body = "\n".join("{} & {} & {} & {} \\\\".format(*row) for row in rows)
    return (
        "% Generated from independently verified JSON; do not edit by hand.\n"
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Native-C0 results. CE and accuracy use separate references; "
        "every recall is paired with its denominator.}\n"
        "\\label{tab:c0-gates}\n"
        "\\begin{tabular}{@{}p{0.29\\linewidth}p{0.23\\linewidth}cc@{}}\n"
        "\\toprule\nMetric & Reference or rule & Unweighted & Action-balanced "
        + r"\\"
        + "\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def render_action_recall_table(values: Mapping[str, Any]) -> str:
    """Return per-action training frequency and recall as a LaTeX fragment."""

    train = values["training_action_histogram"]
    ub = values["c0"]["unweighted"]["balanced"]
    wb = values["c0"]["action_balanced"]["balanced"]
    rows = []
    for action in ACTION_ORDER:
        rows.append(
            (
                DISPLAY_ACTIONS[action],
                str(train[action]),
                _recall_with_denominator(ub, action),
                _recall_with_denominator(wb, action),
            )
        )
    body = "\n".join("{} & {} & {} & {} \\\\".format(*row) for row in rows)
    return (
        "% Generated from independently verified JSON; do not edit by hand.\n"
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Balanced-surface recall follows a frequency-aligned descriptive pattern; "
        "this is an association, not a causal mechanism.}\n"
        "\\label{tab:action-recall}\n"
        "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n"
        "Action & Train targets & Unweighted recall & Action-balanced recall "
        + r"\\"
        + "\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def render_evaluator_control_table(values: Mapping[str, Any]) -> str:
    """Return the non-learning evaluator controls as a LaTeX fragment."""

    rows = []
    for row in values["evaluator_controls"]:
        runs = str(row["run_count"])
        if row["seeds"]:
            runs = f"{runs} seeds"
        rows.append(
            (
                row["label"],
                runs,
                f"{row['episode_count_per_run']}",
                f"{100 * row['success']:.2f}",
                f"{100 * row['spl']:.2f}",
                f"{row['ndtw']:.3f}",
            )
        )
    body = "\n".join("{} & {} & {} & {} & {} & {} \\\\".format(*row) for row in rows)
    return (
        "% Generated from independently verified JSON; do not edit by hand.\n"
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Locked modern-runtime evaluator controls on 61 source-locked "
        "$\\textit{val}_{seen}$ trajectories. Stochastic rows report means over five "
        "fixed seeds.}\n"
        "\\label{tab:evaluator-controls}\n"
        "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        "Control & Runs & Episodes/run & SR (\\%) & SPL (\\%) & nDTW "
        + r"\\"
        + "\n\\midrule\n"
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def build_provenance(
    *,
    repo: Path,
    values_path: Path,
    source_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Build a machine-readable provenance record for generated paper assets."""

    return {
        "schema_version": SCHEMA_VERSION,
        "gate": "native_qwen_c0_paper_asset_provenance_v2",
        "status": "passed",
        "sources": {
            label: {
                "path": str(path.relative_to(repo)),
                "sha256": sha256_file(path),
            }
            for label, path in source_paths.items()
        },
        "paper_values": {
            "path": str(values_path.relative_to(repo)),
            "sha256": sha256_file(values_path),
        },
        "generated_outputs": {
            label: {
                "path": str(path.relative_to(repo)),
                "sha256": sha256_file(path),
            }
            for label, path in output_paths.items()
        },
        "claim_map": {
            "natural_ce_checkpoint_selection_can_prefer_stop_muted_policy": [
                "paper_values.seed_replication.primary_findings.checkpoint_selection_failure_reproduced",
                "paper_values.seed_replication.consistency_counts_out_of_three.unweighted_natural_condition_pass_seed_count",
                "paper_values.seed_replication.consistency_counts_out_of_three.unweighted_zero_selection_stop_prediction_seed_count",
            ],
            "action_weighting_partially_restores_support_with_reproduced_tradeoff": [
                "paper_values.seed_replication.paired_deltas_inverse_frequency_minus_unweighted",
                "paper_values.seed_replication.consistency_counts_out_of_three",
            ],
            "frequency_aligned_association_not_mechanism": [
                "paper_values.training_action_histogram",
                "paper_values.c0.unweighted.balanced.per_action_recall",
            ],
            "original_seed_stop_suppression_signature": ["paper_values.selection_stop_trace"],
            "frozen_existence_gate_never_passed": [
                "paper_values.seed_replication.primary_findings.frozen_existence_gate_ever_passed"
            ],
        },
    }
