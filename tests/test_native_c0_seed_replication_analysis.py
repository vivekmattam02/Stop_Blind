from __future__ import annotations

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "src/spatialvlm/training/native_c0_seed_replication_analysis.py"
SPEC = importlib.util.spec_from_file_location(
    "native_c0_seed_replication_analysis_test", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load seed-replication analysis module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _result(*, stop: int, natural_ce: float, balanced_ce: float, accuracy: float) -> dict:
    actions = {
        action: {"count": 64, "recall": stop / 64 if action == "STOP" else accuracy}
        for action in MODULE.ACTIONS
    }
    summary = {
        "decision_count": 256,
        "mean_restricted_action_ce": balanced_ce,
        "descriptive_accuracy": accuracy,
        "per_target_action": actions,
        "predicted_action_histogram": {
            "MOVE_FORWARD": 256 - stop,
            "TURN_LEFT": 0,
            "TURN_RIGHT": 0,
            "STOP": stop,
        },
    }
    natural = {**summary, "decision_count": 640, "mean_restricted_action_ce": natural_ce}
    selected = {**summary, "decision_count": 256}
    return {
        "status": "not_demonstrated",
        "completed_updates": 192,
        "initial_trainable_state_sha256": "paired",
        "selection": {"selected_update": 160, "selected_mean_restricted_action_ce": natural_ce},
        "calibrations": [{"update_step": 160, "summary": selected}],
        "primary_evaluation": {"summary": natural},
        "balanced_diagnostic": {"summary": summary},
        "existence_gate": {
            "status": "not_demonstrated_at_frozen_budget",
            "conditions": {
                "natural_upper_95_below_blind_prior_minus_margin": True,
                "balanced_upper_95_below_uniform_minus_margin": False,
                "stop_recall_at_or_above_non_mute_floor": False,
            },
            "primary_natural": {"upper_95_restricted_action_ce": natural_ce + 0.1},
            "balanced_diagnostic": {
                "upper_95_restricted_action_ce": balanced_ce + 0.1,
                "stop_correct": stop,
                "stop_count": 64,
                "stop_recall": stop / 64,
            },
        },
    }


def test_three_seed_paired_directions_and_no_selection() -> None:
    runs = {}
    for seed in MODULE.SEEDS:
        unweighted = _result(stop=0, natural_ce=0.7, balanced_ce=1.7, accuracy=0.3)
        weighted = _result(stop=8, natural_ce=0.9, balanced_ce=1.3, accuracy=0.4)
        runs[seed] = {"unweighted": unweighted, "inverse_frequency": weighted}
    analysis = MODULE.build_analysis(
        runs=runs,
        evidence={},
        source_locks={},
        recorded_at_utc="2026-08-25T00:00:00+00:00",
    )
    counts = analysis["consistency_counts_out_of_three"]
    assert counts["unweighted_zero_selection_stop_prediction_seed_count"] == 3
    assert counts["inverse_frequency_natural_ce_worse_seed_count"] == 3
    assert counts["inverse_frequency_balanced_ce_better_seed_count"] == 3
    assert counts["inverse_frequency_stop_recall_better_seed_count"] == 3
    assert math.isclose(
        analysis["paired_deltas_inverse_frequency_minus_unweighted"]["balanced_mean_ce"]["mean"],
        -0.4,
    )


def test_missing_or_unpaired_seed_is_rejected() -> None:
    result = _result(stop=0, natural_ce=0.7, balanced_ce=1.7, accuracy=0.3)
    runs = {seed: {"unweighted": result, "inverse_frequency": result} for seed in MODULE.SEEDS}
    runs[20260827]["inverse_frequency"] = {**result, "initial_trainable_state_sha256": "other"}
    try:
        MODULE.build_analysis(
            runs=runs,
            evidence={},
            source_locks={},
            recorded_at_utc="2026-08-25T00:00:00+00:00",
        )
    except ValueError as exc:
        assert "initial trainable state" in str(exc)
    else:
        raise AssertionError("Expected unpaired initialization to fail")
