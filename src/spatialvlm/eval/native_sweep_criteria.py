"""Offline criteria K1 to K6 of the checkpoint sweep (``PREREG_31_CHECKPOINTS.md`` section 8).

Pure functions over the per-decision records that the training function ``_evaluate`` writes
(the format of ``calibrations[k].per_decision``) and over the output of the gate's own function,
``evaluate_existence_gate``. The offline scorer records these values for each of the 37 policies,
and the analysis script recomputes them from the scorer's per-decision records.

* K1: mean restricted-action CE on the 256 selection decisions.
* K2: mean restricted-action CE on the 640 natural decisions.
* K3: class-balanced CE on the 256 balanced decisions: the observed mean restricted-action CE,
  ``existence_gate.balanced_diagnostic.observed_mean_restricted_action_ce``. With 64 decisions per
  action it is the average of the four per-action CEs. It is not the bootstrap upper bound.
* K4: balanced STOP recall out of 64, counted two ways: with exact ties going to STOP, as the
  scorer and the gate count it (``stop_correct``); and strictly, with ties excluded (the CR-052
  convention: STOP's token NLL is lower than every other action's).
* K5: STOP predictions on the 256 selection decisions, ties going to STOP.
* K6: the existence gate: all three of its conditions.

It also holds the scorer check of section 7.4: the offline step re-scores, on the study GPU, every
decision that training recorded (the selection surface at all 37 policies, the natural and
balanced surfaces at the six selected ones), and ``scorer_check`` decides whether the re-scored
values agree with the records closely enough for K1 to K6 to be used. The study GPU is not the
GB10 that made the records, so bit-identity is reported, not required. ``smoke_gate`` holds the
smoke's two tighter bars for phase 0 (section 6), which the smoke requires before the freeze and
the analysis does not use.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

STOP = "STOP"
ACTIONS = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
SURFACE_SIZES = {"selection": 256, "natural": 640, "balanced": 256}
BALANCED_PER_ACTION = 64
STOP_RECALL_FLOOR_COUNT = 16  # the gate's floor of 0.25, out of 64


def mean_ce(rows: Sequence[Mapping[str, Any]]) -> float:
    """Mean restricted-action CE, summed in record order as the training summary sums it."""
    if not rows:
        raise ValueError("no decisions")
    values = [float(row["restricted_action_ce"]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite restricted-action CE")
    return sum(values) / len(values)


def token_nlls(row: Mapping[str, Any]) -> dict[str, float]:
    """The four full-vocabulary token NLLs of one decision record."""
    scores = row["action_decision"]["scores"]
    return {action: float(scores[action]["token_nll"]) for action in ACTIONS}


def is_strict_stop(row: Mapping[str, Any]) -> bool:
    """STOP's NLL is strictly lower than every other action's: STOP wins without a tie."""
    nll = token_nlls(row)
    return all(nll[STOP] < nll[action] for action in ACTIONS if action != STOP)


def stop_hits(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """STOP recall on the balanced surface: tie-to-STOP count, strict count and the row count."""
    stop_rows = [row for row in rows if row["target_action"] == STOP]
    return {
        "stop_rows": len(stop_rows),
        "tie_to_stop": sum(row["predicted_action"] == STOP for row in stop_rows),
        "strict": sum(is_strict_stop(row) for row in stop_rows),
    }


def criteria(
    *,
    selection_rows: Sequence[Mapping[str, Any]],
    natural_rows: Sequence[Mapping[str, Any]],
    balanced_rows: Sequence[Mapping[str, Any]],
    existence_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """K1 to K6 of one policy, from its per-decision records and its existence-gate result."""
    for name, rows in (
        ("selection", selection_rows),
        ("natural", natural_rows),
        ("balanced", balanced_rows),
    ):
        if len(rows) != SURFACE_SIZES[name]:
            raise ValueError(f"{name} surface has {len(rows)} decisions, not {SURFACE_SIZES[name]}")
    balanced_gate = existence_gate["balanced_diagnostic"]
    k3 = mean_ce(balanced_rows)
    if k3 != float(balanced_gate["observed_mean_restricted_action_ce"]):
        raise ValueError("K3 differs from the gate's observed balanced mean")
    hits = stop_hits(balanced_rows)
    if hits["stop_rows"] != BALANCED_PER_ACTION:
        raise ValueError("the balanced surface does not hold 64 STOP decisions")
    if hits["tie_to_stop"] != int(balanced_gate["stop_correct"]):
        raise ValueError("K4 differs from the gate's own stop_correct")
    conditions = dict(existence_gate["conditions"])
    return {
        "K1_selection_ce": mean_ce(selection_rows),
        "K2_natural_ce": mean_ce(natural_rows),
        "K3_balanced_ce": k3,
        "K4_stop_recall_tie_to_stop": hits["tie_to_stop"],
        "K4_stop_recall_strict": hits["strict"],
        "K5_selection_stop_predictions": sum(
            row["predicted_action"] == STOP for row in selection_rows
        ),
        "K6_gate_passed": existence_gate["status"] == "passed" and all(conditions.values()),
        "K6_conditions": conditions,
        "natural_upper_95_ce": float(
            existence_gate["primary_natural"]["upper_95_restricted_action_ce"]
        ),
        "balanced_upper_95_ce": float(balanced_gate["upper_95_restricted_action_ce"]),
    }


# ── the scorer check (section 7.4) ────────────────────────────────────────────────────────────

# A changed action at a decision whose recorded top-two margin is this large or larger fails the
# check: a change of GPU moves a BF16 logit by a step or a few, which can flip a near-tie but not
# a decision won by a whole nat.
SCORER_CHECK_MARGIN_NATS = 1.0
# The largest allowed difference between a re-scored and a recorded mean restricted-action CE,
# per policy and surface.
SCORER_CHECK_MEAN_CE_NATS = 0.05
MARGIN_BINS = ("0", "0.125", "0.25", "0.375 to 0.875", "1.0 or more")
# Section 6: the smoke's phase 0 requires, beyond the check of 7.4, that no token NLL differs from
# its record by more than this, and that no action changed at a recorded margin of this or more.
SMOKE_GATE_TOKEN_NLL_NATS = 0.5
SMOKE_GATE_MARGIN_NATS = 0.75


def margin_bin(margin: float) -> str:
    """The recorded top-two margin, in BF16 steps of 0.125 nats up to the check's bar."""
    if margin < 0.0625:
        return "0"
    if margin < 0.1875:
        return "0.125"
    if margin < 0.3125:
        return "0.25"
    if margin < SCORER_CHECK_MARGIN_NATS - 1e-6:
        return "0.375 to 0.875"
    return "1.0 or more"


def compare_decisions(
    rescored: Sequence[Mapping[str, Any]], recorded: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare one policy's re-scored decisions on one surface with the training records."""
    if len(rescored) != len(recorded):
        raise ValueError(f"{len(rescored)} re-scored decisions against {len(recorded)} recorded")
    changed = dict.fromkeys(MARGIN_BINS, 0)
    largest = 0.0
    largest_changed_margin: float | None = None
    identical = 0
    for new, old in zip(rescored, recorded, strict=True):
        if (new["sample_index"], new["cache_sha256"]) != (old["sample_index"], old["cache_sha256"]):
            raise ValueError(f"decision {new['sample_index']} is not the recorded decision")
        new_nll, old_nll = token_nlls(new), token_nlls(old)
        difference = max(abs(new_nll[action] - old_nll[action]) for action in ACTIONS)
        if not math.isfinite(difference):
            raise ValueError(f"non-finite token NLL at decision {new['sample_index']}")
        largest = max(largest, difference)
        same_action = new["predicted_action"] == old["predicted_action"]
        identical += int(difference == 0.0 and same_action)
        if not same_action:
            margin = float(old["action_decision"]["top_two_margin_nats"])
            changed[margin_bin(margin)] += 1
            largest_changed_margin = max(margin, largest_changed_margin or 0.0)
    rescored_ce, recorded_ce = mean_ce(rescored), mean_ce(recorded)
    return {
        "decisions": len(rescored),
        "bit_identical_decisions": identical,
        "largest_token_nll_difference": largest,
        "changed_actions_by_recorded_margin": changed,
        "largest_recorded_margin_of_a_changed_action": largest_changed_margin,
        "mean_ce_rescored": rescored_ce,
        "mean_ce_recorded": recorded_ce,
        "mean_ce_difference": rescored_ce - recorded_ce,
    }


def scorer_check(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Section 7.4: the check passes if both hold over every compared policy and surface.

    1. No action changed at a decision whose recorded top-two margin is 1.0 nats or more.
    2. No mean restricted-action CE differs from its record by more than 0.05 nats.
    """
    failures = []
    for key, item in sorted(comparisons.items()):
        if item["changed_actions_by_recorded_margin"]["1.0 or more"]:
            failures.append(
                f"{key}: {item['changed_actions_by_recorded_margin']['1.0 or more']} actions "
                f"changed at a recorded margin of {SCORER_CHECK_MARGIN_NATS} nats or more"
            )
        if abs(item["mean_ce_difference"]) > SCORER_CHECK_MEAN_CE_NATS:
            failures.append(
                f"{key}: mean CE differs from the record by {item['mean_ce_difference']:.4f} nats"
            )
    totals = dict.fromkeys(MARGIN_BINS, 0)
    for item in comparisons.values():
        for name, count in item["changed_actions_by_recorded_margin"].items():
            totals[name] += count
    return {
        "passed": bool(comparisons) and not failures,
        "failures": failures,
        "compared": len(comparisons),
        "decisions": sum(item["decisions"] for item in comparisons.values()),
        "bit_identical_decisions": sum(
            item["bit_identical_decisions"] for item in comparisons.values()
        ),
        "largest_token_nll_difference": max(
            (item["largest_token_nll_difference"] for item in comparisons.values()), default=None
        ),
        "largest_mean_ce_difference": max(
            (abs(item["mean_ce_difference"]) for item in comparisons.values()), default=None
        ),
        "changed_actions_by_recorded_margin": totals,
        "margin_bar_nats": SCORER_CHECK_MARGIN_NATS,
        "mean_ce_bar_nats": SCORER_CHECK_MEAN_CE_NATS,
    }


def smoke_gate(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Section 6: the smoke's two tighter bars on phase 0, beyond the check of 7.4.

    1. No token NLL differs from its record by more than 0.5 nats.
    2. No action changed at a decision whose recorded top-two margin is 0.75 nats or more.
    """
    failures = []
    largest_nll = max(
        (float(item["largest_token_nll_difference"]) for item in comparisons.values()),
        default=None,
    )
    margins = [
        float(item["largest_recorded_margin_of_a_changed_action"])
        for item in comparisons.values()
        if item.get("largest_recorded_margin_of_a_changed_action") is not None
    ]
    largest_margin = max(margins, default=None)
    if largest_nll is None or largest_nll > SMOKE_GATE_TOKEN_NLL_NATS:
        failures.append(
            f"largest token NLL difference {largest_nll} nats is above {SMOKE_GATE_TOKEN_NLL_NATS}"
        )
    if largest_margin is not None and largest_margin >= SMOKE_GATE_MARGIN_NATS - 1e-6:
        failures.append(
            f"an action changed at a recorded margin of {largest_margin} nats, at or above "
            f"{SMOKE_GATE_MARGIN_NATS}"
        )
    return {
        "passed": bool(comparisons) and not failures,
        "failures": failures,
        "largest_token_nll_difference": largest_nll,
        "largest_recorded_margin_of_a_changed_action": largest_margin,
        "token_nll_bar_nats": SMOKE_GATE_TOKEN_NLL_NATS,
        "margin_bar_nats": SMOKE_GATE_MARGIN_NATS,
    }


def recorded_decisions(
    update_step: int, selected: bool, run_result: Mapping[str, Any]
) -> dict[str, list[Mapping[str, Any]]]:
    """The decisions training recorded for one policy of one run: the selection surface at every
    update, the natural and balanced surfaces at the run's selected update only."""
    (calibration,) = [
        item for item in run_result["calibrations"] if int(item["update_step"]) == update_step
    ]
    rows: dict[str, list[Mapping[str, Any]]] = {"selection": calibration["per_decision"]}
    if selected:
        rows["natural"] = run_result["primary_evaluation"]["per_decision"]
        rows["balanced"] = run_result["balanced_diagnostic"]["per_decision"]
    return rows


def compare_policy(
    checkpoint_id: str,
    surfaces: Mapping[str, Mapping[str, Any]],
    recorded: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """``compare_decisions`` for every surface training recorded for this policy."""
    return {
        f"{checkpoint_id}/{surface}": compare_decisions(surfaces[surface]["per_decision"], rows)
        for surface, rows in recorded.items()
    }
