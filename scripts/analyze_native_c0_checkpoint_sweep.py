#!/usr/bin/env python3
"""The analysis of the checkpoint sweep (PREREG_31_CHECKPOINTS.md, sections 5 and 7 to 9).

Everything this program computes is fixed in the preregistration before any rollout: the scorer
check, every rule's picks (A, B, C, D and G), H1 (its screen, and each candidate's test again
with exact-tie STOP successes counted as failures, against the main screen), H2 and H3 with their
scene-level checks, O1 to O3, the descriptions of section 8, the carried list with m, the
cross-check with the Spark recordings and its wording rule, the confirmation test and which row of
section 9 applies. No pick, bar or order is chosen by a person. A not-completed episode never makes
a test easier to pass (section 5): it counts against the candidate in each test, as discordant in
the screen's family, and in m so that m is never below what the completed episode could give; and
a candidate is carried only if it would be carried whatever the missing episodes gave.

It reads committed results only, and refuses before computing anything when:

* a result directory it reads, or the smoke result, is not committed to git, or differs from the
  commit (``git status`` must be clean for it);
* the freeze commit (the commit whose preregistration status line is the freeze line) cannot be
  found, or HEAD, or any commit a job ran from, is not the freeze commit or a descendant that
  differs from it only in what may change after the freeze: result files (``artifacts/torch/``),
  the kit folder (``scripts/hpc/stopblind_sweep/``) and Markdown other than the preregistration;
  every later commit a job ran from is listed. So the offline step and the first main-suite job may
  run from the freeze commit or from such a descendant;
* the smoke's commit differs from the freeze commit in anything but what may change after the
  freeze and the preregistration's status line; or a commit after the freeze changes the committed
  smoke result, or changes or removes any committed result file (results are only added). The
  smoke result is read as the freeze commit holds it, since it decides whether H2 runs;
* a Python file a job recorded does not hash as ``git show <commit>:<path>`` gives it;
* the results hold more than one GPU type, GPU name, GPU driver or image, or more than one
  preregistration;
* an episode has neither a report nor a not-completed record, or a checkpoint has no completed
  result.

Two stages:

* ``main``, after the offline and main runs are committed: the scorer check, the criteria and
  picks, H1, H2, H3, O1 to O3, the cross-check of 7.8, the carried list and the row of section 9
  as far as the main suite can say. It writes ``analysis_main.json`` and ``carried_list.json``.
* ``confirmation``, after the confirmation run is committed: the confirmation test for each
  carried candidate, and the row. It recomputes the main stage and refuses unless its carried
  list is the one the confirmation jobs recorded.

Outputs hold no time, no absolute path and not HEAD (which moves when the outputs are committed),
so a second reader re-derives them byte for byte from any later commit of the study branch, and an
existing output is never replaced by different bytes. The git conditions live in
``spatialvlm.eval.native_sweep_freeze``, which ``bundle.sh`` also runs before it builds a bundle of
a frozen commit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from spatialvlm.eval.native_checkpoint_allowlist import (
    ALLOWED_CHECKPOINTS,
    CANDIDATE_IDS,
    RUN_TAGS,
    RUNS,
    SELECTED_IDS,
    UNWEIGHTED_UNSELECTED_TRAINED_IDS,
    UPDATE_ZERO_DRIVEN_ID,
    SweepRefusalError,
    checkpoint_for,
    comparators_of,
    confirmation_scope_from_carried_list,
    refuse,
    resolve_checkpoint,
)
from spatialvlm.eval.native_sweep_criteria import (
    ACTIONS,
    STOP,
    compare_policy,
    criteria,
    recorded_decisions,
    scorer_check,
    token_nlls,
)
from spatialvlm.eval.native_sweep_freeze import (
    FreezeGuardError,
    blob_sha256,
    check_commit_after_freeze,
    check_smoke_commit,
    freeze_commit,
    git,
    preregistration_status,
)

SCHEMA_VERSION = 4
UPDATES = (0, 32, 64, 96, 128, 160, 192)
TRAINED_UPDATES = UPDATES[1:]
H1_FAMILY = 31
# Section 7.3: the screen's family-wise level. Bonferroni over 31 (0.05/31) is the lowest
# threshold the screen can use, reached when K(0.05) = 31.
H1_SCREEN_ALPHA = Fraction(1, 20)
H1_BONFERRONI_ALPHA = 0.05 / H1_FAMILY
H2_ALPHA = 0.05
O2_TOLERANCE = 1e-12
O2_ORDERINGS_PER_RUN = math.factorial(6)
# Section 8: the simultaneous one-sided 95% upper bound on the 37 success rates (Bonferroni).
FAMILY_BOUND_ALPHA = 0.05 / len(ALLOWED_CHECKPOINTS)
DESCRIPTIVE_P_LABEL = "unadjusted, descriptive"
CARRY_AT_MOST = 3
STOP_RECALL_FLOOR = 16
NEAR_TIE_NATS = 0.25
OFFLINE_CRITERIA_FOR_O2 = (
    ("K1", "K1_selection_ce"),
    ("K2", "K2_natural_ce"),
    ("K3", "K3_balanced_ce"),
    ("K4", "K4_stop_recall_strict"),
    ("K5", "K5_selection_stop_predictions"),
)
RECORDED_CRITERIA_FOR_O2 = (
    ("K1", "K1_selection_ce"),
    ("K5", "K5_selection_stop_predictions"),
)
OUTCOMES = ("Y1", "Y2", "Y3", "Y4")
SPARK_RUN_ID = "native_c0_closed_loop_v1"
SPARK_MANIFEST_SHA256 = "9d8e272d194106b924d4eee7e46e7dc3e478e023020954283773a788368a269a"
SPARK_RESULT_SHA256 = "be7ec8bd979ce587813243b53620b7c367128fc3f75066dec84d92786a3e361c"


# ── statistics ────────────────────────────────────────────────────────────────────────────────


def sign_test_p_exact(b: int, c: int) -> Fraction:
    """One-sided exact McNemar test: P(Binomial(b + c, 1/2) >= b), as an exact fraction."""
    n = b + c
    return Fraction(sum(math.comb(n, k) for k in range(b, n + 1)), 2**n)


def sign_test_p(b: int, c: int) -> float:
    """One-sided exact McNemar test: P(Binomial(b + c, 1/2) >= b). With no discordance, p = 1."""
    return float(sign_test_p_exact(b, c))


def minimum_attainable_p(b: int, c: int) -> Fraction:
    """The smallest p the test can give with b + c discordant units: 0.5^(b + c) (1 if none)."""
    return Fraction(1, 2 ** (b + c))


def tarone_k(p_min: Sequence[Fraction], gamma: Fraction, *, left_limit: bool = False) -> int:
    """Tarone's K(gamma): the smallest k >= 1 with M(gamma, k) <= k, where M(gamma, k) counts the
    hypotheses whose minimum attainable p is at most gamma / k (Zhu and Guo, arXiv 1711.08147,
    Procedure 2.1). With ``left_limit`` M counts those strictly below gamma / k, which is K just
    below gamma."""
    relevant = [p for p in p_min if p <= gamma]
    for k in range(1, len(p_min) + 1):
        bound = gamma / k
        count = sum(1 for p in relevant if (p < bound if left_limit else p <= bound))
        if count <= k:
            return k
    return len(p_min)


def screen_thresholds(
    p_min: Sequence[Fraction], alpha: Fraction = H1_SCREEN_ALPHA
) -> dict[str, Any]:
    """Section 7.3: the screen, Tarone's procedure made alpha-consistent (Hommel and Krummenauer
    1998; Procedure 2.2 of Zhu and Guo): H0_j is rejected if some gamma in (0, alpha] has
    p_j <= gamma / K(gamma).

    gamma / K(gamma) rises with gamma while K is constant and falls where K steps up, and K can
    step only where gamma = k p*_i. So the rule is: p_j <= alpha / K(alpha) (``at_alpha``), or
    p_j < gamma / K(gamma-) at some such step point gamma <= alpha (``below``, the largest of
    those). Exact fractions throughout.
    """
    k_alpha = tarone_k(p_min, alpha)
    at_alpha = alpha / k_alpha
    below = Fraction(0)
    for p_star in sorted({p for p in p_min if p <= alpha}):
        for k in range(1, len(p_min) + 1):
            gamma = k * p_star
            if gamma > alpha:
                break
            below = max(below, gamma / tarone_k(p_min, gamma, left_limit=True))
    return {"K_at_alpha": k_alpha, "at_alpha": at_alpha, "below": below}


def screen_rejects(p: Fraction, thresholds: Mapping[str, Any]) -> bool:
    """Whether a p-value clears the screen of 7.3, given ``screen_thresholds``."""
    return p <= thresholds["at_alpha"] or p < thresholds["below"]


def screen_bar(c: int, thresholds: Mapping[str, Any]) -> int:
    """The smallest b that clears the screen for this c (with the screen's thresholds)."""
    b = 0
    while not screen_rejects(sign_test_p_exact(b, c), thresholds):
        b += 1
    return b


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def smallest_b(c: int, alpha: float) -> int:
    """The smallest b with P(Binomial(b + c, 1/2) >= b) <= alpha: the bar for that c."""
    b = 0
    while sign_test_p(b, c) > alpha:
        b += 1
    return b


def discordance(
    first: Mapping[str, int], second: Mapping[str, int], units: Sequence[str]
) -> tuple[int, int]:
    """(b, c): units only ``first`` succeeds on, and units only ``second`` succeeds on."""
    b = sum(first[unit] == 1 and second[unit] == 0 for unit in units)
    c = sum(first[unit] == 0 and second[unit] == 1 for unit in units)
    return b, c


def scene_outcomes(success: Mapping[str, int], scene_of: Mapping[str, str]) -> dict[str, int]:
    """A policy's scene outcome is 1 if any of its episodes in that scene succeeds (7.7)."""
    scenes: dict[str, int] = {}
    for episode, scene in scene_of.items():
        scenes[scene] = max(scenes.get(scene, 0), int(success[episode]))
    return scenes


def paired_test(
    mine: Mapping[str, int],
    theirs: Mapping[str, int],
    episodes: Sequence[str],
    scene_of: Mapping[str, str],
    alpha: float,
) -> dict[str, Any]:
    """The one-sided exact McNemar test by episode, with the scene-level test beside it."""
    b, c = discordance(mine, theirs, episodes)
    p = sign_test_p(b, c)
    scenes = sorted(set(scene_of.values()))
    bs, cs = discordance(scene_outcomes(mine, scene_of), scene_outcomes(theirs, scene_of), scenes)
    ps = sign_test_p(bs, cs)
    return {
        "b": b,
        "c": c,
        "p": p,
        "alpha": alpha,
        "bar_b_for_this_c": smallest_b(c, alpha),
        "clears": p <= alpha,
        "scene_level": {"b": bs, "c": cs, "p": ps, "clears": ps <= alpha},
    }


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """Ranks starting at 1, with tied values sharing their average rank."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    position = 0
    while position < len(array):
        end = position
        while end + 1 < len(array) and array[order[end + 1]] == array[order[position]]:
            end += 1
        ranks[order[position : end + 1]] = (position + end) / 2.0 + 1.0
        position = end + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    return float((xc * yc).sum() / math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum())))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """The Pearson correlation of average ranks (section 8, O2)."""
    return pearson(average_ranks(x), average_ranks(y))


def spearman_null(criterion_ranks: np.ndarray, outcome_ranks: np.ndarray) -> dict[float, int]:
    """One run's null distribution of the Spearman correlation: its value under each of the 720
    orderings of the six outcomes, as {value: number of orderings}. Centred average ranks are
    multiples of 1/2, so every numerator is exact and equal orderings give equal values."""
    x = criterion_ranks - criterion_ranks.mean()
    y = outcome_ranks - outcome_ranks.mean()
    denominator = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    numerators: dict[float, int] = {}
    for order in itertools.permutations(range(len(y))):
        numerator = float((x * y[list(order)]).sum())
        numerators[numerator] = numerators.get(numerator, 0) + 1
    return {numerator / denominator: count for numerator, count in numerators.items()}


def _sum_distribution(runs: Sequence[Mapping[float, int]]) -> tuple[np.ndarray, np.ndarray]:
    """The distribution of the sum of independent run values: (values, counts), by convolution."""
    values, counts = np.zeros(1), np.ones(1, dtype=np.int64)
    for run in runs:
        run_values = np.array(list(run.keys()))
        run_counts = np.array(list(run.values()), dtype=np.int64)
        values = np.add.outer(values, run_values).ravel()
        counts = np.multiply.outer(counts, run_counts).ravel()
    return values, counts


def _tail_count(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray], bound: float
) -> int:
    """How many orderings give |a + b| >= bound, with a from ``first`` and b from ``second``."""
    if bound <= 0:
        return int(first[1].sum()) * int(second[1].sum())
    order = np.argsort(second[0], kind="mergesort")
    values, counts = second[0][order], second[1][order]
    cumulative = np.concatenate([[0], np.cumsum(counts)])
    total = int(cumulative[-1])
    upper = np.searchsorted(values, bound - first[0], side="left")  # b >= bound - a
    lower = np.searchsorted(values, -bound - first[0], side="right")  # b <= -bound - a
    per_a = (total - cumulative[upper]) + cumulative[lower]
    return int((per_a.astype(object) * first[1].astype(object)).sum())


def o2_cell(
    criterion: Mapping[str, Sequence[float]],
    outcome: Mapping[str, Sequence[float]],
    alpha: float,
) -> dict[str, Any]:
    """One O2 cell: the mean within-run Spearman correlation T and its exact permutation p-value.

    Runs in which the criterion or the outcome is constant are dropped. Under the null every one
    of the 720^R orderings of the kept runs' outcomes (each run's six reordered within the run) is
    equally likely; T's distribution is the convolution of the runs' distributions, and
    p = P(|T| >= |T_observed| - 1e-12), two-sided, exactly. The smallest p the cell can give is
    the same probability at the largest |T| any ordering reaches. A cell whose smallest p is not
    below ``alpha`` is untestable: no outcome could have been called an association.
    """
    kept = [tag for tag in RUN_TAGS if len(set(criterion[tag])) > 1 and len(set(outcome[tag])) > 1]
    dropped = [tag for tag in RUN_TAGS if tag not in kept]
    if not kept:
        return {
            "computed": False,
            "status": "not computed: every run dropped",
            "runs_kept": [],
            "runs_dropped": dropped,
        }
    crit_ranks = {tag: average_ranks(criterion[tag]) for tag in kept}
    out_ranks = {tag: average_ranks(outcome[tag]) for tag in kept}
    observed = sum(pearson(crit_ranks[tag], out_ranks[tag]) for tag in kept) / len(kept)
    nulls = [spearman_null(crit_ranks[tag], out_ranks[tag]) for tag in kept]
    half = len(nulls) // 2
    first, second = _sum_distribution(nulls[:half]), _sum_distribution(nulls[half:])
    orderings = O2_ORDERINGS_PER_RUN ** len(kept)
    runs = len(kept)
    extreme = _tail_count(first, second, runs * (abs(observed) - O2_TOLERANCE))
    largest = (
        max(abs(sum(max(null) for null in nulls)), abs(sum(min(null) for null in nulls))) / runs
    )
    at_largest = _tail_count(first, second, runs * (largest - O2_TOLERANCE))
    p, p_min = Fraction(extreme, orderings), Fraction(at_largest, orderings)
    testable = p_min < alpha
    association = p < alpha
    if not testable:
        status = "untestable"
    elif association:
        status = "association"
    else:
        status = "no association detectable in 36 checkpoints"
    return {
        "computed": True,
        "T": observed,
        "p": float(p),
        "p_exact": fraction_text(p),
        "p_min": float(p_min),
        "p_min_exact": fraction_text(p_min),
        "testable": testable,
        "association": association,
        "status": status,
        "runs_kept": kept,
        "runs_dropped": dropped,
    }


# ── criteria and rule picks (section 8) ───────────────────────────────────────────────────────


def recorded_criteria(results: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """K1 and K5 of all 37 policies from what training recorded (``calibrations[k].summary``).

    Used for O2 when the offline step cannot be used (section 7.4). Update 0 is uw25-u0's.
    """
    table: dict[str, dict[str, Any]] = {}
    for record in ALLOWED_CHECKPOINTS:
        result = results[record.source_run_id]
        (calibration,) = [
            item
            for item in result["calibrations"]
            if int(item["update_step"]) == record.update_step
        ]
        summary = calibration["summary"]
        table[record.checkpoint_id] = {
            "K1_selection_ce": float(summary["mean_restricted_action_ce"]),
            "K5_selection_stop_predictions": int(
                summary["predicted_action_histogram"].get(STOP, 0)
            ),
        }
    return table


def recorded_picks(results: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Rule A: the update each run's preregistered rule selected, as its result.json records."""
    picks = {}
    for run in RUNS:
        selected = int(results[run.source_run_id]["selection"]["selected_update"])
        if selected != run.selected_update:
            raise refuse(f"{run.source_run_id} records selected update {selected}.")
        picks[run.tag] = selected
    return picks


def rule_picks(
    table: Mapping[str, Mapping[str, Any]] | None, rule_a: Mapping[str, int]
) -> dict[str, dict[str, int | None]]:
    """The six picks of each rule (section 8, O1), all seven checkpoints eligible.

    A: the recorded selected update. D: update 192. With the offline table: B, lowest K3, ties to
    the earlier update; C, highest strict K4, ties to the lower K3 and then the earlier update; G,
    lowest K1 among the checkpoints that pass K6, ties to the earlier update, None where none
    passes. Without it, B, C and G are not computed.
    """
    picks: dict[str, dict[str, int | None]] = {
        "A": {tag: int(rule_a[tag]) for tag in RUN_TAGS},
        "D": dict.fromkeys(RUN_TAGS, 192),
    }
    if table is None:
        return picks
    picks.update({"B": {}, "C": {}, "G": {}})
    for tag in RUN_TAGS:
        values = {u: table[checkpoint_for(tag, u).checkpoint_id] for u in UPDATES}
        picks["B"][tag] = min(UPDATES, key=lambda u: (values[u]["K3_balanced_ce"], u))
        picks["C"][tag] = min(
            UPDATES,
            key=lambda u: (-values[u]["K4_stop_recall_strict"], values[u]["K3_balanced_ce"], u),
        )
        passing = [u for u in UPDATES if values[u]["K6_gate_passed"]]
        picks["G"][tag] = (
            min(passing, key=lambda u: (values[u]["K1_selection_ce"], u)) if passing else None
        )
    return {rule: picks[rule] for rule in sorted(picks)}


# ── outcomes and the missing-episode rule (section 5) ────────────────────────────────────────


class Outcomes:
    """Per-episode outcomes of every driven policy on one suite.

    ``reports[checkpoint_id][episode_id]`` is an episode report, or None when the pair was
    recorded as not completed. A not-completed episode is never dropped: in every test it counts
    against the claim at stake, which the caller states with ``missing_counts_as``; its secondary
    outcomes are missing. A report's ``terminal_stop_exact_tie`` (set by ``load_rollouts``) says
    whether the episode ended with a STOP chosen by an exact tie; ``tie_stop_as_failure`` counts
    such a success as a failure (the sensitivity of 7.3).
    """

    def __init__(
        self,
        reports: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
        entries: Sequence[Mapping[str, Any]],
    ) -> None:
        self.reports = reports
        self.episodes = [str(entry["episode_id"]) for entry in entries]
        self.scene_of = {str(entry["episode_id"]): str(entry["scene_id"]) for entry in entries}

    def _metric(self, checkpoint_id: str, episode: str, key: str) -> Any:
        report = self.reports[checkpoint_id][episode]
        return None if report is None else report["evaluation"]["metrics"][key]

    def tie_stop(self, checkpoint_id: str, episode: str) -> bool:
        report = self.reports[checkpoint_id][episode]
        return report is not None and bool(report.get("terminal_stop_exact_tie", False))

    def success(
        self,
        checkpoint_id: str,
        *,
        missing_counts_as: int = 0,
        tie_stop_as_failure: bool = False,
    ) -> dict[str, int]:
        result = {}
        for episode in self.episodes:
            if self.reports[checkpoint_id][episode] is None:
                result[episode] = int(missing_counts_as)
            elif tie_stop_as_failure and self.tie_stop(checkpoint_id, episode):
                result[episode] = 0
            else:
                result[episode] = int(float(self._metric(checkpoint_id, episode, "success")) == 1.0)
        return result

    def tie_stop_successes(self, checkpoint_id: str) -> list[str]:
        """The episodes this policy succeeded on with a STOP chosen by an exact tie (7.1)."""
        won = self.success(checkpoint_id)
        return [e for e in self.episodes if won[e] and self.tie_stop(checkpoint_id, e)]

    def completed(self, checkpoint_id: str) -> list[str]:
        return [e for e in self.episodes if self.reports[checkpoint_id][e] is not None]

    def missing(self, checkpoint_id: str) -> list[str]:
        return [e for e in self.episodes if self.reports[checkpoint_id][e] is None]

    def summary(self, checkpoint_id: str) -> dict[str, Any]:
        done = self.completed(checkpoint_id)
        distances = [float(self._metric(checkpoint_id, e, "navigation_error")) for e in done]
        stops = [e for e in done if bool(self._metric(checkpoint_id, e, "stopped"))]
        successes = sum(self.success(checkpoint_id).values())
        return {
            "Y1": successes,
            "Y2": sum(float(self._metric(checkpoint_id, e, "oracle_success")) == 1.0 for e in done),
            "Y3": len(stops),
            "Y4": sum(distances) / len(distances) if distances else None,
            "clopper_pearson_95": clopper_pearson(successes, len(self.episodes)),
            "stop_presses": [
                {
                    "episode_id": e,
                    "distance_m": float(self._metric(checkpoint_id, e, "navigation_error")),
                    "step": int(self._metric(checkpoint_id, e, "step_count")),
                    "inside_goal_radius": float(self._metric(checkpoint_id, e, "success")) == 1.0,
                    "exact_tie": self.tie_stop(checkpoint_id, e),
                }
                for e in stops
            ],
            "completed_episodes": len(done),
            "not_completed_episodes": self.missing(checkpoint_id),
        }


def _beta_cdf_bound(k: int, n: int, target: float, upper: bool) -> float:
    """Exact Clopper-Pearson bound by bisection on the binomial tail."""

    def tail_at_least(p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_at_most(p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))

    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        if upper:
            # The upper bound solves P(X <= k | p) = target; that tail falls as p rises.
            if tail_at_most(middle) > target:
                low = middle
            else:
                high = middle
        else:
            # The lower bound solves P(X >= k | p) = target; that tail rises with p.
            if tail_at_least(middle) < target:
                low = middle
            else:
                high = middle
    return (low + high) / 2


def clopper_pearson(k: int, n: int) -> list[float]:
    """The exact two-sided 95% interval for k successes of n (section 7.1)."""
    lower = 0.0 if k == 0 else _beta_cdf_bound(k, n, 0.025, upper=False)
    upper = 1.0 if k == n else _beta_cdf_bound(k, n, 0.025, upper=True)
    return [round(lower, 6), round(upper, 6)]


def any_success(
    outcomes: Outcomes, rollouts: Sequence[str], *, missing_counts_as: int
) -> dict[str, int]:
    successes = [outcomes.success(item, missing_counts_as=missing_counts_as) for item in rollouts]
    return {episode: int(any(item[episode] for item in successes)) for episode in outcomes.episodes}


def rollout_of(tag: str, update: int | None) -> str | None:
    """Which driven checkpoint's main-run rollout a rule's pick contributes (section 8)."""
    return None if update is None else checkpoint_for(tag, update).checkpoint_id


# ── H1, H2, H3, O1-O3 and the carried list ────────────────────────────────────────────────────


def missing_comparator_episodes(outcomes: Outcomes) -> dict[str, list[str]]:
    """Section 5: each selected checkpoint's not-completed episodes, which every test of a
    candidate against it counts as successes of the comparator."""
    return {cid: outcomes.missing(cid) for cid in SELECTED_IDS}


def episodes_with_a_missing_side(outcomes: Outcomes, first: str, second: str) -> list[str]:
    """The episodes on which either policy of a pair was recorded as not completed."""
    return [
        episode
        for episode in outcomes.episodes
        if outcomes.reports[first][episode] is None or outcomes.reports[second][episode] is None
    ]


def family_p_min(
    outcomes: Outcomes, checkpoint_id: str, comparator: str, *, best_case: bool = False
) -> Fraction:
    """The smallest attainable p of one pair as the screen's family counts it (7.3, section 5).

    The discordant episodes are counted from the episodes as they came, with no tie-STOP change.
    By default every episode with a missing side counts as discordant, which gives the smallest
    p* that any outcome of the missing episodes could give. With ``best_case`` such an episode
    counts as concordant, which gives the largest."""
    gap = set(episodes_with_a_missing_side(outcomes, checkpoint_id, comparator))
    complete = [episode for episode in outcomes.episodes if episode not in gap]
    b, c = discordance(outcomes.success(checkpoint_id), outcomes.success(comparator), complete)
    return Fraction(1, 2 ** (b + c + (0 if best_case else len(gap))))


def screen_family(outcomes: Outcomes, *, best_case: bool = False) -> dict[str, Fraction]:
    """Each candidate's p* for the screen's family: for update 0 the largest of its six.

    By default (the screen of H1, and of its tie-STOP sensitivity) a missing episode can only
    lower a candidate's p* and so only keep it in the family: it never shrinks K and never lowers
    another candidate's bar. With ``best_case`` it gives the smallest family any outcome of the
    missing episodes could give (used for m, ``counted_for_m``)."""
    return {
        checkpoint_id: max(
            family_p_min(outcomes, checkpoint_id, item.checkpoint_id, best_case=best_case)
            for item in comparators_of(checkpoint_id)
        )
        for checkpoint_id in CANDIDATE_IDS
    }


def best_case(outcomes: Outcomes) -> dict[str, tuple[Fraction, int]]:
    """Each candidate's best case over the outcomes of the missing episodes (sections 5 and 7.5):
    its own not-completed episodes as successes and its comparators' as failures, ties as they
    came. Returns its p then (its smallest over those outcomes; for update 0 the largest of its
    six) and its successes then (its largest)."""
    result: dict[str, tuple[Fraction, int]] = {}
    for checkpoint_id in CANDIDATE_IDS:
        mine = outcomes.success(checkpoint_id, missing_counts_as=1)
        best_p = max(
            sign_test_p_exact(
                *discordance(
                    mine,
                    outcomes.success(item.checkpoint_id, missing_counts_as=0),
                    outcomes.episodes,
                )
            )
            for item in comparators_of(checkpoint_id)
        )
        result[checkpoint_id] = (best_p, sum(mine.values()))
    return result


def counted_for_m(
    outcomes: Outcomes, best: Mapping[str, tuple[Fraction, int]] | None = None
) -> list[str]:
    """Section 7.5: the candidates m counts, so that a missing episode never lowers m.

    A candidate counts if it clears the screen in its best case (``best_case``) against the
    screen's family with every episode that has a missing side counted as concordant (the
    screen's largest bar over the outcomes of the missing episodes). Whatever the missing episodes
    would have given, the candidates that would then clear are among these, and so are the
    candidates H1 clears; with nothing missing they are exactly the candidates H1 clears. Ties
    count as they came, so the tie-STOP sensitivity keeps the main test's m. The carry rule of
    ``carried_list`` ranks each candidate against these."""
    thresholds = screen_thresholds(list(screen_family(outcomes, best_case=True).values()))
    best = best_case(outcomes) if best is None else best
    return [cid for cid in CANDIDATE_IDS if screen_rejects(best[cid][0], thresholds)]


def h1(outcomes: Outcomes, *, tie_stop_as_failure: bool = False) -> dict[str, Any]:
    """Section 7.3: each candidate against its comparators, one-sided exact McNemar, screened by
    Tarone's procedure made alpha-consistent (``screen_thresholds``) over the 31 candidates.

    A trained candidate has one comparator. Update 0 has six (7.2): its p is the largest of its
    six p-values and its minimum attainable p the largest of theirs (an intersection-union test:
    it clears only if it beats each). In each test a missing episode counts as a failure of the
    candidate and a success of the comparator; in the family that sets K it counts as discordant
    (``screen_family``). So a missing episode never lowers a candidate's p and never lowers any
    candidate's bar. With ``tie_stop_as_failure`` a candidate's success that ended with an
    exact-tie STOP counts as a failure in its own test, against the main test's K: the family is
    counted from the episodes as they came, so a tie neither helps its own candidate nor, by
    shrinking K, another (the sensitivity of 7.3). Comparators are unchanged.
    """
    missing = missing_comparator_episodes(outcomes)
    family = screen_family(outcomes)
    thresholds = screen_thresholds([family[cid] for cid in CANDIDATE_IDS])
    best = best_case(outcomes)
    scenes = sorted(set(outcomes.scene_of.values()))
    rows: dict[str, Any] = {}
    for checkpoint_id in CANDIDATE_IDS:
        mine = outcomes.success(
            checkpoint_id, missing_counts_as=0, tie_stop_as_failure=tie_stop_as_failure
        )
        comparisons: dict[str, Any] = {}
        worst: tuple[Fraction, Fraction] | None = None
        binding = ""
        for comparator in (item.checkpoint_id for item in comparators_of(checkpoint_id)):
            theirs = outcomes.success(comparator, missing_counts_as=1)
            b, c = discordance(mine, theirs, outcomes.episodes)
            bs, cs = discordance(
                scene_outcomes(mine, outcomes.scene_of),
                scene_outcomes(theirs, outcomes.scene_of),
                scenes,
            )
            p = sign_test_p_exact(b, c)
            p_scene = sign_test_p_exact(bs, cs)
            comparisons[comparator] = {
                "b": b,
                "c": c,
                "p": float(p),
                "p_exact": fraction_text(p),
                "p_min": float(family_p_min(outcomes, checkpoint_id, comparator)),
                "episodes_with_a_missing_side": len(
                    episodes_with_a_missing_side(outcomes, checkpoint_id, comparator)
                ),
                "comparator_successes_missing_as_success": sum(theirs.values()),
                "comparator_episodes_counted_as_success_because_missing": len(missing[comparator]),
                "scene_level": {"b": bs, "c": cs, "p": float(p_scene)},
            }
            if worst is None:
                worst, binding = (p, p_scene), comparator
            else:
                if p > worst[0]:
                    binding = comparator
                worst = (max(worst[0], p), max(worst[1], p_scene))
        assert worst is not None
        p, p_scene = worst
        only = list(comparisons) if len(comparisons) == 1 else None
        rows[checkpoint_id] = {
            "successes": sum(mine.values()),
            "comparators": list(comparisons),
            "comparator": only[0] if only else None,
            "binding_comparator": binding,
            "b": comparisons[binding]["b"],
            "c": comparisons[binding]["c"],
            "p": float(p),
            "p_exact": fraction_text(p),
            "p_min": float(family[checkpoint_id]),
            "p_min_exact": fraction_text(family[checkpoint_id]),
            "testable": screen_rejects(family[checkpoint_id], thresholds),
            "clears": screen_rejects(p, thresholds),
            "bar_b_for_this_c": screen_bar(comparisons[binding]["c"], thresholds),
            "scene_level": {"p": float(p_scene), "clears": screen_rejects(p_scene, thresholds)},
            # Its own missing episodes as successes, its comparators' as failures (7.5).
            "best_case": {
                "p_exact": fraction_text(best[checkpoint_id][0]),
                "successes": best[checkpoint_id][1],
            },
            "comparisons": comparisons,
        }
    reached = [cid for cid, row in rows.items() if row["clears"]]
    return {
        "screen": (
            "Tarone's procedure made alpha-consistent (Hommel and Krummenauer 1998): reject H0_j "
            "if p_j <= gamma / K(gamma) for some gamma in (0, 0.05]"
        ),
        "family_rule": (
            "p* from the episodes as they came, every episode with a missing side counted as "
            "discordant; the tie-STOP sensitivity uses the same family"
        ),
        "alpha": float(H1_SCREEN_ALPHA),
        "family": H1_FAMILY,
        "K_at_alpha": thresholds["K_at_alpha"],
        "threshold_at_alpha": float(thresholds["at_alpha"]),
        "threshold_at_alpha_exact": fraction_text(thresholds["at_alpha"]),
        "threshold_below": float(thresholds["below"]),
        "threshold_below_exact": fraction_text(thresholds["below"]),
        "tie_stop_successes_counted_as_failures": tie_stop_as_failure,
        "candidates": rows,
        "testable": [cid for cid, row in rows.items() if row["testable"]],
        "reached_bar": reached,
        "holds_on_main_suite": bool(reached),
        "counted_for_m": counted_for_m(outcomes, best),
        "comparator_episodes_counted_as_success_because_missing": missing,
        "comparator_episodes_counted_as_success_because_missing_total": sum(
            len(episodes) for episodes in missing.values()
        ),
    }


NOT_CARRIED_RANK_DEPENDS = "not carried: rank depends on missing episodes"
NOT_CLEARED_DEPENDS = "not cleared: whether it clears depends on missing episodes"
NOT_CONFIRMED_DEPENDS = "not confirmed: confirmation depends on missing episodes"
ROW_2_SELECTION_ONLY = "2, selection reading only"
ROW_5_COMPLETED_ONLY = "5, completed episodes only"


def carried_list(h1_result: Mapping[str, Any]) -> dict[str, Any]:
    """Section 7.5: every candidate at the bar, up to three: smallest p, then most successes,
    then the earlier update, then the run order of section 3. No override.

    A candidate is carried only if it is among the first three whatever the missing episodes of
    the main suite gave (section 5): its rank in its worst case (its own missing episodes as
    failures, its comparators' as successes: its p and successes in H1) must be third or better
    against every other candidate that could clear (``counted_for_m``), each in its best case
    (``best_case``). Whatever the missing episodes would have given, a candidate carried here
    would be carried then, and fewer than three may be carried. A candidate that might have been
    carried had the missing episodes completed is listed: its best-case rank, against every
    candidate that clears H1 in its worst case, is third or better. One that clears H1 is listed as
    "not carried: rank depends on missing episodes", with both ranks and the candidates that might
    rank ahead of it; one that m counts but that does not clear H1 as run is listed as "not
    cleared: whether it clears depends on missing episodes". A candidate whose best-case rank is
    fourth or worse would not be carried whatever the missing episodes gave. With nothing missing,
    the carried list is the first three of the order above and both lists are empty.

    It also fixes m, the number confirmation divides 0.05 by: the candidates ``counted_for_m``,
    at most three, so that a missing episode on the main suite never lowers m. With nothing
    missing, m is the number carried. With nobody carried, m is 0."""
    rows = h1_result["candidates"]
    run_order = [run.source_run_id for run in RUNS]
    counted = list(h1_result["counted_for_m"])

    def key(checkpoint_id: str, case: Mapping[str, Any]) -> tuple[Fraction, int, int, int]:
        record = resolve_checkpoint(checkpoint_id)
        return (
            Fraction(case["p_exact"]),
            -int(case["successes"]),
            record.update_step,
            run_order.index(record.source_run_id),
        )

    worst = {cid: key(cid, rows[cid]) for cid in h1_result["reached_bar"]}
    best = {cid: key(cid, rows[cid]["best_case"]) for cid in counted}
    reached = sorted(worst, key=worst.__getitem__)
    ahead = {
        cid: [other for other in counted if other != cid and best[other] < worst[cid]]
        for cid in reached
    }
    chosen = [cid for cid in reached if len(ahead[cid]) < CARRY_AT_MOST]
    if not set(chosen) <= set(reached[:CARRY_AT_MOST]):
        raise refuse("the carry rule chose a candidate past the third; the analysis is broken.")
    # The best-case rank: against every candidate that clears in every outcome, in its worst case.
    best_rank = {
        cid: 1 + sum(1 for other in reached if other != cid and worst[other] < best[cid])
        for cid in counted
    }
    may_have_been_carried = sorted(
        (cid for cid in counted if cid not in chosen and best_rank[cid] <= CARRY_AT_MOST),
        key=best.__getitem__,
    )
    carried = []
    for checkpoint_id in chosen:
        record = resolve_checkpoint(checkpoint_id)
        comparators = [item.checkpoint_id for item in comparators_of(checkpoint_id)]
        entry: dict[str, Any] = {
            "checkpoint_id": checkpoint_id,
            "source_run_id": record.source_run_id,
            "update_step": record.update_step,
            "comparator_checkpoint_ids": comparators,
            "p_on_main_suite": rows[checkpoint_id]["p"],
            "successes_on_main_suite": rows[checkpoint_id]["successes"],
        }
        if len(comparators) == 1:
            # The single-comparator key, which the kit's job list also reads.
            entry["comparator_checkpoint_id"] = comparators[0]
        carried.append(entry)
    if not set(reached) <= set(counted):
        raise refuse("a candidate that clears H1 is not counted for m; the analysis is broken.")
    return {
        "schema_version": SCHEMA_VERSION,
        "carried": carried,
        "reached_bar_not_carried": [cid for cid in reached if cid not in chosen],
        "not_carried_rank_depends_on_missing_episodes": [
            {
                "checkpoint_id": cid,
                "reason": NOT_CARRIED_RANK_DEPENDS,
                "worst_case_rank": 1 + len(ahead[cid]),
                "best_case_rank": best_rank[cid],
                "may_rank_ahead": ahead[cid],
            }
            for cid in may_have_been_carried
            if cid in worst
        ],
        "not_cleared_clearing_depends_on_missing_episodes": [
            {
                "checkpoint_id": cid,
                "reason": NOT_CLEARED_DEPENDS,
                "best_case_rank": best_rank[cid],
            }
            for cid in may_have_been_carried
            if cid not in worst
        ],
        "m": min(CARRY_AT_MOST, len(counted)) if carried else 0,
        "counted_for_m": counted,
    }


def rules_against_a(
    picks: Mapping[str, Mapping[str, int | None]], outcomes: Outcomes
) -> dict[str, Any]:
    """O1, and H2's statistic: each rule's six picks against rule A's, by episode.

    A missing episode counts as a failure of the rule's picks and a success of rule A's.
    """
    a_rollouts = [rollout_of(tag, picks["A"][tag]) for tag in RUN_TAGS]
    a_joint = any_success(outcomes, [str(item) for item in a_rollouts], missing_counts_as=1)
    result: dict[str, Any] = {}
    for rule, by_run in picks.items():
        rollouts = {tag: rollout_of(tag, update) for tag, update in by_run.items()}
        driven = [item for item in rollouts.values() if item is not None]
        joint = (
            any_success(outcomes, driven, missing_counts_as=0)
            if driven
            else dict.fromkeys(outcomes.episodes, 0)
        )
        entry: dict[str, Any] = {
            "picks": dict(by_run),
            "rollouts": rollouts,
            "outcomes": {
                tag: outcomes.summary(item) if item else None for tag, item in rollouts.items()
            },
            "E": sum(joint.values()),
            "successes_over_run_episodes": sum(
                sum(outcomes.success(item).values()) for item in driven
            ),
            "runs_abstaining": sorted(tag for tag, item in rollouts.items() if item is None),
        }
        if rule == "B":
            entry["against_A"] = {
                **paired_test(joint, a_joint, outcomes.episodes, outcomes.scene_of, H2_ALPHA),
                "p_label": "the H2 test (7.4)",
            }
        elif rule != "A":
            # Rules C, D and G are description only (section 8, O1): no bar, no decision.
            test = paired_test(joint, a_joint, outcomes.episodes, outcomes.scene_of, H2_ALPHA)
            entry["against_A"] = {
                "b": test["b"],
                "c": test["c"],
                "p": test["p"],
                "p_label": DESCRIPTIVE_P_LABEL,
                "scene_level": {key: test["scene_level"][key] for key in ("b", "c", "p")},
            }
        result[rule] = entry
    result["A"]["E_missing_as_success"] = sum(a_joint.values())
    return result


def h2(
    o1: Mapping[str, Any],
    *,
    offline_usable: bool,
    any_driven_success: bool,
    reason: str = "the offline step failed its check or is incomplete",
) -> dict[str, Any]:
    """Section 7.4: rule B's six picks against rule A's, one-sided exact McNemar at 0.05.

    The number k of runs in which rule B picks rule A's checkpoint is always reported. If k = 6
    the two rules drive the same six rollouts, so b = c = 0 and the test says nothing: the status
    is "not informative: same picks in 6 of 6 runs", not "failed".
    """
    if not offline_usable:
        return {"status": "not run", "reason": reason}
    test = dict(o1["B"]["against_A"])
    same = [tag for tag in RUN_TAGS if o1["B"]["picks"][tag] == o1["A"]["picks"][tag]]
    record = {
        "E_B": o1["B"]["E"],
        "E_A": o1["A"]["E"],
        "E_D": o1["D"]["E"],
        **test,
        "same_picks_runs": same,
        "same_picks_count": len(same),
        "last_checkpoint_did_as_well": o1["D"]["E"] >= o1["B"]["E"]
        or o1["D"]["picks"] == o1["B"]["picks"],
    }
    if len(same) == len(RUN_TAGS):
        record["status"] = f"not informative: same picks in {len(same)} of {len(RUN_TAGS)} runs"
    elif not any_driven_success:
        record["status"] = "not informative: no success to rank"
    else:
        record["status"] = "passed" if test["clears"] else "failed"
    return record


def h3(outcomes: Outcomes) -> dict[str, Any]:
    """Section 7.6: the 15 unselected trained unweighted checkpoints never press STOP."""
    rows = {}
    for checkpoint_id in UNWEIGHTED_UNSELECTED_TRAINED_IDS:
        done = outcomes.completed(checkpoint_id)
        presses = [
            e
            for e in done
            if bool(outcomes.reports[checkpoint_id][e]["evaluation"]["metrics"]["stopped"])
        ]
        rows[checkpoint_id] = {
            "completed_episodes": len(done),
            "not_completed_episodes": len(outcomes.episodes) - len(done),
            "stop_presses": presses,
        }
    holds = all(not row["stop_presses"] for row in rows.values())
    missing = sum(row["not_completed_episodes"] for row in rows.values())
    completed = sum(row["completed_episodes"] for row in rows.values())
    selected_unweighted = [
        record.checkpoint_id
        for record in ALLOWED_CHECKPOINTS
        if record.role == "selected" and record.treatment == "unweighted"
    ]
    return {
        "checkpoints": rows,
        "holds": holds,
        "statement": (
            (
                "each of the 15 pressed STOP in 0 of 61 episodes"
                if missing == 0
                else f"no STOP press in the {completed} completed episodes ({missing} missing)"
            )
            if holds
            else "falsified: " + ", ".join(cid for cid, row in rows.items() if row["stop_presses"])
        ),
        "selected_unweighted_on_study_gpu": {
            cid: outcomes.summary(cid)["Y3"] for cid in selected_unweighted
        },
    }


def o2(
    table: Mapping[str, Mapping[str, Any]],
    outcomes: Outcomes,
    criteria_names: Sequence[tuple[str, str]],
    *,
    beside: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Section 8, O2: criteria against Y1-Y4 over the 36 trained checkpoints, within run."""
    alpha = 0.05 / (len(criteria_names) * len(OUTCOMES))

    def trained(tag: str) -> list[str]:
        return [checkpoint_for(tag, update).checkpoint_id for update in TRAINED_UPDATES]

    def outcome_of(checkpoint_id: str, name: str) -> float:
        value = outcomes.summary(checkpoint_id)[name]
        return math.nan if value is None else float(value)

    def cells_for(names: Sequence[tuple[str, str]]) -> dict[str, Any]:
        cells: dict[str, Any] = {}
        for short, key in names:
            crit = {tag: [float(table[cid][key]) for cid in trained(tag)] for tag in RUN_TAGS}
            for outcome in OUTCOMES:
                values = {
                    tag: [outcome_of(cid, outcome) for cid in trained(tag)] for tag in RUN_TAGS
                }
                if any(math.isnan(v) for tag in RUN_TAGS for v in values[tag]):
                    cells[f"{short}x{outcome}"] = {
                        "computed": False,
                        "status": "not computed: missing outcome",
                        "reason": "missing outcome",
                    }
                    continue
                cells[f"{short}x{outcome}"] = o2_cell(crit, values, alpha)
        return cells

    pooled: dict[str, Any] = {}
    for short, key in criteria_names:
        x = [float(table[cid][key]) for tag in RUN_TAGS for cid in trained(tag)]
        for outcome in OUTCOMES:
            y = [outcome_of(cid, outcome) for tag in RUN_TAGS for cid in trained(tag)]
            pooled[f"{short}x{outcome}"] = (
                spearman(x, y)
                if len(set(x)) > 1 and len(set(y)) > 1 and not any(map(math.isnan, y))
                else None
            )
    return {
        "alpha": alpha,
        "cells": cells_for(criteria_names),
        "reported_beside_not_tested": cells_for(beside) if beside else {},
        "pooled_spearman_description_only": pooled,
    }


def o3(table: Mapping[str, Mapping[str, Any]], outcomes: Outcomes) -> dict[str, Any]:
    """Section 8, O3: two-by-two counts over the 37 policies; counts only, no test."""

    def counts(flag_key: str, test: Any) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for threshold in (1, 10):
            cell = {
                "pass_and_reached": 0,
                "pass_not_reached": 0,
                "fail_and_reached": 0,
                "fail_not_reached": 0,
            }
            for record in ALLOWED_CHECKPOINTS:
                y1 = outcomes.summary(record.checkpoint_id)["Y1"]
                flag = "pass" if test(table[record.checkpoint_id][flag_key]) else "fail"
                reached = "and_reached" if y1 >= threshold else "not_reached"
                cell[f"{flag}_{reached}"] += 1
            result[f"Y1_at_least_{threshold}"] = cell
        return result

    return {
        "K6_gate": counts("K6_gate_passed", bool),
        "stop_recall_tie_to_stop_at_least_16": counts(
            "K4_stop_recall_tie_to_stop", lambda value: value >= STOP_RECALL_FLOOR
        ),
        "stop_recall_strict_at_least_16": counts(
            "K4_stop_recall_strict", lambda value: value >= STOP_RECALL_FLOOR
        ),
    }


def descriptive(outcomes: Outcomes) -> dict[str, Any]:
    """Section 8, the pre-specified descriptions (no test, no decision): each run's selection
    regret, the episodes any of the 37 policies solves, the largest success count, and the
    simultaneous one-sided 95% upper bound on the 37 success rates. Y1 counts a not-completed
    episode as a failure (section 5); the bound uses each policy's completed episodes. The title's
    k of section 9 counts each policy's not-completed episodes as successes, and the row-5 title
    is allowed only if none of the 37 succeeds and every episode of the 37 completed."""
    ids = [record.checkpoint_id for record in ALLOWED_CHECKPOINTS]
    y1 = {cid: sum(outcomes.success(cid).values()) for cid in ids}
    title_k = max(sum(outcomes.success(cid, missing_counts_as=1).values()) for cid in ids)
    not_completed = sum(len(outcomes.missing(cid)) for cid in ids)
    regret: dict[str, Any] = {}
    for run in RUNS:
        seven = [checkpoint_for(run.tag, update).checkpoint_id for update in UPDATES]
        selected = checkpoint_for(run.tag, run.selected_update).checkpoint_id
        best = max(y1[cid] for cid in seven)
        regret[run.tag] = {
            "selected": selected,
            "selected_successes": y1[selected],
            "most_successes_of_the_seven": best,
            "checkpoints_with_most": [cid for cid in seven if y1[cid] == best],
            "regret": best - y1[selected],
        }
    solved = any_success(outcomes, ids, missing_counts_as=0)
    bounds = {}
    for cid in ids:
        completed = len(outcomes.completed(cid))
        bounds[cid] = (
            1.0
            if completed == 0 or y1[cid] >= completed
            else round(_beta_cdf_bound(y1[cid], completed, FAMILY_BOUND_ALPHA, upper=True), 6)
        )
    return {
        "selection_regret": {
            "label": "biased upward: the largest of seven counts on the same 61 episodes",
            "runs": regret,
        },
        "episodes_solved_by_any_policy": sum(solved.values()),
        "episodes_solved_by_any_policy_ids": [e for e in outcomes.episodes if solved[e]],
        "largest_successes_any_policy": max(y1.values()),
        # Section 9's k: each policy's not-completed episodes counted as successes.
        "title_k": title_k,
        "not_completed_episodes_of_the_37": not_completed,
        "completed_episodes_of_the_37": len(ids) * len(outcomes.episodes) - not_completed,
        "row_5_title_allowed": max(y1.values()) == 0 and not_completed == 0,
        "family_upper_bound_95": {
            "method": (
                "one-sided Clopper-Pearson upper bound at 0.05/37 for each policy (successes of "
                "its completed episodes), Bonferroni over the 37"
            ),
            "largest": max(bounds.values()),
            "per_policy": bounds,
        },
    }


def section_9_main(h1_result: Mapping[str, Any], carried: Mapping[str, Any], outcomes: Outcomes):
    """The first row of section 9 that applies, as far as the main suite can say.

    Row 5 needs every episode of the 37 completed. If none of the 37 succeeds in a completed
    episode but some are missing, the row is "5, completed episodes only": row 5's statements
    hold only for the completed episodes, and its title is not allowed (section 9)."""
    if h1_result["reached_bar"]:
        return "pending confirmation" if carried["carried"] else "3"
    ids = [record.checkpoint_id for record in ALLOWED_CHECKPOINTS]
    if any(sum(outcomes.success(cid).values()) for cid in ids):
        return "4"
    return "5" if not any(outcomes.missing(cid) for cid in ids) else ROW_5_COMPLETED_ONLY


def confirmation_test(
    carried: Mapping[str, Any], outcomes: Outcomes, *, tie_stop_as_failure: bool = False
) -> dict[str, Any]:
    """Section 7.5: each carried candidate against its comparators on the confirmation suite, at
    0.05/m, with m as the carried list fixes it (``carried_list``): never below the number carried,
    and never lowered by a missing episode on the main suite. Update 0 is confirmed only if it
    clears against each of the six selected checkpoints. With ``tie_stop_as_failure`` a
    candidate's exact-tie STOP successes count as failures, at the same m.

    Row 2's first reading (training removed something the untrained model had) needs every
    trained candidate decided whatever the missing episodes gave. A trained candidate is undecided
    if the carried list says it might have been carried had the main suite's episodes completed,
    or if it was carried and is not confirmed but would be in its best case on this suite (its own
    missing episodes as successes, its comparators' as failures) at 0.05 over the number carried,
    the lowest m any outcome of the main suite's missing episodes could give. With update 0
    confirmed and some trained candidate undecided, the row is "2, selection reading only"."""
    entries = carried["carried"]
    m = carried.get("m")
    if (
        not isinstance(m, int)
        or isinstance(m, bool)
        or not len(entries) <= m <= CARRY_AT_MOST
        or (m == 0) != (not entries)
    ):
        raise refuse(f"the carried list gives m = {m!r} for {len(entries)} carried candidates.")
    alpha = 0.05 / m if m else None
    rows: dict[str, Any] = {}
    for item in entries:
        checkpoint_id = item["checkpoint_id"]
        comparators = [record.checkpoint_id for record in comparators_of(checkpoint_id)]
        listed = item.get("comparator_checkpoint_ids")
        if listed is not None and list(listed) != comparators:
            raise refuse(f"the carried list names other comparators for {checkpoint_id}.")
        mine = outcomes.success(
            checkpoint_id, missing_counts_as=0, tie_stop_as_failure=tie_stop_as_failure
        )
        comparisons: dict[str, Any] = {}
        level = float(alpha or 0)
        for comparator in comparators:
            theirs = outcomes.success(comparator, missing_counts_as=1)
            comparisons[comparator] = {
                **paired_test(mine, theirs, outcomes.episodes, outcomes.scene_of, level),
                "comparator_successes_missing_as_success": sum(theirs.values()),
                "comparator_episodes_counted_as_success_because_missing": outcomes.missing(
                    comparator
                ),
                "comparator_summary": outcomes.summary(comparator),
            }
        binding = max(comparators, key=lambda cid: (comparisons[cid]["p"], -comparators.index(cid)))
        confirmed = all(test["clears"] for test in comparisons.values())
        # Its best case here: its own missing episodes as successes, its comparators' as
        # failures, at the lowest m any outcome of the main suite could give.
        best_mine = outcomes.success(
            checkpoint_id, missing_counts_as=1, tie_stop_as_failure=tie_stop_as_failure
        )
        lenient = 0.05 / len(entries)
        best_confirmed = all(
            paired_test(
                best_mine,
                outcomes.success(comparator, missing_counts_as=0),
                outcomes.episodes,
                outcomes.scene_of,
                lenient,
            )["clears"]
            for comparator in comparators
        )
        rows[checkpoint_id] = {
            "b": comparisons[binding]["b"],
            "c": comparisons[binding]["c"],
            "p": comparisons[binding]["p"],
            "binding_comparator": binding,
            "comparisons": comparisons,
            "confirmed": confirmed,
            "confirmed_in_best_case": best_confirmed,
            "successes": sum(mine.values()),
            "summary": outcomes.summary(checkpoint_id),
        }
    trained = [
        cid for cid, row in rows.items() if row["confirmed"] and cid != UPDATE_ZERO_DRIVEN_ID
    ]
    update_zero = rows.get(UPDATE_ZERO_DRIVEN_ID, {}).get("confirmed", False)
    undecided = [
        {"checkpoint_id": str(item["checkpoint_id"]), "reason": str(item["reason"])}
        for key in (
            "not_carried_rank_depends_on_missing_episodes",
            "not_cleared_clearing_depends_on_missing_episodes",
        )
        for item in carried.get(key, [])
        if item["checkpoint_id"] != UPDATE_ZERO_DRIVEN_ID
    ] + [
        {"checkpoint_id": cid, "reason": NOT_CONFIRMED_DEPENDS}
        for cid, row in rows.items()
        if cid != UPDATE_ZERO_DRIVEN_ID and not row["confirmed"] and row["confirmed_in_best_case"]
    ]
    if trained:
        row = "1"
    elif update_zero:
        row = ROW_2_SELECTION_ONLY if undecided else "2"
    else:
        row = "3"
    return {
        "m": m,
        "counted_for_m": list(carried.get("counted_for_m", [])),
        "alpha": alpha,
        "candidates": rows,
        "trained_candidates_undecided_by_missing_episodes": undecided,
        "section_9_row": row,
    }


# ── the cross-check with the Spark recordings (7.8) ──────────────────────────────────────────


def _decision_nlls(report: Mapping[str, Any]) -> list[dict[str, float]]:
    return [
        token_nlls(decision["diagnostics"]) for decision in report["policy_telemetry"]["decisions"]
    ]


def terminal_stop_exact_tie(report: Mapping[str, Any]) -> bool:
    """Whether an episode ended with a STOP chosen by an exact tie (sections 7.1 and 7.3).

    An exact tie goes to the lowest action id, STOP (section 4.1). The episode ended with STOP if
    ``stopped`` is set; that STOP was a tie if, at the last decision, another action's token NLL
    equals STOP's (STOP is not strictly preferred: the CR-052 reading).
    """
    if not bool(report["evaluation"]["metrics"]["stopped"]):
        return False
    decisions = report["policy_telemetry"]["decisions"]
    if not decisions:
        raise refuse("a stopped episode records no decision.")
    nll = token_nlls(decisions[-1]["diagnostics"])
    if nll[STOP] != min(nll.values()):
        raise refuse("the last decision of a stopped episode did not choose STOP.")
    return any(nll[action] == nll[STOP] for action in ACTIONS if action != STOP)


def _margins(report: Mapping[str, Any]) -> list[float]:
    return [
        float(decision["diagnostics"]["action_decision"]["top_two_margin_nats"])
        for decision in report["policy_telemetry"]["decisions"]
    ]


def compare_rollouts(study: Mapping[str, Any], spark: Mapping[str, Any]) -> dict[str, Any]:
    """One episode pair: identical actions, first difference with both margins, flags, NLLs."""
    study_actions = list(study["evaluation"]["trace"]["actions"])
    spark_actions = list(spark["evaluation"]["trace"]["actions"])
    first = next(
        (i for i, (a, b) in enumerate(zip(study_actions, spark_actions, strict=False)) if a != b),
        None,
    )
    if first is None and len(study_actions) != len(spark_actions):
        first = min(len(study_actions), len(spark_actions))
    study_nll, spark_nll = _decision_nlls(study), _decision_nlls(spark)
    upto = len(study_actions) if first is None else first
    nll_difference = max(
        (
            abs(study_nll[i][action] - spark_nll[i][action])
            for i in range(min(upto, len(study_nll), len(spark_nll)))
            for action in study_nll[i]
        ),
        default=0.0,
    )
    study_metrics = study["evaluation"]["metrics"]
    spark_metrics = spark["evaluation"]["metrics"]
    flags = ("success", "oracle_success", "stopped")
    spark_margins, study_margins = _margins(spark), _margins(study)
    return {
        "identical_actions": first is None,
        "first_difference_step": first,
        "spark_margin_at_first_difference": (
            spark_margins[first] if first is not None and first < len(spark_margins) else None
        ),
        "study_margin_at_first_difference": (
            study_margins[first] if first is not None and first < len(study_margins) else None
        ),
        "flags_agree": {flag: study_metrics[flag] == spark_metrics[flag] for flag in flags},
        "success_differs": study_metrics["success"] != spark_metrics["success"],
        "final_distance_difference_m": float(study_metrics["navigation_error"])
        - float(spark_metrics["navigation_error"]),
        "largest_token_nll_difference_before_first_difference": nll_difference,
    }


def cross_check(
    outcomes: Outcomes,
    entries: Sequence[Mapping[str, Any]],
    spark: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Section 7.8: the six selected checkpoints, main run against the Spark, by episode.

    Reports are read one pair at a time (each is megabytes) and only the comparison is kept.
    """
    pairs: dict[str, dict[str, Any]] = {}
    for checkpoint_id in SELECTED_IDS:
        legacy = str(resolve_checkpoint(checkpoint_id).legacy_checkpoint_id)
        pairs[checkpoint_id] = {}
        for entry in entries:
            episode = str(entry["episode_id"])
            slim = outcomes.reports[checkpoint_id][episode]
            if slim is None:
                pairs[checkpoint_id][episode] = None
                continue
            study = _load_json(Path(slim["path"]))
            pairs[checkpoint_id][episode] = compare_rollouts(study, spark(legacy, entry))
    compared = [row for rows in pairs.values() for row in rows.values() if row is not None]
    differing = [
        (cid, episode, row)
        for cid, rows in pairs.items()
        for episode, row in rows.items()
        if row is not None and not row["identical_actions"]
    ]
    beyond = [
        {
            "checkpoint_id": cid,
            "episode_id": episode,
            "spark_margin": row["spark_margin_at_first_difference"],
        }
        for cid, episode, row in differing
        if row["spark_margin_at_first_difference"] is None
        or row["spark_margin_at_first_difference"] > NEAR_TIE_NATS + 1e-9
    ]
    success_differs = [
        {"checkpoint_id": cid, "episode_id": episode}
        for cid, rows in pairs.items()
        for episode, row in rows.items()
        if row is not None and row["success_differs"]
    ]
    all_compared = len(compared) == len(SELECTED_IDS) * len(entries)
    return {
        "episode_pairs_compared": len(compared),
        "episode_pairs_not_compared": [
            {"checkpoint_id": cid, "episode_id": episode}
            for cid, rows in pairs.items()
            for episode, row in rows.items()
            if row is None
        ],
        "identical_actions": sum(row["identical_actions"] for row in compared),
        "identical_flags": sum(all(row["flags_agree"].values()) for row in compared),
        # Section 7.8: the phrase needs every pair compared; otherwise it is limited to the
        # compared pairs, and the pairs not compared are named.
        "may_say_agree_up_to_near_ties": not beyond and all_compared,
        "may_say_agree_up_to_near_ties_in_the_compared_pairs": not beyond,
        "first_differences_beyond_near_tie": beyond,
        "success_differs_between_machines": success_differs,
        "near_tie_nats": NEAR_TIE_NATS,
        "pairs": pairs,
    }


# ── git: committed results and recorded code (the freeze guard is native_sweep_freeze) ────────


def check_committed(repo: Path, relative: str) -> None:
    """Refuse a result path that is not tracked by git or differs from HEAD."""
    tracked = git(repo, "ls-files", "--", relative).strip()
    if not tracked:
        raise refuse(f"{relative} is not committed; results are committed before analysis.")
    dirty = git(repo, "status", "--porcelain", "--untracked-files=all", "--", relative).strip()
    if dirty:
        raise refuse(f"{relative} differs from the commit:\n{dirty}")


def check_code_records(repo: Path, records: Sequence[Mapping[str, Any]]) -> int:
    """Every recorded Python file must hash as the commit its record names gives it."""
    checked: dict[tuple[str, str], str] = {}
    for record in records:
        commit = str(record["commit"])
        for path, digest in dict(record["files"]).items():
            key = (commit, path)
            if key not in checked:
                checked[key] = blob_sha256(repo, commit, path)
            if checked[key] != digest:
                raise refuse(f"{path} ran with sha256 {digest}, but commit {commit} holds another.")
    return len(checked)


# ── reading committed results ─────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise refuse(f"{path} is not a JSON object.")
    return payload


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class Provenance:
    """Everything the results record about where they ran, gathered for the refusals."""

    def __init__(self) -> None:
        self.gpu_types: set[str] = set()
        self.gpu_names: set[str] = set()
        self.gpu_drivers: set[str] = set()
        self.images: set[str] = set()
        self.preregistrations: set[str] = set()
        self.commits: dict[str, set[str]] = {}
        self.code_records: list[dict[str, Any]] = []
        self.smoke_results: set[str] = set()
        self.main_manifest_commits: list[tuple[str, str]] = []
        self.failed_attempts: dict[str, dict[str, dict[str, list[list[str]]]]] = {}

    def add(self, step: str, item: Mapping[str, Any] | None, *, prereg_key: str | None) -> None:
        if not item:
            return
        for key, target in (
            ("gpu_type", self.gpu_types),
            ("gpu_name", self.gpu_names),
            ("gpu_driver", self.gpu_drivers),
            ("image_sha256", self.images),
        ):
            if item.get(key) is not None:
                target.add(str(item[key]))
        if prereg_key is not None and item.get(prereg_key) is not None:
            self.preregistrations.add(str(item[prereg_key]))
        if item.get("code_commit") is not None:
            self.commits.setdefault(step, set()).add(str(item["code_commit"]))

    def add_manifest(self, step: str, manifest: Mapping[str, Any]) -> None:
        self.add(step, manifest.get("platform"), prereg_key="preregistration_sha256")
        self.preregistrations.add(str(manifest["preregistration"]["sha256"]))
        self.code_records.append(dict(manifest["code"]))
        if manifest.get("smoke_result"):
            self.smoke_results.add(str(manifest["smoke_result"]["sha256"]))
        if step == "main":
            self.main_manifest_commits.append(
                (str(manifest["recorded_at_utc"]), str(manifest["code"]["commit"]))
            )

    def check_single(self) -> dict[str, Any]:
        for name, values in (
            ("GPU type", self.gpu_types),
            ("GPU name", self.gpu_names),
            # Section 4.2: the host driver supplies the renderer that draws the agent's pixels and
            # libcuda, so one study runs on one driver.
            ("GPU driver", self.gpu_drivers),
            ("image", self.images),
            ("preregistration", self.preregistrations),
            ("smoke result", self.smoke_results),
        ):
            if len(values) > 1:
                raise refuse(f"the results hold more than one {name}: {sorted(values)}")
            if not values:
                raise refuse(f"the results record no {name}.")
        return {
            "gpu_type": next(iter(self.gpu_types)),
            "gpu_name": next(iter(self.gpu_names)),
            "gpu_driver": next(iter(self.gpu_drivers)),
            "image_sha256": next(iter(self.images)),
            "preregistration_sha256": next(iter(self.preregistrations)),
            "smoke_result_sha256": next(iter(self.smoke_results)),
        }


def _checkpoint_dirs(root: Path, expected: Sequence[str]) -> dict[str, Path]:
    dirs = {}
    for checkpoint_id in expected:
        directory = root / checkpoint_id
        result = directory / "result.json"
        if not result.is_file() or _load_json(result).get("status") != "completed":
            raise refuse(f"{checkpoint_id} has no completed result in {root}.")
        dirs[checkpoint_id] = directory
    return dirs


def load_rollouts(
    root: Path, step: str, expected: Sequence[str], provenance: Provenance
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Episode reports of one runner step; a not-completed pair is None. Records provenance."""
    reports: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] | None = None
    for checkpoint_id, directory in _checkpoint_dirs(root, expected).items():
        manifest = _load_json(directory / "manifest.json")
        result = _load_json(directory / "result.json")
        if (manifest.get("step"), manifest.get("checkpoint_id")) != (step, checkpoint_id):
            raise refuse(f"{directory} is not {step}/{checkpoint_id}.")
        if result.get("manifest_sha256") != _sha256(directory / "manifest.json"):
            raise refuse(f"{directory}'s result is not linked to its manifest.")
        if entries is None:
            entries = list(manifest["episodes"])
        elif manifest["episodes"] != entries:
            raise refuse(f"{directory} drove other episodes than the rest of {step}.")
        provenance.add_manifest(step, manifest)
        provenance.code_records.append(dict(result["code"]))
        provenance.add(step, result.get("platform"), prereg_key="preregistration_sha256")
        invocation_lines = _lines(directory / "invocations.jsonl")
        for line in invocation_lines:
            provenance.add(step, line.get("provenance"), prereg_key="preregistration_sha256")
        failed = failed_attempt_starts(invocation_lines)
        if failed:
            provenance.failed_attempts.setdefault(step, {})[checkpoint_id] = failed
        not_completed = {
            str(row["episode_id"])
            for row in _lines(directory / "not_completed.jsonl")
            if row.get("checkpoint_id") == checkpoint_id
        }
        manifest_sha = _sha256(directory / "manifest.json")
        reports[checkpoint_id] = {}
        for entry in entries:
            episode = str(entry["episode_id"])
            stem = str(entry["artifact_stem"])
            run_dir = directory / "runs" / checkpoint_id
            path = run_dir / "episodes" / f"{stem}.json"
            if path.is_file():
                report = _load_json(path)
                if report.get("manifest_sha256") != manifest_sha or report.get("status") != (
                    "completed"
                ):
                    raise refuse(f"{path} is not a completed report of its manifest.")
                # The report is admitted only as the episode its file name says, with the step and
                # timing files it names, as the locked runner checks before it reuses a report
                # (run_spark_native_c0_closed_loop.py, _load_existing_reports). A report copied
                # under another episode's name, or one whose per-step record is missing or changed,
                # would otherwise count (CR-054, finding 2).
                identity = {
                    "checkpoint": report.get("checkpoint_id") == checkpoint_id,
                    "episode": str(report.get("episode_id")) == episode,
                    "trajectory": str(report.get("trajectory_id")) == str(entry["trajectory_id"]),
                    "scene": str(report.get("scene_id")) == str(entry["scene_id"]),
                    "step": report.get("step", step) == step,
                }
                for kind in ("steps", "timings"):
                    linked = run_dir / kind / f"{stem}.jsonl"
                    named = str(report.get(f"{kind}_path") or "")
                    identity[kind] = (
                        named.endswith(f"/runs/{checkpoint_id}/{kind}/{stem}.jsonl")
                        and linked.is_file()
                        and report.get(f"{kind}_sha256") == _sha256(linked)
                    )
                failed_checks = sorted(name for name, ok in identity.items() if not ok)
                if failed_checks:
                    raise refuse(
                        f"{path} is not the report of {checkpoint_id}/{episode} with its own "
                        f"step and timing files: {', '.join(failed_checks)} failed."
                    )
                provenance.add(step, report.get("provenance"), prereg_key="preregistration_sha256")
                # Only the metrics and the tie flag stay in memory; the cross-check reads the
                # file again.
                reports[checkpoint_id][episode] = {
                    "evaluation": {"metrics": dict(report["evaluation"]["metrics"])},
                    "terminal_stop_exact_tie": terminal_stop_exact_tie(report),
                    "path": str(path),
                }
            elif episode in not_completed:
                reports[checkpoint_id][episode] = None
            else:
                raise refuse(f"no report and no not-completed record: {checkpoint_id}/{episode}")
    if entries is None:
        raise refuse(f"{root} holds no checkpoint.")
    return reports, entries


def load_offline(
    root: Path, provenance: Provenance, results: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any] | None]:
    """K1 to K6 of each scored policy, recomputed from its decisions, and its comparison with the
    training records; a policy that could not be scored is None."""
    policies: dict[str, dict[str, Any] | None] = {}
    expected = [record.checkpoint_id for record in ALLOWED_CHECKPOINTS]
    for checkpoint_id, directory in _checkpoint_dirs(root, expected).items():
        manifest = _load_json(directory / "manifest.json")
        result = _load_json(directory / "result.json")
        if (manifest.get("step"), manifest.get("checkpoint_id")) != ("offline", checkpoint_id):
            raise refuse(f"{directory} is not offline/{checkpoint_id}.")
        if result.get("manifest_sha256") != _sha256(directory / "manifest.json"):
            raise refuse(f"{directory}'s result is not linked to its manifest.")
        provenance.add_manifest("offline", manifest)
        provenance.code_records.append(dict(result["code"]))
        provenance.add("offline", result.get("platform"), prereg_key="preregistration_sha256")
        for line in _lines(directory / "invocations.jsonl"):
            provenance.add("offline", line.get("provenance"), prereg_key="preregistration_sha256")
        if not result.get("policy_scored"):
            policies[checkpoint_id] = None
            continue
        policy_path = directory / "policy.json"
        if result.get("policy_record_sha256") != _sha256(policy_path):
            raise refuse(f"{policy_path} is not the record its result names.")
        payload = _load_json(policy_path)
        provenance.add("offline", payload.get("provenance"), prereg_key="preregistration_sha256")
        surfaces = payload["surfaces"]
        values = criteria(
            selection_rows=surfaces["selection"]["per_decision"],
            natural_rows=surfaces["natural"]["per_decision"],
            balanced_rows=surfaces["balanced"]["per_decision"],
            existence_gate=payload["existence_gate"],
        )
        if values != payload["criteria"]:
            raise refuse(f"stored criteria do not recompute: {checkpoint_id}")
        record = resolve_checkpoint(checkpoint_id)
        recorded = recorded_decisions(
            record.update_step, record.role == "selected", results[record.source_run_id]
        )
        policies[checkpoint_id] = {
            "criteria": values,
            "comparisons": compare_policy(checkpoint_id, surfaces, recorded),
        }
    return policies


def smoke_gate_passed(smoke: Mapping[str, Any]) -> bool:
    """Section 6: whether the smoke's tighter phase-0 bars passed, as the committed smoke result
    records it. If they failed, H2, rules B, C and G and O3 are not run, whatever the offline step
    gives."""
    passed = (smoke.get("smoke_gate") or {}).get("passed")
    if not isinstance(passed, bool):
        raise refuse(
            "the committed smoke result does not record whether phase 0's tighter bars passed."
        )
    return passed


def offline_check(
    policies: Mapping[str, Mapping[str, Any] | None], *, smoke_gate: bool = True
) -> dict[str, Any]:
    """Section 7.4: the scorer check over every decision training recorded. The offline criteria
    are used only if it passes, every policy was scored, and the smoke's tighter bars passed."""
    comparisons = {
        key: value
        for payload in policies.values()
        if payload is not None
        for key, value in payload["comparisons"].items()
    }
    check = scorer_check(comparisons)
    missing = sorted(cid for cid, payload in policies.items() if payload is None)
    if not smoke_gate:
        reason = "the smoke's tighter phase-0 bars failed (section 6)"
    elif not check["passed"] or missing:
        reason = "the offline step failed its check or is incomplete"
    else:
        reason = None
    return {
        **check,
        "policies_not_scored": missing,
        "smoke_gate_passed": smoke_gate,
        "usable": reason is None,
        "not_used_because": reason,
        "comparisons": comparisons,
    }


def failed_attempt_starts(lines: Sequence[Mapping[str, Any]]) -> dict[str, list[list[str]]]:
    """Every episode (or the model load) with a failed attempt, and the job start of each attempt:
    an exit that was neither clean (0) nor resumable (75) and not an interruption."""
    found: dict[str, list[list[str]]] = {}
    for line in lines:
        if line.get("exit_code") in (0, 75) or line.get("stop_by_hand"):
            continue
        pair = line.get("pair_in_progress_at_exit") or {}
        if pair:
            key = str(pair["episode_id"])
        elif line.get("checkpoint_loading_at_exit"):
            key = "model load"
        else:
            continue
        prov = line.get("provenance") or {}
        found.setdefault(key, []).append([str(prov.get("job_id")), str(prov.get("node"))])
    return found


def spark_loader(root: Path) -> Callable[[str, Mapping[str, Any]], dict[str, Any]]:
    """The recorded Spark reports, read one at a time, each checked against its steps and timings
    files by the locked runner's own loader, after the run's manifest, result and verification
    are checked against their recorded hashes."""
    from scripts.run_spark_native_c0_closed_loop import _load_existing_reports

    if _sha256(root / "manifest.json") != SPARK_MANIFEST_SHA256:
        raise refuse("the Spark closed loop's manifest is not the recorded one.")
    if _sha256(root / "result.json") != SPARK_RESULT_SHA256:
        raise refuse("the Spark closed loop's result is not the recorded one.")
    verification = _load_json(root / "verification.json")
    if verification.get("status") != "passed" or verification.get("result_sha256") != (
        SPARK_RESULT_SHA256
    ):
        raise refuse("the Spark closed loop's verification is not passed and linked.")

    def get(legacy: str, entry: Mapping[str, Any]) -> dict[str, Any]:
        try:
            loaded = _load_existing_reports(root, [legacy], [entry], SPARK_MANIFEST_SHA256)
        except RuntimeError as error:
            raise refuse(f"a Spark report is inconsistent: {error}") from error
        report = loaded.get((legacy, str(entry["episode_id"])))
        if report is None:
            raise refuse(f"the Spark recording lacks {legacy}/{entry['episode_id']}.")
        return report

    return get


def training_results(source_root: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for run in RUNS:
        path = source_root / run.source_run_id / "result.json"
        if _sha256(path) != run.source_result_sha256:
            raise refuse(f"{path} is not the result native_c0_closed_loop_v1 pinned.")
        results[run.source_run_id] = _load_json(path)
    return results


def write_output(path: Path, payload: Mapping[str, Any]) -> str:
    """Write deterministic JSON; refuse to replace an existing file with different bytes."""
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.is_file():
        if path.read_bytes() != data:
            raise refuse(f"{path} exists with different content; outputs are never replaced.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return hashlib.sha256(data).hexdigest()


# ── the stages ────────────────────────────────────────────────────────────────────────────────


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--stage", choices=("main", "confirmation"), required=True)
    parser.add_argument("--preregistration", required=True, help="Repository-relative path.")
    parser.add_argument("--smoke-result", required=True, help="Repository-relative path.")
    parser.add_argument("--offline-root", required=True, help="Repository-relative path.")
    parser.add_argument("--main-root", required=True, help="Repository-relative path.")
    parser.add_argument("--confirmation-root", default=None, help="Repository-relative path.")
    parser.add_argument("--aborted-root", action="append", default=[])
    parser.add_argument("--source-artifact-root", default="artifacts/spark")
    parser.add_argument("--output-dir", required=True, help="Where the outputs are written.")
    return parser.parse_args(argv)


def _aborted_summary(root: Path) -> dict[str, Any]:
    """An aborted run (section 4.2), counted and reported beside, never pooled."""
    summary: dict[str, Any] = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        episodes = sorted((directory / "runs" / directory.name / "episodes").glob("*.json"))
        reports = [_load_json(path) for path in episodes]
        summary[directory.name] = {
            "completed_episodes": len(reports),
            "successes": sum(
                float(report["evaluation"]["metrics"]["success"]) == 1.0 for report in reports
            ),
            "gpu_names": sorted(
                {str(report.get("provenance", {}).get("gpu_name")) for report in reports}
            ),
        }
    return summary


def guard(args: argparse.Namespace, roots: Sequence[str]) -> dict[str, Any]:
    """The git refusals that come before any number."""
    repo = args.repo.resolve()
    if preregistration_status((repo / args.preregistration).read_text("utf-8")) != "frozen":
        raise refuse("the preregistration in the working tree is not frozen.")
    freeze = freeze_commit(repo, args.preregistration)
    head = git(repo, "rev-parse", "HEAD").strip()
    head_changes = check_commit_after_freeze(
        repo, freeze, head, args.preregistration, smoke_result=args.smoke_result
    )
    for relative in (*roots, args.smoke_result):
        check_committed(repo, relative)
    return {"repo": repo, "freeze": freeze, "head": head, "head_changes": head_changes}


def stage_main(args: argparse.Namespace) -> dict[str, Any]:
    roots = [args.offline_root, args.main_root, *args.aborted_root]
    checked = guard(args, roots)
    repo, freeze = checked["repo"], checked["freeze"]
    provenance = Provenance()
    results = training_results(repo / args.source_artifact_root)
    policies = load_offline(repo / args.offline_root, provenance, results)
    all_ids = [record.checkpoint_id for record in ALLOWED_CHECKPOINTS]
    reports, entries = load_rollouts(repo / args.main_root, "main", all_ids, provenance)
    return _main_payload(
        args, checked, provenance, results, policies, reports, entries, repo, freeze
    )


def _main_payload(
    args: argparse.Namespace,
    checked: Mapping[str, Any],
    provenance: Provenance,
    results: Mapping[str, Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any] | None],
    reports: Mapping[str, Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    repo: Path,
    freeze: str,
) -> dict[str, Any]:
    # The smoke result as the freeze commit holds it: it decides whether H2 runs (section 6), so
    # it is never read from a later commit, and the working copy must be the same bytes.
    smoke_bytes: bytes = git(repo, "show", f"{freeze}:{args.smoke_result}", binary=True)
    smoke_sha256 = hashlib.sha256(smoke_bytes).hexdigest()
    if _sha256(repo / args.smoke_result) != smoke_sha256:
        raise refuse("the smoke result in the working tree is not the one the freeze commit holds.")
    smoke = json.loads(smoke_bytes)
    smoke_changes = check_smoke_commit(
        repo, str(smoke["code_commit"]), freeze, args.preregistration
    )
    # The smoke ran under the DRAFT preregistration, so only its GPU, image and commit count.
    provenance.add("smoke", smoke, prereg_key=None)
    single = provenance.check_single()
    if single["smoke_result_sha256"] != smoke_sha256:
        raise refuse("the jobs checked another smoke result than the one the freeze commit holds.")
    if single["preregistration_sha256"] != blob_sha256(repo, freeze, args.preregistration):
        raise refuse("the jobs ran under another preregistration than the freeze commit's.")
    later: dict[str, list[str]] = {}
    for step, commits in sorted(provenance.commits.items()):
        if step == "smoke":
            continue
        for commit in sorted(commits):
            # The offline step and the first main-suite job included: the freeze commit or an
            # allowed descendant (Freeze, and section 5, step 4).
            changed = check_commit_after_freeze(
                repo, freeze, commit, args.preregistration, smoke_result=args.smoke_result
            )
            if commit != freeze:
                later[commit] = changed
    first_main = min(provenance.main_manifest_commits)[1]
    code_files_checked = check_code_records(repo, provenance.code_records)

    source_root = repo / args.source_artifact_root
    rule_a = recorded_picks(results)
    check = offline_check(policies, smoke_gate=smoke_gate_passed(smoke))
    table = (
        {cid: dict(payload["criteria"]) for cid, payload in policies.items() if payload}
        if check["usable"]
        else None
    )
    picks = rule_picks(table, rule_a)
    outcomes = Outcomes(reports, entries)
    h1_result = h1(outcomes)
    carried = carried_list(h1_result)
    o1 = rules_against_a(picks, outcomes)
    any_driven = any(sum(outcomes.success(cid).values()) for cid in reports)
    if table is not None:
        o2_result = o2(
            table,
            outcomes,
            OFFLINE_CRITERIA_FOR_O2,
            beside=(("K4tie", "K4_stop_recall_tie_to_stop"),),
        )
        o3_result: dict[str, Any] = o3(table, outcomes)
        criteria_source = "offline step on the study GPU"
    else:
        o2_result = o2(recorded_criteria(results), outcomes, RECORDED_CRITERIA_FOR_O2)
        o3_result = {"status": "not run", "reason": str(check["not_used_because"])}
        criteria_source = "training records (K1 and K5 only)"
    spark = spark_loader(source_root / SPARK_RUN_ID)
    aborted = {root: _aborted_summary(repo / root) for root in args.aborted_root}
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "main",
        "inputs": {
            "freeze_commit": freeze,
            "first_main_job_commit": first_main,
            "later_commits_and_what_they_changed": later,
            "smoke_commit": smoke["code_commit"],
            "smoke_to_freeze_changes": smoke_changes,
            "code_files_checked_against_git": code_files_checked,
            "preregistration": args.preregistration,
            "offline_root": args.offline_root,
            "main_root": args.main_root,
            **single,
        },
        "scorer_check": {k: v for k, v in check.items() if k != "comparisons"},
        "scorer_check_by_policy_and_surface": check["comparisons"],
        "criteria_source": criteria_source,
        "criteria": table,
        "picks": picks,
        "h1": h1_result,
        # Section 7.3: H1 again with each candidate's exact-tie STOP successes as failures.
        "h1_tie_stop_sensitivity": h1(outcomes, tie_stop_as_failure=True),
        "tie_stop_successes": {cid: outcomes.tie_stop_successes(cid) for cid in reports},
        "h2": h2(
            o1,
            offline_usable=table is not None,
            any_driven_success=any_driven,
            reason=str(check["not_used_because"]),
        ),
        "h3": h3(outcomes),
        "o1": o1,
        "o2": o2_result,
        "o3": o3_result,
        "per_policy_outcomes": {cid: outcomes.summary(cid) for cid in reports},
        "descriptive": descriptive(outcomes),
        "failed_attempts_with_job_starts": provenance.failed_attempts.get("main", {}),
        "cross_check": cross_check(outcomes, entries, spark),
        "carried": carried,
        "aborted_runs": aborted,
        "section_9_row": section_9_main(h1_result, carried, outcomes),
    }


def stage_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirmation_root is None:
        raise refuse("the confirmation stage needs --confirmation-root.")
    roots = [args.offline_root, args.main_root, args.confirmation_root, *args.aborted_root]
    checked = guard(args, roots)
    repo, freeze = checked["repo"], checked["freeze"]
    provenance = Provenance()
    results = training_results(repo / args.source_artifact_root)
    policies = load_offline(repo / args.offline_root, provenance, results)
    all_ids = [record.checkpoint_id for record in ALLOWED_CHECKPOINTS]
    reports, entries = load_rollouts(repo / args.main_root, "main", all_ids, provenance)
    main_payload = _main_payload(
        args, checked, provenance, results, policies, reports, entries, repo, freeze
    )
    carried = main_payload["carried"]
    output = repo / args.output_dir
    carried_sha = write_output(output / "carried_list.json", carried)
    scope = [record.checkpoint_id for record in confirmation_scope_from_carried_list(carried)]
    confirmation_reports, confirmation_entries = load_rollouts(
        repo / args.confirmation_root, "confirmation", scope, provenance
    )
    for checkpoint_id in scope:
        manifest = _load_json(repo / args.confirmation_root / checkpoint_id / "manifest.json")
        if (manifest.get("carried_list") or {}).get("sha256") != carried_sha:
            raise refuse(f"{checkpoint_id} ran from another carried list than the analysis's.")
    single = provenance.check_single()
    if single["gpu_name"] != main_payload["inputs"]["gpu_name"]:
        raise refuse("the confirmation ran on another GPU name than the main run.")
    for commit in sorted(provenance.commits.get("confirmation", set())):
        check_commit_after_freeze(
            repo, freeze, commit, args.preregistration, smoke_result=args.smoke_result
        )
    check_code_records(repo, provenance.code_records)
    outcomes = Outcomes(confirmation_reports, confirmation_entries)
    result = confirmation_test(carried, outcomes)
    strict = confirmation_test(carried, outcomes, tie_stop_as_failure=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "confirmation",
        "inputs": {
            "freeze_commit": freeze,
            "carried_list_sha256": carried_sha,
            "confirmation_root": args.confirmation_root,
            "gpu_name": single["gpu_name"],
            "gpu_driver": single["gpu_driver"],
            "image_sha256": single["image_sha256"],
        },
        "failed_attempts_with_job_starts": provenance.failed_attempts.get("confirmation", {}),
        "tie_stop_sensitivity": tie_stop_survival(
            main_payload["h1_tie_stop_sensitivity"], result, strict
        ),
        "tie_stop_successes": {
            cid: outcomes.tie_stop_successes(cid) for cid in confirmation_reports
        },
        **result,
    }


def tie_stop_survival(
    main_strict: Mapping[str, Any], result: Mapping[str, Any], strict: Mapping[str, Any]
) -> dict[str, Any]:
    """Section 7.3: whether each confirmed candidate also clears the main-suite screen and the
    confirmation test when its exact-tie STOP successes count as failures. A row-1 or row-2 claim
    says whether it survives."""
    candidates = {}
    for checkpoint_id, row in result["candidates"].items():
        clears = checkpoint_id in main_strict["reached_bar"]
        confirmed = bool(strict["candidates"][checkpoint_id]["confirmed"])
        candidates[checkpoint_id] = {
            "confirmed": bool(row["confirmed"]),
            "clears_main_screen_strict": clears,
            "confirmed_strict": confirmed,
            "survives": bool(row["confirmed"]) and clears and confirmed,
        }
    return {
        "candidates": candidates,
        "confirmed_and_survives": sorted(c for c, row in candidates.items() if row["survives"]),
        "confirmed_and_does_not_survive": sorted(
            c for c, row in candidates.items() if row["confirmed"] and not row["survives"]
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        if args.stage == "main":
            payload = stage_main(args)
            output = args.repo.resolve() / args.output_dir
            carried_sha = write_output(output / "carried_list.json", payload["carried"])
            sha = write_output(output / "analysis_main.json", payload)
            summary = {
                "analysis_main_sha256": sha,
                "carried_list_sha256": carried_sha,
                "head_commit": git(args.repo.resolve(), "rev-parse", "HEAD").strip(),
                "section_9_row": payload["section_9_row"],
            }
        else:
            payload = stage_confirmation(args)
            output = args.repo.resolve() / args.output_dir
            sha = write_output(output / "analysis_confirmation.json", payload)
            summary = {
                "analysis_confirmation_sha256": sha,
                "section_9_row": payload["section_9_row"],
            }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except (SweepRefusalError, FreezeGuardError) as refusal:
        print(json.dumps({"status": "refused", "stage": args.stage, "reason": str(refusal)}))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
