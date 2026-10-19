"""Frozen native-C0 closed-loop latency and rollout contract.

The first real learned-policy use of the R2R-CE evaluator is deliberately split in two:

1. a 32-decision integration/latency gate that cannot emit scientific navigation results; and
2. an official 500-step evaluation whose checkpoint/episode scope is selected only from the
   preregistered latency statistic.

This module contains the pure contract and instrumented episode loop.  Spark launchers own model,
dataset, checkpoint, and artifact provenance; this code owns timing boundaries and metric parity.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from spatialvlm.data.datasets import R2RCEAction, R2RCEEpisode, R2RCEGroundTruth
from spatialvlm.eval.metrics import R2RCEEpisodeTrace, evaluate_r2r_ce_trace
from spatialvlm.eval.r2r_ce import (
    R2R_CE_MAX_STEPS,
    R2R_CE_SUCCESS_DISTANCE_M,
    R2RCEEpisodeEnvironment,
    R2RCEEvaluationResult,
    R2RCEPolicy,
    R2RCEStepRecord,
)

CLOSED_LOOP_SCHEMA_VERSION = 1
CLOSED_LOOP_STRESS_DECISIONS = 32
CLOSED_LOOP_STRESS_MAX_SECONDS = 30 * 60


@dataclass(frozen=True)
class ClosedLoopCheckpoint:
    """One immutable selected checkpoint and its producing evidence directory."""

    checkpoint_id: str
    treatment: str
    replication_seed: int
    source_run_id: str
    checkpoint_gate: str
    selected_update: int
    checkpoint_relative_path: str
    checkpoint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FROZEN_CHECKPOINTS: tuple[ClosedLoopCheckpoint, ...] = (
    ClosedLoopCheckpoint(
        checkpoint_id="unweighted_s20260825",
        treatment="unweighted",
        replication_seed=20260825,
        source_run_id="native_c0_development_v1",
        checkpoint_gate="native_qwen_c0_development_v1",
        selected_update=160,
        checkpoint_relative_path="step_0160.pt",
        checkpoint_sha256="a4a58cd34409e24c805c847249cd921da17d11c2bef96e37a20bf7c50a86987d",
    ),
    ClosedLoopCheckpoint(
        checkpoint_id="inverse_frequency_s20260825",
        treatment="inverse_frequency",
        replication_seed=20260825,
        source_run_id="native_c0_action_balance_v1",
        checkpoint_gate="native_qwen_c0_action_balance_v1",
        selected_update=192,
        checkpoint_relative_path="step_0192.pt",
        checkpoint_sha256="f21496af3bc35bcc9f680b142bb19b4eff7f92fb5e8b405837e50ff7181872dc",
    ),
    ClosedLoopCheckpoint(
        checkpoint_id="unweighted_s20260826",
        treatment="unweighted",
        replication_seed=20260826,
        source_run_id="native_c0_seed_replication_unweighted_s20260826_v1",
        checkpoint_gate="native_qwen_c0_development_v1",
        selected_update=160,
        checkpoint_relative_path="step_0160.pt",
        checkpoint_sha256="ac5b106093c3b7e0a30851f907bba094b0db35f382940394bbc6d95fd4b89413",
    ),
    ClosedLoopCheckpoint(
        checkpoint_id="inverse_frequency_s20260826",
        treatment="inverse_frequency",
        replication_seed=20260826,
        source_run_id="native_c0_seed_replication_inverse_frequency_s20260826_v1",
        checkpoint_gate="native_qwen_c0_action_balance_v1",
        selected_update=128,
        checkpoint_relative_path="step_0128.pt",
        checkpoint_sha256="55380852648df9581c06b887f5399240f981245d4da7b868019fb6ea8a544260",
    ),
    ClosedLoopCheckpoint(
        checkpoint_id="unweighted_s20260827",
        treatment="unweighted",
        replication_seed=20260827,
        source_run_id="native_c0_seed_replication_unweighted_s20260827_v1",
        checkpoint_gate="native_qwen_c0_development_v1",
        selected_update=160,
        checkpoint_relative_path="step_0160.pt",
        checkpoint_sha256="7264fdd47d9554ce9d21da8f29d7ccb73dbe3903560b587c26106297cf86da21",
    ),
    ClosedLoopCheckpoint(
        checkpoint_id="inverse_frequency_s20260827",
        treatment="inverse_frequency",
        replication_seed=20260827,
        source_run_id="native_c0_seed_replication_inverse_frequency_s20260827_v1",
        checkpoint_gate="native_qwen_c0_action_balance_v1",
        selected_update=192,
        checkpoint_relative_path="step_0192.pt",
        checkpoint_sha256="77f6eb8695323db53cf7394125f33a4a78be0b9608c500dfe575b4078eec81fc",
    ),
)


@dataclass(frozen=True)
class ClosedLoopScope:
    """Latency-selected evaluation scope, frozen before any learned rollout outcome."""

    branch_id: str
    measured_seconds_per_step: float
    checkpoint_ids: tuple[str, ...]
    episode_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "measured_seconds_per_step": self.measured_seconds_per_step,
            "checkpoint_ids": list(self.checkpoint_ids),
            "episode_count": self.episode_count,
        }


def select_closed_loop_scope(measured_seconds_per_step: float) -> ClosedLoopScope:
    """Apply the preregistered latency-only branch rule.

    The medium branch is the original paired treatment comparison plus one unweighted replicate.
    That exact identity is frozen here rather than chosen after seeing navigation outcomes.
    """
    if not math.isfinite(measured_seconds_per_step) or measured_seconds_per_step <= 0.0:
        raise ValueError("Measured closed-loop seconds per step must be finite and positive.")
    if measured_seconds_per_step <= 0.5:
        return ClosedLoopScope(
            branch_id="all_six_61_episodes",
            measured_seconds_per_step=measured_seconds_per_step,
            checkpoint_ids=tuple(item.checkpoint_id for item in FROZEN_CHECKPOINTS),
            episode_count=61,
        )
    if measured_seconds_per_step <= 1.5:
        return ClosedLoopScope(
            branch_id="paired_original_plus_unweighted_replication_61_episodes",
            measured_seconds_per_step=measured_seconds_per_step,
            checkpoint_ids=(
                "unweighted_s20260825",
                "inverse_frequency_s20260825",
                "unweighted_s20260826",
            ),
            episode_count=61,
        )
    return ClosedLoopScope(
        branch_id="single_unweighted_50_episodes",
        measured_seconds_per_step=measured_seconds_per_step,
        checkpoint_ids=("unweighted_s20260825",),
        episode_count=50,
    )


@dataclass(frozen=True)
class ClosedLoopDecisionTiming:
    """One sequential policy-forward plus simulator-step timing record."""

    episode_id: str
    episode_index: int
    step_index: int
    action_id: int
    action_name: str
    policy_seconds: float
    environment_seconds: float
    closed_loop_seconds: float

    def __post_init__(self) -> None:
        values = (self.policy_seconds, self.environment_seconds, self.closed_loop_seconds)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Closed-loop timing values must be finite and non-negative.")
        if not math.isclose(
            self.closed_loop_seconds,
            self.policy_seconds + self.environment_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("Closed-loop timing must equal policy plus environment time.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_closed_loop_timings(
    records: Sequence[ClosedLoopDecisionTiming],
) -> dict[str, Any]:
    """Summarize the exact 32 decision-wall measurements used for scope selection."""
    if len(records) != CLOSED_LOOP_STRESS_DECISIONS:
        raise ValueError(
            f"Latency admission requires exactly {CLOSED_LOOP_STRESS_DECISIONS} decisions."
        )
    closed = [record.closed_loop_seconds for record in records]
    ordered = sorted(closed)
    # Nearest-rank p95, recorded alongside the arithmetic mean that drives the branch.
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    mean_seconds = statistics.fmean(closed)
    return {
        "statistic_id": "arithmetic_mean_32_sequential_policy_plus_environment_steps_v1",
        "decision_count": len(records),
        "mean_seconds_per_step": mean_seconds,
        "median_seconds_per_step": statistics.median(closed),
        "p95_seconds_per_step": ordered[p95_index],
        "mean_policy_seconds": statistics.fmean(record.policy_seconds for record in records),
        "mean_environment_seconds": statistics.fmean(
            record.environment_seconds for record in records
        ),
        "includes": [
            "native H4 processor construction",
            "batch-1 Qwen forward",
            "constrained one-action scoring",
            "Habitat action or task-level STOP",
            "next RGB-D render",
            "runtime geodesic computation",
        ],
        "excludes": [
            "one-time model and checkpoint load",
            "episode reset and scene initialization",
            "artifact hashing and serialization",
        ],
        "scope": select_closed_loop_scope(mean_seconds).to_dict(),
    }


@dataclass(frozen=True)
class InstrumentedEpisode:
    """Canonical evaluation result plus non-overlapping runtime measurements."""

    evaluation: R2RCEEvaluationResult
    decision_timings: tuple[ClosedLoopDecisionTiming, ...]
    environment_reset_seconds: float
    policy_reset_seconds: float


@dataclass(frozen=True)
class ClosedLoopDiagnosticRun:
    """A fixed decision budget with no scientific episode metrics."""

    decision_timings: tuple[ClosedLoopDecisionTiming, ...]
    episode_resets: tuple[dict[str, Any], ...]


def _normalize_action(raw_action: Any) -> tuple[R2RCEAction | None, str | None]:
    if isinstance(raw_action, bool):
        return None, repr(raw_action)
    try:
        return R2RCEAction(int(raw_action)), None
    except (TypeError, ValueError, OverflowError):
        return None, repr(raw_action)


def run_instrumented_r2r_ce_episode(
    episode: R2RCEEpisode,
    ground_truth: R2RCEGroundTruth,
    environment: R2RCEEpisodeEnvironment,
    policy: R2RCEPolicy,
    *,
    episode_index: int,
    max_steps: int = R2R_CE_MAX_STEPS,
    success_distance: float = R2R_CE_SUCCESS_DISTANCE_M,
) -> InstrumentedEpisode:
    """Run the official episode semantics while timing each sequential closed-loop decision."""
    if episode.episode_id != ground_truth.episode_id:
        raise ValueError("Episode and ground-truth IDs must match.")
    if max_steps <= 0:
        raise ValueError("R2R-CE max_steps must be positive.")

    started = time.perf_counter()
    state = environment.reset(episode)
    environment_reset_seconds = time.perf_counter() - started
    started = time.perf_counter()
    policy.reset(episode, ground_truth)
    policy_reset_seconds = time.perf_counter() - started

    positions = [state.position]
    distances = [state.distance_to_goal]
    actions: list[R2RCEAction] = []
    collisions: list[bool] = []
    timings: list[ClosedLoopDecisionTiming] = []
    invalid_action_count = 0
    steps = [
        R2RCEStepRecord(
            episode_id=episode.episode_id,
            step_index=0,
            event="reset",
            action_id=None,
            action_name=None,
            executed=False,
            position=state.position,
            distance_to_goal=state.distance_to_goal,
            collision=False,
            done=state.done,
        )
    ]

    for step_index in range(1, max_steps + 1):
        if state.done:
            break
        decision_started = time.perf_counter()
        raw_action = policy.act(state.observation)
        action_finished = time.perf_counter()
        action, invalid_output = _normalize_action(raw_action)
        if action is None:
            invalid_action_count += 1
            steps.append(
                R2RCEStepRecord(
                    episode_id=episode.episode_id,
                    step_index=step_index,
                    event="invalid_action",
                    action_id=None,
                    action_name=None,
                    executed=False,
                    position=state.position,
                    distance_to_goal=state.distance_to_goal,
                    collision=False,
                    done=True,
                    invalid_output=invalid_output,
                )
            )
            break

        state = environment.step(action)
        step_finished = time.perf_counter()
        policy_seconds = action_finished - decision_started
        environment_seconds = step_finished - action_finished
        timings.append(
            ClosedLoopDecisionTiming(
                episode_id=episode.episode_id,
                episode_index=episode_index,
                step_index=step_index,
                action_id=int(action),
                action_name=action.name,
                policy_seconds=policy_seconds,
                environment_seconds=environment_seconds,
                closed_loop_seconds=step_finished - decision_started,
            )
        )
        actions.append(action)
        positions.append(state.position)
        distances.append(state.distance_to_goal)
        collisions.append(state.collision)
        steps.append(
            R2RCEStepRecord(
                episode_id=episode.episode_id,
                step_index=step_index,
                event="action",
                action_id=int(action),
                action_name=action.name,
                executed=True,
                position=state.position,
                distance_to_goal=state.distance_to_goal,
                collision=state.collision,
                done=state.done,
            )
        )
        if action == R2RCEAction.STOP or state.done:
            break

    trace = R2RCEEpisodeTrace(
        episode_id=episode.episode_id,
        positions=tuple(positions),
        distances_to_goal=tuple(distances),
        actions=tuple(actions),
        shortest_path_length=distances[0],
        reference_path=ground_truth.locations,
        collisions=tuple(collisions),
        invalid_action_count=invalid_action_count,
    )
    evaluation = R2RCEEvaluationResult(
        trace=trace,
        metrics=evaluate_r2r_ce_trace(trace, success_distance=success_distance),
        steps=tuple(steps),
    )
    return InstrumentedEpisode(
        evaluation=evaluation,
        decision_timings=tuple(timings),
        environment_reset_seconds=environment_reset_seconds,
        policy_reset_seconds=policy_reset_seconds,
    )


def run_closed_loop_decision_budget(
    episodes: Sequence[R2RCEEpisode],
    ground_truth: Mapping[str, R2RCEGroundTruth],
    environment: R2RCEEpisodeEnvironment,
    policy: R2RCEPolicy,
    *,
    decision_budget: int = CLOSED_LOOP_STRESS_DECISIONS,
    hard_deadline_monotonic: float | None = None,
    per_episode_step_cap: int = R2R_CE_MAX_STEPS,
) -> ClosedLoopDiagnosticRun:
    """Execute exactly ``decision_budget`` sequential decisions across the frozen episode order.

    The final episode may be intentionally truncated after the last admitted decision.  Therefore
    this function emits timing/integration evidence only and never computes episode metrics.
    """
    if decision_budget <= 0:
        raise ValueError("Diagnostic decision budget must be positive.")
    if per_episode_step_cap <= 0:
        raise ValueError("Per-episode step cap must be positive.")

    timings: list[ClosedLoopDecisionTiming] = []
    reset_records: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes):
        if len(timings) >= decision_budget:
            break
        if hard_deadline_monotonic is not None and time.monotonic() >= hard_deadline_monotonic:
            raise TimeoutError("Closed-loop diagnostic exceeded its frozen wall-clock ceiling.")
        if episode.episode_id not in ground_truth:
            raise KeyError(f"Missing ground truth for episode {episode.episode_id}.")

        started = time.perf_counter()
        state = environment.reset(episode)
        environment_reset_seconds = time.perf_counter() - started
        started = time.perf_counter()
        policy.reset(episode, ground_truth[episode.episode_id])
        policy_reset_seconds = time.perf_counter() - started
        reset_records.append(
            {
                "episode_id": episode.episode_id,
                "episode_index": episode_index,
                "environment_reset_seconds": environment_reset_seconds,
                "policy_reset_seconds": policy_reset_seconds,
                "reset_distance_to_goal": state.distance_to_goal,
            }
        )

        for step_index in range(1, per_episode_step_cap + 1):
            if len(timings) >= decision_budget or state.done:
                break
            if hard_deadline_monotonic is not None and time.monotonic() >= hard_deadline_monotonic:
                raise TimeoutError("Closed-loop diagnostic exceeded its frozen wall-clock ceiling.")
            decision_started = time.perf_counter()
            raw_action = policy.act(state.observation)
            action_finished = time.perf_counter()
            action, invalid_output = _normalize_action(raw_action)
            if action is None:
                raise RuntimeError(
                    "Closed-loop diagnostic policy returned an invalid action: "
                    f"{invalid_output}."
                )
            state = environment.step(action)
            step_finished = time.perf_counter()
            policy_seconds = action_finished - decision_started
            environment_seconds = step_finished - action_finished
            timings.append(
                ClosedLoopDecisionTiming(
                    episode_id=episode.episode_id,
                    episode_index=episode_index,
                    step_index=step_index,
                    action_id=int(action),
                    action_name=action.name,
                    policy_seconds=policy_seconds,
                    environment_seconds=environment_seconds,
                    closed_loop_seconds=step_finished - decision_started,
                )
            )
            if action is R2RCEAction.STOP or state.done:
                break

    if len(timings) != decision_budget:
        raise RuntimeError(
            f"Frozen manifest exhausted after {len(timings)} of {decision_budget} decisions."
        )
    return ClosedLoopDiagnosticRun(
        decision_timings=tuple(timings), episode_resets=tuple(reset_records)
    )


def closed_loop_protocol_record() -> dict[str, Any]:
    """Return the frozen values that launchers and independent verifiers must reproduce."""
    return {
        "schema_version": CLOSED_LOOP_SCHEMA_VERSION,
        "history_mode": "H4_CAUSAL",
        "chunk_length": 1,
        "official_max_steps": R2R_CE_MAX_STEPS,
        "success_distance_m": R2R_CE_SUCCESS_DISTANCE_M,
        "split": "val_seen",
        "sealed_split": "val_unseen",
        "stress": {
            "decision_count": CLOSED_LOOP_STRESS_DECISIONS,
            "hard_max_seconds": CLOSED_LOOP_STRESS_MAX_SECONDS,
            "checkpoint_id": "unweighted_s20260825",
            "scientific_metrics_emitted": False,
        },
        "latency_scope_rule": [
            {"upper_inclusive_seconds": 0.5, "branch": "all_six_61_episodes"},
            {
                "lower_exclusive_seconds": 0.5,
                "upper_inclusive_seconds": 1.5,
                "branch": "paired_original_plus_unweighted_replication_61_episodes",
            },
            {"lower_exclusive_seconds": 1.5, "branch": "single_unweighted_50_episodes"},
        ],
        "checkpoints": [checkpoint.to_dict() for checkpoint in FROZEN_CHECKPOINTS],
        "headline_metrics": ["oracle_success_rate", "success_rate"],
        "secondary_metrics": [
            "spl",
            "navigation_error",
            "ndtw",
            "sdtw",
            "stop_rate",
            "step_cap_rate",
            "collision_count",
        ],
        "evidence_boundary": {
            "stress_pass_establishes": (
                "One selected native-C0 checkpoint completed 32 sequential H4 batch-1 policy "
                "decisions and simulator steps with finite telemetry inside the infrastructure cap."
            ),
            "stress_pass_does_not_establish": (
                "Navigation competence, success, oracle success, treatment ordering, "
                "generalization, or any C1-C4 claim."
            ),
        },
    }
