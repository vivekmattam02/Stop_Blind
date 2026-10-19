"""Deterministic R2R-CE rollout contract and machine-readable artifacts.

The concrete Habitat adapter will implement :class:`R2RCEEpisodeEnvironment`. Keeping the rollout
logic behind this small protocol lets us test action, termination, trajectory, and metric behavior
without pretending that a synthetic environment validates Habitat-Sim itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from spatialvlm.data.datasets import (
    Point3D,
    R2RCEAction,
    R2RCEEpisode,
    R2RCEGroundTruth,
    R2RCEReleaseManifest,
)
from spatialvlm.eval.metrics import (
    R2RCEAggregateMetrics,
    R2RCEEpisodeMetrics,
    R2RCEEpisodeTrace,
    aggregate_r2r_ce_metrics,
    evaluate_r2r_ce_trace,
)

R2R_CE_MAX_STEPS = 500
R2R_CE_SUCCESS_DISTANCE_M = 3.0
R2R_CE_FORWARD_STEP_M = 0.25
R2R_CE_TURN_ANGLE_DEG = 15
R2R_CE_RGB_SIZE = (224, 224)
R2R_CE_DEPTH_SIZE = (256, 256)


@dataclass(frozen=True)
class R2RCERolloutState:
    """One simulator state exposed to the canonical evaluator."""

    observation: Mapping[str, Any]
    position: Point3D
    distance_to_goal: float
    collision: bool = False
    done: bool = False

    def __post_init__(self) -> None:
        if len(self.position) != 3 or not all(math.isfinite(value) for value in self.position):
            raise ValueError("R2R-CE rollout position must contain three finite coordinates.")
        if self.distance_to_goal < 0.0 or not math.isfinite(self.distance_to_goal):
            raise ValueError("R2R-CE distance_to_goal must be finite and non-negative.")


class R2RCEEpisodeEnvironment(Protocol):
    """Minimum simulator surface consumed by the evaluator."""

    def reset(self, episode: R2RCEEpisode) -> R2RCERolloutState: ...

    def step(self, action: R2RCEAction) -> R2RCERolloutState: ...

    def close(self) -> None: ...


class R2RCEPolicy(Protocol):
    """Policy surface shared by learned and conformance agents."""

    def reset(self, episode: R2RCEEpisode, ground_truth: R2RCEGroundTruth) -> None: ...

    def act(self, observation: Mapping[str, Any]) -> int | R2RCEAction: ...


@dataclass(frozen=True)
class R2RCEStepRecord:
    """One reset or policy-decision event for episode-level JSONL."""

    episode_id: str
    step_index: int
    event: str
    action_id: int | None
    action_name: str | None
    executed: bool
    position: Point3D
    distance_to_goal: float
    collision: bool
    done: bool
    invalid_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class R2RCEEvaluationResult:
    """Trace, metrics, and step events for one episode."""

    trace: R2RCEEpisodeTrace
    metrics: R2RCEEpisodeMetrics
    steps: tuple[R2RCEStepRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        trace = {
            "episode_id": self.trace.episode_id,
            "positions": [list(point) for point in self.trace.positions],
            "distances_to_goal": list(self.trace.distances_to_goal),
            "actions": [int(action) for action in self.trace.actions],
            "shortest_path_length": self.trace.shortest_path_length,
            "reference_path": [list(point) for point in self.trace.reference_path],
            "collisions": list(self.trace.collisions),
            "invalid_action_count": self.trace.invalid_action_count,
        }
        return {
            "episode_id": self.trace.episode_id,
            "trace": trace,
            "metrics": self.metrics.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class R2RCEEvaluationRun:
    """Results and aggregate for one fixed episode manifest."""

    results: tuple[R2RCEEvaluationResult, ...]
    aggregate: R2RCEAggregateMetrics


class GroundTruthActionPolicy:
    """Replay official ground-truth actions as an evaluator conformance policy."""

    def reset(self, episode: R2RCEEpisode, ground_truth: R2RCEGroundTruth) -> None:
        if episode.episode_id != ground_truth.episode_id:
            raise ValueError("Episode and ground-truth IDs must match.")
        self._actions = ground_truth.actions
        self._index = 0

    def act(self, observation: Mapping[str, Any]) -> R2RCEAction:
        del observation
        if self._index >= len(self._actions):
            return R2RCEAction.STOP
        action = self._actions[self._index]
        self._index += 1
        return action


class AlwaysForwardPolicy:
    """Deliberately weak policy that never calls STOP."""

    def reset(self, episode: R2RCEEpisode, ground_truth: R2RCEGroundTruth) -> None:
        del episode, ground_truth

    def act(self, observation: Mapping[str, Any]) -> R2RCEAction:
        del observation
        return R2RCEAction.MOVE_FORWARD


class ImmediateStopPolicy:
    """Termination-only control that stops at the reset pose."""

    def reset(self, episode: R2RCEEpisode, ground_truth: R2RCEGroundTruth) -> None:
        del episode, ground_truth

    def act(self, observation: Mapping[str, Any]) -> R2RCEAction:
        del observation
        return R2RCEAction.STOP


class InvalidActionPolicy:
    """Deliberately broken policy used to test fail-closed evaluation."""

    def reset(self, episode: R2RCEEpisode, ground_truth: R2RCEGroundTruth) -> None:
        del episode, ground_truth

    def act(self, observation: Mapping[str, Any]) -> int:
        del observation
        return 999


class SeededHandcraftedPolicy:
    """Episode-deterministic port of the official VLN-CE handcrafted agent.

    The reference agent samples a right-turn heading, advances 37 times (the released training
    split's mean 9.27 m path length divided by the 0.25 m forward step), and then stops.  Stable
    per-episode seeding makes repeated Spark runs auditable without changing that action prior.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        turn_angle_degrees: int = R2R_CE_TURN_ANGLE_DEG,
        forward_steps: int = 37,
    ) -> None:
        if turn_angle_degrees <= 0 or 360 % turn_angle_degrees != 0:
            raise ValueError("Handcrafted policy turn angle must evenly divide 360 degrees.")
        if forward_steps < 0:
            raise ValueError("Handcrafted policy forward_steps must be non-negative.")
        self.seed = int(seed)
        self.turn_angle_degrees = int(turn_angle_degrees)
        self.forward_steps = int(forward_steps)
        self._actions: tuple[R2RCEAction, ...] = ()
        self._index = 0

    def reset(self, episode: R2RCEEpisode, ground_truth: R2RCEGroundTruth) -> None:
        del ground_truth
        digest = hashlib.sha256(f"{self.seed}:{episode.episode_id}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        # The official NumPy call uses randint(0, 360 / turn_angle + 1), so both 0 and one
        # complete 360-degree turn are possible.
        right_turns = rng.randrange(360 // self.turn_angle_degrees + 1)
        self._actions = (
            (R2RCEAction.TURN_RIGHT,) * right_turns
            + (R2RCEAction.MOVE_FORWARD,) * self.forward_steps
            + (R2RCEAction.STOP,)
        )
        self._index = 0

    def act(self, observation: Mapping[str, Any]) -> R2RCEAction:
        del observation
        if self._index >= len(self._actions):
            return R2RCEAction.STOP
        action = self._actions[self._index]
        self._index += 1
        return action


class SeededRandomPolicy:
    """Episode-deterministic random policy with the official baseline's action prior."""

    def __init__(
        self,
        seed: int = 0,
        probabilities: Sequence[float] = (0.02, 0.68, 0.15, 0.15),
    ) -> None:
        if len(probabilities) != len(R2RCEAction):
            raise ValueError("Random policy needs one probability per R2R-CE action.")
        if any(probability < 0.0 for probability in probabilities):
            raise ValueError("Random policy probabilities must be non-negative.")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Random policy probabilities must sum to one.")
        self.seed = int(seed)
        self.probabilities = tuple(float(value) for value in probabilities)
        self._rng = random.Random(self.seed)

    def reset(self, episode: R2RCEEpisode, ground_truth: R2RCEGroundTruth) -> None:
        del ground_truth
        digest = hashlib.sha256(f"{self.seed}:{episode.episode_id}".encode()).digest()
        self._rng = random.Random(int.from_bytes(digest[:8], "big"))

    def act(self, observation: Mapping[str, Any]) -> R2RCEAction:
        del observation
        return self._rng.choices(list(R2RCEAction), weights=self.probabilities, k=1)[0]


def _normalize_action(raw_action: Any) -> tuple[R2RCEAction | None, str | None]:
    if isinstance(raw_action, bool):
        return None, repr(raw_action)
    try:
        return R2RCEAction(int(raw_action)), None
    except (TypeError, ValueError, OverflowError):
        return None, repr(raw_action)


def run_r2r_ce_episode(
    episode: R2RCEEpisode,
    ground_truth: R2RCEGroundTruth,
    environment: R2RCEEpisodeEnvironment,
    policy: R2RCEPolicy,
    *,
    max_steps: int = R2R_CE_MAX_STEPS,
    success_distance: float = R2R_CE_SUCCESS_DISTANCE_M,
) -> R2RCEEvaluationResult:
    """Run one episode and terminate invalid outputs without executing them."""
    if episode.episode_id != ground_truth.episode_id:
        raise ValueError("Episode and ground-truth IDs must match.")
    if max_steps <= 0:
        raise ValueError("R2R-CE max_steps must be positive.")

    state = environment.reset(episode)
    policy.reset(episode, ground_truth)
    positions = [state.position]
    distances = [state.distance_to_goal]
    actions: list[R2RCEAction] = []
    collisions: list[bool] = []
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
        raw_action = policy.act(state.observation)
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
        # Habitat-Lab v0.1.7 SPL captures DistanceToGoal at reset.  The released
        # episode.info geodesic is provenance/diagnostic metadata, not the SPL
        # denominator for a newly instantiated simulator runtime.
        shortest_path_length=distances[0],
        reference_path=ground_truth.locations,
        collisions=tuple(collisions),
        invalid_action_count=invalid_action_count,
    )
    metrics = evaluate_r2r_ce_trace(trace, success_distance=success_distance)
    return R2RCEEvaluationResult(trace=trace, metrics=metrics, steps=tuple(steps))


def evaluate_r2r_ce_policy(
    episodes: Sequence[R2RCEEpisode],
    ground_truth: Mapping[str, R2RCEGroundTruth],
    environment_factory: Callable[[R2RCEEpisode], R2RCEEpisodeEnvironment],
    policy: R2RCEPolicy,
    *,
    max_steps: int = R2R_CE_MAX_STEPS,
    success_distance: float = R2R_CE_SUCCESS_DISTANCE_M,
) -> R2RCEEvaluationRun:
    """Evaluate a policy on an explicit, ordered episode manifest."""
    if not episodes:
        raise ValueError("R2R-CE evaluation manifest must not be empty.")
    results: list[R2RCEEvaluationResult] = []
    for episode in episodes:
        if episode.episode_id not in ground_truth:
            raise KeyError(f"Missing ground truth for episode {episode.episode_id}.")
        environment = environment_factory(episode)
        try:
            result = run_r2r_ce_episode(
                episode,
                ground_truth[episode.episode_id],
                environment,
                policy,
                max_steps=max_steps,
                success_distance=success_distance,
            )
        finally:
            environment.close()
        results.append(result)

    aggregate = aggregate_r2r_ce_metrics([result.metrics for result in results])
    return R2RCEEvaluationRun(results=tuple(results), aggregate=aggregate)


def build_r2r_ce_artifact(
    run: R2RCEEvaluationRun,
    release_manifest: R2RCEReleaseManifest,
    *,
    policy_id: str,
    max_steps: int = R2R_CE_MAX_STEPS,
    success_distance: float = R2R_CE_SUCCESS_DISTANCE_M,
) -> dict[str, Any]:
    """Build a stable evaluation artifact; callers add Git/container identity at the run layer."""
    if not policy_id.strip():
        raise ValueError("R2R-CE policy_id must not be empty.")
    return {
        "schema_version": 1,
        "benchmark": "R2R-CE",
        "release": release_manifest.to_dict(),
        "protocol": {
            "policy_id": policy_id,
            "max_steps": max_steps,
            "success_distance_m": success_distance,
            "success_requires_stop": True,
            "forward_step_m": R2R_CE_FORWARD_STEP_M,
            "turn_angle_degrees": R2R_CE_TURN_ANGLE_DEG,
            "rgb_height_width": list(R2R_CE_RGB_SIZE),
            "depth_height_width": list(R2R_CE_DEPTH_SIZE),
            "action_mapping": {action.name: int(action) for action in R2RCEAction},
            "invalid_action_behavior": "terminate_failure_without_execution",
        },
        "episode_order": [result.trace.episode_id for result in run.results],
        "aggregate": run.aggregate.to_dict(),
        "episodes": [result.to_dict() for result in run.results],
    }


def write_r2r_ce_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    """Write an evaluation artifact as canonical, human-readable JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_r2r_ce_step_jsonl(
    path: str | Path,
    results: Sequence[R2RCEEvaluationResult],
) -> None:
    """Write ordered per-step records for trajectory inspection and metric reconstruction."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(step.to_dict(), sort_keys=True, allow_nan=False)
        for result in results
        for step in result.steps
    ]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
