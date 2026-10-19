"""Core evaluation metrics for navigation and spatial reasoning."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from spatialvlm.data.datasets import Point3D, R2RCEAction


@dataclass(frozen=True)
class NavigationEpisodeResult:
    """Single navigation episode summary."""

    success: bool
    path_length: float
    shortest_path_length: float


@dataclass(frozen=True)
class MetricBundle:
    """Standardized metric tuple for experiment reporting."""

    success_rate: float
    spl: float
    permutation_sensitivity: float
    composite: float


@dataclass(frozen=True)
class R2RCEEpisodeTrace:
    """Policy trace with evaluator-only state needed for official R2R-CE metrics.

    Positions and geodesic distances include the reset state followed by one state per executed
    action. Invalid policy outputs are not executed and are counted separately.
    """

    episode_id: str
    positions: tuple[Point3D, ...]
    distances_to_goal: tuple[float, ...]
    actions: tuple[R2RCEAction, ...]
    shortest_path_length: float
    reference_path: tuple[Point3D, ...]
    collisions: tuple[bool, ...] = ()
    invalid_action_count: int = 0

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("R2R-CE trace episode_id must not be empty.")
        if len(self.positions) != len(self.actions) + 1:
            raise ValueError("R2R-CE trace must contain one more position than executed actions.")
        if len(self.distances_to_goal) != len(self.positions):
            raise ValueError("R2R-CE trace needs one distance-to-goal value per position.")
        if self.collisions and len(self.collisions) != len(self.actions):
            raise ValueError("R2R-CE collision flags must align with executed actions.")
        if not self.reference_path:
            raise ValueError("R2R-CE reference path must not be empty.")
        if self.shortest_path_length < 0.0 or not math.isfinite(self.shortest_path_length):
            raise ValueError("R2R-CE shortest_path_length must be finite and non-negative.")
        if self.invalid_action_count < 0:
            raise ValueError("R2R-CE invalid_action_count must be non-negative.")
        for distance in self.distances_to_goal:
            if distance < 0.0 or not math.isfinite(distance):
                raise ValueError("R2R-CE distances_to_goal must be finite and non-negative.")


@dataclass(frozen=True)
class R2RCEEpisodeMetrics:
    """Official-style metrics and diagnostics for one R2R-CE episode."""

    episode_id: str
    navigation_error: float
    oracle_success: float
    success: float
    path_length: float
    shortest_path_length: float
    spl: float
    ndtw: float
    sdtw: float
    step_count: int
    collision_count: int
    invalid_action_count: int
    stopped: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class R2RCEAggregateMetrics:
    """Unweighted mean over a fixed episode manifest."""

    episode_count: int
    navigation_error: float
    oracle_success_rate: float
    success_rate: float
    spl: float
    ndtw: float
    sdtw: float
    mean_path_length: float
    mean_steps: float
    collision_count: int
    invalid_action_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def success_rate(successes: Sequence[bool]) -> float:
    if len(successes) == 0:
        return 0.0
    return sum(1.0 for s in successes if s) / float(len(successes))


def euclidean_distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    """Return 3D Euclidean distance, matching the official VLN-CE path/nDTW code."""
    if len(point_a) != 3 or len(point_b) != 3:
        raise ValueError("R2R-CE points must contain exactly three coordinates.")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(point_a, point_b)))


def trajectory_length(positions: Sequence[Sequence[float]]) -> float:
    """Sum Euclidean motion between consecutive simulator positions."""
    if len(positions) < 2:
        return 0.0
    return sum(euclidean_distance(a, b) for a, b in zip(positions[:-1], positions[1:]))


def _deduplicate_consecutive_positions(positions: Sequence[Point3D]) -> tuple[Point3D, ...]:
    """Match VLN-CE nDTW behavior, which skips unchanged simulator positions."""
    unique: list[Point3D] = []
    for point in positions:
        if not unique or point != unique[-1]:
            unique.append(point)
    return tuple(unique)


def dynamic_time_warping_distance(
    path: Sequence[Sequence[float]],
    reference_path: Sequence[Sequence[float]],
) -> float:
    """Compute exact DTW with Euclidean point cost and no window approximation."""
    if not path or not reference_path:
        raise ValueError("DTW paths must both be non-empty.")

    previous = [math.inf] * (len(reference_path) + 1)
    previous[0] = 0.0
    for point in path:
        current = [math.inf] * (len(reference_path) + 1)
        for index, reference_point in enumerate(reference_path, start=1):
            cost = euclidean_distance(point, reference_point)
            current[index] = cost + min(current[index - 1], previous[index], previous[index - 1])
        previous = current
    return previous[-1]


def normalized_dtw(
    path: Sequence[Point3D],
    reference_path: Sequence[Point3D],
    success_distance: float = 3.0,
) -> float:
    """Compute VLN-CE nDTW: ``exp(-DTW / (len(reference) * radius))``."""
    if success_distance <= 0.0 or not math.isfinite(success_distance):
        raise ValueError("nDTW success_distance must be finite and positive.")
    deduplicated = _deduplicate_consecutive_positions(path)
    distance = dynamic_time_warping_distance(deduplicated, reference_path)
    return math.exp(-distance / (len(reference_path) * success_distance))


def evaluate_r2r_ce_trace(
    trace: R2RCEEpisodeTrace,
    success_distance: float = 3.0,
    eps: float = 1e-8,
) -> R2RCEEpisodeMetrics:
    """Evaluate one trace using the official R2R-CE success, SPL, nDTW, and SDTW rules."""
    if success_distance <= 0.0 or not math.isfinite(success_distance):
        raise ValueError("R2R-CE success_distance must be finite and positive.")
    stopped = bool(trace.actions and trace.actions[-1] == R2RCEAction.STOP)
    navigation_error = trace.distances_to_goal[-1]
    success = float(stopped and navigation_error < success_distance)
    oracle_success = float(min(trace.distances_to_goal) < success_distance)
    path_length = trajectory_length(trace.positions)
    spl_score = (
        success
        * trace.shortest_path_length
        / max(
            trace.shortest_path_length,
            path_length,
            eps,
        )
    )
    ndtw_score = normalized_dtw(
        trace.positions,
        trace.reference_path,
        success_distance=success_distance,
    )
    return R2RCEEpisodeMetrics(
        episode_id=trace.episode_id,
        navigation_error=navigation_error,
        oracle_success=oracle_success,
        success=success,
        path_length=path_length,
        shortest_path_length=trace.shortest_path_length,
        spl=spl_score,
        ndtw=ndtw_score,
        sdtw=success * ndtw_score,
        step_count=len(trace.actions) + trace.invalid_action_count,
        collision_count=sum(trace.collisions),
        invalid_action_count=trace.invalid_action_count,
        stopped=stopped,
    )


def aggregate_r2r_ce_metrics(
    episodes: Sequence[R2RCEEpisodeMetrics],
) -> R2RCEAggregateMetrics:
    """Aggregate episode metrics without silently accepting an empty manifest."""
    if not episodes:
        raise ValueError("Cannot aggregate an empty R2R-CE episode manifest.")
    count = len(episodes)

    def mean(field: str) -> float:
        return sum(float(getattr(episode, field)) for episode in episodes) / count

    return R2RCEAggregateMetrics(
        episode_count=count,
        navigation_error=mean("navigation_error"),
        oracle_success_rate=mean("oracle_success"),
        success_rate=mean("success"),
        spl=mean("spl"),
        ndtw=mean("ndtw"),
        sdtw=mean("sdtw"),
        mean_path_length=mean("path_length"),
        mean_steps=mean("step_count"),
        collision_count=sum(episode.collision_count for episode in episodes),
        invalid_action_count=sum(episode.invalid_action_count for episode in episodes),
    )


def spl(episodes: Sequence[NavigationEpisodeResult], eps: float = 1e-8) -> float:
    """Compute Success weighted by Path Length (SPL)."""
    if len(episodes) == 0:
        return 0.0

    vals: list[float] = []
    for ep in episodes:
        if not ep.success:
            vals.append(0.0)
            continue
        denom = max(ep.path_length, ep.shortest_path_length, eps)
        vals.append(ep.shortest_path_length / denom)
    return sum(vals) / float(len(vals))


def permutation_sensitivity_index(
    baseline_score: float,
    permuted_score: float,
    eps: float = 1e-8,
) -> float:
    """Relative drop from baseline under token permutation.

    PSI = max(0, (baseline - permuted) / max(|baseline|, eps))
    """
    denom = max(abs(baseline_score), eps)
    return max(0.0, (baseline_score - permuted_score) / denom)


def weighted_composite(
    metrics: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Weighted average of scalar metrics."""
    if len(metrics) == 0:
        return 0.0

    if weights is None:
        return sum(metrics.values()) / float(len(metrics))

    total_weight = 0.0
    weighted_sum = 0.0
    for key, value in metrics.items():
        w = float(weights.get(key, 0.0))
        if w <= 0.0:
            continue
        weighted_sum += w * float(value)
        total_weight += w
    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight


def compute_metric_bundle(
    episodes: Sequence[NavigationEpisodeResult],
    baseline_score: float,
    permuted_score: float,
    composite_weights: Mapping[str, float] | None = None,
) -> MetricBundle:
    """Compute SR, SPL, PSI, and a weighted composite metric."""
    sr = success_rate([ep.success for ep in episodes])
    spl_score = spl(episodes)
    psi = permutation_sensitivity_index(
        baseline_score=baseline_score,
        permuted_score=permuted_score,
    )
    cmb = weighted_composite(
        metrics={
            "success_rate": sr,
            "spl": spl_score,
            "permutation_sensitivity": psi,
        },
        weights=composite_weights,
    )
    return MetricBundle(
        success_rate=sr,
        spl=spl_score,
        permutation_sensitivity=psi,
        composite=cmb,
    )
