"""Pure comparison logic for modern Habitat-Sim versus released R2R-CE traces."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from spatialvlm.data.datasets import R2RCEAction, R2RCEEpisode, R2RCEGroundTruth
from spatialvlm.eval.r2r_ce import R2RCEEvaluationResult


@dataclass(frozen=True)
class HabitatConformanceThresholds:
    """Fail-closed numerical tolerances for the first modern-runtime parity gate."""

    reset_position_m: float = 1e-4
    reset_rotation_radians: float = 1e-5
    release_geodesic_drift_m: float = 1e-3
    replay_position_m: float = 1e-3
    ndtw_minimum: float = 0.9999

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("Habitat conformance thresholds must be finite and non-negative.")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def quaternion_angular_error_radians(
    actual_xyzw: tuple[float, float, float, float],
    expected_xyzw: tuple[float, float, float, float],
) -> float:
    """Return sign-invariant angular distance between two XYZW quaternions."""
    actual = np.asarray(actual_xyzw, dtype=np.float64)
    expected = np.asarray(expected_xyzw, dtype=np.float64)
    actual_norm = float(np.linalg.norm(actual))
    expected_norm = float(np.linalg.norm(expected))
    if actual_norm == 0.0 or expected_norm == 0.0:
        raise ValueError("Quaternion norms must be non-zero.")
    dot = abs(float(np.dot(actual / actual_norm, expected / expected_norm)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def describe_observation_contract(observation: dict[str, Any]) -> dict[str, Any]:
    """Validate the native sensor contract plus explicit metric-depth stream."""
    expected = {
        "rgb": ((224, 224, 3), np.dtype(np.uint8)),
        "depth": ((256, 256, 1), np.dtype(np.float32)),
        "depth_meters": ((256, 256), np.dtype(np.float32)),
    }
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for name, (shape, dtype) in expected.items():
        if name not in observation:
            checks[f"{name}_present"] = False
            continue
        array = np.asarray(observation[name])
        checks[f"{name}_present"] = True
        checks[f"{name}_shape"] = tuple(array.shape) == shape
        checks[f"{name}_dtype"] = array.dtype == dtype
        checks[f"{name}_finite"] = bool(np.isfinite(array).all())
        details[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }

    if "depth" in observation:
        depth = np.asarray(observation["depth"])
        checks["depth_normalized_range"] = bool(
            depth.size and float(depth.min()) >= 0.0 and float(depth.max()) <= 1.0
        )
    if "depth_meters" in observation:
        metric = np.asarray(observation["depth_meters"])
        checks["depth_metric_range"] = bool(
            metric.size and float(metric.min()) >= 0.0 and float(metric.max()) <= 10.0
        )
    return {
        "status": "passed" if checks and all(checks.values()) else "failed",
        "checks": checks,
        "observed": details,
    }


def compare_ground_truth_replay(
    episode: R2RCEEpisode,
    ground_truth: R2RCEGroundTruth,
    result: R2RCEEvaluationResult,
    *,
    reset_rotation_error_radians: float,
    thresholds: HabitatConformanceThresholds,
) -> dict[str, Any]:
    """Compare a real simulator rollout with positions released by VLN-CE v1-3."""
    if episode.episode_id != ground_truth.episode_id:
        raise ValueError("Episode and ground-truth IDs must match for conformance.")
    if result.trace.episode_id != episode.episode_id:
        raise ValueError("Evaluation result does not match the conformance episode.")

    observed_move_positions = [result.trace.positions[0]]
    observed_move_collisions: list[bool] = []
    for index, action in enumerate(result.trace.actions):
        if action == R2RCEAction.MOVE_FORWARD:
            observed_move_positions.append(result.trace.positions[index + 1])
            observed_move_collisions.append(result.trace.collisions[index])

    expected_locations = ground_truth.locations
    location_count_matches = len(observed_move_positions) == len(expected_locations)
    location_errors = (
        [
            math.dist(actual, expected)
            for actual, expected in zip(observed_move_positions, expected_locations)
        ]
        if location_count_matches
        else []
    )
    max_location_error = max(location_errors, default=math.inf)
    mean_location_error = (
        sum(location_errors) / len(location_errors) if location_errors else math.inf
    )
    release_geodesic_drift = abs(result.trace.distances_to_goal[0] - episode.geodesic_distance)
    reset_position_error = math.dist(result.trace.positions[0], episode.start_position)

    expected_displacements = [
        math.dist(start, end) for start, end in zip(expected_locations[:-1], expected_locations[1:])
    ]
    observed_displacements = [
        math.dist(start, end)
        for start, end in zip(observed_move_positions[:-1], observed_move_positions[1:])
    ]
    displacement_errors = (
        [
            abs(actual - expected)
            for actual, expected in zip(observed_displacements, expected_displacements)
        ]
        if location_count_matches
        else []
    )

    checks = {
        "ground_truth_location_count_is_forward_steps_plus_reset": len(expected_locations)
        == ground_truth.forward_steps + 1,
        "executed_actions_match_release": result.trace.actions == ground_truth.actions,
        "executed_forward_count_matches_release": len(observed_move_collisions)
        == ground_truth.forward_steps,
        "replay_location_count_matches_release": location_count_matches,
        "reset_position_matches_release": reset_position_error <= thresholds.reset_position_m,
        "reset_rotation_matches_release": reset_rotation_error_radians
        <= thresholds.reset_rotation_radians,
        "all_replay_positions_match_release": max_location_error <= thresholds.replay_position_m,
        "stopped": result.metrics.stopped,
        "ground_truth_policy_succeeds": result.metrics.success == 1.0,
        "ndtw_matches_release_path": result.metrics.ndtw >= thresholds.ndtw_minimum,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "thresholds": thresholds.to_dict(),
        "checks": checks,
        "diagnostics": {
            "release_metadata_geodesic_within_tolerance": release_geodesic_drift
            <= thresholds.release_geodesic_drift_m,
        },
        "observed": {
            "episode_id": episode.episode_id,
            "scene_id": episode.scene_id,
            "action_count": len(result.trace.actions),
            "forward_count": len(observed_move_collisions),
            "release_location_count": len(expected_locations),
            "replay_location_count": len(observed_move_positions),
            "reset_position_error_m": reset_position_error,
            "reset_rotation_error_radians": reset_rotation_error_radians,
            "release_metadata_geodesic_m": episode.geodesic_distance,
            "runtime_reset_geodesic_m": result.trace.distances_to_goal[0],
            "release_metadata_geodesic_drift_m": release_geodesic_drift,
            "maximum_location_error_m": max_location_error if location_errors else None,
            "mean_location_error_m": mean_location_error if location_errors else None,
            "maximum_move_displacement_error_m": (
                max(displacement_errors) if displacement_errors else None
            ),
            "simulator_collision_count_on_forward": sum(observed_move_collisions),
            "metrics": result.metrics.to_dict(),
        },
    }
