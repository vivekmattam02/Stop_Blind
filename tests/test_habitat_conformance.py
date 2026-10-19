"""Tests for pure released-trace versus runtime comparison logic."""

from __future__ import annotations

import numpy as np
import pytest

from spatialvlm.data.datasets import R2RCEAction, R2RCEEpisode, R2RCEGoal, R2RCEGroundTruth
from spatialvlm.eval.habitat_conformance import (
    HabitatConformanceThresholds,
    compare_ground_truth_replay,
    describe_observation_contract,
    quaternion_angular_error_radians,
)
from spatialvlm.eval.metrics import R2RCEEpisodeTrace, evaluate_r2r_ce_trace
from spatialvlm.eval.r2r_ce import R2RCEEvaluationResult


def _episode() -> R2RCEEpisode:
    return R2RCEEpisode(
        episode_id="1",
        trajectory_id="101",
        scene_id="mp3d/example/example.glb",
        instruction_text="Walk forward.",
        instruction_tokens=(),
        start_position=(0.0, 0.0, 0.0),
        start_rotation=(0.0, 0.0, 0.0, 1.0),
        goals=(R2RCEGoal(position=(2.0, 0.0, 0.0), radius=3.0),),
        reference_path=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        geodesic_distance=2.0,
        raw={"episode_id": 1},
    )


def _ground_truth() -> R2RCEGroundTruth:
    return R2RCEGroundTruth(
        episode_id="1",
        actions=(
            R2RCEAction.MOVE_FORWARD,
            R2RCEAction.TURN_LEFT,
            R2RCEAction.MOVE_FORWARD,
            R2RCEAction.STOP,
        ),
        forward_steps=2,
        locations=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )


def _result(
    last_position: float = 2.0,
    runtime_reset_geodesic: float = 2.0,
) -> R2RCEEvaluationResult:
    positions = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (last_position, 0.0, 0.0),
        (last_position, 0.0, 0.0),
    )
    trace = R2RCEEpisodeTrace(
        episode_id="1",
        positions=positions,
        distances_to_goal=(
            runtime_reset_geodesic,
            1.0,
            1.0,
            abs(2.0 - last_position),
            abs(2.0 - last_position),
        ),
        actions=_ground_truth().actions,
        shortest_path_length=runtime_reset_geodesic,
        reference_path=_ground_truth().locations,
        collisions=(False, False, False, False),
    )
    return R2RCEEvaluationResult(
        trace=trace,
        metrics=evaluate_r2r_ce_trace(trace),
        steps=(),
    )


def test_observation_contract_requires_native_sensors_and_metric_depth():
    report = describe_observation_contract(
        {
            "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "depth": np.zeros((256, 256, 1), dtype=np.float32),
            "depth_meters": np.ones((256, 256), dtype=np.float32),
        }
    )
    assert report["status"] == "passed"
    assert all(report["checks"].values())


def test_quaternion_error_is_sign_invariant():
    error = quaternion_angular_error_radians(
        (0.0, 0.0, 0.0, -1.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert error == pytest.approx(0.0)


def test_exact_ground_truth_replay_passes_every_check():
    report = compare_ground_truth_replay(
        _episode(),
        _ground_truth(),
        _result(),
        reset_rotation_error_radians=0.0,
        thresholds=HabitatConformanceThresholds(),
    )
    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["diagnostics"]["release_metadata_geodesic_within_tolerance"] is True


def test_release_geodesic_drift_is_diagnostic_not_a_parity_failure():
    report = compare_ground_truth_replay(
        _episode(),
        _ground_truth(),
        _result(runtime_reset_geodesic=1.5),
        reset_rotation_error_radians=0.0,
        thresholds=HabitatConformanceThresholds(),
    )
    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert report["diagnostics"]["release_metadata_geodesic_within_tolerance"] is False
    assert report["observed"]["runtime_reset_geodesic_m"] == pytest.approx(1.5)
    assert report["observed"]["release_metadata_geodesic_m"] == pytest.approx(2.0)
    assert report["observed"]["release_metadata_geodesic_drift_m"] == pytest.approx(0.5)


def test_position_drift_fails_closed():
    report = compare_ground_truth_replay(
        _episode(),
        _ground_truth(),
        _result(last_position=2.01),
        reset_rotation_error_radians=0.0,
        thresholds=HabitatConformanceThresholds(),
    )
    assert report["status"] == "failed"
    assert report["checks"]["all_replay_positions_match_release"] is False
