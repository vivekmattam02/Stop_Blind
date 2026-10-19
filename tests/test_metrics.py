"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.metrics import (
    NavigationEpisodeResult,
    R2RCEEpisodeTrace,
    aggregate_r2r_ce_metrics,
    compute_metric_bundle,
    dynamic_time_warping_distance,
    evaluate_r2r_ce_trace,
    normalized_dtw,
    permutation_sensitivity_index,
    spl,
    success_rate,
    weighted_composite,
)


def test_success_rate():
    assert success_rate([True, False, True, True]) == pytest.approx(0.75)


def test_spl():
    episodes = [
        NavigationEpisodeResult(success=True, path_length=10.0, shortest_path_length=8.0),
        NavigationEpisodeResult(success=False, path_length=5.0, shortest_path_length=4.0),
        NavigationEpisodeResult(success=True, path_length=6.0, shortest_path_length=6.0),
    ]
    score = spl(episodes)
    assert score == pytest.approx((0.8 + 0.0 + 1.0) / 3.0)


def test_permutation_sensitivity_index():
    psi = permutation_sensitivity_index(baseline_score=0.8, permuted_score=0.6)
    assert psi == pytest.approx(0.25)


def test_weighted_composite():
    cmb = weighted_composite(
        metrics={"a": 0.5, "b": 1.0},
        weights={"a": 1.0, "b": 3.0},
    )
    assert cmb == pytest.approx(0.875)


def test_compute_metric_bundle():
    episodes = [
        NavigationEpisodeResult(success=True, path_length=10.0, shortest_path_length=8.0),
        NavigationEpisodeResult(success=True, path_length=5.0, shortest_path_length=5.0),
    ]
    bundle = compute_metric_bundle(
        episodes=episodes,
        baseline_score=0.8,
        permuted_score=0.6,
        composite_weights={"success_rate": 0.4, "spl": 0.4, "permutation_sensitivity": 0.2},
    )
    assert bundle.success_rate == pytest.approx(1.0)
    assert bundle.spl == pytest.approx((0.8 + 1.0) / 2.0)
    assert bundle.permutation_sensitivity == pytest.approx(0.25)
    assert 0.0 <= bundle.composite <= 1.0


def _perfect_trace() -> R2RCEEpisodeTrace:
    return R2RCEEpisodeTrace(
        episode_id="1",
        positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        distances_to_goal=(2.0, 1.0, 0.0, 0.0),
        actions=(
            R2RCEAction.MOVE_FORWARD,
            R2RCEAction.MOVE_FORWARD,
            R2RCEAction.STOP,
        ),
        shortest_path_length=2.0,
        reference_path=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        collisions=(False, False, False),
    )


def test_exact_dtw_and_ndtw_are_one_for_identical_path():
    path = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert dynamic_time_warping_distance(path, path) == pytest.approx(0.0)
    assert normalized_dtw(path, path, success_distance=3.0) == pytest.approx(1.0)


def test_r2r_ce_perfect_trace_matches_success_spl_ndtw_sdtw_contract():
    metrics = evaluate_r2r_ce_trace(_perfect_trace())
    assert metrics.navigation_error == pytest.approx(0.0)
    assert metrics.oracle_success == pytest.approx(1.0)
    assert metrics.success == pytest.approx(1.0)
    assert metrics.path_length == pytest.approx(2.0)
    assert metrics.spl == pytest.approx(1.0)
    assert metrics.ndtw == pytest.approx(1.0)
    assert metrics.sdtw == pytest.approx(1.0)
    assert metrics.step_count == 3


def test_r2r_ce_requires_stop_even_after_reaching_goal():
    trace = R2RCEEpisodeTrace(
        episode_id="no-stop",
        positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        distances_to_goal=(2.0, 1.0, 0.0),
        actions=(R2RCEAction.MOVE_FORWARD, R2RCEAction.MOVE_FORWARD),
        shortest_path_length=2.0,
        reference_path=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    metrics = evaluate_r2r_ce_trace(trace)
    assert metrics.oracle_success == pytest.approx(1.0)
    assert metrics.success == pytest.approx(0.0)
    assert metrics.spl == pytest.approx(0.0)
    assert metrics.ndtw == pytest.approx(1.0)
    assert metrics.sdtw == pytest.approx(0.0)


def test_r2r_ce_success_radius_is_strictly_less_than_three_metres():
    trace = R2RCEEpisodeTrace(
        episode_id="boundary",
        positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        distances_to_goal=(3.0, 3.0),
        actions=(R2RCEAction.STOP,),
        shortest_path_length=3.0,
        reference_path=((0.0, 0.0, 0.0),),
    )
    metrics = evaluate_r2r_ce_trace(trace, success_distance=3.0)
    assert metrics.success == pytest.approx(0.0)


def test_r2r_ce_aggregate_is_episode_unweighted_and_preserves_counts():
    success_metrics = evaluate_r2r_ce_trace(_perfect_trace())
    failure_trace = R2RCEEpisodeTrace(
        episode_id="2",
        positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        distances_to_goal=(5.0, 5.0),
        actions=(R2RCEAction.STOP,),
        shortest_path_length=5.0,
        reference_path=((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)),
        collisions=(True,),
        invalid_action_count=2,
    )
    failure_metrics = evaluate_r2r_ce_trace(failure_trace)
    aggregate = aggregate_r2r_ce_metrics([success_metrics, failure_metrics])
    assert aggregate.episode_count == 2
    assert aggregate.success_rate == pytest.approx(0.5)
    assert aggregate.collision_count == 1
    assert aggregate.invalid_action_count == 2
