from __future__ import annotations

import importlib.util
from pathlib import Path


def _analysis_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/spatialvlm/eval/native_closed_loop_analysis.py"
    )
    spec = importlib.util.spec_from_file_location("gridnav_native_closed_loop_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = _analysis_module()


def _report(episode_id: str, *, navigation_error: float, stopped: bool = False):
    return {
        "episode_id": episode_id,
        "evaluation": {
            "metrics": {
                "navigation_error": navigation_error,
                "stopped": stopped,
                "step_count": 500,
            }
        },
    }


def test_paired_delta_summary_preserves_pairing_and_direction():
    unweighted = {
        "a": _report("a", navigation_error=5.0),
        "b": _report("b", navigation_error=3.0),
        "c": _report("c", navigation_error=4.0),
    }
    weighted = {
        "a": _report("a", navigation_error=4.0),
        "b": _report("b", navigation_error=5.0),
        "c": _report("c", navigation_error=4.0),
    }
    summary = analysis.paired_delta_summary(unweighted, weighted, "navigation_error")
    assert summary["direction"] == "inverse_frequency_minus_unweighted"
    assert summary["mean_delta"] == 1.0 / 3.0
    assert summary["negative_count"] == 1
    assert summary["positive_count"] == 1
    assert summary["zero_count"] == 1


def test_scene_cluster_bootstrap_is_deterministic_and_keeps_scene_clusters():
    rows = [
        {"episode_id": "a", "scene_id": "one", "deltas": {"navigation_error": -1.0}},
        {"episode_id": "b", "scene_id": "one", "deltas": {"navigation_error": -3.0}},
        {"episode_id": "c", "scene_id": "two", "deltas": {"navigation_error": 2.0}},
    ]
    first = analysis.scene_cluster_bootstrap(
        rows, metrics=("navigation_error",), seed=7, replicates=100
    )
    second = analysis.scene_cluster_bootstrap(
        rows, metrics=("navigation_error",), seed=7, replicates=100
    )
    assert first == second
    assert first["scene_count"] == 2
    assert first["episode_count"] == 3
    assert first["metrics"]["navigation_error"]["point_delta"] == -2.0 / 3.0


def test_paired_stop_delta_uses_boolean_as_zero_one():
    unweighted = {"a": _report("a", navigation_error=1.0, stopped=False)}
    weighted = {"a": _report("a", navigation_error=1.0, stopped=True)}
    summary = analysis.paired_delta_summary(unweighted, weighted, "stopped")
    assert summary["mean_delta"] == 1.0
    assert summary["positive_count"] == 1


def test_episode_behavior_distinguishes_aggregate_diversity_from_switching():
    report = {
        "evaluation": {
            "trace": {
                "actions": [2, 2, 1, 1],
                "collisions": [False, False, False, True],
                "distances_to_goal": [5.0, 5.0, 5.0, 4.8, 4.8],
                "positions": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.2],
                    [0.0, 0.0, 0.2],
                ],
            }
        }
    }
    behavior = analysis.derive_episode_behavior(report)
    assert behavior["action_support"] == ["MOVE_FORWARD", "TURN_LEFT"]
    assert behavior["constant_action"] is None
    assert behavior["action_switch_count"] == 1
    assert behavior["longest_action_run"] == 2
    assert behavior["best_goal_progress"] == 0.20000000000000018
    assert behavior["forward_command_count"] == 2
    assert behavior["forward_collision_count"] == 1
    assert behavior["forward_movement_count"] == 1
    assert behavior["stationary_step_count"] == 3


def test_goal_radius_decision_audit_uses_pre_action_distance():
    def decision(stop_probability):
        return {
            "diagnostics": {
                "action_decision": {
                    "scores": {
                        "STOP": {"restricted_probability": stop_probability}
                    }
                }
            }
        }

    report = {
        "evaluation": {
            "trace": {
                "actions": [1, 1, 0],
                "distances_to_goal": [3.2, 2.9, 2.8, 2.8],
            }
        },
        "policy_telemetry": {
            "decisions": [decision(0.1), decision(0.2), decision(0.6)]
        },
    }
    audit = analysis.derive_goal_radius_decisions(report)
    assert audit["inside_radius_decision_count"] == 2
    assert audit["inside_radius_stop_count"] == 1
    assert audit["inside_radius_first_decision_step"] == 2
    assert audit["inside_radius_last_decision_step"] == 3
    assert audit["inside_radius_mean_stop_probability"] == 0.4
