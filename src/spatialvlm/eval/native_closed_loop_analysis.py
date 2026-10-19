"""Deterministic episode-level analysis of the frozen native-C0 closed-loop suite.

This module deliberately uses only the already verified JSON artifacts.  It does not
load a model, render Habitat, or open either sealed split.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
GATE_ID = "native_qwen_c0_closed_loop_episode_analysis_v1"
ANALYSIS_SEED = 20260826
BOOTSTRAP_REPLICATES = 10_000
SUCCESS_RADIUS_METERS = 3.0
ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
PAIR_METRICS = (
    "navigation_error",
    "ndtw",
    "path_length",
    "step_count",
    "collision_count",
    "oracle_success",
    "success",
    "spl",
    "sdtw",
    "stopped",
    "step_cap",
    "minimum_distance",
    "best_goal_progress",
    "final_goal_progress",
    "stationary_fraction",
    "action_switch_count",
    "action_entropy_nats",
    "unique_position_count",
    "closest_approach_step",
)
DERIVED_VALUE_METRICS = frozenset(
    {
        "minimum_distance",
        "best_goal_progress",
        "final_goal_progress",
        "stationary_fraction",
        "action_switch_count",
        "action_entropy_nats",
        "unique_position_count",
        "closest_approach_step",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sequence.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Quantile probability must be in [0, 1].")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _treatment_and_seed(checkpoint_id: str) -> tuple[str, str]:
    if checkpoint_id.startswith("unweighted_s"):
        return "unweighted", checkpoint_id.removeprefix("unweighted_s")
    if checkpoint_id.startswith("inverse_frequency_s"):
        return "inverse_frequency", checkpoint_id.removeprefix("inverse_frequency_s")
    raise ValueError(f"Unrecognized frozen checkpoint ID: {checkpoint_id}")


def _report_value(report: Mapping[str, Any], metric: str) -> float:
    metrics = report["evaluation"]["metrics"]
    if metric == "step_cap":
        return float(int(metrics["step_count"] == 500))
    if metric in DERIVED_VALUE_METRICS:
        return float(derive_episode_behavior(report)[metric])
    return float(metrics[metric])


def _trace_actions(report: Mapping[str, Any]) -> list[int]:
    return [int(action) for action in report["evaluation"]["trace"]["actions"]]


def derive_episode_behavior(report: Mapping[str, Any]) -> dict[str, Any]:
    """Derive deterministic behavior diagnostics from one completed episode trace."""

    trace = report["evaluation"]["trace"]
    actions = [int(action) for action in trace["actions"]]
    collisions = [bool(value) for value in trace["collisions"]]
    distances = [float(value) for value in trace["distances_to_goal"]]
    positions = [[float(value) for value in point] for point in trace["positions"]]
    if len(collisions) != len(actions):
        raise ValueError("Collision and action traces must have equal length.")
    if len(distances) != len(actions) + 1 or len(positions) != len(actions) + 1:
        raise ValueError("Distance and position traces must include initial state.")

    displacements = [
        math.dist(positions[index], positions[index + 1])
        for index in range(len(actions))
    ]
    counts = Counter(actions)
    run_lengths: list[tuple[int, int]] = []
    for action in actions:
        if run_lengths and run_lengths[-1][0] == action:
            prior_action, prior_length = run_lengths[-1]
            run_lengths[-1] = (prior_action, prior_length + 1)
        else:
            run_lengths.append((action, 1))
    closest_step = min(range(len(distances)), key=distances.__getitem__)
    total = len(actions)
    entropy = 0.0
    if total:
        entropy = -sum(
            (count / total) * math.log(count / total) for count in counts.values()
        )
    forward_indices = [index for index, action in enumerate(actions) if action == 1]
    action_conditioned: dict[str, dict[str, Any]] = {}
    for action_id, action_name in ACTION_NAMES.items():
        indices = [index for index, action in enumerate(actions) if action == action_id]
        if not indices:
            continue
        collision_count = sum(collisions[index] for index in indices)
        action_conditioned[action_name] = {
            "command_count": len(indices),
            "collision_count": collision_count,
            "collision_fraction": collision_count / len(indices),
        }
    support = sorted(ACTION_NAMES[action] for action in counts)
    return {
        "initial_distance": distances[0],
        "minimum_distance": min(distances),
        "final_distance": distances[-1],
        "best_goal_progress": distances[0] - min(distances),
        "final_goal_progress": distances[0] - distances[-1],
        "closest_approach_step": closest_step,
        "steps_after_closest_approach": total - closest_step,
        "entered_success_radius": min(distances) <= SUCCESS_RADIUS_METERS,
        "finished_inside_success_radius": distances[-1] <= SUCCESS_RADIUS_METERS,
        "action_support": support,
        "action_support_size": len(support),
        "constant_action": support[0] if len(support) == 1 else None,
        "action_switch_count": max(0, len(run_lengths) - 1),
        "longest_action_run": max((length for _, length in run_lengths), default=0),
        "action_entropy_nats": entropy,
        "forward_command_count": len(forward_indices),
        "forward_collision_count": sum(collisions[index] for index in forward_indices),
        "forward_movement_count": sum(
            displacements[index] > 1e-5 for index in forward_indices
        ),
        "stationary_step_count": sum(value <= 1e-5 for value in displacements),
        "stationary_fraction": (
            sum(value <= 1e-5 for value in displacements) / total if total else 0.0
        ),
        "unique_position_count": len(
            {
                tuple(round(value, 6) for value in position)
                for position in positions
            }
        ),
        "turn_command_count": counts[2] + counts[3],
        "commanded_turn_degrees": 15 * (counts[2] + counts[3]),
        "net_commanded_turn_degrees": 15 * (counts[3] - counts[2]),
        "action_conditioned_collisions": action_conditioned,
    }


def derive_goal_radius_decisions(report: Mapping[str, Any]) -> dict[str, Any]:
    """Audit model decisions made while the agent is already inside success radius.

    Counts are repeated decision opportunities, not independent statistical samples.
    """

    trace = report["evaluation"]["trace"]
    actions = [int(action) for action in trace["actions"]]
    distances = [float(value) for value in trace["distances_to_goal"]]
    decisions = report["policy_telemetry"]["decisions"]
    if len(decisions) != len(actions) or len(distances) != len(actions) + 1:
        raise ValueError("Goal-radius audit requires aligned action, distance, and decision rows.")
    inside_probabilities: list[float] = []
    outside_probabilities: list[float] = []
    inside_stop_count = 0
    inside_steps: list[int] = []
    for index, (action, distance, decision) in enumerate(
        zip(actions, distances, decisions)
    ):
        stop_probability = float(
            decision["diagnostics"]["action_decision"]["scores"]["STOP"][
                "restricted_probability"
            ]
        )
        if distance <= SUCCESS_RADIUS_METERS:
            inside_probabilities.append(stop_probability)
            inside_stop_count += int(action == 0)
            inside_steps.append(index + 1)
        else:
            outside_probabilities.append(stop_probability)
    return {
        "inside_radius_decision_count": len(inside_probabilities),
        "inside_radius_stop_count": inside_stop_count,
        "inside_radius_first_decision_step": inside_steps[0] if inside_steps else None,
        "inside_radius_last_decision_step": inside_steps[-1] if inside_steps else None,
        "inside_radius_mean_stop_probability": (
            mean(inside_probabilities) if inside_probabilities else None
        ),
        "outside_radius_decision_count": len(outside_probabilities),
        "outside_radius_mean_stop_probability": (
            mean(outside_probabilities) if outside_probabilities else None
        ),
    }


def _report_digest(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = report["evaluation"]["metrics"]
    actions = _trace_actions(report)
    distances = [
        float(value) for value in report["evaluation"]["trace"]["distances_to_goal"]
    ]
    return {
        "episode_id": str(report["episode_id"]),
        "scene_id": str(report["scene_id"]),
        "metrics": metrics,
        "actions": actions,
        "distances_to_goal": distances,
        "action_histogram": dict(report["action_histogram"]),
    }


def summarize_checkpoint(
    checkpoint_id: str,
    reports: Sequence[Mapping[str, Any]],
    always_forward_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize behavior that aggregate navigation metrics cannot expose."""

    if not reports:
        raise ValueError("Checkpoint report list cannot be empty.")
    treatment, seed = _treatment_and_seed(checkpoint_id)
    action_histogram: Counter[str] = Counter()
    success_ids: list[str] = []
    oracle_ids: list[str] = []
    stopped_ids: list[str] = []
    capped_ids: list[str] = []
    all_forward_ids: list[str] = []
    exact_control_action_ids: list[str] = []
    exact_control_metric_ids: list[str] = []
    stop_events: list[dict[str, Any]] = []
    oracle_events: list[dict[str, Any]] = []
    total_steps = 0
    total_collisions = 0
    behavior_rows: list[dict[str, Any]] = []
    goal_radius_rows: list[dict[str, Any]] = []
    constant_actions: Counter[str] = Counter()
    support_sizes: Counter[int] = Counter()
    conditioned_commands: Counter[str] = Counter()
    conditioned_collisions: Counter[str] = Counter()

    metric_keys = tuple(reports[0]["evaluation"]["metrics"].keys())
    for report in reports:
        digest = _report_digest(report)
        episode_id = digest["episode_id"]
        metrics = digest["metrics"]
        actions = digest["actions"]
        distances = digest["distances_to_goal"]
        behavior = derive_episode_behavior(report)
        behavior_rows.append(behavior)
        goal_radius_rows.append(derive_goal_radius_decisions(report))
        support_sizes[int(behavior["action_support_size"])] += 1
        if behavior["constant_action"] is not None:
            constant_actions[str(behavior["constant_action"])] += 1
        for action_name, record in behavior["action_conditioned_collisions"].items():
            conditioned_commands[action_name] += int(record["command_count"])
            conditioned_collisions[action_name] += int(record["collision_count"])
        action_histogram.update(report["action_histogram"])
        total_steps += int(metrics["step_count"])
        total_collisions += int(metrics["collision_count"])
        if float(metrics["success"]) == 1.0:
            success_ids.append(episode_id)
        if float(metrics["oracle_success"]) == 1.0:
            oracle_ids.append(episode_id)
            oracle_events.append(
                {
                    "episode_id": episode_id,
                    "final_distance_m": float(metrics["navigation_error"]),
                    "minimum_distance_m": min(distances),
                    "finished_inside_success_radius": (
                        float(metrics["navigation_error"]) <= SUCCESS_RADIUS_METERS
                    ),
                    "entered_then_finished_outside_success_radius": (
                        min(distances) <= SUCCESS_RADIUS_METERS
                        and float(metrics["navigation_error"]) > SUCCESS_RADIUS_METERS
                    ),
                    "step_count": int(metrics["step_count"]),
                    "stopped": bool(metrics["stopped"]),
                    "action_histogram": dict(report["action_histogram"]),
                }
            )
        if bool(metrics["stopped"]):
            stopped_ids.append(episode_id)
            stop_step = actions.index(0) + 1
            final_distance = float(metrics["navigation_error"])
            stop_events.append(
                {
                    "episode_id": episode_id,
                    "stop_step": stop_step,
                    "final_distance_m": final_distance,
                    "minimum_distance_m": min(distances),
                    "outside_success_radius": final_distance > SUCCESS_RADIUS_METERS,
                    "pre_stop_action_histogram": dict(
                        sorted(
                            Counter(
                                ACTION_NAMES[action] for action in actions[:stop_step]
                            ).items()
                        )
                    ),
                }
            )
        if int(metrics["step_count"]) == 500:
            capped_ids.append(episode_id)
        if actions and all(action == 1 for action in actions):
            all_forward_ids.append(episode_id)

        control = always_forward_reports[episode_id]
        if actions == _trace_actions(control):
            exact_control_action_ids.append(episode_id)
        learned_metrics = report["evaluation"]["metrics"]
        control_metrics = control["evaluation"]["metrics"]
        if all(learned_metrics[key] == control_metrics[key] for key in metric_keys):
            exact_control_metric_ids.append(episode_id)

    action_total = sum(action_histogram.values())
    proportions = {
        action: count / action_total for action, count in sorted(action_histogram.items())
    }
    action_conditioned_collisions = {
        action_name: {
            "command_count": conditioned_commands[action_name],
            "collision_count": conditioned_collisions[action_name],
            "collision_fraction": (
                conditioned_collisions[action_name] / conditioned_commands[action_name]
            ),
        }
        for action_name in sorted(conditioned_commands)
    }
    return {
        "checkpoint_id": checkpoint_id,
        "treatment": treatment,
        "seed": seed,
        "episode_count": len(reports),
        "success_episode_ids": success_ids,
        "oracle_success_episode_ids": oracle_ids,
        "stopped_episode_ids": stopped_ids,
        "step_cap_episode_ids": capped_ids,
        "all_move_forward_episode_count": len(all_forward_ids),
        "all_move_forward_episode_ids": all_forward_ids,
        "exact_always_forward_action_sequence_count": len(exact_control_action_ids),
        "exact_always_forward_action_sequence_episode_ids": exact_control_action_ids,
        "exact_always_forward_metric_count": len(exact_control_metric_ids),
        "exact_always_forward_metric_episode_ids": exact_control_metric_ids,
        "action_histogram": dict(sorted(action_histogram.items())),
        "action_proportions": proportions,
        "dominant_action": max(action_histogram, key=action_histogram.get),
        "dominant_action_fraction": max(action_histogram.values()) / action_total,
        "total_steps": total_steps,
        "total_collisions": total_collisions,
        "collisions_per_step": total_collisions / total_steps,
        "constant_action_episode_count": sum(constant_actions.values()),
        "constant_action_episode_histogram": dict(sorted(constant_actions.items())),
        "nonconstant_action_episode_count": len(reports) - sum(constant_actions.values()),
        "action_support_size_histogram": {
            str(size): count for size, count in sorted(support_sizes.items())
        },
        "total_action_switches": sum(
            int(row["action_switch_count"]) for row in behavior_rows
        ),
        "median_action_switches": median(
            int(row["action_switch_count"]) for row in behavior_rows
        ),
        "maximum_action_switches": max(
            int(row["action_switch_count"]) for row in behavior_rows
        ),
        "mean_action_entropy_nats": mean(
            float(row["action_entropy_nats"]) for row in behavior_rows
        ),
        "median_longest_action_run": median(
            int(row["longest_action_run"]) for row in behavior_rows
        ),
        "goal_distance_dynamics": {
            "mean_initial_distance_m": mean(
                float(row["initial_distance"]) for row in behavior_rows
            ),
            "mean_minimum_distance_m": mean(
                float(row["minimum_distance"]) for row in behavior_rows
            ),
            "mean_final_distance_m": mean(
                float(row["final_distance"]) for row in behavior_rows
            ),
            "mean_best_goal_progress_m": mean(
                float(row["best_goal_progress"]) for row in behavior_rows
            ),
            "mean_final_goal_progress_m": mean(
                float(row["final_goal_progress"]) for row in behavior_rows
            ),
            "positive_best_progress_episode_count": sum(
                float(row["best_goal_progress"]) > 1e-12 for row in behavior_rows
            ),
            "positive_final_progress_episode_count": sum(
                float(row["final_goal_progress"]) > 1e-12 for row in behavior_rows
            ),
            "median_closest_approach_step": median(
                int(row["closest_approach_step"]) for row in behavior_rows
            ),
            "median_steps_after_closest_approach": median(
                int(row["steps_after_closest_approach"]) for row in behavior_rows
            ),
            "minimum_distance_threshold_counts": {
                str(threshold): sum(
                    float(row["minimum_distance"]) <= threshold
                    for row in behavior_rows
                )
                for threshold in (3.0, 5.0, 10.0)
            },
        },
        "locomotion_dynamics": {
            "action_conditioned_collisions": action_conditioned_collisions,
            "forward_movement_count": sum(
                int(row["forward_movement_count"]) for row in behavior_rows
            ),
            "forward_command_count": sum(
                int(row["forward_command_count"]) for row in behavior_rows
            ),
            "forward_movement_fraction": (
                sum(int(row["forward_movement_count"]) for row in behavior_rows)
                / sum(int(row["forward_command_count"]) for row in behavior_rows)
                if sum(int(row["forward_command_count"]) for row in behavior_rows)
                else 0.0
            ),
            "stationary_step_count": sum(
                int(row["stationary_step_count"]) for row in behavior_rows
            ),
            "stationary_step_fraction": mean(
                float(row["stationary_fraction"]) for row in behavior_rows
            ),
            "median_unique_position_count": median(
                int(row["unique_position_count"]) for row in behavior_rows
            ),
            "mean_turn_commands_per_episode": mean(
                int(row["turn_command_count"]) for row in behavior_rows
            ),
            "mean_commanded_turn_degrees_per_episode": mean(
                int(row["commanded_turn_degrees"]) for row in behavior_rows
            ),
        },
        "goal_radius_decision_audit": {
            "interpretation": "repeated closed-loop opportunities, not independent samples",
            "episode_count_with_inside_radius_decision": sum(
                int(row["inside_radius_decision_count"] > 0)
                for row in goal_radius_rows
            ),
            "inside_radius_decision_count": sum(
                int(row["inside_radius_decision_count"]) for row in goal_radius_rows
            ),
            "inside_radius_stop_count": sum(
                int(row["inside_radius_stop_count"]) for row in goal_radius_rows
            ),
            "inside_radius_mean_stop_probability": (
                sum(
                    float(row["inside_radius_mean_stop_probability"])
                    * int(row["inside_radius_decision_count"])
                    for row in goal_radius_rows
                    if row["inside_radius_mean_stop_probability"] is not None
                )
                / sum(int(row["inside_radius_decision_count"]) for row in goal_radius_rows)
                if sum(int(row["inside_radius_decision_count"]) for row in goal_radius_rows)
                else None
            ),
            "episodes": [
                {
                    "episode_id": str(report["episode_id"]),
                    **row,
                }
                for report, row in zip(reports, goal_radius_rows)
                if row["inside_radius_decision_count"] > 0
            ],
        },
        "stop_events": stop_events,
        "oracle_success_events": oracle_events,
    }


def paired_delta_summary(
    unweighted: Mapping[str, Mapping[str, Any]],
    weighted: Mapping[str, Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    """Return weighted-minus-unweighted paired differences on the same episodes."""

    if set(unweighted) != set(weighted):
        raise ValueError("Paired treatments must contain exactly the same episode IDs.")
    episode_ids = list(unweighted)
    deltas = [
        _report_value(weighted[episode_id], metric)
        - _report_value(unweighted[episode_id], metric)
        for episode_id in episode_ids
    ]
    tolerance = 1e-12
    return {
        "metric": metric,
        "direction": "inverse_frequency_minus_unweighted",
        "pair_count": len(deltas),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "negative_count": sum(delta < -tolerance for delta in deltas),
        "positive_count": sum(delta > tolerance for delta in deltas),
        "zero_count": sum(abs(delta) <= tolerance for delta in deltas),
        "minimum_delta": min(deltas),
        "maximum_delta": max(deltas),
    }


def scene_cluster_bootstrap(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = PAIR_METRICS,
    seed: int = ANALYSIS_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Descriptive scene-cluster bootstrap of treatment mean differences.

    Each row contains one episode and treatment-mean metric values across the three
    frozen seeds.  Scenes are resampled with replacement; all episodes belonging to a
    sampled scene remain together.  The interval is descriptive, not a population-CI
    claim, because the 53 scenes were not sampled from a declared superpopulation.
    """

    if replicates <= 0:
        raise ValueError("Bootstrap replicate count must be positive.")
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        clusters[str(row["scene_id"])].append(row)
    scenes = sorted(clusters)
    if not scenes:
        raise ValueError("At least one scene is required for the bootstrap.")
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(replicates):
        sampled_scenes = [rng.choice(scenes) for _ in scenes]
        sampled_rows = [row for scene in sampled_scenes for row in clusters[scene]]
        for metric in metrics:
            deltas = [float(row["deltas"][metric]) for row in sampled_rows]
            distributions[metric].append(mean(deltas))
    return {
        "method": "episode_weighted_scene_cluster_percentile_bootstrap",
        "interpretation": "descriptive interval over the frozen 53-scene surface",
        "seed": seed,
        "replicates": replicates,
        "scene_count": len(scenes),
        "episode_count": len(episode_rows),
        "metrics": {
            metric: {
                "point_delta": mean(float(row["deltas"][metric]) for row in episode_rows),
                "lower_95": _quantile(values, 0.025),
                "upper_95": _quantile(values, 0.975),
            }
            for metric, values in distributions.items()
        },
    }


def _control_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    controls = result["control_comparison"]
    ordered: list[dict[str, Any]] = []
    for run_id, record in controls.items():
        aggregate = record["aggregate"]
        ordered.append(
            {
                "run_id": run_id,
                "family": record["family"],
                "seed": record["seed"],
                "oracle_success_rate": aggregate["oracle_success_rate"],
                "success_rate": aggregate["success_rate"],
                "spl": aggregate["spl"],
                "navigation_error": aggregate["navigation_error"],
            }
        )
    return {
        "controls": sorted(ordered, key=lambda item: item["run_id"]),
        "positive_success_controls": sorted(
            item["run_id"] for item in ordered if float(item["success_rate"]) > 0.0
        ),
        "ground_truth_replay_success_rate": next(
            item["success_rate"]
            for item in ordered
            if item["run_id"] == "released_ground_truth_replay"
        ),
    }


def cross_seed_consistency(
    reports_by_checkpoint: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Measure exact closed-loop trace agreement across the three frozen seeds."""

    output: dict[str, Any] = {}
    for treatment in ("unweighted", "inverse_frequency"):
        checkpoint_ids = sorted(
            checkpoint_id
            for checkpoint_id in reports_by_checkpoint
            if _treatment_and_seed(checkpoint_id)[0] == treatment
        )
        if len(checkpoint_ids) != 3:
            raise ValueError(f"Expected three checkpoints for treatment {treatment}.")
        episode_ids = list(reports_by_checkpoint[checkpoint_ids[0]])
        pairwise: list[dict[str, Any]] = []
        for first_index in range(len(checkpoint_ids)):
            for second_index in range(first_index + 1, len(checkpoint_ids)):
                first = checkpoint_ids[first_index]
                second = checkpoint_ids[second_index]
                exact_ids = [
                    episode_id
                    for episode_id in episode_ids
                    if _trace_actions(reports_by_checkpoint[first][episode_id])
                    == _trace_actions(reports_by_checkpoint[second][episode_id])
                ]
                pairwise.append(
                    {
                        "first_checkpoint_id": first,
                        "second_checkpoint_id": second,
                        "exact_action_sequence_count": len(exact_ids),
                        "exact_action_sequence_episode_ids": exact_ids,
                    }
                )
        all_three_ids = [
            episode_id
            for episode_id in episode_ids
            if len(
                {
                    tuple(_trace_actions(reports_by_checkpoint[checkpoint_id][episode_id]))
                    for checkpoint_id in checkpoint_ids
                }
            )
            == 1
        ]
        output[treatment] = {
            "checkpoint_ids": checkpoint_ids,
            "pairwise": pairwise,
            "all_three_exact_action_sequence_count": len(all_three_ids),
            "all_three_exact_action_sequence_episode_ids": all_three_ids,
        }
    return output


def analyze_closed_loop_artifacts(
    result_root: Path,
    control_root: Path,
) -> dict[str, Any]:
    """Reconstruct and analyze the complete frozen closed-loop suite."""

    manifest_path = result_root / "manifest.json"
    result_path = result_root / "result.json"
    verification_path = result_root / "verification.json"
    for path in (manifest_path, result_path, verification_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required closed-loop artifact is missing: {path}")
    manifest = load_json(manifest_path)
    result = load_json(result_path)
    verification = load_json(verification_path)
    if result.get("status") != "completed":
        raise ValueError("Closed-loop result is not completed.")
    if verification.get("status") != "passed" or verification.get("error_count") != 0:
        raise ValueError("Closed-loop independent verification did not pass.")
    if result.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("Closed-loop result does not link to the manifest bytes.")
    if verification.get("result_sha256") != sha256(result_path):
        raise ValueError("Closed-loop verifier does not link to the result bytes.")

    entries = manifest["episodes"]
    checkpoint_ids = manifest["selected_scope"]["checkpoint_ids"]
    if len(entries) != 61 or len(checkpoint_ids) != 6:
        raise ValueError("Analysis requires the frozen six-checkpoint, 61-episode scope.")

    control_reports: dict[str, dict[str, Any]] = {}
    control_report_hashes: dict[str, str] = {}
    for entry in entries:
        episode_id = str(entry["episode_id"])
        path = (
            control_root
            / "runs"
            / "always_forward"
            / "episodes"
            / f"{entry['artifact_stem']}.json"
        )
        control_reports[episode_id] = load_json(path)
        control_report_hashes[episode_id] = sha256(path)
    recorded_control_hashes = result["control_comparison"]["always_forward"][
        "episode_report_sha256"
    ]
    if control_report_hashes != recorded_control_hashes:
        raise ValueError("Always-forward report hashes differ from the verified result links.")

    reports_by_checkpoint: dict[str, dict[str, dict[str, Any]]] = {}
    checkpoint_summaries: dict[str, Any] = {}
    report_hashes: dict[str, dict[str, str]] = {}
    for checkpoint_id in checkpoint_ids:
        reports: dict[str, dict[str, Any]] = {}
        hashes: dict[str, str] = {}
        for entry in entries:
            episode_id = str(entry["episode_id"])
            path = (
                result_root
                / "runs"
                / checkpoint_id
                / "episodes"
                / f"{entry['artifact_stem']}.json"
            )
            report = load_json(path)
            if report.get("status") != "completed":
                raise ValueError(f"Episode report is not completed: {path}")
            reports[episode_id] = report
            hashes[episode_id] = sha256(path)
        reports_by_checkpoint[checkpoint_id] = reports
        report_hashes[checkpoint_id] = hashes
        checkpoint_summaries[checkpoint_id] = summarize_checkpoint(
            checkpoint_id, list(reports.values()), control_reports
        )

    pairs: dict[str, Any] = {}
    pair_rows_by_seed: dict[str, list[dict[str, Any]]] = {}
    for checkpoint_id in checkpoint_ids:
        treatment, seed = _treatment_and_seed(checkpoint_id)
        pairs.setdefault(seed, {})[treatment] = reports_by_checkpoint[checkpoint_id]
    for seed, treatment_reports in sorted(pairs.items()):
        unweighted = treatment_reports["unweighted"]
        weighted = treatment_reports["inverse_frequency"]
        metric_summaries = {
            metric: paired_delta_summary(unweighted, weighted, metric)
            for metric in PAIR_METRICS
        }
        rows: list[dict[str, Any]] = []
        for entry in entries:
            episode_id = str(entry["episode_id"])
            rows.append(
                {
                    "episode_id": episode_id,
                    "scene_id": entry["scene_id"],
                    "deltas": {
                        metric: _report_value(weighted[episode_id], metric)
                        - _report_value(unweighted[episode_id], metric)
                        for metric in PAIR_METRICS
                    },
                }
            )
        pair_rows_by_seed[seed] = rows
        pairs[seed] = {"metrics": metric_summaries}

    treatment_mean_rows: list[dict[str, Any]] = []
    for entry in entries:
        episode_id = str(entry["episode_id"])
        deltas: dict[str, float] = {}
        for metric in PAIR_METRICS:
            unweighted_values = []
            weighted_values = []
            for checkpoint_id in checkpoint_ids:
                treatment, _ = _treatment_and_seed(checkpoint_id)
                value = _report_value(reports_by_checkpoint[checkpoint_id][episode_id], metric)
                if treatment == "unweighted":
                    unweighted_values.append(value)
                else:
                    weighted_values.append(value)
            deltas[metric] = mean(weighted_values) - mean(unweighted_values)
        treatment_mean_rows.append(
            {
                "episode_id": episode_id,
                "scene_id": entry["scene_id"],
                "deltas": deltas,
            }
        )

    stop_events = [
        {"checkpoint_id": checkpoint_id, **event}
        for checkpoint_id, summary in checkpoint_summaries.items()
        for event in summary["stop_events"]
    ]
    oracle_sets = {
        checkpoint_id: summary["oracle_success_episode_ids"]
        for checkpoint_id, summary in checkpoint_summaries.items()
    }
    unweighted_oracle = sorted(
        {
            episode_id
            for checkpoint_id, ids in oracle_sets.items()
            if _treatment_and_seed(checkpoint_id)[0] == "unweighted"
            for episode_id in ids
        }
    )
    weighted_oracle = sorted(
        {
            episode_id
            for checkpoint_id, ids in oracle_sets.items()
            if _treatment_and_seed(checkpoint_id)[0] == "inverse_frequency"
            for episode_id in ids
        }
    )
    goal_radius_audits = {
        checkpoint_id: summary["goal_radius_decision_audit"]
        for checkpoint_id, summary in checkpoint_summaries.items()
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "gate": GATE_ID,
        "source": {
            "result_root": str(result_root),
            "control_root": str(control_root),
            "manifest_sha256": sha256(manifest_path),
            "result_sha256": sha256(result_path),
            "verification_sha256": sha256(verification_path),
            "verification_status": verification["status"],
            "verification_error_count": verification["error_count"],
            "episode_report_sha256": report_hashes,
            "always_forward_control_report_sha256": control_report_hashes,
        },
        "scope": {
            "split": manifest["protocol"]["split"],
            "episode_count": len(entries),
            "scene_count": len({entry["scene_id"] for entry in entries}),
            "checkpoint_count": len(checkpoint_ids),
            "episode_run_count": len(entries) * len(checkpoint_ids),
            "history_mode": manifest["protocol"]["history_mode"],
            "chunk_length": manifest["protocol"]["chunk_length"],
            "step_cap": manifest["protocol"]["official_max_steps"],
            "success_radius_m": SUCCESS_RADIUS_METERS,
            "val_unseen_status": "sealed_not_used",
        },
        "checkpoint_behavior": checkpoint_summaries,
        "cross_seed_behavior_consistency": cross_seed_consistency(
            reports_by_checkpoint
        ),
        "same_seed_paired_differences": pairs,
        "treatment_mean_scene_cluster_bootstrap": scene_cluster_bootstrap(
            treatment_mean_rows
        ),
        "stop_event_audit": {
            "stop_event_count": len(stop_events),
            "outside_success_radius_count": sum(
                bool(event["outside_success_radius"]) for event in stop_events
            ),
            "immediate_step_1_count": sum(event["stop_step"] == 1 for event in stop_events),
            "events": stop_events,
        },
        "oracle_success_identity_audit": {
            "unweighted_episode_ids": unweighted_oracle,
            "inverse_frequency_episode_ids": weighted_oracle,
            "intersection": sorted(set(unweighted_oracle) & set(weighted_oracle)),
            "same_scalar_rate_masks_different_episodes": (
                bool(unweighted_oracle)
                and bool(weighted_oracle)
                and set(unweighted_oracle) != set(weighted_oracle)
            ),
        },
        "closed_loop_goal_radius_decision_audit": {
            "interpretation": (
                "repeated sequential decision opportunities, not independent samples"
            ),
            "checkpoint_episode_count": sum(
                int(audit["episode_count_with_inside_radius_decision"])
                for audit in goal_radius_audits.values()
            ),
            "unique_episode_ids": sorted(
                {
                    str(event["episode_id"])
                    for audit in goal_radius_audits.values()
                    for event in audit["episodes"]
                }
            ),
            "inside_radius_decision_count": sum(
                int(audit["inside_radius_decision_count"])
                for audit in goal_radius_audits.values()
            ),
            "inside_radius_stop_count": sum(
                int(audit["inside_radius_stop_count"])
                for audit in goal_radius_audits.values()
            ),
            "by_checkpoint": goal_radius_audits,
        },
        "matched_controls": _control_summary(result),
        "evidence_labels": {
            "established": [
                "All six learned checkpoints achieved 0/61 SR, 0 SPL, and 0 SDTW.",
                "Each learned checkpoint achieved 1/61 OSR on the frozen surface.",
                (
                    "Two unweighted seeds exactly reproduced always-forward on 61/61 "
                    "episodes; the third did so on 60/61."
                ),
                (
                    "Inverse-frequency weighting produced seven STOP episodes across 183 "
                    "runs; all seven STOPs occurred outside the 3 m success radius."
                ),
                (
                    "The unweighted and weighted treatments reached the success radius on "
                    "different single episodes despite identical aggregate OSR."
                ),
                (
                    "Most weighted episodes still used exactly one action type throughout: "
                    "47/61, 42/61, and 55/61 across the three seeds."
                ),
                (
                    "Across 1,539 sequential decisions made inside the 3 m success radius, "
                    "none selected STOP; these are repeated opportunities across only two "
                    "unique episode identities, not independent samples."
                ),
            ],
            "interpretation": [
                (
                    "The unweighted closed-loop behavior is an always-forward collapse on "
                    "this frozen protocol."
                ),
                (
                    "Inverse-frequency weighting increases action diversity but does not "
                    "recover goal-conditioned stopping or success."
                ),
                (
                    "Weighted diversity is primarily across episodes, not sustained within-"
                    "episode adaptation."
                ),
                (
                    "The closed-loop failure is broader than STOP alone because only one "
                    "episode per checkpoint ever enters the success radius."
                ),
                (
                    "Final navigation error can move without improving SR, SPL, SDTW, or "
                    "consistently improving nDTW."
                ),
            ],
            "not_established": [
                "val_unseen generalization",
                "a causal mechanism for collapse",
                "Qwen3-VL behavior outside this exact fine-tuning protocol",
                "C1-C4 geometry efficacy",
                "statistical population inference from the descriptive bootstrap",
            ],
        },
    }
