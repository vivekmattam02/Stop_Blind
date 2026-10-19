#!/usr/bin/env python3
"""Independently verify a completed Spark policy-ordering evidence directory.

This checker intentionally uses only the Python standard library.  It recomputes hashes,
trajectory metrics, aggregates, policy schedules, and pre-registered contrasts from the copied
per-episode artifacts instead of trusting ``suite.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUCCESS_DISTANCE_M = 3.0
MAX_STEPS = 500
ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}.")
        rows.append(payload)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _close(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _nested_close(left: Any, right: Any) -> bool:
    """Compare JSON-like structures with tight tolerance for cross-Python float sums."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_nested_close(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _nested_close(left_value, right_value) for left_value, right_value in zip(left, right)
        )
    return _close(left, right)


def _point_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 3 or len(right) != 3:
        raise ValueError("Every trajectory point must contain exactly three coordinates.")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _path_length(positions: Sequence[Sequence[float]]) -> float:
    return sum(_point_distance(a, b) for a, b in zip(positions[:-1], positions[1:]))


def _ndtw(path: Sequence[Sequence[float]], reference: Sequence[Sequence[float]]) -> float:
    deduplicated: list[Sequence[float]] = []
    for point in path:
        if not deduplicated or point != deduplicated[-1]:
            deduplicated.append(point)
    if not deduplicated or not reference:
        raise ValueError("nDTW paths must not be empty.")
    previous = [math.inf] * (len(reference) + 1)
    previous[0] = 0.0
    for point in deduplicated:
        current = [math.inf] * (len(reference) + 1)
        for index, reference_point in enumerate(reference, start=1):
            cost = _point_distance(point, reference_point)
            current[index] = cost + min(current[index - 1], previous[index], previous[index - 1])
        previous = current
    return math.exp(-previous[-1] / (len(reference) * SUCCESS_DISTANCE_M))


def _recompute_metrics(trace: Mapping[str, Any]) -> dict[str, Any]:
    actions = [int(action) for action in trace["actions"]]
    positions = trace["positions"]
    distances = [float(value) for value in trace["distances_to_goal"]]
    collisions = [bool(value) for value in trace["collisions"]]
    invalid_count = int(trace["invalid_action_count"])
    if len(positions) != len(actions) + 1:
        raise ValueError("Trace position count does not equal executed actions plus reset.")
    if len(distances) != len(positions):
        raise ValueError("Trace distance count does not equal position count.")
    if len(collisions) != len(actions):
        raise ValueError("Trace collision count does not equal executed action count.")
    if any(action not in ACTION_NAMES for action in actions):
        raise ValueError("Trace contains an action outside the locked R2R-CE vocabulary.")
    shortest = float(trace["shortest_path_length"])
    if not _close(shortest, distances[0]):
        raise ValueError("SPL denominator is not the simulator geodesic measured at reset.")
    stopped = bool(actions and actions[-1] == 0)
    navigation_error = distances[-1]
    success = float(stopped and navigation_error < SUCCESS_DISTANCE_M)
    oracle_success = float(min(distances) < SUCCESS_DISTANCE_M)
    path_length = _path_length(positions)
    spl = success * shortest / max(shortest, path_length, 1e-8)
    ndtw = _ndtw(positions, trace["reference_path"])
    return {
        "episode_id": str(trace["episode_id"]),
        "navigation_error": navigation_error,
        "oracle_success": oracle_success,
        "success": success,
        "path_length": path_length,
        "shortest_path_length": shortest,
        "spl": spl,
        "ndtw": ndtw,
        "sdtw": success * ndtw,
        "step_count": len(actions) + invalid_count,
        "collision_count": sum(collisions),
        "invalid_action_count": invalid_count,
        "stopped": stopped,
    }


def _metrics_match(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return set(observed) == set(expected) and all(
        _close(observed[key], expected[key]) for key in expected
    )


def _expected_policy_actions(spec: Mapping[str, Any], episode_id: str) -> list[int] | None:
    policy_run_id = str(spec["policy_run_id"])
    family = str(spec["family"])
    if policy_run_id == "immediate_stop":
        return [0]
    if policy_run_id == "always_forward":
        return [1] * MAX_STEPS
    if policy_run_id == "invalid_action":
        return []
    if family not in {"official_random", "official_handcrafted"}:
        return None
    seed = int(spec["seed"])
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    if family == "official_handcrafted":
        return [3] * rng.randrange(25) + [1] * 37 + [0]
    actions: list[int] = []
    for _ in range(MAX_STEPS):
        action = int(rng.choices([0, 1, 2, 3], weights=(0.02, 0.68, 0.15, 0.15), k=1)[0])
        actions.append(action)
        if action == 0:
            break
    return actions


def _verify_steps(
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    expected_actions: Sequence[int] | None,
) -> None:
    evaluation = report["evaluation"]
    trace = evaluation["trace"]
    actions = [int(action) for action in trace["actions"]]
    invalid_count = int(trace["invalid_action_count"])
    episode_id = str(report["episode_id"])
    if expected_actions is not None and actions != list(expected_actions):
        raise ValueError("Executed policy actions differ from the deterministic policy schedule.")
    if len(rows) != 1 + len(actions) + invalid_count:
        raise ValueError("JSONL row count does not match reset, actions, and invalid outputs.")
    if [int(row["step_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("JSONL step indices are not contiguous from zero.")
    if any(str(row["episode_id"]) != episode_id for row in rows):
        raise ValueError("JSONL episode identity differs from its report.")
    reset = rows[0]
    if reset["event"] != "reset" or reset["executed"] or reset["action_id"] is not None:
        raise ValueError("First JSONL row is not a non-executed reset event.")
    if not _close(reset["position"], trace["positions"][0]) or not _close(
        reset["distance_to_goal"], trace["distances_to_goal"][0]
    ):
        raise ValueError("Reset JSONL state differs from the evaluator trace.")
    for offset, action in enumerate(actions, start=1):
        row = rows[offset]
        if (
            row["event"] != "action"
            or not row["executed"]
            or int(row["action_id"]) != action
            or row["action_name"] != ACTION_NAMES[action]
        ):
            raise ValueError("Executed JSONL action differs from the evaluator trace.")
        if not _close(row["position"], trace["positions"][offset]):
            raise ValueError("Executed JSONL position differs from the evaluator trace.")
        if not _close(row["distance_to_goal"], trace["distances_to_goal"][offset]):
            raise ValueError("Executed JSONL geodesic differs from the evaluator trace.")
        if bool(row["collision"]) != bool(trace["collisions"][offset - 1]):
            raise ValueError("Executed JSONL collision differs from the evaluator trace.")
    invalid_rows = rows[1 + len(actions) :]
    if len(invalid_rows) != invalid_count or any(
        row["event"] != "invalid_action" or row["executed"] for row in invalid_rows
    ):
        raise ValueError("Invalid-output JSONL events differ from the evaluator trace.")


def _summarize(reports: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    episode_ids = {str(row["episode_id"]) for row in manifest["episodes"]}
    specs = {str(row["policy_run_id"]): row for row in manifest["policies"]}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for report in reports:
        identity = (str(report["policy_run_id"]), str(report["episode_id"]))
        if identity in seen or identity[0] not in specs or identity[1] not in episode_ids:
            raise ValueError(f"Duplicate or out-of-manifest report identity: {identity}.")
        seen.add(identity)
        grouped[identity[0]].append(report)
    metric_names = (
        "navigation_error",
        "oracle_success",
        "success",
        "spl",
        "ndtw",
        "sdtw",
        "path_length",
        "step_count",
    )
    run_summaries: dict[str, dict[str, Any]] = {}
    for policy_run_id, spec in specs.items():
        rows = grouped.get(policy_run_id, [])
        metrics = [row["evaluation"]["metrics"] for row in rows if row.get("status") == "passed"]
        summary: dict[str, Any] = {
            "family": spec["family"],
            "seed": spec["seed"],
            "completed_episode_count": len(rows),
            "passed_episode_count": len(metrics),
            "failed_episode_count": sum(row.get("status") != "passed" for row in rows),
        }
        if metrics:
            for name in metric_names:
                summary[name] = sum(float(row[name]) for row in metrics) / len(metrics)
            summary["stopped_rate"] = sum(bool(row["stopped"]) for row in metrics) / len(metrics)
            summary["collision_count"] = sum(int(row["collision_count"]) for row in metrics)
            summary["invalid_action_count"] = sum(
                int(row["invalid_action_count"]) for row in metrics
            )
            summary["minimum_ndtw"] = min(float(row["ndtw"]) for row in metrics)
        run_summaries[policy_run_id] = summary
    family_runs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for summary in run_summaries.values():
        if summary["passed_episode_count"]:
            family_runs[str(summary["family"])].append(summary)
    family_summaries: dict[str, dict[str, Any]] = {}
    for family, rows in sorted(family_runs.items()):
        family_summary: dict[str, Any] = {
            "run_count": len(rows),
            "seeds": [row["seed"] for row in rows if row["seed"] is not None],
        }
        for name in ("success", "spl", "ndtw", "sdtw", "navigation_error"):
            values = [float(row[name]) for row in rows]
            family_summary[name] = {
                "mean_across_seed_runs": sum(values) / len(values),
                "minimum_seed_run": min(values),
                "maximum_seed_run": max(values),
            }
        family_summaries[family] = family_summary
    target = len(episode_ids) * len(specs)
    return {
        "target_report_count": target,
        "completed_report_count": len(reports),
        "remaining_report_count": target - len(reports),
        "policy_runs": run_summaries,
        "families": family_summaries,
    }


def _pre_registered_analysis(
    reports: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for report in reports:
        grouped[str(report["policy_run_id"])].append(report)

    def metrics(policy_run_id: str) -> list[Mapping[str, Any]]:
        return [row["evaluation"]["metrics"] for row in grouped[policy_run_id]]

    gt = metrics("released_ground_truth_replay")
    immediate = metrics("immediate_stop")
    forward = metrics("always_forward")
    invalid_reports = grouped["invalid_action"]
    invalid = [row["evaluation"]["metrics"] for row in invalid_reports]
    checks: dict[str, bool] = {
        "released_replay_all_success": all(row["success"] == 1.0 for row in gt),
        "released_replay_all_stopped": all(row["stopped"] for row in gt),
        "released_replay_no_invalid": all(row["invalid_action_count"] == 0 for row in gt),
        "released_replay_minimum_ndtw": min(row["ndtw"] for row in gt) >= 0.9999,
        "immediate_stop_all_stopped": all(row["stopped"] for row in immediate),
        "immediate_stop_zero_motion": all(
            math.isclose(row["path_length"], 0.0, abs_tol=1e-12) for row in immediate
        ),
        "immediate_stop_zero_success": all(row["success"] == 0.0 for row in immediate),
        "always_forward_never_stopped": all(not row["stopped"] for row in forward),
        "always_forward_reaches_step_cap": all(row["step_count"] == MAX_STEPS for row in forward),
        "always_forward_zero_success": all(row["success"] == 0.0 for row in forward),
        "invalid_action_exactly_one_per_episode": all(
            row["invalid_action_count"] == 1 for row in invalid
        ),
        "invalid_action_zero_executed_actions": all(
            not report["evaluation"]["trace"]["actions"] for report in invalid_reports
        ),
        "invalid_action_zero_success": all(row["success"] == 0.0 for row in invalid),
    }
    replay = summary["policy_runs"]["released_ground_truth_replay"]
    for family in ("official_random", "official_handcrafted"):
        rows = [row for row in summary["policy_runs"].values() if row["family"] == family]
        for metric_name in ("success", "spl", "ndtw", "sdtw"):
            checks[f"{family}_every_seed_below_replay_{metric_name}"] = all(
                row[metric_name] < replay[metric_name] for row in rows
            )
        checks[f"{family}_no_invalid_actions"] = all(
            row["invalid_action_count"] == 0 for row in rows
        )
    return {
        "status": "passed" if all(checks.values()) else "failed_scientific_gate",
        "all_required_checks_pass": all(checks.values()),
        "checks": checks,
        "summary": dict(summary),
        "non_required_comparison": (
            "No random-versus-handcrafted ordering was tested or required."
        ),
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    root = args.artifact_root.resolve()
    parent_root = args.parent_root.resolve()
    manifest_path = root / "suite_manifest.json"
    suite_path = root / "suite.json"
    progress_path = root / "progress.json"
    manifest = _load_json(manifest_path)
    suite = _load_json(suite_path)
    progress = _load_json(progress_path)
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    unhashed = dict(manifest)
    declared_manifest_hash = str(unhashed.pop("manifest_sha256", ""))
    canonical_manifest_hash = _canonical_sha256(unhashed)
    require(declared_manifest_hash == canonical_manifest_hash, "canonical manifest hash mismatch")
    require(suite.get("status") == "passed", "suite status is not passed")
    require(progress.get("status") == "passed", "progress status is not passed")
    require(suite.get("manifest_sha256") == declared_manifest_hash, "suite manifest link mismatch")
    require(
        progress.get("manifest_sha256") == declared_manifest_hash,
        "progress manifest link mismatch",
    )
    require(suite.get("summary") == progress.get("summary"), "suite/progress summary mismatch")

    parent_hashes: dict[str, str] = {}
    for name in ("suite_manifest.json", "suite.json"):
        path = parent_root / name
        if path.is_file():
            parent_hashes[name] = _sha256(path)
            require(
                parent_hashes[name] == suite.get("parent_artifact_sha256", {}).get(name),
                f"parent {name} host hash mismatch",
            )
        else:
            errors.append(f"parent artifact missing: {name}")
    parent_manifest: dict[str, Any] | None = None
    if (parent_root / "suite_manifest.json").is_file():
        parent_manifest = _load_json(parent_root / "suite_manifest.json")
        require(
            parent_manifest.get("manifest_sha256")
            == manifest.get("parent_evidence", {}).get("manifest_sha256"),
            "parent canonical manifest link mismatch",
        )

    parent_replay_actions: dict[str, list[int]] = {}
    if parent_manifest is not None:
        selected_episode_ids = {str(entry["episode_id"]) for entry in manifest["episodes"]}
        for entry in parent_manifest["episodes"]:
            episode_id = str(entry["episode_id"])
            if episode_id not in selected_episode_ids:
                continue
            steps_path = parent_root / "steps" / f"{entry['artifact_stem']}.jsonl"
            if not steps_path.is_file():
                errors.append(f"selected parent replay trace missing: {episode_id}")
                continue
            parent_replay_actions[episode_id] = [
                int(row["action_id"])
                for row in _load_jsonl(steps_path)
                if row["event"] == "action" and row["executed"]
            ]

    software_verified = 0
    for relative, expected_hash in manifest.get("software_sha256", {}).items():
        path = repo / relative
        if path.is_file() and _sha256(path) == expected_hash:
            software_verified += 1
        else:
            errors.append(f"locked software hash mismatch: {relative}")

    expected_reports: set[str] = set()
    expected_steps: set[str] = set()
    for entry in manifest["episodes"]:
        for spec in manifest["policies"]:
            prefix = f"runs/{spec['policy_run_id']}"
            expected_reports.add(f"{prefix}/episodes/{entry['artifact_stem']}.json")
            expected_steps.add(f"{prefix}/steps/{entry['artifact_stem']}.jsonl")
    actual_reports = {
        path.relative_to(root).as_posix() for path in (root / "runs").glob("*/episodes/*.json")
    }
    actual_steps = {
        path.relative_to(root).as_posix() for path in (root / "runs").glob("*/steps/*.jsonl")
    }
    require(actual_reports == expected_reports, "report filename set differs from manifest")
    require(actual_steps == expected_steps, "step filename set differs from manifest")

    reports: list[dict[str, Any]] = []
    verified_trace_hashes = 0
    recomputed_metric_bundles = 0
    verified_policy_schedules = 0
    for entry in manifest["episodes"]:
        for spec in manifest["policies"]:
            prefix = Path("runs") / str(spec["policy_run_id"])
            report_relative = prefix / "episodes" / f"{entry['artifact_stem']}.json"
            steps_relative = prefix / "steps" / f"{entry['artifact_stem']}.jsonl"
            report_path = root / report_relative
            steps_path = root / steps_relative
            if not report_path.is_file() or not steps_path.is_file():
                continue
            report = _load_json(report_path)
            reports.append(report)
            identity_checks = {
                "schema": report.get("schema_version") == 1,
                "status": report.get("status") == "passed",
                "manifest": report.get("manifest_sha256") == declared_manifest_hash,
                "index": report.get("index") == entry["index"],
                "episode": str(report.get("episode_id")) == str(entry["episode_id"]),
                "trajectory": str(report.get("trajectory_id")) == str(entry["trajectory_id"]),
                "scene": report.get("scene_id") == entry["scene_id"],
                "policy": report.get("policy_run_id") == spec["policy_run_id"],
                "family": report.get("family") == spec["family"],
                "seed": report.get("seed") == spec["seed"],
                "role": report.get("role") == spec["role"],
                "runtime": bool(report.get("runtime_checks"))
                and all(report["runtime_checks"].values()),
                "steps_path": str(report.get("steps_path", "")).endswith(steps_relative.as_posix()),
            }
            if not all(identity_checks.values()):
                errors.append(
                    f"report identity/runtime mismatch {report_relative}: {identity_checks}"
                )
                continue
            observed_steps_hash = _sha256(steps_path)
            if observed_steps_hash != report.get("steps_sha256"):
                errors.append(f"step trace hash mismatch: {steps_relative}")
                continue
            verified_trace_hashes += 1
            try:
                trace = report["evaluation"]["trace"]
                if str(report["evaluation"]["episode_id"]) != str(entry["episode_id"]):
                    raise ValueError("Evaluation episode identity differs from the manifest.")
                recomputed = _recompute_metrics(trace)
                if not _metrics_match(report["evaluation"]["metrics"], recomputed):
                    raise ValueError("Reported metrics differ from recomputed trajectory metrics.")
                recomputed_metric_bundles += 1
                expected_actions = _expected_policy_actions(spec, str(entry["episode_id"]))
                if spec["policy_run_id"] == "released_ground_truth_replay":
                    expected_actions = parent_replay_actions.get(str(entry["episode_id"]))
                _verify_steps(_load_jsonl(steps_path), report, expected_actions)
                if expected_actions is not None:
                    verified_policy_schedules += 1
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"trace/metric verification failed {report_relative}: {error}")

    try:
        recomputed_summary = _summarize(reports, manifest)
        recomputed_analysis = _pre_registered_analysis(reports, recomputed_summary)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"aggregate recomputation failed: {error}")
        recomputed_summary = {}
        recomputed_analysis = {}
    summary_matches = _nested_close(recomputed_summary, suite.get("summary"))
    analysis_matches = _nested_close(recomputed_analysis, suite.get("pre_registered_analysis"))
    require(summary_matches, "recomputed suite summary mismatch")
    require(
        analysis_matches,
        "recomputed pre-registered analysis mismatch",
    )
    require(
        bool(recomputed_analysis.get("all_required_checks_pass")),
        "a recomputed pre-registered check did not pass",
    )

    top_hashes = {
        name: _sha256(root / name)
        for name in ("suite_manifest.json", "suite.json", "progress.json")
    }
    result = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact_root": str(args.artifact_root),
        "checks": {
            "canonical_manifest_hash_matches": declared_manifest_hash == canonical_manifest_hash,
            "suite_and_progress_links_match": (
                suite.get("manifest_sha256")
                == progress.get("manifest_sha256")
                == declared_manifest_hash
            ),
            "report_filename_set_matches_manifest": actual_reports == expected_reports,
            "step_filename_set_matches_manifest": actual_steps == expected_steps,
            "episode_reports_verified": len(reports),
            "step_trace_hashes_verified": verified_trace_hashes,
            "trajectory_metric_bundles_recomputed": recomputed_metric_bundles,
            "deterministic_policy_schedules_verified": verified_policy_schedules,
            "locked_software_hashes_verified": software_verified,
            "parent_artifact_hashes_verified": len(parent_hashes),
            "aggregate_summary_recomputed": summary_matches,
            "pre_registered_analysis_recomputed": analysis_matches,
            "errors": errors,
        },
        "source_sha256": {
            **top_hashes,
            "canonical_manifest_payload": canonical_manifest_hash,
        },
        "recomputed_summary": recomputed_summary,
        "recomputed_pre_registered_analysis": recomputed_analysis,
        "evidence_boundary": suite.get("evidence_boundary"),
    }
    return result


def main() -> int:
    args = _arguments()
    result = verify(args)
    _write_json(args.output, result)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"Policy-ordering artifact verification: {result['status']}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
