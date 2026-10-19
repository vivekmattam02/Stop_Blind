#!/usr/bin/env python3
"""Independently reconstruct a completed latency-selected native-C0 closed-loop suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.metrics import (
    R2RCEEpisodeMetrics,
    R2RCEEpisodeTrace,
    aggregate_r2r_ce_metrics,
    evaluate_r2r_ce_trace,
)
from spatialvlm.eval.native_closed_loop import (
    FROZEN_CHECKPOINTS,
    ClosedLoopDecisionTiming,
    closed_loop_protocol_record,
    summarize_closed_loop_timings,
)

SCHEMA_VERSION = 1
GATE_ID = "independent_native_qwen_c0_latency_selected_closed_loop_verification_v1"
RUNNER_GATE_ID = "native_qwen_c0_latency_selected_closed_loop_evaluation_v1"
CONTROL_MANIFEST_FILE_SHA256 = (
    "cb75d6259e42decccc161d069df2e04fdadbb7b4f20c6c480bcbca575ec44af4"
)
CONTROL_SUITE_FILE_SHA256 = (
    "2fab8d96958f4abc9377a204f2e43bc1e49d58fc1c7422f60315cf35479557b9"
)
CONTROL_VERIFICATION_FILE_SHA256 = (
    "9d07c3a20a335910d91b338d5e22eea715ae004aa2e998d8ad968967ffa889e5"
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must all be objects: {path}")
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--control-verification", type=Path, required=True)
    parser.add_argument("--stress-root", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--source-artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _trace(payload: Mapping[str, Any]) -> R2RCEEpisodeTrace:
    return R2RCEEpisodeTrace(
        episode_id=str(payload["episode_id"]),
        positions=tuple(tuple(float(value) for value in point) for point in payload["positions"]),
        distances_to_goal=tuple(float(value) for value in payload["distances_to_goal"]),
        actions=tuple(R2RCEAction(int(value)) for value in payload["actions"]),
        shortest_path_length=float(payload["shortest_path_length"]),
        reference_path=tuple(
            tuple(float(value) for value in point) for point in payload["reference_path"]
        ),
        collisions=tuple(bool(value) for value in payload["collisions"]),
        invalid_action_count=int(payload["invalid_action_count"]),
    )


def _checkpoint_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [R2RCEEpisodeMetrics(**report["evaluation"]["metrics"]) for report in reports]
    histogram: Counter[str] = Counter()
    timing_values: list[float] = []
    for report in reports:
        histogram.update(report["action_histogram"])
        timing_values.extend(float(value) for value in report["closed_loop_seconds"])
    aggregate = aggregate_r2r_ce_metrics(metrics).to_dict()
    aggregate.update(
        {
            "stop_rate": sum(metric.stopped for metric in metrics) / len(metrics),
            "step_cap_rate": sum(metric.step_count == 500 for metric in metrics) / len(metrics),
            "action_histogram": dict(sorted(histogram.items())),
            "mean_closed_loop_seconds_per_step": (
                sum(timing_values) / len(timing_values) if timing_values else 0.0
            ),
        }
    )
    return aggregate


def _control_summary(
    control_root: Path, manifest: Mapping[str, Any], episode_count: int
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for spec in manifest["policies"]:
        run_id = str(spec["policy_run_id"])
        metrics: list[R2RCEEpisodeMetrics] = []
        hashes: dict[str, str] = {}
        for entry in manifest["episodes"][:episode_count]:
            path = (
                control_root
                / "runs"
                / run_id
                / "episodes"
                / f"{entry['artifact_stem']}.json"
            )
            report = _load_json(path)
            if report.get("status") != "passed":
                raise ValueError(f"control report is not passed: {path}")
            metrics.append(R2RCEEpisodeMetrics(**report["evaluation"]["metrics"]))
            hashes[str(entry["episode_id"])] = _sha256(path)
        summaries[run_id] = {
            "family": spec["family"],
            "seed": spec["seed"],
            "aggregate": aggregate_r2r_ce_metrics(metrics).to_dict(),
            "episode_report_sha256": hashes,
        }
    return summaries


def verify(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    root = args.result_root
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    for path in (manifest_path, result_path):
        if not path.is_file():
            errors.append(f"required main artifact is missing: {path}")
    if errors:
        return _payload(args, errors)

    manifest = _load_json(manifest_path)
    result = _load_json(result_path)
    control_manifest_path = args.control_root / "suite_manifest.json"
    control_suite_path = args.control_root / "suite.json"
    stress_result_path = args.stress_root / "result.json"
    stress_timings_path = args.stress_root / "timings.jsonl"
    stress_verification_path = args.stress_root / "verification.json"
    for path in (
        control_manifest_path,
        control_suite_path,
        args.control_verification,
        stress_result_path,
        stress_timings_path,
        stress_verification_path,
    ):
        if not path.is_file():
            errors.append(f"required upstream artifact is missing: {path}")
    if errors:
        return _payload(args, errors)

    control_manifest = _load_json(control_manifest_path)
    control_suite = _load_json(control_suite_path)
    control_verification = _load_json(args.control_verification)
    stress_result = _load_json(stress_result_path)
    stress_verification = _load_json(stress_verification_path)
    stress_timings = [ClosedLoopDecisionTiming(**row) for row in _load_jsonl(stress_timings_path)]
    reconstructed_scope = summarize_closed_loop_timings(stress_timings)["scope"]
    if _sha256(control_manifest_path) != CONTROL_MANIFEST_FILE_SHA256:
        errors.append("control manifest differs from the frozen 61-trajectory input")
    if _sha256(control_suite_path) != CONTROL_SUITE_FILE_SHA256:
        errors.append("control suite bytes differ from the independently verified input")
    if _sha256(args.control_verification) != CONTROL_VERIFICATION_FILE_SHA256:
        errors.append("independent control verification bytes differ from the frozen input")
    if control_suite.get("status") != "passed":
        errors.append("control suite is not passed")
    control_checks = control_verification.get("checks", {})
    if control_verification.get("status") != "passed" or control_checks.get("errors") != []:
        errors.append("independent control verification is not passed with zero errors")
    if control_checks.get("episode_reports_verified") != 854:
        errors.append("independent control verification does not cover 854 episode reports")
    if control_checks.get("step_trace_hashes_verified") != 854:
        errors.append("independent control verification does not cover 854 step traces")
    source_hashes = control_verification.get("source_sha256", {})
    if source_hashes.get("suite.json") != CONTROL_SUITE_FILE_SHA256:
        errors.append("control verification does not link the frozen suite bytes")
    if source_hashes.get("suite_manifest.json") != CONTROL_MANIFEST_FILE_SHA256:
        errors.append("control verification does not link the frozen manifest bytes")
    if stress_result.get("status") != "passed" or stress_verification.get("status") != "passed":
        errors.append("stress result and verification are not both passed")
    if stress_verification.get("error_count") != 0:
        errors.append("stress verification contains errors")
    if stress_result.get("selected_scope") != reconstructed_scope:
        errors.append("stress scope does not reconstruct from the 32 timing records")

    if manifest.get("gate") != RUNNER_GATE_ID or manifest.get("status") != "frozen":
        errors.append("main manifest is not the frozen runner gate")
    if manifest.get("protocol") != closed_loop_protocol_record():
        errors.append("main protocol differs from the frozen contract")
    if manifest.get("selected_scope") != reconstructed_scope:
        errors.append("main scope differs from the preregistered latency branch")
    expected_control_locks = {
        "manifest_sha256": CONTROL_MANIFEST_FILE_SHA256,
        "suite_sha256": CONTROL_SUITE_FILE_SHA256,
        "verification_sha256": CONTROL_VERIFICATION_FILE_SHA256,
    }
    if manifest.get("control_locks") != expected_control_locks:
        errors.append("main manifest does not bind the independently verified control suite")
    if manifest.get("outcome_used_for_scope_selection") is not False:
        errors.append("main manifest does not deny outcome-based scope selection")
    episode_count = int(reconstructed_scope["episode_count"])
    if manifest.get("episodes") != control_manifest["episodes"][:episode_count]:
        errors.append("main episodes are not the exact frozen control-manifest prefix")
    selected_ids = list(reconstructed_scope["checkpoint_ids"])
    if [row["checkpoint_id"] for row in manifest.get("selected_checkpoints", [])] != selected_ids:
        errors.append("main checkpoint order differs from the latency-selected scope")

    for relative, expected_hash in manifest.get("software_sha256", {}).items():
        path = args.repo / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            errors.append(f"executed source lock mismatch: {relative}")

    checkpoint_by_id = {item.checkpoint_id: item for item in FROZEN_CHECKPOINTS}
    reports_by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for checkpoint_id in selected_ids:
        checkpoint = checkpoint_by_id[checkpoint_id]
        checkpoint_path = (
            args.checkpoint_archive
            / checkpoint.source_run_id
            / checkpoint.checkpoint_relative_path
        )
        source_result_path = args.source_artifact_root / checkpoint.source_run_id / "result.json"
        source_verification_path = (
            args.source_artifact_root / checkpoint.source_run_id / "verification.json"
        )
        if not all(
            path.is_file()
            for path in (checkpoint_path, source_result_path, source_verification_path)
        ):
            errors.append(f"checkpoint provenance inputs are missing: {checkpoint_id}")
            continue
        if _sha256(checkpoint_path) != checkpoint.checkpoint_sha256:
            errors.append(f"checkpoint bytes differ: {checkpoint_id}")
        source_result = _load_json(source_result_path)
        source_verification = _load_json(source_verification_path)
        if source_verification.get("status") != "passed":
            errors.append(f"checkpoint source verification not passed: {checkpoint_id}")
        if source_verification.get("result_sha256") != _sha256(source_result_path):
            errors.append(f"checkpoint source linkage mismatch: {checkpoint_id}")
        if source_result.get("selection", {}).get("checkpoint", {}).get(
            "sha256"
        ) != checkpoint.checkpoint_sha256:
            errors.append(f"checkpoint selection record differs: {checkpoint_id}")

        checkpoint_reports: list[dict[str, Any]] = []
        for entry in manifest.get("episodes", []):
            stem = str(entry["artifact_stem"])
            report_path = root / "runs" / checkpoint_id / "episodes" / f"{stem}.json"
            steps_path = root / "runs" / checkpoint_id / "steps" / f"{stem}.jsonl"
            timings_path = root / "runs" / checkpoint_id / "timings" / f"{stem}.jsonl"
            if not all(path.is_file() for path in (report_path, steps_path, timings_path)):
                errors.append(f"episode artifact missing: {checkpoint_id}/{stem}")
                continue
            report = _load_json(report_path)
            steps = _load_jsonl(steps_path)
            timings = _load_jsonl(timings_path)
            if report.get("status") != "completed":
                errors.append(f"episode report not completed: {checkpoint_id}/{stem}")
            if report.get("manifest_sha256") != _sha256(manifest_path):
                errors.append(f"episode manifest linkage mismatch: {checkpoint_id}/{stem}")
            if report.get("steps_sha256") != _sha256(steps_path):
                errors.append(f"episode steps linkage mismatch: {checkpoint_id}/{stem}")
            if report.get("timings_sha256") != _sha256(timings_path):
                errors.append(f"episode timings linkage mismatch: {checkpoint_id}/{stem}")
            trace = _trace(report["evaluation"]["trace"])
            reconstructed_metrics = evaluate_r2r_ce_trace(trace).to_dict()
            if report["evaluation"]["metrics"] != reconstructed_metrics:
                errors.append(f"episode metrics reconstruction mismatch: {checkpoint_id}/{stem}")
            action_steps = [row for row in steps if row.get("event") == "action"]
            if [row.get("action_id") for row in action_steps] != [
                int(action) for action in trace.actions
            ]:
                errors.append(f"episode action/step mismatch: {checkpoint_id}/{stem}")
            if len(timings) != len(trace.actions):
                errors.append(f"episode action/timing count mismatch: {checkpoint_id}/{stem}")
            for timing in timings:
                try:
                    ClosedLoopDecisionTiming(**timing)
                except (TypeError, ValueError) as error:
                    errors.append(f"invalid timing {checkpoint_id}/{stem}: {error}")
            decisions = report.get("policy_telemetry", {}).get("decisions", [])
            if len(decisions) != len(trace.actions):
                errors.append(f"episode action/telemetry count mismatch: {checkpoint_id}/{stem}")
            else:
                for index, (decision, action) in enumerate(zip(decisions, trace.actions)):
                    if decision.get("chunk_actions") != [action.name]:
                        errors.append(
                            f"telemetry action mismatch: {checkpoint_id}/{stem}/{index}"
                        )
                    diagnostic = decision.get("diagnostics", {})
                    if diagnostic.get("checkpoint_sha256") != checkpoint.checkpoint_sha256:
                        errors.append(
                            f"telemetry checkpoint mismatch: {checkpoint_id}/{stem}/{index}"
                        )
                    if diagnostic.get("chunk_length") != 1:
                        errors.append(
                            f"telemetry chunk mismatch: {checkpoint_id}/{stem}/{index}"
                        )
            observed_histogram = Counter(action.name for action in trace.actions)
            if report.get("action_histogram") != dict(sorted(observed_histogram.items())):
                errors.append(f"episode action histogram mismatch: {checkpoint_id}/{stem}")
            closed = report.get("closed_loop_seconds", [])
            if len(closed) != len(timings) or any(
                not math.isclose(float(value), float(timing["closed_loop_seconds"]), abs_tol=0.0)
                for value, timing in zip(closed, timings)
            ):
                errors.append(f"episode compact timing mismatch: {checkpoint_id}/{stem}")
            checkpoint_reports.append(report)
        reports_by_checkpoint[checkpoint_id] = checkpoint_reports

    if result.get("status") != "completed" or result.get("gate") != RUNNER_GATE_ID:
        errors.append("main result is not completed")
    if result.get("manifest_sha256") != _sha256(manifest_path):
        errors.append("main result/manifest linkage mismatch")
    if result.get("selected_scope") != reconstructed_scope:
        errors.append("main result scope differs from stress-selected scope")
    target = len(selected_ids) * episode_count
    if result.get("completed_episode_runs") != target:
        errors.append("main result completed count differs from frozen target")
    for checkpoint_id, reports in reports_by_checkpoint.items():
        if len(reports) == episode_count:
            reconstructed = _checkpoint_summary(reports)
            if result.get("checkpoint_summaries", {}).get(checkpoint_id) != reconstructed:
                errors.append(f"checkpoint aggregate reconstruction mismatch: {checkpoint_id}")
    controls = _control_summary(args.control_root, control_manifest, episode_count)
    if result.get("control_comparison") != controls:
        errors.append("control comparison does not reconstruct from matched episode reports")
    if result.get("headline_contrast") != "oracle_success_rate versus success_rate":
        errors.append("headline OSR-versus-SR contrast is missing")
    runtime_config = result.get("runtime", {}).get("config", {})
    if runtime_config.get("turn_angle_degrees") != 15:
        errors.append("main Habitat turn angle is not the official 15 degrees")
    if runtime_config.get("forward_step_m") != 0.25:
        errors.append("main Habitat forward step is not the official 0.25 m")
    if result.get("runtime", {}).get("habitat_sim_build_commit_observed") is None:
        errors.append("main Habitat build identity is missing")

    return _payload(args, errors)


def _payload(args: argparse.Namespace, errors: list[str]) -> dict[str, Any]:
    manifest_path = args.result_root / "manifest.json"
    result_path = args.result_root / "result.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "gate": GATE_ID,
        "recorded_at_utc": _timestamp(),
        "error_count": len(errors),
        "errors": errors,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path) if result_path.is_file() else None,
        "metric_reconstruction_performed": result_path.is_file(),
        "control_reconstruction_performed": result_path.is_file(),
        "all_checkpoint_files_fully_hashed": result_path.is_file(),
        "scope_reconstruction_performed": result_path.is_file(),
    }


def main() -> int:
    args = _arguments()
    payload = verify(args)
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
