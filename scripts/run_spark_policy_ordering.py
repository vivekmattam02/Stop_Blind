#!/usr/bin/env python3
"""Run the resumable, trajectory-level R2R-CE control-policy gate on DGX Spark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatialvlm.data.datasets import R2RCEDataset
from spatialvlm.data.habitat_sim_r2r import (
    HABITAT_SIM_COMMIT,
    HABITAT_SIM_VERSION,
    HabitatSimR2RConfig,
    HabitatSimR2REnvironment,
)
from spatialvlm.eval.policy_ordering import (
    POLICY_ORDERING_SCHEMA_VERSION,
    PolicyRunSpec,
    build_policy_ordering_manifest,
    evaluate_pre_registered_checks,
    make_policy,
    policy_run_specs,
    summarize_policy_reports,
)
from spatialvlm.eval.r2r_ce import run_r2r_ce_episode

RUNNER_SCHEMA_VERSION = 1
GATE_ID = "spark_r2r_ce_trajectory_level_policy_controls_v1"
_SOFTWARE_PATHS = (
    "scripts/run_spark_policy_ordering.py",
    "scripts/start_spark_policy_ordering.sh",
    "src/spatialvlm/data/datasets.py",
    "src/spatialvlm/data/habitat_sim_r2r.py",
    "src/spatialvlm/eval/habitat_suite.py",
    "src/spatialvlm/eval/metrics.py",
    "src/spatialvlm/eval/policy_ordering.py",
    "src/spatialvlm/eval/r2r_ce.py",
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_steps(path: Path, steps: Sequence[Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    lines = [json.dumps(step.to_dict(), sort_keys=True, allow_nan=False) for step in steps]
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(path)
    return _sha256(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--parent-suite-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--require-architecture", default="aarch64")
    parser.add_argument("--expected-parent-episodes", type=int, default=778)
    parser.add_argument("--expected-parent-scenes", type=int, default=53)
    parser.add_argument(
        "--habitat-build-lock",
        type=Path,
        default=Path("/opt/gridnav/habitat-sim-commit"),
    )
    return parser.parse_args()


def _software_hashes(repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _SOFTWARE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Policy-ordering software input is missing: {path}")
        hashes[relative] = _sha256(path)
    return hashes


def _verify_parent_suite(
    parent_root: Path,
    *,
    expected_episode_count: int,
    expected_scene_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Re-verify the full parent suite before using it to select any policy episode."""
    parent_manifest_path = parent_root / "suite_manifest.json"
    parent_suite_path = parent_root / "suite.json"
    parent_manifest = _load_json(parent_manifest_path)
    parent_suite = _load_json(parent_suite_path)
    if parent_suite.get("status") != "passed":
        raise RuntimeError("Parent Habitat conformance suite has not passed.")
    if parent_suite.get("manifest_sha256") != parent_manifest.get("manifest_sha256"):
        raise RuntimeError("Parent suite and manifest hashes disagree.")
    summary = parent_suite.get("summary", {})
    if summary.get("passed_episode_count") != expected_episode_count:
        raise RuntimeError(
            "Parent conformance episode count mismatch: "
            f"expected {expected_episode_count}, found {summary.get('passed_episode_count')}."
        )
    if summary.get("target_scene_count") != expected_scene_count:
        raise RuntimeError(
            "Parent conformance scene count mismatch: "
            f"expected {expected_scene_count}, found {summary.get('target_scene_count')}."
        )

    reports: list[dict[str, Any]] = []
    for entry in parent_manifest["episodes"]:
        report_path = parent_root / "episodes" / f"{entry['artifact_stem']}.json"
        steps_path = parent_root / "steps" / f"{entry['artifact_stem']}.jsonl"
        report = _load_json(report_path)
        checks = {
            "schema_version": report.get("schema_version") == 2,
            "status": report.get("status") == "passed",
            "manifest": report.get("manifest_sha256") == parent_manifest["manifest_sha256"],
            "index": report.get("index") == entry["index"],
            "episode_id": str(report.get("episode_id")) == str(entry["episode_id"]),
            "steps_present": steps_path.is_file(),
            "steps_hash": steps_path.is_file()
            and _sha256(steps_path) == report.get("steps_sha256"),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Parent conformance checkpoint failed verification: {checks}")
        reports.append(report)
    locks = {
        "suite_manifest.json": _sha256(parent_manifest_path),
        "suite.json": _sha256(parent_suite_path),
    }
    return parent_manifest, reports, locks


def _artifact_paths(
    output_root: Path,
    spec: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> tuple[Path, Path]:
    run_root = output_root / "runs" / str(spec["policy_run_id"])
    stem = str(entry["artifact_stem"])
    return run_root / "episodes" / f"{stem}.json", run_root / "steps" / f"{stem}.jsonl"


def _validate_existing_report(
    report_path: Path,
    steps_path: Path,
    spec: Mapping[str, Any],
    entry: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    report = _load_json(report_path)
    checks = {
        "schema_version": report.get("schema_version") == RUNNER_SCHEMA_VERSION,
        "manifest": report.get("manifest_sha256") == manifest_sha256,
        "policy_run_id": report.get("policy_run_id") == spec["policy_run_id"],
        "family": report.get("family") == spec["family"],
        "seed": report.get("seed") == spec["seed"],
        "index": report.get("index") == entry["index"],
        "episode_id": str(report.get("episode_id")) == str(entry["episode_id"]),
        "status": report.get("status") in {"passed", "failed"},
    }
    if report.get("status") == "passed":
        checks["steps_present"] = steps_path.is_file()
        checks["steps_hash"] = steps_path.is_file() and _sha256(steps_path) == report.get(
            "steps_sha256"
        )
    if not all(checks.values()):
        raise RuntimeError(f"Existing policy checkpoint is inconsistent: {checks}")
    return report


def _completed_reports(
    output_root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for entry in manifest["episodes"]:
        for spec in manifest["policies"]:
            report_path, steps_path = _artifact_paths(output_root, spec, entry)
            if report_path.is_file():
                reports.append(
                    _validate_existing_report(
                        report_path,
                        steps_path,
                        spec,
                        entry,
                        str(manifest["manifest_sha256"]),
                    )
                )
    return reports


def _suite_payload(
    *,
    status: str,
    manifest: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    parent_artifact_sha256: Mapping[str, str],
    first_started_at_utc: str,
    invocation_started_at_utc: str,
    max_hours: float,
    runtime: Mapping[str, Any] | None,
    reason: str | None,
    gate_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": status,
        "gate": GATE_ID,
        "recorded_at_utc": _timestamp(),
        "first_started_at_utc": first_started_at_utc,
        "invocation_started_at_utc": invocation_started_at_utc,
        "max_hours_this_invocation": max_hours,
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_artifact_sha256": dict(parent_artifact_sha256),
        "reason": reason,
        "summary": summarize_policy_reports(reports, manifest),
        "pre_registered_analysis": dict(gate_analysis) if gate_analysis is not None else None,
        "runtime": dict(runtime) if runtime is not None else None,
        "evidence_boundary": {
            "established_if_passed": (
                "The locked modern Spark evaluator distinguishes released replay, STOP, "
                "missing-STOP, invalid-action, and recognized non-learning controls on 61 "
                "unique val_seen trajectories spanning all 53 scenes and all observed geodesic-"
                "drift trajectories. Stochastic results are descriptive across fixed seeds."
            ),
            "not_established": (
                "Learned-policy quality, legacy Habitat 0.1.7 numerical reproduction, CMA or "
                "Qwen baseline performance, val_unseen generalization, training correctness, "
                "or any paper superiority claim."
            ),
        },
    }


def _write_state(
    output_root: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = _suite_payload(**kwargs)
    _write_json(output_root / "suite.json", payload)
    _write_json(
        output_root / "progress.json",
        {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "status": payload["status"],
            "recorded_at_utc": payload["recorded_at_utc"],
            "manifest_sha256": payload["manifest_sha256"],
            "summary": payload["summary"],
            "reason": payload["reason"],
        },
    )
    return payload


def _spec_objects(manifest: Mapping[str, Any]) -> tuple[PolicyRunSpec, ...]:
    current = policy_run_specs()
    if [spec.to_dict() for spec in current] != manifest["policies"]:
        raise RuntimeError("Current policy specifications differ from the frozen manifest.")
    return current


def run(args: argparse.Namespace) -> int:
    if not math.isfinite(args.max_hours) or args.max_hours <= 0.0:
        raise ValueError("Policy-ordering max-hours must be finite and positive.")
    if args.split != "val_seen":
        raise ValueError("The v1 control manifest is locked to val_seen development evidence.")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    invocation_started_at_utc = _timestamp()
    invocation_started = time.monotonic()

    parent_manifest, parent_reports, parent_locks = _verify_parent_suite(
        args.parent_suite_root.resolve(),
        expected_episode_count=args.expected_parent_episodes,
        expected_scene_count=args.expected_parent_scenes,
    )
    manifest = build_policy_ordering_manifest(
        parent_manifest,
        parent_reports,
        software_sha256=_software_hashes(args.repo_root.resolve()),
    )
    if manifest["schema_version"] != POLICY_ORDERING_SCHEMA_VERSION:
        raise RuntimeError("Policy-ordering manifest schema mismatch.")
    manifest_path = output_root / "suite_manifest.json"
    if manifest_path.exists():
        if _load_json(manifest_path) != manifest:
            raise RuntimeError(
                "Existing policy manifest differs from the current evidence/software lock. "
                "Preserve the old artifact directory and use a new versioned output root."
            )
    else:
        _write_json(manifest_path, manifest)

    dataset = R2RCEDataset.from_release(
        args.episodes,
        args.ground_truth,
        split=args.split,
        limit=None,
    )
    if dataset.release_manifest is None:
        raise RuntimeError("Policy ordering requires release-locked R2R-CE data.")
    observed_release = dataset.release_manifest.to_dict()
    for key in (
        "release_id",
        "split",
        "episodes_sha256",
        "source_episode_count",
        "ground_truth_sha256",
        "ground_truth_count",
    ):
        if observed_release[key] != manifest["release"][key]:
            raise RuntimeError(f"R2R-CE release lock mismatch for {key}.")

    suite_path = output_root / "suite.json"
    previous_suite = _load_json(suite_path) if suite_path.exists() else None
    first_started_at_utc = (
        str(previous_suite["first_started_at_utc"])
        if previous_suite is not None
        else invocation_started_at_utc
    )
    reports = _completed_reports(output_root, manifest)
    if any(report["status"] == "failed" for report in reports):
        _write_state(
            output_root,
            status="failed_infrastructure",
            manifest=manifest,
            reports=reports,
            parent_artifact_sha256=parent_locks,
            first_started_at_utc=first_started_at_utc,
            invocation_started_at_utc=invocation_started_at_utc,
            max_hours=args.max_hours,
            runtime=None,
            reason="A preserved policy-episode failure requires diagnosis before resume.",
        )
        return 1

    target = len(manifest["episodes"]) * len(manifest["policies"])
    if len(reports) == target:
        analysis = evaluate_pre_registered_checks(reports, manifest)
        final_status = str(analysis["status"])
        _write_state(
            output_root,
            status=final_status,
            manifest=manifest,
            reports=reports,
            parent_artifact_sha256=parent_locks,
            first_started_at_utc=first_started_at_utc,
            invocation_started_at_utc=invocation_started_at_utc,
            max_hours=args.max_hours,
            runtime=previous_suite.get("runtime") if previous_suite is not None else None,
            reason="All frozen policy checkpoints were revalidated.",
            gate_analysis=analysis,
        )
        print(f"Policy-control suite is already complete: {final_status}.", flush=True)
        return 0 if final_status == "passed" else 2

    architecture = platform.machine().lower()
    build_commit = args.habitat_build_lock.read_text(encoding="utf-8").strip()
    if architecture != args.require_architecture.lower():
        raise RuntimeError(
            f"Architecture mismatch: expected {args.require_architecture}, found {architecture}."
        )
    if build_commit != HABITAT_SIM_COMMIT:
        raise RuntimeError(
            f"Habitat build-lock mismatch: expected {HABITAT_SIM_COMMIT}, found {build_commit}."
        )

    spec_objects = _spec_objects(manifest)
    report_by_identity = {
        (str(report["policy_run_id"]), str(report["episode_id"])): report for report in reports
    }
    environment = HabitatSimR2REnvironment(
        HabitatSimR2RConfig(scene_root=args.scene_root, gpu_device_id=args.gpu_device_id)
    )
    runtime: dict[str, Any] | None = None
    try:
        for entry in manifest["episodes"]:
            episode_id = str(entry["episode_id"])
            episode = dataset.episode_record(episode_id)
            ground_truth = dataset.ground_truth_record(episode_id)
            for spec in spec_objects:
                identity = (spec.policy_run_id, episode_id)
                if identity in report_by_identity:
                    continue
                elapsed_hours = (time.monotonic() - invocation_started) / 3600.0
                if elapsed_hours >= args.max_hours:
                    _write_state(
                        output_root,
                        status="partial_time_budget",
                        manifest=manifest,
                        reports=reports,
                        parent_artifact_sha256=parent_locks,
                        first_started_at_utc=first_started_at_utc,
                        invocation_started_at_utc=invocation_started_at_utc,
                        max_hours=args.max_hours,
                        runtime=runtime,
                        reason="Invocation time budget reached before the next policy episode.",
                    )
                    print("Policy-control suite reached its budget and is resumable.", flush=True)
                    return 75

                report_started = time.monotonic()
                report_path, steps_path = _artifact_paths(output_root, spec.to_dict(), entry)
                try:
                    result = run_r2r_ce_episode(
                        episode,
                        ground_truth,
                        environment,
                        make_policy(spec),
                    )
                    runtime = {
                        **environment.runtime_metadata(),
                        "platform": platform.platform(),
                        "architecture": architecture,
                        "python": platform.python_version(),
                        "habitat_sim_build_commit_observed": build_commit,
                    }
                    runtime_checks = {
                        "habitat_runtime_version_matches": runtime["habitat_sim_version"]
                        == HABITAT_SIM_VERSION,
                        "navmesh_loaded": runtime["navmesh_loaded"],
                    }
                    if not all(runtime_checks.values()):
                        raise RuntimeError(f"Habitat runtime checks failed: {runtime_checks}")
                    steps_sha256 = _write_steps(steps_path, result.steps)
                    evaluation = result.to_dict()
                    evaluation.pop("steps")
                    report = {
                        "schema_version": RUNNER_SCHEMA_VERSION,
                        "status": "passed",
                        "recorded_at_utc": _timestamp(),
                        "duration_seconds": time.monotonic() - report_started,
                        "manifest_sha256": manifest["manifest_sha256"],
                        "index": entry["index"],
                        "episode_id": episode_id,
                        "trajectory_id": entry["trajectory_id"],
                        "scene_id": episode.scene_id,
                        "policy_run_id": spec.policy_run_id,
                        "family": spec.family,
                        "seed": spec.seed,
                        "role": spec.role,
                        "runtime_checks": runtime_checks,
                        "evaluation": evaluation,
                        "steps_path": str(steps_path),
                        "steps_sha256": steps_sha256,
                    }
                except Exception as error:
                    report = {
                        "schema_version": RUNNER_SCHEMA_VERSION,
                        "status": "failed",
                        "recorded_at_utc": _timestamp(),
                        "duration_seconds": time.monotonic() - report_started,
                        "manifest_sha256": manifest["manifest_sha256"],
                        "index": entry["index"],
                        "episode_id": episode_id,
                        "trajectory_id": entry["trajectory_id"],
                        "scene_id": episode.scene_id,
                        "policy_run_id": spec.policy_run_id,
                        "family": spec.family,
                        "seed": spec.seed,
                        "steps_sha256": None,
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc().splitlines()[-40:],
                        },
                    }
                _write_json(report_path, report)
                reports.append(report)
                report_by_identity[identity] = report
                state = _write_state(
                    output_root,
                    status="running" if report["status"] == "passed" else "failed_infrastructure",
                    manifest=manifest,
                    reports=reports,
                    parent_artifact_sha256=parent_locks,
                    first_started_at_utc=first_started_at_utc,
                    invocation_started_at_utc=invocation_started_at_utc,
                    max_hours=args.max_hours,
                    runtime=runtime,
                    reason=(
                        None
                        if report["status"] == "passed"
                        else "First policy-episode infrastructure failure preserved."
                    ),
                )
                completed = state["summary"]["completed_report_count"]
                print(
                    f"[{completed}/{target}] episode={episode_id} "
                    f"policy={spec.policy_run_id} status={report['status']} "
                    f"seconds={report['duration_seconds']:.2f}",
                    flush=True,
                )
                if report["status"] != "passed":
                    return 1
    finally:
        environment.close()

    analysis = evaluate_pre_registered_checks(reports, manifest)
    final_status = str(analysis["status"])
    _write_state(
        output_root,
        status=final_status,
        manifest=manifest,
        reports=reports,
        parent_artifact_sha256=parent_locks,
        first_started_at_utc=first_started_at_utc,
        invocation_started_at_utc=invocation_started_at_utc,
        max_hours=args.max_hours,
        runtime=runtime,
        reason=(
            "All pre-registered evaluator-control checks passed."
            if final_status == "passed"
            else "All runs completed, but at least one pre-registered contrast failed."
        ),
        gate_analysis=analysis,
    )
    print(f"Policy-control suite completed: {final_status}.", flush=True)
    return 0 if final_status == "passed" else 2


def main() -> int:
    args = _arguments()
    try:
        return run(args)
    except Exception as error:
        args.output_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "status": "failed_infrastructure",
            "recorded_at_utc": _timestamp(),
            "gate": GATE_ID,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc().splitlines()[-40:],
            },
        }
        _write_json(args.output_root / "runner_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
