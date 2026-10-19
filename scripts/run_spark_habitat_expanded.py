#!/usr/bin/env python3
"""Run a resumable, full-val_seen Habitat-Sim released-trajectory parity suite."""

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
from spatialvlm.eval.habitat_conformance import (
    HabitatConformanceThresholds,
    compare_ground_truth_replay,
    describe_observation_contract,
    quaternion_angular_error_radians,
)
from spatialvlm.eval.habitat_suite import build_full_split_manifest
from spatialvlm.eval.r2r_ce import GroundTruthActionPolicy, run_r2r_ce_episode

RUNNER_SCHEMA_VERSION = 2
GATE_ID = "spark_habitat_sim_r2r_ce_full_val_seen_runtime_geodesic_v2"

_SOFTWARE_PATHS = (
    "scripts/run_spark_habitat_expanded.py",
    "src/spatialvlm/data/datasets.py",
    "src/spatialvlm/data/habitat_sim_r2r.py",
    "src/spatialvlm/eval/habitat_conformance.py",
    "src/spatialvlm/eval/habitat_suite.py",
    "src/spatialvlm/eval/metrics.py",
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--require-architecture", default="aarch64")
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
            raise FileNotFoundError(f"Suite software input is missing: {path}")
        hashes[relative] = _sha256(path)
    return hashes


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _artifact_paths(output_root: Path, entry: Mapping[str, Any]) -> tuple[Path, Path]:
    stem = str(entry["artifact_stem"])
    return output_root / "episodes" / f"{stem}.json", output_root / "steps" / f"{stem}.jsonl"


def _validate_existing_report(
    report_path: Path,
    steps_path: Path,
    entry: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    report = _load_json(report_path)
    checks = {
        "schema_version": report.get("schema_version") == RUNNER_SCHEMA_VERSION,
        "manifest": report.get("manifest_sha256") == manifest_sha256,
        "index": report.get("index") == entry["index"],
        "episode_id": report.get("episode_id") == entry["episode_id"],
        "status": report.get("status") in {"passed", "failed"},
    }
    if report.get("status") == "passed":
        checks["steps_sha256_declared"] = (
            isinstance(report.get("steps_sha256"), str) and len(report["steps_sha256"]) == 64
        )
        checks["steps_present"] = steps_path.is_file()
        checks["steps_sha256"] = steps_path.is_file() and _sha256(steps_path) == report.get(
            "steps_sha256"
        )
    if not all(checks.values()):
        raise RuntimeError(
            f"Existing episode checkpoint is inconsistent: {report_path}; checks={checks}"
        )
    return report


def _completed_reports(
    output_root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    manifest_sha256 = str(manifest["manifest_sha256"])
    for entry in manifest["episodes"]:
        report_path, steps_path = _artifact_paths(output_root, entry)
        if report_path.is_file():
            reports.append(
                _validate_existing_report(report_path, steps_path, entry, manifest_sha256)
            )
    return reports


def _summary(reports: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    passed = [report for report in reports if report.get("status") == "passed"]
    failed = [report for report in reports if report.get("status") == "failed"]
    replay_rows = [report["ground_truth_replay"]["observed"] for report in passed]
    replay_diagnostics = [report["ground_truth_replay"]["diagnostics"] for report in passed]
    metrics = [row["metrics"] for row in replay_rows]
    release_geodesic_drifts = [row["release_metadata_geodesic_drift_m"] for row in replay_rows]
    target = int(manifest["selection"]["target_episode_count"])
    return {
        "target_episode_count": target,
        "completed_episode_count": len(reports),
        "passed_episode_count": len(passed),
        "failed_episode_count": len(failed),
        "remaining_episode_count": target - len(reports),
        "completed_scene_count": len({report["scene_id"] for report in reports}),
        "target_scene_count": manifest["selection"]["scene_count"],
        "runtime_collision_count": sum(
            row["simulator_collision_count_on_forward"] for row in replay_rows
        ),
        "maximum_location_error_m": max(
            (row["maximum_location_error_m"] for row in replay_rows), default=None
        ),
        "release_metadata_geodesic_drift_episode_count": sum(
            not row["release_metadata_geodesic_within_tolerance"] for row in replay_diagnostics
        ),
        "maximum_release_metadata_geodesic_drift_m": max(release_geodesic_drifts, default=None),
        "mean_release_metadata_geodesic_drift_m": (
            sum(release_geodesic_drifts) / len(release_geodesic_drifts)
            if release_geodesic_drifts
            else None
        ),
        "minimum_ndtw": min((row["metrics"]["ndtw"] for row in replay_rows), default=None),
        "mean_success_rate": (
            sum(row["success"] for row in metrics) / len(metrics) if metrics else None
        ),
        "mean_spl": sum(row["spl"] for row in metrics) / len(metrics) if metrics else None,
    }


def _suite_payload(
    *,
    status: str,
    manifest: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    first_started_at_utc: str,
    invocation_started_at_utc: str,
    max_hours: float,
    runtime: Mapping[str, Any] | None,
    reason: str | None = None,
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
        "reason": reason,
        "summary": _summary(reports, manifest),
        "runtime": dict(runtime) if runtime is not None else None,
        "evidence_boundary": {
            "established_if_passed": (
                "Every released val_seen ground-truth trajectory reproduced its actions, native "
                "sensor contract, reset pose, forward-step positions, STOP, success, and nDTW "
                "under the locked runtime and tolerances. SPL used the simulator geodesic measured "
                "at reset, and drift from released geodesic metadata was quantified."
            ),
            "not_established": (
                "Identity between legacy and modern navmesh geodesic values, legacy collision-flag "
                "identity, val_unseen parity, learned-policy performance, training correctness, "
                "baseline ordering, or paper-level model claims."
            ),
        },
    }


def _write_state(
    output_root: Path,
    *,
    status: str,
    manifest: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    first_started_at_utc: str,
    invocation_started_at_utc: str,
    max_hours: float,
    runtime: Mapping[str, Any] | None,
    reason: str | None = None,
) -> dict[str, Any]:
    payload = _suite_payload(
        status=status,
        manifest=manifest,
        reports=reports,
        first_started_at_utc=first_started_at_utc,
        invocation_started_at_utc=invocation_started_at_utc,
        max_hours=max_hours,
        runtime=runtime,
        reason=reason,
    )
    _write_json(output_root / "suite.json", payload)
    _write_json(
        output_root / "progress.json",
        {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "status": status,
            "recorded_at_utc": payload["recorded_at_utc"],
            "manifest_sha256": manifest["manifest_sha256"],
            "summary": payload["summary"],
            "reason": reason,
        },
    )
    return payload


def run(args: argparse.Namespace) -> int:
    if not math.isfinite(args.max_hours) or args.max_hours <= 0.0:
        raise ValueError("Expanded conformance max-hours must be finite and positive.")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    invocation_started_at_utc = _timestamp()
    invocation_started = time.monotonic()

    dataset = R2RCEDataset.from_release(
        args.episodes,
        args.ground_truth,
        split=args.split,
        limit=None,
    )
    manifest = build_full_split_manifest(
        dataset,
        expected_split=args.split,
        software_sha256=_software_hashes(args.repo_root.resolve()),
    )
    manifest_path = output_root / "suite_manifest.json"
    if manifest_path.exists():
        existing_manifest = _load_json(manifest_path)
        if existing_manifest != manifest:
            raise RuntimeError(
                "Existing suite manifest differs from the current release/software lock. "
                "Preserve the old artifact directory and start a new one."
            )
    else:
        _write_json(manifest_path, manifest)

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
            status="failed",
            manifest=manifest,
            reports=reports,
            first_started_at_utc=first_started_at_utc,
            invocation_started_at_utc=invocation_started_at_utc,
            max_hours=args.max_hours,
            runtime=None,
            reason="A preserved episode failure requires diagnosis before resume.",
        )
        return 1
    if len(reports) == manifest["selection"]["target_episode_count"]:
        preserved_runtime = previous_suite.get("runtime") if previous_suite is not None else None
        _write_state(
            output_root,
            status="passed",
            manifest=manifest,
            reports=reports,
            first_started_at_utc=first_started_at_utc,
            invocation_started_at_utc=invocation_started_at_utc,
            max_hours=args.max_hours,
            runtime=preserved_runtime,
            reason="All manifest episodes already have validated passing checkpoints.",
        )
        print("Expanded Habitat suite is already complete and passed.", flush=True)
        return 0

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

    thresholds = HabitatConformanceThresholds()
    environment = HabitatSimR2REnvironment(
        HabitatSimR2RConfig(scene_root=args.scene_root, gpu_device_id=args.gpu_device_id)
    )
    runtime: dict[str, Any] | None = None
    report_by_id = {str(report["episode_id"]): report for report in reports}
    try:
        for entry in manifest["episodes"]:
            episode_id = str(entry["episode_id"])
            if episode_id in report_by_id:
                continue
            elapsed_hours = (time.monotonic() - invocation_started) / 3600.0
            if elapsed_hours >= args.max_hours:
                _write_state(
                    output_root,
                    status="partial_time_budget",
                    manifest=manifest,
                    reports=reports,
                    first_started_at_utc=first_started_at_utc,
                    invocation_started_at_utc=invocation_started_at_utc,
                    max_hours=args.max_hours,
                    runtime=runtime,
                    reason="Invocation time budget reached before the next episode.",
                )
                print("Expanded suite reached its time budget and is safely resumable.", flush=True)
                return 75

            episode_started = time.monotonic()
            episode = dataset.episode_record(episode_id)
            ground_truth = dataset.ground_truth_record(episode_id)
            report_path, steps_path = _artifact_paths(output_root, entry)
            try:
                reset_state = environment.reset(episode)
                observation_report = describe_observation_contract(dict(reset_state.observation))
                rotation_error = quaternion_angular_error_radians(
                    environment.agent_rotation_xyzw(), episode.start_rotation
                )
                result = run_r2r_ce_episode(
                    episode,
                    ground_truth,
                    environment,
                    GroundTruthActionPolicy(),
                )
                replay_report = compare_ground_truth_replay(
                    episode,
                    ground_truth,
                    result,
                    reset_rotation_error_radians=rotation_error,
                    thresholds=thresholds,
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
                status = (
                    "passed"
                    if observation_report["status"] == "passed"
                    and replay_report["status"] == "passed"
                    and all(runtime_checks.values())
                    else "failed"
                )
                steps_sha256 = _write_steps(steps_path, result.steps)
                evaluation = result.to_dict()
                evaluation.pop("steps")
                report = {
                    "schema_version": RUNNER_SCHEMA_VERSION,
                    "status": status,
                    "recorded_at_utc": _timestamp(),
                    "duration_seconds": time.monotonic() - episode_started,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "index": entry["index"],
                    "phase": entry["phase"],
                    "episode_id": episode_id,
                    "scene_id": episode.scene_id,
                    "released_features": entry["features"],
                    "runtime_checks": runtime_checks,
                    "observation_contract": observation_report,
                    "ground_truth_replay": replay_report,
                    "evaluation": evaluation,
                    "steps_path": str(steps_path),
                    "steps_sha256": steps_sha256,
                }
            except Exception as error:
                report = {
                    "schema_version": RUNNER_SCHEMA_VERSION,
                    "status": "failed",
                    "recorded_at_utc": _timestamp(),
                    "duration_seconds": time.monotonic() - episode_started,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "index": entry["index"],
                    "phase": entry["phase"],
                    "episode_id": episode_id,
                    "scene_id": episode.scene_id,
                    "released_features": entry["features"],
                    "steps_sha256": None,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc().splitlines()[-40:],
                    },
                }

            _write_json(report_path, report)
            reports.append(report)
            report_by_id[episode_id] = report
            progress = _write_state(
                output_root,
                status="running" if report["status"] == "passed" else "failed",
                manifest=manifest,
                reports=reports,
                first_started_at_utc=first_started_at_utc,
                invocation_started_at_utc=invocation_started_at_utc,
                max_hours=args.max_hours,
                runtime=runtime,
                reason=None if report["status"] == "passed" else "First episode failure preserved.",
            )
            summary = progress["summary"]
            print(
                f"[{summary['completed_episode_count']}/{summary['target_episode_count']}] "
                f"episode={episode_id} scene={episode.scene_id} status={report['status']} "
                f"seconds={report['duration_seconds']:.2f}",
                flush=True,
            )
            if report["status"] != "passed":
                return 1
    finally:
        environment.close()

    _write_state(
        output_root,
        status="passed",
        manifest=manifest,
        reports=reports,
        first_started_at_utc=first_started_at_utc,
        invocation_started_at_utc=invocation_started_at_utc,
        max_hours=args.max_hours,
        runtime=runtime,
        reason="Every episode in the immutable full-split manifest passed.",
    )
    print("Expanded full-val_seen Habitat conformance PASSED.", flush=True)
    return 0


def main() -> int:
    args = _arguments()
    try:
        return run(args)
    except Exception as error:
        args.output_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "status": "failed",
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
