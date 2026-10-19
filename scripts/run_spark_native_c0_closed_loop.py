#!/usr/bin/env python3
"""Run the latency-selected, resumable official-cap native-C0 closed-loop evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEDataset
from spatialvlm.data.habitat_sim_r2r import (
    HABITAT_SIM_COMMIT,
    HabitatSimR2RConfig,
    HabitatSimR2REnvironment,
)
from spatialvlm.data.history import HistoryMode
from spatialvlm.eval.metrics import (
    R2RCEEpisodeMetrics,
    aggregate_r2r_ce_metrics,
)
from spatialvlm.eval.native_closed_loop import (
    FROZEN_CHECKPOINTS,
    ClosedLoopCheckpoint,
    ClosedLoopDecisionTiming,
    closed_loop_protocol_record,
    run_instrumented_r2r_ce_episode,
    summarize_closed_loop_timings,
)
from spatialvlm.eval.native_policy import NativePolicyConfig, NativeQwenPolicy
from spatialvlm.eval.qwen_action_scorer import NativeQwenActionScorer

SCHEMA_VERSION = 1
GATE_ID = "native_qwen_c0_latency_selected_closed_loop_evaluation_v1"
RUN_ID = "native_c0_closed_loop_v1"
CONTROL_MANIFEST_FILE_SHA256 = (
    "cb75d6259e42decccc161d069df2e04fdadbb7b4f20c6c480bcbca575ec44af4"
)
CONTROL_SUITE_FILE_SHA256 = (
    "2fab8d96958f4abc9377a204f2e43bc1e49d58fc1c7422f60315cf35479557b9"
)
CONTROL_VERIFICATION_FILE_SHA256 = (
    "9d07c3a20a335910d91b338d5e22eea715ae004aa2e998d8ad968967ffa889e5"
)
_SOFTWARE_PATHS = (
    "scripts/run_spark_native_c0_closed_loop.py",
    "scripts/check_native_c0_closed_loop_artifacts.py",
    "scripts/run_spark_native_c0_closed_loop.sh",
    "scripts/start_spark_native_c0_closed_loop.sh",
    "src/spatialvlm/data/habitat_sim_r2r.py",
    "src/spatialvlm/data/history.py",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/eval/action_decision.py",
    "src/spatialvlm/eval/metrics.py",
    "src/spatialvlm/eval/native_closed_loop.py",
    "src/spatialvlm/eval/native_policy.py",
    "src/spatialvlm/eval/qwen_action_scorer.py",
    "src/spatialvlm/eval/r2r_ce.py",
    "src/spatialvlm/training/native_c0_gate.py",
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
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(
            json.dumps(dict(row), sort_keys=True, allow_nan=False) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return _sha256(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--control-verification", type=Path, required=True)
    parser.add_argument("--stress-root", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--source-artifact-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--max-hours", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--habitat-build-lock", type=Path, default=Path("/opt/gridnav/habitat-sim-commit")
    )
    return parser.parse_args()


def _software_hashes(repo: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _SOFTWARE_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Closed-loop source is missing: {path}")
        result[relative] = _sha256(path)
    return result


def _load_timings(path: Path) -> list[ClosedLoopDecisionTiming]:
    records: list[ClosedLoopDecisionTiming] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(ClosedLoopDecisionTiming(**json.loads(line)))
    return records


def _validate_stress(stress_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = stress_root / "manifest.json"
    result_path = stress_root / "result.json"
    timings_path = stress_root / "timings.jsonl"
    verification_path = stress_root / "verification.json"
    for path in (manifest_path, result_path, timings_path, verification_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required stress artifact is missing: {path}")
    result = _load_json(result_path)
    verification = _load_json(verification_path)
    checks = {
        "result_passed": result.get("status") == "passed",
        "verification_passed": verification.get("status") == "passed",
        "verification_errors": verification.get("error_count") == 0,
        "manifest_link": result.get("manifest_sha256") == _sha256(manifest_path),
        "timings_link": result.get("timings_sha256") == _sha256(timings_path),
        "verification_result_link": verification.get("result_sha256") == _sha256(result_path),
    }
    timings = _load_timings(timings_path)
    reconstructed = summarize_closed_loop_timings(timings)
    checks["summary_reconstruction"] = result.get("latency_summary") == reconstructed
    checks["scope_reconstruction"] = result.get("selected_scope") == reconstructed["scope"]
    if not all(checks.values()):
        raise RuntimeError(f"Closed-loop stress evidence did not pass reconstruction: {checks}")
    return result, {
        "manifest_sha256": _sha256(manifest_path),
        "result_sha256": _sha256(result_path),
        "timings_sha256": _sha256(timings_path),
        "verification_sha256": _sha256(verification_path),
    }


def _checkpoint_path(archive: Path, checkpoint: ClosedLoopCheckpoint) -> Path:
    return archive / checkpoint.source_run_id / checkpoint.checkpoint_relative_path


def _verify_checkpoint_source(
    source_root: Path, archive: Path, checkpoint: ClosedLoopCheckpoint
) -> dict[str, Any]:
    run_root = source_root / checkpoint.source_run_id
    result_path = run_root / "result.json"
    verification_path = run_root / "verification.json"
    checkpoint_path = _checkpoint_path(archive, checkpoint)
    for path in (result_path, verification_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Closed-loop checkpoint input is missing: {path}")
    result = _load_json(result_path)
    verification = _load_json(verification_path)
    selection = result.get("selection", {})
    selected = selection.get("checkpoint", {})
    checks = {
        "run_id": result.get("run_id") == checkpoint.source_run_id,
        "checkpoint_gate": result.get("gate") == checkpoint.checkpoint_gate,
        "selected_update": selection.get("selected_update") == checkpoint.selected_update,
        "relative_path": selected.get("relative_path") == checkpoint.checkpoint_relative_path,
        "recorded_hash": selected.get("sha256") == checkpoint.checkpoint_sha256,
        "recorded_state_hash": isinstance(selected.get("trainable_state_sha256"), str)
        and len(selected["trainable_state_sha256"]) == 64,
        "actual_hash": _sha256(checkpoint_path) == checkpoint.checkpoint_sha256,
        "verification_passed": verification.get("status") == "passed",
        "verification_errors": verification.get("error_count") == 0,
        "verification_link": verification.get("result_sha256") == _sha256(result_path),
        "manifest_link": verification.get("manifest_sha256") == result.get("manifest_sha256"),
        "manifest_hash": isinstance(result.get("manifest_sha256"), str)
        and len(result["manifest_sha256"]) == 64,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Checkpoint source verification failed for {checkpoint.checkpoint_id}: {checks}"
        )
    return {
        "source_result_sha256": _sha256(result_path),
        "source_verification_sha256": _sha256(verification_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_gate": checkpoint.checkpoint_gate,
        "checkpoint_update_step": checkpoint.selected_update,
        "checkpoint_trainable_state_sha256": selected.get("trainable_state_sha256"),
        "source_manifest_sha256": result.get("manifest_sha256"),
    }


def _control_summary(
    control_root: Path, control_manifest: Mapping[str, Any], episode_count: int
) -> dict[str, Any]:
    selected_entries = control_manifest["episodes"][:episode_count]
    summaries: dict[str, Any] = {}
    for spec in control_manifest["policies"]:
        run_id = str(spec["policy_run_id"])
        metrics: list[R2RCEEpisodeMetrics] = []
        report_hashes: dict[str, str] = {}
        for entry in selected_entries:
            path = (
                control_root
                / "runs"
                / run_id
                / "episodes"
                / f"{entry['artifact_stem']}.json"
            )
            report = _load_json(path)
            if report.get("status") != "passed":
                raise RuntimeError(f"Control report is not passed: {path}")
            metrics.append(R2RCEEpisodeMetrics(**report["evaluation"]["metrics"]))
            report_hashes[str(entry["episode_id"])] = _sha256(path)
        summaries[run_id] = {
            "family": spec["family"],
            "seed": spec["seed"],
            "aggregate": aggregate_r2r_ce_metrics(metrics).to_dict(),
            "episode_report_sha256": report_hashes,
        }
    return summaries


def _report_paths(output_root: Path, checkpoint_id: str, stem: str) -> tuple[Path, Path, Path]:
    root = output_root / "runs" / checkpoint_id
    return (
        root / "episodes" / f"{stem}.json",
        root / "steps" / f"{stem}.jsonl",
        root / "timings" / f"{stem}.jsonl",
    )


def _load_existing_reports(
    output_root: Path,
    checkpoint_ids: Sequence[str],
    entries: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    for checkpoint_id in checkpoint_ids:
        for entry in entries:
            report_path, steps_path, timings_path = _report_paths(
                output_root, checkpoint_id, str(entry["artifact_stem"])
            )
            if not report_path.exists():
                continue
            report = _load_json(report_path)
            checks = {
                "status": report.get("status") == "completed",
                "manifest": report.get("manifest_sha256") == manifest_sha256,
                "checkpoint": report.get("checkpoint_id") == checkpoint_id,
                "episode": str(report.get("episode_id")) == str(entry["episode_id"]),
                "steps": steps_path.is_file()
                and report.get("steps_sha256") == _sha256(steps_path),
                "timings": timings_path.is_file()
                and report.get("timings_sha256") == _sha256(timings_path),
            }
            if not all(checks.values()):
                raise RuntimeError(f"Existing closed-loop report is inconsistent: {checks}")
            reports[(checkpoint_id, str(entry["episode_id"]))] = report
    return reports


def _checkpoint_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [R2RCEEpisodeMetrics(**report["evaluation"]["metrics"]) for report in reports]
    action_histogram: Counter[str] = Counter()
    step_cap_count = 0
    timing_values: list[float] = []
    for report in reports:
        action_histogram.update(report["action_histogram"])
        step_cap_count += int(report["evaluation"]["metrics"]["step_count"] == 500)
        timing_values.extend(report["closed_loop_seconds"])
    aggregate = aggregate_r2r_ce_metrics(metrics).to_dict()
    aggregate.update(
        {
            "stop_rate": sum(metric.stopped for metric in metrics) / len(metrics),
            "step_cap_rate": step_cap_count / len(metrics),
            "action_histogram": dict(sorted(action_histogram.items())),
            "mean_closed_loop_seconds_per_step": (
                sum(timing_values) / len(timing_values) if timing_values else 0.0
            ),
        }
    )
    return aggregate


def _write_progress(
    output_root: Path,
    *,
    status: str,
    completed: int,
    target: int,
    current_checkpoint: str | None,
    current_episode: str | None,
) -> None:
    _write_json(
        output_root / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "recorded_at_utc": _timestamp(),
            "completed_episode_runs": completed,
            "target_episode_runs": target,
            "current_checkpoint": current_checkpoint,
            "current_episode": current_episode,
        },
    )


def run(args: argparse.Namespace) -> int:
    if args.run_id != RUN_ID:
        raise ValueError(f"Frozen main run ID must be {RUN_ID}.")
    if not math.isfinite(args.max_hours) or args.max_hours <= 0.0:
        raise ValueError("Main closed-loop max-hours must be finite and positive.")
    if platform.machine().lower() != "aarch64":
        raise RuntimeError("Closed-loop evaluation requires the Spark aarch64 runtime.")
    build_commit = args.habitat_build_lock.read_text(encoding="utf-8").strip()
    if build_commit != HABITAT_SIM_COMMIT:
        raise RuntimeError("Habitat-Sim build-lock mismatch.")
    if not (args.model_snapshot / "config.json").is_file():
        raise FileNotFoundError("Pinned Qwen snapshot is incomplete.")

    invocation_started = time.monotonic()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stress_result, stress_locks = _validate_stress(args.stress_root.resolve())
    selected_scope = stress_result["selected_scope"]
    checkpoint_map = {item.checkpoint_id: item for item in FROZEN_CHECKPOINTS}
    selected_checkpoints = [checkpoint_map[item] for item in selected_scope["checkpoint_ids"]]
    episode_count = int(selected_scope["episode_count"])

    control_manifest_path = args.control_root / "suite_manifest.json"
    control_suite_path = args.control_root / "suite.json"
    control_manifest = _load_json(control_manifest_path)
    control_suite = _load_json(control_suite_path)
    control_verification = _load_json(args.control_verification)
    if _sha256(control_manifest_path) != CONTROL_MANIFEST_FILE_SHA256:
        raise RuntimeError("Control manifest differs from the frozen 61-trajectory suite.")
    if _sha256(control_suite_path) != CONTROL_SUITE_FILE_SHA256:
        raise RuntimeError("Control suite bytes differ from the independently verified input.")
    if _sha256(args.control_verification) != CONTROL_VERIFICATION_FILE_SHA256:
        raise RuntimeError("Independent control verification bytes differ from the frozen input.")
    if control_suite.get("status") != "passed":
        raise RuntimeError("Control suite is not independently admitted as passed.")
    control_checks = control_verification.get("checks", {})
    if (
        control_verification.get("status") != "passed"
        or control_checks.get("errors") != []
        or control_checks.get("episode_reports_verified") != 854
        or control_checks.get("step_trace_hashes_verified") != 854
        or control_verification.get("source_sha256", {}).get("suite.json")
        != CONTROL_SUITE_FILE_SHA256
        or control_verification.get("source_sha256", {}).get("suite_manifest.json")
        != CONTROL_MANIFEST_FILE_SHA256
    ):
        raise RuntimeError("Independent policy-control verification is not fully admitted.")
    entries = control_manifest["episodes"][:episode_count]

    checkpoint_locks = {
        item.checkpoint_id: _verify_checkpoint_source(
            args.source_artifact_root, args.checkpoint_archive, item
        )
        for item in selected_checkpoints
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "gate": GATE_ID,
        "run_id": RUN_ID,
        "recorded_at_utc": _timestamp(),
        "protocol": closed_loop_protocol_record(),
        "selected_scope": selected_scope,
        "selected_checkpoints": [item.to_dict() for item in selected_checkpoints],
        "episodes": entries,
        "control_locks": {
            "manifest_sha256": _sha256(control_manifest_path),
            "suite_sha256": _sha256(control_suite_path),
            "verification_sha256": _sha256(args.control_verification),
        },
        "stress_locks": stress_locks,
        "checkpoint_locks": checkpoint_locks,
        "software_sha256": _software_hashes(args.repo.resolve()),
        "docker": {"image_ref": args.docker_image_ref, "image_id": args.docker_image_id},
        "max_hours_per_invocation": args.max_hours,
        "resume_unit": "completed episode report with linked steps and timings",
        "outcome_used_for_scope_selection": False,
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        # Invocation budget is operational and may change on resume; everything else is immutable.
        for payload in (existing, manifest):
            payload.pop("recorded_at_utc", None)
            payload.pop("max_hours_per_invocation", None)
        if existing != manifest:
            raise RuntimeError("Existing main manifest differs; preserve it and use a new version.")
    else:
        _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path)

    dataset = R2RCEDataset.from_release(
        args.episodes, args.ground_truth, split="val_seen", limit=None
    )
    reports = _load_existing_reports(
        output_root,
        [item.checkpoint_id for item in selected_checkpoints],
        entries,
        manifest_sha256,
    )
    target = len(selected_checkpoints) * len(entries)
    _write_progress(
        output_root,
        status="running",
        completed=len(reports),
        target=target,
        current_checkpoint=None,
        current_episode=None,
    )

    environment = HabitatSimR2REnvironment(
        HabitatSimR2RConfig(scene_root=args.scene_root, gpu_device_id=args.gpu_device_id)
    )
    last_runtime: dict[str, Any] | None = None
    try:
        for checkpoint in selected_checkpoints:
            checkpoint_path = _checkpoint_path(args.checkpoint_archive, checkpoint)
            pending = [
                entry
                for entry in entries
                if (checkpoint.checkpoint_id, str(entry["episode_id"])) not in reports
            ]
            if not pending:
                continue
            if (time.monotonic() - invocation_started) / 3600.0 >= args.max_hours:
                _write_progress(
                    output_root,
                    status="partial_time_budget",
                    completed=len(reports),
                    target=target,
                    current_checkpoint=checkpoint.checkpoint_id,
                    current_episode=None,
                )
                return 75
            model_started = time.monotonic()
            scorer = NativeQwenActionScorer.from_native_c0_checkpoint(
                checkpoint_path,
                device=torch.device(args.device),
                expected_checkpoint_sha256=checkpoint_locks[checkpoint.checkpoint_id][
                    "checkpoint_sha256"
                ],
                expected_checkpoint_gate=checkpoint_locks[checkpoint.checkpoint_id][
                    "checkpoint_gate"
                ],
                expected_manifest_sha256=checkpoint_locks[checkpoint.checkpoint_id][
                    "source_manifest_sha256"
                ],
                expected_trainable_state_sha256=checkpoint_locks[checkpoint.checkpoint_id][
                    "checkpoint_trainable_state_sha256"
                ],
                expected_update_step=int(
                    checkpoint_locks[checkpoint.checkpoint_id]["checkpoint_update_step"]
                ),
                local_files_only=True,
            )
            model_load_seconds = time.monotonic() - model_started
            for entry in pending:
                if (time.monotonic() - invocation_started) / 3600.0 >= args.max_hours:
                    _write_progress(
                        output_root,
                        status="partial_time_budget",
                        completed=len(reports),
                        target=target,
                        current_checkpoint=checkpoint.checkpoint_id,
                        current_episode=str(entry["episode_id"]),
                    )
                    return 75
                episode_id = str(entry["episode_id"])
                episode = dataset.episode_record(episode_id)
                ground_truth = dataset.ground_truth_record(episode_id)
                # A fresh lightweight policy wrapper per episode prevents completed history and
                # image-window telemetry from accumulating in RAM. The loaded scorer/backbone is
                # shared for the whole checkpoint, so this does not add model-load overhead.
                policy = NativeQwenPolicy(
                    scorer=scorer,
                    config=NativePolicyConfig.for_mode(
                        HistoryMode.H4_SHORT_WINDOW, chunk_length=1, step_cap=500
                    ),
                )
                episode_started = time.monotonic()
                instrumented = run_instrumented_r2r_ce_episode(
                    episode,
                    ground_truth,
                    environment,
                    policy,
                    episode_index=int(entry["index"]),
                    max_steps=500,
                )
                last_runtime = environment.runtime_metadata()
                report_path, steps_path, timings_path = _report_paths(
                    output_root, checkpoint.checkpoint_id, str(entry["artifact_stem"])
                )
                steps_sha256 = _write_jsonl(
                    steps_path, [step.to_dict() for step in instrumented.evaluation.steps]
                )
                timing_rows = [item.to_dict() for item in instrumented.decision_timings]
                timings_sha256 = _write_jsonl(timings_path, timing_rows)
                evaluation = instrumented.evaluation.to_dict()
                evaluation.pop("steps")
                actions = [action.name for action in instrumented.evaluation.trace.actions]
                telemetry = policy.episode_record()
                report = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "completed",
                    "recorded_at_utc": _timestamp(),
                    "manifest_sha256": manifest_sha256,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "episode_index": entry["index"],
                    "episode_id": episode_id,
                    "trajectory_id": entry["trajectory_id"],
                    "scene_id": entry["scene_id"],
                    "duration_seconds": time.monotonic() - episode_started,
                    "model_load_seconds_this_process": model_load_seconds,
                    "environment_reset_seconds": instrumented.environment_reset_seconds,
                    "policy_reset_seconds": instrumented.policy_reset_seconds,
                    "evaluation": evaluation,
                    "action_histogram": dict(sorted(Counter(actions).items())),
                    "closed_loop_seconds": [item["closed_loop_seconds"] for item in timing_rows],
                    "policy_telemetry": telemetry,
                    "steps_path": str(steps_path),
                    "steps_sha256": steps_sha256,
                    "timings_path": str(timings_path),
                    "timings_sha256": timings_sha256,
                }
                _write_json(report_path, report)
                reports[(checkpoint.checkpoint_id, episode_id)] = report
                _write_progress(
                    output_root,
                    status="running",
                    completed=len(reports),
                    target=target,
                    current_checkpoint=checkpoint.checkpoint_id,
                    current_episode=episode_id,
                )
                print(
                    f"[{len(reports)}/{target}] checkpoint={checkpoint.checkpoint_id} "
                    f"episode={episode_id} steps={evaluation['metrics']['step_count']} "
                    f"SR={evaluation['metrics']['success']:.0f} "
                    f"OSR={evaluation['metrics']['oracle_success']:.0f}",
                    flush=True,
                )
    finally:
        environment.close()

    summaries: dict[str, Any] = {}
    for checkpoint in selected_checkpoints:
        checkpoint_reports = [
            reports[(checkpoint.checkpoint_id, str(entry["episode_id"]))] for entry in entries
        ]
        summaries[checkpoint.checkpoint_id] = _checkpoint_summary(checkpoint_reports)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "gate": GATE_ID,
        "run_id": RUN_ID,
        "recorded_at_utc": _timestamp(),
        "manifest_path": "/artifacts/manifest.json",
        "manifest_sha256": manifest_sha256,
        "selected_scope": selected_scope,
        "completed_episode_runs": len(reports),
        "checkpoint_summaries": summaries,
        "control_comparison": _control_summary(
            args.control_root, control_manifest, episode_count
        ),
        "headline_contrast": "oracle_success_rate versus success_rate",
        "combined_invocation_seconds": time.monotonic() - invocation_started,
        "runtime": {
            **(last_runtime or {}),
            "habitat_sim_build_commit_observed": build_commit,
            "platform": platform.platform(),
            "architecture": platform.machine().lower(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "evidence_boundary": {
            "established_if_verified": (
                "The latency-selected native-C0 checkpoint scope was executed on the frozen "
                "val_seen control episodes with H4, one-step scoring, and the official 500-step "
                "cap; learned SR/OSR and secondary metrics are measured outcomes."
            ),
            "not_established": (
                "val_unseen generalization, C1-C4 geometry efficacy, legacy-runtime parity, "
                "causal explanation, or superiority beyond the executed controls."
            ),
        },
    }
    _write_json(output_root / "result.json", result)
    _write_progress(
        output_root,
        status="completed",
        completed=len(reports),
        target=target,
        current_checkpoint=None,
        current_episode=None,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


def main() -> int:
    args = _arguments()
    try:
        return run(args)
    except Exception as error:
        args.output_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_infrastructure",
            "gate": GATE_ID,
            "recorded_at_utc": _timestamp(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(args.output_root / "failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
