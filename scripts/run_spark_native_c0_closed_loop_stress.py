#!/usr/bin/env python3
"""Run the frozen 32-decision native-C0 closed-loop latency/integration gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from collections.abc import Mapping
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
from spatialvlm.eval.native_closed_loop import (
    CLOSED_LOOP_STRESS_DECISIONS,
    CLOSED_LOOP_STRESS_MAX_SECONDS,
    FROZEN_CHECKPOINTS,
    closed_loop_protocol_record,
    run_closed_loop_decision_budget,
    summarize_closed_loop_timings,
)
from spatialvlm.eval.native_policy import NativePolicyConfig, NativeQwenPolicy
from spatialvlm.eval.qwen_action_scorer import NativeQwenActionScorer

SCHEMA_VERSION = 1
GATE_ID = "native_qwen_c0_closed_loop_latency_integration_stress_v1"
RUN_ID = "native_c0_closed_loop_stress_v3"
CONTROL_MANIFEST_ID = "ad3a9a55ebcbb3a0d8119e76e603fedbc71a0ad744c05a6fd005e4d43458fa93"
CONTROL_MANIFEST_FILE_SHA256 = (
    "cb75d6259e42decccc161d069df2e04fdadbb7b4f20c6c480bcbca575ec44af4"
)
_SOFTWARE_PATHS = (
    "scripts/run_spark_native_c0_closed_loop_stress.py",
    "scripts/check_native_c0_closed_loop_stress_artifacts.py",
    "scripts/run_spark_native_c0_closed_loop_stress.sh",
    "scripts/start_spark_native_c0_closed_loop_stress.sh",
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows) + "\n",
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
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-verification", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--habitat-build-lock", type=Path, default=Path("/opt/gridnav/habitat-sim-commit")
    )
    parser.add_argument("--hard-max-seconds", type=float, default=CLOSED_LOOP_STRESS_MAX_SECONDS)
    return parser.parse_args()


def _software_hashes(repo: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _SOFTWARE_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Closed-loop stress source is missing: {path}")
        result[relative] = _sha256(path)
    return result


def _validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.run_id != RUN_ID:
        raise ValueError(f"Frozen stress run ID must be {RUN_ID}.")
    if args.hard_max_seconds != CLOSED_LOOP_STRESS_MAX_SECONDS:
        raise ValueError("Frozen stress wall-clock ceiling must be exactly 1,800 seconds.")
    if platform.machine().lower() != "aarch64":
        raise RuntimeError("Closed-loop stress requires the Spark aarch64 runtime.")
    build_commit = args.habitat_build_lock.read_text(encoding="utf-8").strip()
    if build_commit != HABITAT_SIM_COMMIT:
        raise RuntimeError("Habitat-Sim build-lock mismatch.")

    control = _load_json(args.control_manifest)
    if _sha256(args.control_manifest) != CONTROL_MANIFEST_FILE_SHA256:
        raise RuntimeError(
            "Policy-control manifest bytes differ from the frozen 61-trajectory input."
        )
    checks = {
        "manifest_field": control.get("manifest_sha256") == CONTROL_MANIFEST_ID,
        "split": control.get("release", {}).get("split") == "val_seen",
        "episode_count": len(control.get("episodes", [])) == 61,
        "trajectory_count": len({row["trajectory_id"] for row in control.get("episodes", [])})
        == 61,
        "scene_count": len({row["scene_id"] for row in control.get("episodes", [])}) == 53,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen policy-control manifest checks failed: {checks}")

    expected = FROZEN_CHECKPOINTS[0]
    source_result = _load_json(args.source_result)
    source_verification = _load_json(args.source_verification)
    selection = source_result.get("selection", {})
    checkpoint = selection.get("checkpoint", {})
    source_checks = {
        "run_id": source_result.get("run_id") == expected.source_run_id,
        "checkpoint_gate": source_result.get("gate") == expected.checkpoint_gate,
        "selected_update": selection.get("selected_update") == expected.selected_update,
        "relative_path": checkpoint.get("relative_path") == expected.checkpoint_relative_path,
        "recorded_hash": checkpoint.get("sha256") == expected.checkpoint_sha256,
        "recorded_state_hash": isinstance(checkpoint.get("trainable_state_sha256"), str)
        and len(checkpoint["trainable_state_sha256"]) == 64,
        "checkpoint_hash": _sha256(args.checkpoint) == expected.checkpoint_sha256,
        "verification_passed": source_verification.get("status") == "passed",
        "verification_errors": source_verification.get("error_count") == 0,
        "verification_link": source_verification.get("result_sha256")
        == _sha256(args.source_result),
        "manifest_link": source_verification.get("manifest_sha256")
        == source_result.get("manifest_sha256"),
        "manifest_hash": isinstance(source_result.get("manifest_sha256"), str)
        and len(source_result["manifest_sha256"]) == 64,
    }
    if not all(source_checks.values()):
        raise RuntimeError(f"Selected checkpoint provenance checks failed: {source_checks}")
    if not (args.model_snapshot / "config.json").is_file():
        raise FileNotFoundError("Pinned Qwen snapshot is incomplete.")
    return control, {
        "habitat_sim_build_commit": build_commit,
        "control_manifest_file_sha256": CONTROL_MANIFEST_FILE_SHA256,
        "control_manifest_id": CONTROL_MANIFEST_ID,
        "source_result_sha256": _sha256(args.source_result),
        "source_verification_sha256": _sha256(args.source_verification),
        "checkpoint_sha256": expected.checkpoint_sha256,
        "checkpoint_gate": expected.checkpoint_gate,
        "checkpoint_update_step": expected.selected_update,
        "checkpoint_trainable_state_sha256": checkpoint.get("trainable_state_sha256"),
        "source_manifest_sha256": source_result.get("manifest_sha256"),
        "model_snapshot_config_sha256": _sha256(args.model_snapshot / "config.json"),
    }


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    hard_deadline = started + args.hard_max_seconds
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "result.json", "verification.json", "failure.json"):
        if (output_root / name).exists():
            raise FileExistsError(
                f"Closed-loop stress output already exists; preserve the directory: {name}"
            )

    control, input_locks = _validate_inputs(args)
    dataset = R2RCEDataset.from_release(
        args.episodes, args.ground_truth, split="val_seen", limit=None
    )
    ordered_ids = [str(entry["episode_id"]) for entry in control["episodes"]]
    episodes = [dataset.episode_record(episode_id) for episode_id in ordered_ids]
    ground_truth = {
        episode_id: dataset.ground_truth_record(episode_id) for episode_id in ordered_ids
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "gate": GATE_ID,
        "run_id": RUN_ID,
        "recorded_at_utc": _timestamp(),
        "protocol": closed_loop_protocol_record(),
        "control_manifest_path": str(args.control_manifest),
        "input_locks": input_locks,
        "software_sha256": _software_hashes(args.repo.resolve()),
        "docker": {"image_ref": args.docker_image_ref, "image_id": args.docker_image_id},
        "stress_episode_order": ordered_ids,
    }
    _write_json(output_root / "manifest.json", manifest)
    manifest_sha256 = _sha256(output_root / "manifest.json")

    device = torch.device(args.device)
    model_started = time.monotonic()
    scorer = NativeQwenActionScorer.from_native_c0_checkpoint(
        args.checkpoint,
        device=device,
        expected_checkpoint_sha256=input_locks["checkpoint_sha256"],
        expected_checkpoint_gate=input_locks["checkpoint_gate"],
        expected_manifest_sha256=input_locks["source_manifest_sha256"],
        expected_trainable_state_sha256=input_locks["checkpoint_trainable_state_sha256"],
        expected_update_step=input_locks["checkpoint_update_step"],
        local_files_only=True,
    )
    model_load_seconds = time.monotonic() - model_started
    policy = NativeQwenPolicy(
        scorer=scorer,
        config=NativePolicyConfig.for_mode(
            HistoryMode.H4_SHORT_WINDOW, chunk_length=1, step_cap=500
        ),
    )
    environment = HabitatSimR2REnvironment(
        HabitatSimR2RConfig(scene_root=args.scene_root, gpu_device_id=args.gpu_device_id)
    )
    try:
        diagnostic = run_closed_loop_decision_budget(
            episodes,
            ground_truth,
            environment,
            policy,
            decision_budget=CLOSED_LOOP_STRESS_DECISIONS,
            hard_deadline_monotonic=hard_deadline,
            per_episode_step_cap=500,
        )
        runtime = environment.runtime_metadata()
    finally:
        environment.close()

    timing_rows = [record.to_dict() for record in diagnostic.decision_timings]
    timings_sha256 = _write_jsonl(output_root / "timings.jsonl", timing_rows)
    summary = summarize_closed_loop_timings(diagnostic.decision_timings)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "gate": GATE_ID,
        "run_id": RUN_ID,
        "recorded_at_utc": _timestamp(),
        "manifest_path": "/artifacts/manifest.json",
        "manifest_sha256": manifest_sha256,
        "timings_path": "/artifacts/timings.jsonl",
        "timings_sha256": timings_sha256,
        "latency_summary": summary,
        "selected_scope": summary["scope"],
        "model_load_seconds": model_load_seconds,
        "combined_elapsed_seconds": time.monotonic() - started,
        "episode_resets": list(diagnostic.episode_resets),
        "policy_telemetry": list(policy.collected_episode_records()),
        "runtime": {
            **runtime,
            "platform": platform.platform(),
            "architecture": platform.machine().lower(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "scientific_metrics_emitted": False,
        "evidence_boundary": closed_loop_protocol_record()["evidence_boundary"],
    }
    _write_json(output_root / "result.json", result)
    _write_json(
        output_root / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "recorded_at_utc": result["recorded_at_utc"],
            "completed_decisions": CLOSED_LOOP_STRESS_DECISIONS,
            "mean_seconds_per_step": summary["mean_seconds_per_step"],
            "selected_scope": summary["scope"],
        },
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
            "status": "failed",
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
