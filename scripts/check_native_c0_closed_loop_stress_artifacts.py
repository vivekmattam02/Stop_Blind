#!/usr/bin/env python3
"""Independently verify the native-C0 32-decision closed-loop stress artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatialvlm.eval.native_closed_loop import (
    CLOSED_LOOP_STRESS_DECISIONS,
    FROZEN_CHECKPOINTS,
    ClosedLoopDecisionTiming,
    closed_loop_protocol_record,
    summarize_closed_loop_timings,
)

SCHEMA_VERSION = 1
GATE_ID = "independent_native_qwen_c0_closed_loop_stress_verification_v1"
RUNNER_GATE_ID = "native_qwen_c0_closed_loop_latency_integration_stress_v1"
CONTROL_MANIFEST_ID = "ad3a9a55ebcbb3a0d8119e76e603fedbc71a0ad744c05a6fd005e4d43458fa93"
CONTROL_MANIFEST_FILE_SHA256 = (
    "cb75d6259e42decccc161d069df2e04fdadbb7b4f20c6c480bcbca575ec44af4"
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-verification", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []

    for path in (
        args.control_manifest,
        args.source_result,
        args.source_verification,
        args.checkpoint,
        args.manifest,
        args.result,
        args.timings,
    ):
        if not path.is_file():
            errors.append(f"required artifact is missing: {path}")
    if errors:
        return _payload(args, errors)

    control = _load_json(args.control_manifest)
    source = _load_json(args.source_result)
    source_verification = _load_json(args.source_verification)
    manifest = _load_json(args.manifest)
    result = _load_json(args.result)
    expected_checkpoint = FROZEN_CHECKPOINTS[0]

    if _sha256(args.control_manifest) != CONTROL_MANIFEST_FILE_SHA256:
        errors.append("control manifest bytes differ from the frozen input")
    if control.get("manifest_sha256") != CONTROL_MANIFEST_ID:
        errors.append("control manifest self-identity differs")
    if len(control.get("episodes", [])) != 61:
        errors.append("control manifest must contain 61 episodes")
    if control.get("release", {}).get("split") != "val_seen":
        errors.append("control manifest is not val_seen")

    if source_verification.get("status") != "passed":
        errors.append("source checkpoint verification is not passed")
    if source_verification.get("error_count") != 0:
        errors.append("source checkpoint verification has errors")
    if source_verification.get("result_sha256") != _sha256(args.source_result):
        errors.append("source result/verification hash linkage mismatch")
    selection = source.get("selection", {})
    selected_checkpoint = selection.get("checkpoint", {})
    if source.get("run_id") != expected_checkpoint.source_run_id:
        errors.append("source run ID differs from the frozen stress checkpoint")
    if selection.get("selected_update") != expected_checkpoint.selected_update:
        errors.append("source selected update differs")
    if selected_checkpoint.get("sha256") != expected_checkpoint.checkpoint_sha256:
        errors.append("source selected checkpoint digest differs")
    if _sha256(args.checkpoint) != expected_checkpoint.checkpoint_sha256:
        errors.append("executed checkpoint bytes differ")

    if manifest.get("status") != "frozen" or manifest.get("gate") != RUNNER_GATE_ID:
        errors.append("stress manifest is not the frozen runner gate")
    if manifest.get("protocol") != closed_loop_protocol_record():
        errors.append("stress protocol differs from the independently reconstructed contract")
    if (
        manifest.get("input_locks", {}).get("control_manifest_file_sha256")
        != CONTROL_MANIFEST_FILE_SHA256
    ):
        errors.append("manifest does not bind the frozen control manifest")
    if manifest.get("input_locks", {}).get("checkpoint_sha256") != _sha256(args.checkpoint):
        errors.append("manifest does not bind the executed checkpoint")
    source_locks = manifest.get("software_sha256", {})
    for relative, expected_hash in source_locks.items():
        path = args.repo / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            errors.append(f"executed source lock mismatch: {relative}")

    if result.get("status") != "passed" or result.get("gate") != RUNNER_GATE_ID:
        errors.append("stress result is not passed")
    if result.get("manifest_sha256") != _sha256(args.manifest):
        errors.append("result/manifest hash linkage mismatch")
    if result.get("timings_sha256") != _sha256(args.timings):
        errors.append("result/timings hash linkage mismatch")
    if result.get("scientific_metrics_emitted") is not False:
        errors.append("stress result must explicitly refuse scientific metrics")
    forbidden_result_keys = {"aggregate", "success_rate", "oracle_success_rate", "spl"}
    if forbidden_result_keys.intersection(result):
        errors.append("stress result contains prohibited scientific metric fields")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(args.timings.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"timings JSONL line {line_number} is invalid JSON")
            continue
        if not isinstance(row, dict):
            errors.append(f"timings JSONL line {line_number} is not an object")
            continue
        rows.append(row)
    if len(rows) != CLOSED_LOOP_STRESS_DECISIONS:
        errors.append("stress timing ledger does not contain exactly 32 decisions")

    timing_objects: list[ClosedLoopDecisionTiming] = []
    if len(rows) == CLOSED_LOOP_STRESS_DECISIONS:
        for index, row in enumerate(rows):
            try:
                record = ClosedLoopDecisionTiming(**row)
            except (TypeError, ValueError) as error:
                errors.append(f"invalid timing row {index}: {error}")
                continue
            if record.action_id not in (0, 1, 2, 3):
                errors.append(f"timing row {index} contains a non-official action")
            timing_objects.append(record)

    if len(timing_objects) == CLOSED_LOOP_STRESS_DECISIONS:
        reconstructed = summarize_closed_loop_timings(timing_objects)
        if result.get("latency_summary") != reconstructed:
            errors.append("latency summary does not match exact timing reconstruction")
        if result.get("selected_scope") != reconstructed["scope"]:
            errors.append("selected scope does not match the preregistered latency branch")

    telemetry = result.get("policy_telemetry", [])
    decisions = [decision for episode in telemetry for decision in episode.get("decisions", [])]
    if len(decisions) != CLOSED_LOOP_STRESS_DECISIONS:
        errors.append("policy telemetry does not contain exactly 32 scored decisions")
    elif len(rows) == CLOSED_LOOP_STRESS_DECISIONS:
        for index, (decision, timing) in enumerate(zip(decisions, rows)):
            actions = decision.get("chunk_actions")
            if actions != [timing.get("action_name")]:
                errors.append(f"policy/timing action mismatch at decision {index}")
            diagnostics = decision.get("diagnostics", {})
            if diagnostics.get("checkpoint_sha256") != expected_checkpoint.checkpoint_sha256:
                errors.append(f"scorer checkpoint mismatch at decision {index}")
            if diagnostics.get("chunk_length") != 1:
                errors.append(f"scorer chunk length mismatch at decision {index}")
            components = (
                diagnostics.get("processor_seconds"), diagnostics.get("model_forward_seconds")
            )
            if any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in components
            ):
                errors.append(f"nonfinite scorer telemetry at decision {index}")
            window = decision.get("window", {})
            contract = window.get("contract", {})
            if contract.get("mode") != "H4" or len(window.get("frames", [])) != 4:
                errors.append(f"non-H4 history at decision {index}")
            excluded = set(contract.get("explicitly_excluded_fields", []))
            if excluded != {
                "simulator_pose",
                "goal_coordinate",
                "geodesic_distance",
                "future_action",
            }:
                errors.append(f"policy-visible exclusion contract mismatch at decision {index}")

    combined = result.get("combined_elapsed_seconds")
    if not isinstance(combined, (int, float)) or not (0.0 < float(combined) <= 1800.0):
        errors.append("combined stress wall time violates the 30-minute ceiling")
    if result.get("runtime", {}).get("config", {}).get("turn_angle_degrees") != 15:
        errors.append("Habitat turn angle is not the official 15 degrees")
    if result.get("runtime", {}).get("config", {}).get("forward_step_m") != 0.25:
        errors.append("Habitat forward step is not the official 0.25 m")

    return _payload(args, errors)


def _payload(args: argparse.Namespace, errors: list[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "gate": GATE_ID,
        "recorded_at_utc": _timestamp(),
        "error_count": len(errors),
        "errors": errors,
        "manifest_path": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest) if args.manifest.is_file() else None,
        "result_path": str(args.result),
        "result_sha256": _sha256(args.result) if args.result.is_file() else None,
        "timings_path": str(args.timings),
        "timings_sha256": _sha256(args.timings) if args.timings.is_file() else None,
        "checkpoint_fully_hashed": args.checkpoint.is_file(),
        "timing_reconstruction_performed": args.timings.is_file(),
        "scope_reconstruction_performed": args.timings.is_file(),
    }


def main() -> int:
    args = _arguments()
    payload = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
