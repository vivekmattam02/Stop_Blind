#!/usr/bin/env python3
"""Independently verify the native-C0 inventory and causal-H4 benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from scripts.run_spark_native_development_measurements import (
    MEASUREMENTS_VERIFIER_GATE,
    _load_dataset,
    _prerequisites,
    _select_samples,
    _source_locks,
)
from scripts.run_spark_native_history_select import (
    _canonical_sha256,
    _validate_cache_entry,
)
from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.action_decision import validate_action_decision_record
from spatialvlm.training.native_c0_gate import (
    file_sha256,
    model_weight_locks,
    tensor_state_sha256,
)
from spatialvlm.training.native_development_measurements import (
    NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
    NATIVE_DEVELOPMENT_MEASUREMENTS_SCHEMA_VERSION,
    NATIVE_H4_HISTORY,
    NATIVE_H4_OBJECTIVE,
    NativeH4BenchmarkProtocol,
    h4_training_sample_order,
    summarize_h4_timings,
    validate_h4_protocol_record,
    validate_inventory_record,
)
from spatialvlm.training.native_optimizer_admission import summarize_calibration


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def verify_h4_event_records(
    events: list[dict[str, Any]],
    *,
    protocol: NativeH4BenchmarkProtocol,
    samples: list[dict[str, Any]],
    initial_trainable_sha256: str,
    final_trainable_sha256: str,
) -> list[str]:
    """Check exact units, ordering, supervision, timing, and trainable hash chain."""
    errors: list[str] = []
    if len(events) != protocol.planned_optimizer_updates:
        errors.append("H4 event count differs from the protocol")
    train = [row for row in samples if row.get("split") == "train"]
    train_by_index = {int(row["sample_index"]): row for row in train}
    expected_order = h4_training_sample_order(train, protocol)
    previous_hash = initial_trainable_sha256
    for expected_step, event in enumerate(events, start=1):
        label = f"update {expected_step}"
        if event.get("gate") != NATIVE_DEVELOPMENT_MEASUREMENTS_GATE:
            errors.append(f"{label} gate mismatch")
        if event.get("history_contract") != NATIVE_H4_HISTORY:
            errors.append(f"{label} is not causal H4")
        if event.get("objective") != NATIVE_H4_OBJECTIVE:
            errors.append(f"{label} objective mismatch")
        if event.get("update_step") != expected_step:
            errors.append(f"{label} step mismatch")
        start = (expected_step - 1) * protocol.accumulation_steps
        end = start + protocol.accumulation_steps
        if (
            event.get("sample_exposure_start") != start
            or event.get("sample_exposure_end_exclusive") != end
            or event.get("microbatch_count") != protocol.accumulation_steps
            or event.get("accumulation_steps") != protocol.accumulation_steps
        ):
            errors.append(f"{label} exposure/accumulation units mismatch")
        if event.get("before_trainable_state_sha256") != previous_hash:
            errors.append(f"{label} trainable hash chain broke")
        after_hash = event.get("after_trainable_state_sha256")
        if not isinstance(after_hash, str) or len(after_hash) != 64:
            errors.append(f"{label} after-update trainable hash is invalid")
        if after_hash == event.get("before_trainable_state_sha256"):
            errors.append(f"{label} changed no trainable bytes")
        previous_hash = str(after_hash)
        elapsed = float(event.get("elapsed_seconds", float("nan")))
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            errors.append(f"{label} elapsed time is invalid")
        expected_role = (
            "measured" if expected_step in protocol.timed_update_steps else "warmup_excluded"
        )
        if event.get("timing_role") != expected_role:
            errors.append(f"{label} timing role mismatch")
        if float(event.get("learning_rate", float("nan"))) != protocol.learning_rate:
            errors.append(f"{label} learning rate mismatch")
        if not math.isfinite(float(event.get("mean_restricted_action_ce", float("nan")))):
            errors.append(f"{label} restricted CE is non-finite")
        gradient_norm = float(event.get("global_gradient_norm_before_clip", float("nan")))
        if (
            not math.isfinite(gradient_norm)
            or gradient_norm <= 0.0
            or gradient_norm > protocol.hard_max_gradient_norm
        ):
            errors.append(f"{label} global gradient norm is invalid")
        gradient = event.get("gradient_report", {})
        update = event.get("update_report", {})
        if (
            gradient.get("gradient_tensor_count") != gradient.get("trainable_tensor_count")
            or gradient.get("nonzero_gradient_tensor_count", 0) <= 0
            or gradient.get("all_gradients_finite") is not True
        ):
            errors.append(f"{label} gradient report is invalid")
        if (
            update.get("changed_tensor_count", 0) <= 0
            or update.get("all_updates_finite") is not True
        ):
            errors.append(f"{label} update report is invalid")
        optimizer = event.get("optimizer_state", {})
        if not optimizer or any(row.get("all_finite") is not True for row in optimizer.values()):
            errors.append(f"{label} optimizer state is missing or non-finite")
        microbatches = event.get("microbatches")
        if not isinstance(microbatches, list) or len(microbatches) != protocol.accumulation_steps:
            errors.append(f"{label} microbatch ledger mismatch")
        else:
            if [row.get("sample_index") for row in microbatches] != list(expected_order[start:end]):
                errors.append(f"{label} sample order mismatch")
            if any(row.get("supervised_token_count") != 1 for row in microbatches):
                errors.append(f"{label} supervision count mismatch")
            for row in microbatches:
                sample_index = row.get("sample_index")
                expected_sample = train_by_index.get(sample_index)
                if expected_sample is None:
                    errors.append(f"{label} references an unknown training sample")
                    continue
                if (
                    row.get("episode_id") != expected_sample["episode_id"]
                    or row.get("trajectory_id") != expected_sample["trajectory_id"]
                    or row.get("target_action") != expected_sample["target_action"]
                ):
                    errors.append(f"{label} microbatch sample metadata mismatch")
                restricted_ce = float(row.get("restricted_action_ce", float("nan")))
                if (
                    not math.isfinite(restricted_ce)
                    or restricted_ce < 0.0
                    or restricted_ce > protocol.hard_max_restricted_ce
                ):
                    errors.append(f"{label} microbatch restricted CE is invalid")
                if not isinstance(row.get("action_token_position"), int):
                    errors.append(f"{label} microbatch action-token position is invalid")
                if not isinstance(row.get("supervised_token_id"), int):
                    errors.append(f"{label} microbatch supervised token ID is invalid")
            observed_mean = sum(
                float(row.get("restricted_action_ce", float("nan"))) for row in microbatches
            ) / len(microbatches)
            if not math.isclose(
                float(event.get("mean_restricted_action_ce", float("nan"))),
                observed_mean,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                errors.append(f"{label} mean restricted CE does not reconstruct")
    if previous_hash != final_trainable_sha256:
        errors.append("final trainable hash does not close the update chain")
    return errors


def verify_h4_checkpoint(
    path: Path,
    *,
    record: dict[str, Any],
    manifest_sha256: str,
    cache_manifest_sha256: str,
    protocol: NativeH4BenchmarkProtocol,
) -> list[str]:
    """Load and validate the measured checkpoint boundary."""
    errors: list[str] = []
    if not path.is_file():
        return ["H4 benchmark checkpoint is missing"]
    if record.get("sha256") != file_sha256(path) or record.get("bytes") != path.stat().st_size:
        errors.append("H4 benchmark checkpoint file identity mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return [*errors, f"H4 benchmark checkpoint cannot load: {exc}"]
    if payload.get("gate") != NATIVE_DEVELOPMENT_MEASUREMENTS_GATE:
        errors.append("H4 benchmark checkpoint gate mismatch")
    if payload.get("manifest_sha256") != manifest_sha256:
        errors.append("H4 benchmark checkpoint manifest link mismatch")
    if payload.get("cache_manifest_sha256") != cache_manifest_sha256:
        errors.append("H4 benchmark checkpoint cache link mismatch")
    if payload.get("protocol") != protocol.to_record():
        errors.append("H4 benchmark checkpoint protocol mismatch")
    cursor = payload.get("cursor", {})
    if cursor != {
        "update_step": protocol.planned_optimizer_updates,
        "sample_exposure_end_exclusive": protocol.unique_train_decisions,
    }:
        errors.append("H4 benchmark checkpoint cursor mismatch")
    state = payload.get("trainable_state_dict")
    if not isinstance(state, dict) or not state:
        errors.append("H4 benchmark checkpoint trainable state is missing")
    else:
        observed = tensor_state_sha256(state)
        if observed != payload.get("trainable_state_sha256"):
            errors.append("H4 benchmark checkpoint internal trainable hash mismatch")
        if observed != record.get("trainable_state_sha256"):
            errors.append("H4 benchmark result/checkpoint trainable hash mismatch")
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, dict) or not optimizer.get("state"):
        errors.append("H4 benchmark checkpoint optimizer state is missing")
    elif any(
        isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item())
        for state in optimizer["state"].values()
        if isinstance(state, dict)
        for value in state.values()
    ):
        errors.append("H4 benchmark checkpoint optimizer state is non-finite")
    if not isinstance(payload.get("rng_state"), dict):
        errors.append("H4 benchmark checkpoint RNG state is missing")
    return errors


def verify_native_development_measurements(args: argparse.Namespace) -> list[str]:
    """Fully reconstruct the compact scientific and runtime artifact surface."""
    errors: list[str] = []
    root = args.result_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    inventory_path = root / "inventory.json"
    result_path = root / "result.json"
    for path in (manifest_path, inventory_path, result_path):
        if not path.is_file():
            errors.append(f"required measurement artifact is missing: {path}")
    if errors:
        return errors
    manifest = _load_json(manifest_path)
    inventory = _load_json(inventory_path)
    result = _load_json(result_path)
    protocol = NativeH4BenchmarkProtocol()
    if manifest.get("gate") != NATIVE_DEVELOPMENT_MEASUREMENTS_GATE:
        errors.append("measurement manifest gate mismatch")
    if manifest.get("run_id") != args.run_id or result.get("run_id") != args.run_id:
        errors.append("measurement run identity mismatch")
    if (
        result.get("status") != "passed"
        or result.get("gate") != NATIVE_DEVELOPMENT_MEASUREMENTS_GATE
    ):
        errors.append("measurement result did not pass")
    protocol_record = manifest.get("protocol")
    if not isinstance(protocol_record, dict):
        errors.append("measurement protocol is missing")
    else:
        errors.extend(validate_h4_protocol_record(protocol_record))
    if protocol_record != protocol.to_record() or result.get("protocol") != protocol.to_record():
        errors.append("measurement protocol differs from deterministic reconstruction")
    if result.get("manifest_sha256") != file_sha256(manifest_path):
        errors.append("result manifest hash mismatch")
    if manifest.get("inventory_file_sha256") != file_sha256(inventory_path):
        errors.append("manifest inventory hash mismatch")
    if result.get("inventory_sha256") != file_sha256(inventory_path):
        errors.append("result inventory hash mismatch")

    try:
        expected_locks = _source_locks(args.repo.expanduser().resolve())
    except Exception as exc:
        errors.append(f"source locks cannot reconstruct: {exc}")
        expected_locks = {}
    if (
        manifest.get("source_locks") != expected_locks
        or result.get("source_locks") != expected_locks
    ):
        errors.append("measurement source locks do not reconstruct")
    try:
        expected_prerequisites = _prerequisites(args)
    except Exception as exc:
        errors.append(f"measurement prerequisites cannot reconstruct: {exc}")
        expected_prerequisites = {}
    if (
        manifest.get("prerequisites") != expected_prerequisites
        or result.get("prerequisites") != expected_prerequisites
    ):
        errors.append("measurement prerequisites do not reconstruct")
    if manifest.get("dataset", {}).get("episodes_sha256") != file_sha256(args.episodes):
        errors.append("measurement episode-release hash mismatch")
    if manifest.get("dataset", {}).get("ground_truth_sha256") != file_sha256(args.ground_truth):
        errors.append("measurement ground-truth hash mismatch")

    try:
        dataset = _load_dataset(args)
        expected_samples = _select_samples(dataset, protocol)
        errors.extend(
            validate_inventory_record(
                inventory,
                dataset,
                protocol=protocol.inventory_protocol,
            )
        )
    except Exception as exc:
        errors.append(f"measurement dataset reconstruction failed: {exc}")
        dataset = None
        expected_samples = []
    if manifest.get("samples") != expected_samples:
        errors.append("measurement samples do not reconstruct")
    if manifest.get("sample_records_sha256") != _canonical_sha256(expected_samples):
        errors.append("measurement sample hash mismatch")

    cache_manifest_path = args.cache_root / "cache_manifest.json"
    if not cache_manifest_path.is_file():
        errors.append("measurement cache manifest is missing")
        cache_manifest: dict[str, Any] = {}
    else:
        cache_manifest = _load_json(cache_manifest_path)
        if result.get("cache_manifest_sha256") != file_sha256(cache_manifest_path):
            errors.append("measurement cache-manifest hash mismatch")
        if cache_manifest.get("gate") != NATIVE_DEVELOPMENT_MEASUREMENTS_GATE:
            errors.append("measurement cache gate mismatch")
        if cache_manifest.get("manifest_sha256") != file_sha256(manifest_path):
            errors.append("measurement cache manifest link mismatch")
        entries = cache_manifest.get("entries", [])
        if len(entries) != len(expected_samples):
            errors.append("measurement cache entry count mismatch")
        else:
            for entry, sample in zip(entries, expected_samples):
                path = args.cache_root / str(entry.get("relative_path"))
                if not path.is_file() or file_sha256(path) != entry.get("sha256"):
                    errors.append(f"measurement cache entry missing or changed: {path}")
                    break
                try:
                    _validate_cache_entry(
                        path,
                        sample,
                        gate_name=NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
                    )
                except Exception as exc:
                    errors.append(f"measurement cache content invalid: {exc}")
                    break

    events_path = root / "updates.jsonl"
    if not events_path.is_file() or result.get("events_sha256") != file_sha256(events_path):
        errors.append("measurement event log is missing or changed")
        events: list[dict[str, Any]] = []
    else:
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    errors.extend(
        verify_h4_event_records(
            events,
            protocol=protocol,
            samples=expected_samples,
            initial_trainable_sha256=str(result.get("initial_trainable_state_sha256")),
            final_trainable_sha256=str(result.get("final_trainable_state_sha256")),
        )
    )
    timing = result.get("timing", {})
    try:
        expected_timing = summarize_h4_timings(
            events,
            protocol=protocol,
            model_load_seconds=float(timing["model_load_seconds"]),
            cache_prepare_seconds=float(timing["cache_prepare_seconds"]),
            calibration_seconds=float(timing["calibration_seconds"]),
            checkpoint_seconds=float(timing["checkpoint_seconds"]),
        )
        if timing != expected_timing:
            errors.append("measurement timing summary does not reconstruct")
    except Exception as exc:
        errors.append(f"measurement timing summary cannot reconstruct: {exc}")

    calibration = result.get("calibration", {})
    rows = calibration.get("per_decision", []) if isinstance(calibration, dict) else []
    try:
        expected_calibration = summarize_calibration(rows)
        if calibration.get("summary") != expected_calibration:
            errors.append("measurement calibration summary does not reconstruct")
        if calibration.get("update_step") != protocol.planned_optimizer_updates:
            errors.append("measurement calibration update mismatch")
        for row in rows:
            errors.extend(
                f"calibration sample {row.get('sample_index')}: {error}"
                for error in validate_action_decision_record(row.get("action_decision", {}))
            )
            if row.get("supervised_token_count") != 1:
                errors.append("measurement calibration supervision count mismatch")
                break
    except Exception as exc:
        errors.append(f"measurement calibration cannot reconstruct: {exc}")
    if len(rows) != protocol.unique_calibration_decisions:
        errors.append("measurement calibration decision count mismatch")
    if {row.get("target_action") for row in rows} != {action.name for action in R2RCEAction}:
        errors.append("measurement calibration lacks an official target action")
    if any(
        sum(row.get("target_action") == action.name for row in rows)
        != protocol.calibration_samples_per_action
        for action in R2RCEAction
    ):
        errors.append("measurement calibration is not exactly action-balanced")

    checkpoint_record = result.get("checkpoint", {})
    if checkpoint_record.get("trainable_state_sha256") != result.get(
        "final_trainable_state_sha256"
    ):
        errors.append("measurement result/checkpoint does not close the trainable hash chain")
    checkpoint_path = args.checkpoint_root / str(checkpoint_record.get("relative_path"))
    errors.extend(
        verify_h4_checkpoint(
            checkpoint_path,
            record=checkpoint_record,
            manifest_sha256=file_sha256(manifest_path),
            cache_manifest_sha256=(
                file_sha256(cache_manifest_path) if cache_manifest_path.is_file() else ""
            ),
            protocol=protocol,
        )
    )
    try:
        expected_weights = model_weight_locks(args.model_snapshot)
    except Exception as exc:
        errors.append(f"measurement model weights cannot be hashed: {exc}")
        expected_weights = {}
    if result.get("model_weight_locks") != expected_weights:
        errors.append("measurement model-weight locks mismatch")
    runtime = result.get("runtime", {})
    if runtime.get("docker_image_id") != args.docker_image_id:
        errors.append("measurement runtime Docker identity mismatch")
    if runtime.get("peak_allocated_bytes", 0) <= 0 or runtime.get("peak_reserved_bytes", 0) <= 0:
        errors.append("measurement peak accelerator memory is missing")
    supervision = result.get("supervision", {})
    if supervision.get("supervised_token_count_per_decision") != 1:
        errors.append("measurement result does not declare one-token supervision")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--prior-failure", type=Path, required=True)
    parser.add_argument("--processor-result", type=Path, required=True)
    parser.add_argument("--processor-verification", type=Path, required=True)
    parser.add_argument("--optimizer-suite", type=Path, required=True)
    parser.add_argument("--optimizer-verification", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors: list[str]
    try:
        errors = verify_native_development_measurements(args)
    except Exception as exc:
        errors = [f"independent verifier crashed: {exc}"]
    output = args.output.expanduser().resolve()
    record = {
        "schema_version": NATIVE_DEVELOPMENT_MEASUREMENTS_SCHEMA_VERSION,
        "gate": MEASUREMENTS_VERIFIER_GATE,
        "run_id": args.run_id,
        "status": "passed" if not errors else "failed",
        "recorded_at_utc": _utc_now(),
        "error_count": len(errors),
        "errors": errors,
        "inventory_path": str((args.result_root / "inventory.json").resolve()),
        "inventory_sha256": (
            file_sha256(args.result_root / "inventory.json")
            if (args.result_root / "inventory.json").is_file()
            else None
        ),
        "manifest_path": str((args.result_root / "manifest.json").resolve()),
        "manifest_sha256": (
            file_sha256(args.result_root / "manifest.json")
            if (args.result_root / "manifest.json").is_file()
            else None
        ),
        "result_path": str((args.result_root / "result.json").resolve()),
        "result_sha256": (
            file_sha256(args.result_root / "result.json")
            if (args.result_root / "result.json").is_file()
            else None
        ),
    }
    _write = json.dumps(record, indent=2, sort_keys=True) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(_write, encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
