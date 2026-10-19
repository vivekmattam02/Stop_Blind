#!/usr/bin/env python3
"""Independently verify the frozen native-Qwen C0 development result and checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from scripts.run_spark_native_c0_development import (
    DEVELOPMENT_VERIFIER_GATE,
    _expected_manifest,
    _load_json,
    _source_locks,
    summarize_decisions,
)
from spatialvlm.training.native_c0_development import (
    NATIVE_C0_DEVELOPMENT_GATE,
    NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION,
    NATIVE_C0_HISTORY,
    NATIVE_C0_OBJECTIVE,
    NativeC0DevelopmentProtocol,
    canonical_sha256,
    evaluate_existence_gate,
    select_checkpoint,
    training_sample_order,
    validate_partition_records,
    validate_protocol_record,
)
from spatialvlm.training.native_c0_gate import (
    file_sha256,
    model_weight_locks,
    tensor_state_sha256,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row is not an object: {path}")
            rows.append(payload)
    return rows


def _verify_events(
    events: list[dict[str, Any]],
    *,
    samples: list[dict[str, Any]],
    protocol: NativeC0DevelopmentProtocol,
    initial_sha256: str,
    final_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if len(events) != protocol.planned_optimizer_updates:
        return ["C0 update-event count differs from the unit ledger"]
    train = [row for row in samples if row["split"] == "train"]
    by_index = {int(row["sample_index"]): row for row in train}
    order = training_sample_order(samples, protocol)
    previous = initial_sha256
    for expected_step, event in enumerate(events, start=1):
        label = f"update {expected_step}"
        if (
            event.get("gate") != NATIVE_C0_DEVELOPMENT_GATE
            or event.get("history_contract") != NATIVE_C0_HISTORY
            or event.get("objective") != NATIVE_C0_OBJECTIVE
            or event.get("update_step") != expected_step
        ):
            errors.append(f"{label} identity mismatch")
        start = (expected_step - 1) * protocol.accumulation_steps
        end = start + protocol.accumulation_steps
        if (
            event.get("sample_exposure_start") != start
            or event.get("sample_exposure_end_exclusive") != end
            or event.get("microbatch_count") != protocol.accumulation_steps
            or event.get("accumulation_steps") != protocol.accumulation_steps
        ):
            errors.append(f"{label} exposure units mismatch")
        if event.get("before_trainable_state_sha256") != previous:
            errors.append(f"{label} trainable hash chain broke")
        after = event.get("after_trainable_state_sha256")
        if not isinstance(after, str) or len(after) != 64 or after == previous:
            errors.append(f"{label} after-update trainable hash is invalid")
        previous = str(after)
        if float(event.get("learning_rate", float("nan"))) != protocol.learning_rate:
            errors.append(f"{label} learning-rate mismatch")
        if not math.isfinite(float(event.get("elapsed_seconds", float("nan")))):
            errors.append(f"{label} elapsed time is invalid")
        gradients = event.get("gradient_report", {})
        updates = event.get("update_report", {})
        if (
            gradients.get("gradient_tensor_count") != gradients.get("trainable_tensor_count")
            or gradients.get("nonzero_gradient_tensor_count", 0) <= 0
            or gradients.get("all_gradients_finite") is not True
        ):
            errors.append(f"{label} gradient report is invalid")
        if updates.get("changed_tensor_count", 0) <= 0 or updates.get(
            "all_updates_finite"
        ) is not True:
            errors.append(f"{label} update report is invalid")
        microbatches = event.get("microbatches")
        if not isinstance(microbatches, list) or len(microbatches) != protocol.accumulation_steps:
            errors.append(f"{label} microbatch ledger mismatch")
            continue
        if [row.get("sample_index") for row in microbatches] != list(order[start:end]):
            errors.append(f"{label} sample order mismatch")
        for row in microbatches:
            sample = by_index.get(row.get("sample_index"))
            if sample is None:
                errors.append(f"{label} references unknown training sample")
                continue
            if any(
                row.get(name) != sample[name]
                for name in ("episode_id", "trajectory_id", "scene_id", "target_action")
            ):
                errors.append(f"{label} sample metadata mismatch")
            if row.get("supervised_token_count") != 1:
                errors.append(f"{label} does not supervise exactly one token")
            if not isinstance(row.get("supervised_token_id"), int):
                errors.append(f"{label} supervised token ID is invalid")
            ce = float(row.get("restricted_action_ce", float("nan")))
            if not math.isfinite(ce) or ce < 0.0 or ce > protocol.hard_max_restricted_ce:
                errors.append(f"{label} restricted CE is invalid")
        observed_mean = sum(float(row["restricted_action_ce"]) for row in microbatches) / len(
            microbatches
        )
        if not math.isclose(
            observed_mean,
            float(event.get("mean_restricted_action_ce", float("nan"))),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(f"{label} mean restricted CE does not reconstruct")
    if previous != final_sha256:
        errors.append("C0 final training hash does not close the update chain")
    return errors


def _verify_checkpoint(
    path: Path,
    record: dict[str, Any],
    *,
    protocol: NativeC0DevelopmentProtocol,
    manifest_sha256: str,
    cache_index_sha256: str,
) -> tuple[list[str], str | None]:
    errors: list[str] = []
    if not path.is_file():
        return [f"C0 checkpoint is missing: {path}"], None
    if record.get("sha256") != file_sha256(path) or record.get("bytes") != path.stat().st_size:
        errors.append(f"C0 checkpoint file identity mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("gate") != NATIVE_C0_DEVELOPMENT_GATE
        or payload.get("protocol") != protocol.to_record()
        or payload.get("manifest_sha256") != manifest_sha256
        or payload.get("cache_index_sha256") != cache_index_sha256
    ):
        errors.append(f"C0 checkpoint provenance mismatch: {path}")
    state = payload.get("trainable_state_dict")
    observed: str | None = None
    if not isinstance(state, dict) or not state:
        errors.append(f"C0 checkpoint trainable state is missing: {path}")
    else:
        observed = tensor_state_sha256(state)
        if observed != payload.get("trainable_state_sha256"):
            errors.append(f"C0 checkpoint internal trainable hash mismatch: {path}")
        if observed != record.get("trainable_state_sha256"):
            errors.append(f"C0 result/checkpoint trainable hash mismatch: {path}")
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        errors.append(f"C0 checkpoint optimizer state is missing: {path}")
    if not isinstance(payload.get("rng_state"), dict):
        errors.append(f"C0 checkpoint RNG state is missing: {path}")
    step = int(record.get("update_step", -1))
    if payload.get("cursor") != {
        "update_step": step,
        "sample_exposure_end_exclusive": step * protocol.accumulation_steps,
    }:
        errors.append(f"C0 checkpoint cursor mismatch: {path}")
    return errors, observed


def _verify_evaluation(
    record: dict[str, Any], expected_samples: list[dict[str, Any]], label: str
) -> list[str]:
    errors: list[str] = []
    rows = record.get("per_decision")
    if not isinstance(rows, list) or len(rows) != len(expected_samples):
        return [f"{label} decision count mismatch"]
    for observed, expected in zip(rows, expected_samples):
        if any(
            observed.get(name) != expected[name]
            for name in (
                "split",
                "sample_index",
                "episode_id",
                "trajectory_id",
                "scene_id",
                "target_action",
            )
        ):
            errors.append(f"{label} sample identity/order mismatch")
            break
        if observed.get("supervised_token_count") != 1:
            errors.append(f"{label} supervision count mismatch")
        ce = float(observed.get("restricted_action_ce", float("nan")))
        probability = float(observed.get("target_restricted_probability", float("nan")))
        if (
            not math.isfinite(ce)
            or not math.isfinite(probability)
            or probability <= 0.0
            or not math.isclose(ce, -math.log(probability), rel_tol=0.0, abs_tol=1e-12)
        ):
            errors.append(f"{label} CE/probability mismatch")
    if rows and record.get("summary") != summarize_decisions(rows):
        errors.append(f"{label} summary does not reconstruct")
    return errors


def verify_development(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    root = args.result_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    events_path = root / "updates.jsonl"
    for path in (manifest_path, result_path, events_path):
        if not path.is_file():
            errors.append(f"required C0 development artifact is missing: {path}")
    if errors:
        return errors
    manifest = _load_json(manifest_path, "development manifest")
    result = _load_json(result_path, "development result")
    expected, _ = _expected_manifest(args)
    if manifest != expected:
        errors.append("C0 development manifest does not independently reconstruct")
    protocol = NativeC0DevelopmentProtocol()
    errors.extend(validate_protocol_record(manifest.get("protocol", {})))
    try:
        validate_partition_records(manifest.get("samples", []), protocol=protocol)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"C0 partitions are invalid: {exc}")
    manifest_sha = file_sha256(manifest_path)
    if (
        result.get("gate") != NATIVE_C0_DEVELOPMENT_GATE
        or result.get("schema_version") != NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION
        or result.get("run_id") != args.run_id
        or result.get("status") not in {"passed", "not_demonstrated"}
        or result.get("manifest_sha256") != manifest_sha
        or result.get("completed_updates") != protocol.planned_optimizer_updates
        or result.get("protocol") != protocol.to_record()
        or result.get("updates_sha256") != file_sha256(events_path)
    ):
        errors.append("C0 development result identity/status/provenance mismatch")
    events = _read_jsonl(events_path)
    errors.extend(
        _verify_events(
            events,
            samples=manifest["samples"],
            protocol=protocol,
            initial_sha256=str(result.get("initial_trainable_state_sha256")),
            final_sha256=str(result.get("final_training_trainable_state_sha256")),
        )
    )
    calibrations = result.get("calibrations")
    if not isinstance(calibrations, list) or [
        row.get("update_step") for row in calibrations
    ] != list(protocol.calibration_updates):
        errors.append("C0 calibration cadence differs from the frozen protocol")
        calibrations = []
    selection_samples = [row for row in manifest["samples"] if row["split"] == "selection"]
    for calibration in calibrations:
        errors.extend(_verify_evaluation(calibration, selection_samples, "selection calibration"))
    if calibrations and result.get("calibrations_sha256") != canonical_sha256(calibrations):
        errors.append("C0 calibration hash mismatch")
    checkpoints = result.get("checkpoints")
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    if not isinstance(checkpoints, list) or [row.get("update_step") for row in checkpoints] != list(
        protocol.checkpoint_updates
    ):
        errors.append("C0 checkpoint cadence differs from the frozen protocol")
        checkpoints = []
    for checkpoint in checkpoints:
        step = int(checkpoint["update_step"])
        observed_errors, _ = _verify_checkpoint(
            checkpoint_root / str(checkpoint["relative_path"]),
            checkpoint,
            protocol=protocol,
            manifest_sha256=manifest_sha,
            cache_index_sha256=manifest["cache"]["cache_index_sha256"],
        )
        errors.extend(observed_errors)
    if calibrations:
        rebuilt_selection = select_checkpoint(calibrations)
        observed_selection = dict(result.get("selection", {}))
        selected_checkpoint = observed_selection.pop("checkpoint", None)
        if observed_selection != rebuilt_selection:
            errors.append("C0 checkpoint selection does not reconstruct")
        selected_step = int(rebuilt_selection["selected_update"])
        if not isinstance(selected_checkpoint, dict) or selected_checkpoint != next(
            (row for row in checkpoints if row["update_step"] == selected_step), None
        ):
            errors.append("C0 selected-checkpoint record mismatch")
    primary_samples = [
        row for row in manifest["samples"] if row["split"] == "primary_evaluation"
    ]
    diagnostic_samples = [
        row for row in manifest["samples"] if row["split"] == "balanced_diagnostic"
    ]
    primary = result.get("primary_evaluation", {})
    diagnostic = result.get("balanced_diagnostic", {})
    errors.extend(_verify_evaluation(primary, primary_samples, "primary evaluation"))
    errors.extend(_verify_evaluation(diagnostic, diagnostic_samples, "balanced diagnostic"))
    if primary.get("per_decision") and diagnostic.get("per_decision"):
        rebuilt_outcome = evaluate_existence_gate(
            primary_decisions=primary["per_decision"],
            diagnostic_decisions=diagnostic["per_decision"],
            protocol=protocol,
        )
        if result.get("existence_gate") != rebuilt_outcome:
            errors.append("C0 existence-gate outcome does not reconstruct")
        expected_status = "passed" if rebuilt_outcome["status"] == "passed" else "not_demonstrated"
        if result.get("status") != expected_status:
            errors.append("C0 result status disagrees with the conjunctive outcome")
    runtime = result.get("runtime", {})
    if (
        float(runtime.get("combined_elapsed_seconds", math.inf))
        > float(runtime.get("hard_total_seconds", -math.inf))
        or float(runtime.get("hard_total_seconds", -1)) != protocol.hard_total_hours * 3600
    ):
        errors.append("C0 run exceeded or changed the frozen combined hard cap")
    if result.get("source_locks") != _source_locks(args.repo.expanduser().resolve()):
        errors.append("C0 source locks changed after freeze")
    if result.get("model_weight_locks") != model_weight_locks(args.model_snapshot):
        errors.append("C0 model weight locks changed")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--measurements-result", type=Path, required=True)
    parser.add_argument("--measurements-verification", type=Path, required=True)
    parser.add_argument("--cache-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-object-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors = verify_development(args)
    root = args.result_root.expanduser().resolve()
    payload = {
        "schema_version": NATIVE_C0_DEVELOPMENT_SCHEMA_VERSION,
        "gate": DEVELOPMENT_VERIFIER_GATE,
        "recorded_at_utc": _utc_now(),
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors,
        "manifest_path": str(root / "manifest.json"),
        "manifest_sha256": file_sha256(root / "manifest.json")
        if (root / "manifest.json").is_file()
        else None,
        "result_path": str(root / "result.json"),
        "result_sha256": file_sha256(root / "result.json")
        if (root / "result.json").is_file()
        else None,
        "updates_path": str(root / "updates.jsonl"),
        "updates_sha256": file_sha256(root / "updates.jsonl")
        if (root / "updates.jsonl").is_file()
        else None,
        "dataset_reconstruction_performed": True,
        "all_checkpoint_files_fully_loaded_and_hashed": True,
        "outcome_reconstruction_performed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
