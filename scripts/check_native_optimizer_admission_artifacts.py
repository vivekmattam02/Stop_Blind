#!/usr/bin/env python3
"""Independently verify native-Qwen optimizer-admission artifacts."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.run_spark_native_optimizer_admission import (
    _load_dataset,
    _prerequisite_records,
    _protocol,
    _select_samples,
    _source_locks,
)
from spatialvlm.eval.action_decision import validate_action_decision_record
from spatialvlm.training.native_c0_gate import file_sha256, model_weight_locks
from spatialvlm.training.native_optimizer_admission import (
    NATIVE_OPTIMIZER_ADMISSION_GATE,
    NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
    NATIVE_OPTIMIZER_OBJECTIVE,
    learning_rate_multiplier,
    select_admitted_rate,
    summarize_calibration,
    training_sample_order,
    validate_protocol_record,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _canonical_sha256(payload: Any) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_native_optimizer_admission(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    suite: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    protocol = _protocol(args.profile)
    protocol_record = manifest.get("protocol")
    if not isinstance(protocol_record, dict):
        errors.append("manifest protocol is missing")
    else:
        errors.extend(validate_protocol_record(protocol_record))
        if protocol_record != protocol.to_record():
            errors.append("manifest protocol differs from the requested profile")
    if manifest.get("gate") != NATIVE_OPTIMIZER_ADMISSION_GATE:
        errors.append("manifest gate mismatch")
    if manifest.get("profile") != args.profile or manifest.get("run_id") != args.run_id:
        errors.append("manifest run/profile mismatch")
    runtime = manifest.get("runtime", {})
    if (
        runtime.get("docker_image_ref") != args.docker_image_ref
        or runtime.get("docker_image_id") != args.docker_image_id
    ):
        errors.append("manifest Docker identity differs from the verifier invocation")
    repo = args.repo.expanduser().resolve()
    try:
        expected_locks = _source_locks(repo)
    except Exception as exc:
        errors.append(f"source locks cannot reconstruct: {exc}")
        expected_locks = {}
    if manifest.get("source_locks") != expected_locks:
        errors.append("manifest source locks differ from the verifier checkout")
    if suite.get("source_locks") != expected_locks:
        errors.append("suite source locks differ from the verifier checkout")
    try:
        expected_prerequisites = _prerequisite_records(args)
    except Exception as exc:
        errors.append(f"prerequisites cannot reverify: {exc}")
        expected_prerequisites = {}
    if manifest.get("prerequisites") != expected_prerequisites:
        errors.append("manifest prerequisite records do not reconstruct")
    if suite.get("prerequisites") != expected_prerequisites:
        errors.append("suite prerequisite records do not reconstruct")

    if manifest.get("dataset", {}).get("episodes_sha256") != file_sha256(args.episodes):
        errors.append("episode release hash mismatch")
    if manifest.get("dataset", {}).get("ground_truth_sha256") != file_sha256(
        args.ground_truth
    ):
        errors.append("ground-truth release hash mismatch")
    try:
        dataset = _load_dataset(args)
        expected_samples = _select_samples(dataset, protocol)
    except Exception as exc:
        errors.append(f"dataset/sample reconstruction failed: {exc}")
        expected_samples = []
    if manifest.get("samples") != expected_samples:
        errors.append("manifest samples do not reconstruct")
    if manifest.get("sample_records_sha256") != _canonical_sha256(expected_samples):
        errors.append("manifest sample-record hash mismatch")
    expected_train_samples = [row for row in expected_samples if row.get("split") == "train"]
    try:
        expected_exposure_order = training_sample_order(expected_train_samples, protocol)
    except Exception as exc:
        errors.append(f"training exposure order cannot reconstruct: {exc}")
        expected_exposure_order = ()

    manifest_path = args.result_root / "manifest.json"
    cache_manifest_path = args.cache_root / "cache_manifest.json"
    if suite.get("manifest_sha256") != file_sha256(manifest_path):
        errors.append("suite manifest hash mismatch")
    if not cache_manifest_path.is_file():
        errors.append("cache manifest is missing")
        cache_manifest = {}
    else:
        cache_manifest = _load_json(cache_manifest_path)
        if suite.get("cache_manifest_sha256") != file_sha256(cache_manifest_path):
            errors.append("suite cache-manifest hash mismatch")
        if cache_manifest.get("manifest_sha256") != file_sha256(manifest_path):
            errors.append("cache manifest does not bind the frozen manifest")
        if cache_manifest.get("entry_count") != len(expected_samples):
            errors.append("cache entry count differs from frozen samples")
        for row in cache_manifest.get("entries", []):
            path = args.cache_root / str(row.get("relative_path"))
            if not path.is_file() or file_sha256(path) != row.get("sha256"):
                errors.append(f"cache entry missing or changed: {path}")
                break

    try:
        expected_weight_locks = model_weight_locks(args.model_snapshot)
    except Exception as exc:
        errors.append(f"model weights cannot be hashed: {exc}")
        expected_weight_locks = {}
    if suite.get("model_weight_locks") != expected_weight_locks:
        errors.append("suite model-weight locks mismatch")
    if suite.get("gate") != NATIVE_OPTIMIZER_ADMISSION_GATE:
        errors.append("suite gate mismatch")
    if suite.get("status") != "passed" or suite.get("profile") != args.profile:
        errors.append("suite did not pass the requested profile")
    if suite.get("protocol") != protocol.to_record():
        errors.append("suite protocol differs from deterministic reconstruction")
    if suite.get("runtime", {}).get("docker_image_id") != args.docker_image_id:
        errors.append("suite Docker image identity mismatch")
    supervision = suite.get("supervision", {})
    if supervision.get("supervised_token_count_per_decision") != 1:
        errors.append("suite does not declare exact one-token supervision")

    arm_results: list[dict[str, Any]] = []
    initial_hashes: set[str] = set()
    arm_index = suite.get("arm_index")
    if not isinstance(arm_index, list) or len(arm_index) != len(protocol.learning_rates):
        errors.append("suite arm index is incomplete")
        arm_index = []
    for expected_rate, index_row in zip(protocol.learning_rates, arm_index, strict=False):
        try:
            relative = str(index_row["relative_path"])
            result_path = args.result_root / relative
            result = _load_json(result_path)
        except Exception as exc:
            errors.append(f"arm result cannot load: {exc}")
            continue
        if file_sha256(result_path) != index_row.get("sha256"):
            errors.append(f"arm result hash mismatch: {result_path}")
        if float(result.get("learning_rate", float("nan"))) != expected_rate:
            errors.append(f"arm learning rate mismatch: {result_path}")
        if result.get("status") != "passed":
            errors.append(f"arm result did not pass: {result_path}")
        if result.get("completed_updates") != protocol.planned_optimizer_updates:
            errors.append(f"arm update count mismatch: {result_path}")
        if result.get("changed_update_count") != protocol.planned_optimizer_updates:
            errors.append(f"arm contains an unchanged update: {result_path}")
        if result.get("all_updates_finite") is not True:
            errors.append(f"arm update finiteness failed: {result_path}")
        initial_hashes.add(str(result.get("initial_trainable_state_sha256")))

        events_path = args.result_root / "arms" / str(result["arm_id"]) / "updates.jsonl"
        if not events_path.is_file() or file_sha256(events_path) != result.get("events_sha256"):
            errors.append(f"arm event file missing or changed: {events_path}")
            events: list[dict[str, Any]] = []
        else:
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if len(events) != protocol.planned_optimizer_updates:
            errors.append(f"arm event count mismatch: {events_path}")
        previous_hash = str(result.get("initial_trainable_state_sha256"))
        for expected_step, event in enumerate(events, start=1):
            if event.get("gate") != NATIVE_OPTIMIZER_ADMISSION_GATE:
                errors.append(f"event gate mismatch at {result['arm_id']}/{expected_step}")
            if event.get("objective") != NATIVE_OPTIMIZER_OBJECTIVE:
                errors.append(f"event objective mismatch at {result['arm_id']}/{expected_step}")
            if event.get("update_step") != expected_step:
                errors.append(f"event step mismatch at {result['arm_id']}/{expected_step}")
            if event.get("microbatch_count") != protocol.accumulation_steps:
                errors.append(f"event accumulation mismatch at {result['arm_id']}/{expected_step}")
            exposure_start = (expected_step - 1) * protocol.accumulation_steps
            exposure_end = exposure_start + protocol.accumulation_steps
            if (
                event.get("sample_exposure_start") != exposure_start
                or event.get("sample_exposure_end_exclusive") != exposure_end
                or event.get("epoch_index")
                != exposure_start // protocol.unique_train_decisions
            ):
                errors.append(
                    f"event exposure units mismatch at {result['arm_id']}/{expected_step}"
                )
            if event.get("before_trainable_state_sha256") != previous_hash:
                errors.append(
                    f"event trainable hash chain broke at {result['arm_id']}/{expected_step}"
                )
            previous_hash = str(event.get("after_trainable_state_sha256"))
            expected_multiplier = learning_rate_multiplier(expected_step, protocol)
            if not math.isclose(
                float(event.get("learning_rate_multiplier", float("nan"))),
                expected_multiplier,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                errors.append(f"event schedule mismatch at {result['arm_id']}/{expected_step}")
            if not math.isclose(
                float(event.get("applied_learning_rate", float("nan"))),
                expected_rate * expected_multiplier,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                errors.append(f"event applied LR mismatch at {result['arm_id']}/{expected_step}")
            if not math.isfinite(float(event.get("mean_restricted_action_ce", float("nan")))):
                errors.append(f"event CE is non-finite at {result['arm_id']}/{expected_step}")
            if not math.isfinite(float(event.get("update_to_weight_ratio", float("nan")))):
                errors.append(
                    f"event update ratio is non-finite at {result['arm_id']}/{expected_step}"
                )
            microbatches = event.get("microbatches")
            if (
                not isinstance(microbatches, list)
                or len(microbatches) != protocol.accumulation_steps
            ):
                errors.append(
                    f"event microbatch ledger mismatch at {result['arm_id']}/{expected_step}"
                )
            elif any(row.get("supervised_token_count") != 1 for row in microbatches):
                errors.append(
                    f"event supervision count mismatch at {result['arm_id']}/{expected_step}"
                )
            elif [row.get("sample_index") for row in microbatches] != list(
                expected_exposure_order[exposure_start:exposure_end]
            ):
                errors.append(f"event sample order mismatch at {result['arm_id']}/{expected_step}")
            gradients = event.get("gradient_report", {})
            if (
                gradients.get("all_gradients_finite") is not True
                or gradients.get("gradient_tensor_count")
                != gradients.get("trainable_tensor_count")
                or int(gradients.get("nonzero_gradient_tensor_count", 0)) <= 0
                or not math.isfinite(float(gradients.get("gradient_l2_norm", float("nan"))))
            ):
                errors.append(
                    f"event gradient diagnostics failed at {result['arm_id']}/{expected_step}"
                )
            update = event.get("update_report", {})
            if (
                update.get("all_updates_finite") is not True
                or int(update.get("changed_tensor_count", 0)) <= 0
                or not math.isfinite(float(update.get("update_l2_norm", float("nan"))))
            ):
                errors.append(
                    f"event update diagnostics failed at {result['arm_id']}/{expected_step}"
                )
            optimizer_state = event.get("optimizer_state")
            if not isinstance(optimizer_state, dict) or not optimizer_state:
                errors.append(
                    f"event optimizer state is missing at {result['arm_id']}/{expected_step}"
                )
            elif any(
                not isinstance(field, dict)
                or field.get("all_finite") is not True
                or field.get("nonfinite_element_count") != 0
                or int(field.get("tensor_count", 0)) <= 0
                for field in optimizer_state.values()
            ):
                errors.append(f"event optimizer state failed at {result['arm_id']}/{expected_step}")

        if previous_hash != result.get("final_trainable_state_sha256"):
            errors.append(f"arm final trainable hash mismatch: {result_path}")

        calibrations = result.get("calibrations")
        if not isinstance(calibrations, list) or [
            row.get("update_step") for row in calibrations
        ] != list(protocol.calibration_updates):
            errors.append(f"arm calibration cadence mismatch: {result_path}")
            calibrations = []
        for calibration in calibrations:
            decisions = calibration.get("per_decision")
            if (
                not isinstance(decisions, list)
                or len(decisions) != protocol.unique_calibration_decisions
            ):
                errors.append(f"calibration decision count mismatch: {result_path}")
                continue
            if any(row.get("supervised_token_count") != 1 for row in decisions):
                errors.append(f"calibration supervision count mismatch: {result_path}")
            for row in decisions:
                errors.extend(validate_action_decision_record(row.get("action_decision", {})))
            try:
                expected_summary = summarize_calibration(decisions)
            except Exception as exc:
                errors.append(f"calibration summary cannot recompute: {exc}")
            else:
                if calibration.get("summary") != expected_summary:
                    errors.append(f"calibration summary mismatch: {result_path}")

        expected_checkpoint_steps = set(protocol.checkpoint_updates)
        observed_checkpoint_steps = set()
        for checkpoint in result.get("checkpoints", []):
            path = Path(str(checkpoint.get("path")))
            if not path.is_file() or file_sha256(path) != checkpoint.get("sha256"):
                errors.append(f"checkpoint missing or changed: {path}")
                continue
            observed_checkpoint_steps.add(int(checkpoint.get("update_step", -1)))
        if observed_checkpoint_steps != expected_checkpoint_steps:
            errors.append(f"checkpoint cadence mismatch: {result_path}")
        arm_results.append(result)

    if len(initial_hashes) != 1 or suite.get(
        "identical_initial_trainable_state_sha256"
    ) not in initial_hashes:
        errors.append("learning-rate arms do not share one initial trainable state")
    if len(arm_results) == len(protocol.learning_rates):
        try:
            expected_selection = select_admitted_rate(arm_results, protocol)
        except Exception as exc:
            errors.append(f"suite selection cannot recompute: {exc}")
        else:
            if suite.get("selection") != expected_selection:
                errors.append("suite selection differs from independent recomputation")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("stress", "main"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--processor-result", type=Path, required=True)
    parser.add_argument("--processor-verification", type=Path, required=True)
    parser.add_argument("--sensitivity-result", type=Path, required=True)
    parser.add_argument("--sensitivity-verification", type=Path, required=True)
    parser.add_argument("--stress-suite", type=Path)
    parser.add_argument("--stress-verification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.repo = args.repo.expanduser().resolve()
    args.episodes = args.episodes.expanduser().resolve()
    args.ground_truth = args.ground_truth.expanduser().resolve()
    args.model_snapshot = args.model_snapshot.expanduser().resolve()
    args.result_root = args.result_root.expanduser().resolve()
    args.cache_root = args.cache_root.expanduser().resolve()
    manifest_path = args.result_root / "manifest.json"
    suite_path = args.result_root / "suite.json"
    manifest = _load_json(manifest_path)
    suite = _load_json(suite_path)
    errors = verify_native_optimizer_admission(args=args, manifest=manifest, suite=suite)
    verification = {
        "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
        "gate": "independent_native_qwen_optimizer_admission_verification_v1",
        "status": "passed" if not errors else "failed",
        "run_id": args.run_id,
        "profile": args.profile,
        "recorded_at_utc": _utc_now(),
        "error_count": len(errors),
        "errors": errors,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "suite_path": str(suite_path),
        "suite_sha256": file_sha256(suite_path),
        "checkpoint_root": str(args.cache_root.parent / "arms"),
        "dataset_reconstruction_performed": True,
        "all_model_weight_files_fully_hashed": True,
        "evidence_boundary": suite.get("evidence_boundary"),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite verification: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
