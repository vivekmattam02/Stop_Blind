#!/usr/bin/env python3
"""Independently reconstruct and verify the CR-025 native-Qwen sensitivity artifact."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
)
from spatialvlm.eval.action_decision import validate_action_decision_record
from spatialvlm.eval.native_sensitivity import (
    NATIVE_SENSITIVITY_CONDITIONS,
    NATIVE_SENSITIVITY_GATE_NAME,
    NATIVE_SENSITIVITY_SCHEMA_VERSION,
    NATIVE_SENSITIVITY_SOURCE_LOCK_PATHS,
    NativeSensitivityProtocol,
    select_native_sensitivity_samples,
    summarize_native_sensitivity,
)
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    file_sha256,
    model_weight_locks,
)

VERIFIER_SCHEMA_VERSION = 1
VERIFIER_GATE_NAME = "independent_native_qwen_c0_visual_sensitivity_verification_v1"
EXPECTED_DOCKER_IMAGE_ID = "sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_native_sensitivity_result(
    result: dict[str, Any],
    manifest: dict[str, Any],
    *,
    repo: Path,
    episodes: Path,
    ground_truth: Path,
    model_snapshot: Path,
) -> list[str]:
    """Return every independent source, data, objective, and metric error."""
    errors: list[str] = []
    protocol = NativeSensitivityProtocol()
    if manifest.get("schema_version") != NATIVE_SENSITIVITY_SCHEMA_VERSION:
        errors.append("sensitivity manifest schema mismatch")
    if manifest.get("gate") != NATIVE_SENSITIVITY_GATE_NAME or manifest.get("status") != "frozen":
        errors.append("sensitivity manifest gate/status mismatch")
    if manifest.get("protocol") != protocol.to_record():
        errors.append("sensitivity manifest protocol mismatch")
    if manifest.get("dataset", {}).get("episodes_sha256") != file_sha256(episodes):
        errors.append("sensitivity episodes hash mismatch")
    if manifest.get("dataset", {}).get("ground_truth_sha256") != file_sha256(ground_truth):
        errors.append("sensitivity ground-truth hash mismatch")
    dataset = R2RCEDataset.from_release(episodes, ground_truth, split=protocol.split)
    reconstructed = [
        sample.to_record() for sample in select_native_sensitivity_samples(dataset, protocol)
    ]
    if manifest.get("samples") != reconstructed:
        errors.append("sensitivity sample selection does not reconstruct")

    locks = manifest.get("source_locks")
    if not isinstance(locks, dict) or set(locks) != set(NATIVE_SENSITIVITY_SOURCE_LOCK_PATHS):
        errors.append("sensitivity source-lock path set mismatch")
    else:
        for relative in NATIVE_SENSITIVITY_SOURCE_LOCK_PATHS:
            path = repo / relative
            if not path.is_file() or file_sha256(path) != locks.get(relative):
                errors.append(f"sensitivity source-lock mismatch: {relative}")

    if result.get("schema_version") != NATIVE_SENSITIVITY_SCHEMA_VERSION:
        errors.append("sensitivity result schema mismatch")
    if result.get("gate") != NATIVE_SENSITIVITY_GATE_NAME or result.get("status") != "passed":
        errors.append("sensitivity result gate/status mismatch")
    if result.get("protocol") != protocol.to_record():
        errors.append("sensitivity result protocol mismatch")
    if result.get("source_locks") != locks:
        errors.append("sensitivity result/manifest source-lock mismatch")
    if result.get("runtime", {}).get("architecture") != "aarch64":
        errors.append("sensitivity result did not run on aarch64")
    if result.get("runtime", {}).get("gpu") != "NVIDIA GB10":
        errors.append("sensitivity result did not run on NVIDIA GB10")
    if result.get("runtime", {}).get("docker_image_id") != EXPECTED_DOCKER_IMAGE_ID:
        errors.append("sensitivity Docker image identity mismatch")
    model = result.get("model", {})
    if (
        model.get("model_id") != NATIVE_C0_MODEL_ID
        or model.get("model_revision") != NATIVE_C0_MODEL_REVISION
        or model.get("optimization_updates") != 0
        or model.get("initialization") != "fresh_seeded_public_pretrained_lora"
    ):
        errors.append("sensitivity model/initialization contract mismatch")
    try:
        if model.get("weight_locks") != model_weight_locks(model_snapshot):
            errors.append("sensitivity frozen model-weight locks mismatch")
    except Exception as exc:
        errors.append(f"sensitivity model-weight verification failed: {type(exc).__name__}: {exc}")
    if result.get("supervision") != {
        "schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
        "method": NATIVE_QWEN_SUPERVISION_METHOD,
        "supervised_token_count_per_decision": 1,
    }:
        errors.append("sensitivity top-level one-token objective mismatch")

    codebook = result.get("action_codebook")
    decisions = result.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != protocol.sample_count:
        errors.append("sensitivity decision count mismatch")
        return errors
    expected_by_index = {row["sample_index"]: row for row in reconstructed}
    for row in decisions:
        sample_index = row.get("sample_index")
        expected = expected_by_index.get(sample_index)
        if expected is None:
            errors.append("sensitivity decision has an invalid sample index")
            continue
        for key in (
            "episode_id",
            "trajectory_id",
            "scene_id",
            "target_action_id",
            "target_action",
            "donor_sample_index",
        ):
            if row.get(key) != expected.get(key):
                errors.append(f"sensitivity sample {sample_index} identity mismatch: {key}")
        conditions = row.get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != set(
            NATIVE_SENSITIVITY_CONDITIONS
        ):
            errors.append(f"sensitivity sample {sample_index} condition surface mismatch")
            continue
        for condition, decision in conditions.items():
            if not isinstance(decision, dict):
                errors.append(f"sensitivity sample {sample_index}/{condition} decision missing")
                continue
            decision_errors = validate_action_decision_record(
                decision, codebook_record=codebook
            )
            errors.extend(
                f"sensitivity sample {sample_index}/{condition}: {error}"
                for error in decision_errors
            )
            if (
                decision.get("supervision_schema_version")
                != NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION
                or decision.get("supervision_method") != NATIVE_QWEN_SUPERVISION_METHOD
                or decision.get("supervised_token_count") != 1
                or decision.get("supervised_token_position") is None
                or decision.get("supervised_token_id") is None
            ):
                errors.append(
                    f"sensitivity sample {sample_index}/{condition} one-token objective mismatch"
                )
            logits = decision.get("action_logits")
            if not isinstance(logits, dict) or set(logits) != {
                action.name for action in R2RCEAction
            }:
                errors.append(f"sensitivity sample {sample_index}/{condition} logits missing")
            elif not all(math.isfinite(float(value)) for value in logits.values()):
                errors.append(f"sensitivity sample {sample_index}/{condition} logits non-finite")
    try:
        expected_summary = summarize_native_sensitivity(decisions, protocol)
    except Exception as exc:
        errors.append(f"sensitivity summary reconstruction failed: {type(exc).__name__}: {exc}")
    else:
        if result.get("summary") != expected_summary:
            errors.append("sensitivity aggregate summary mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    errors = verify_native_sensitivity_result(
        result,
        manifest,
        repo=args.repo.resolve(),
        episodes=args.episodes.resolve(),
        ground_truth=args.ground_truth.resolve(),
        model_snapshot=args.model_snapshot.resolve(),
    )
    if result.get("manifest_sha256") != file_sha256(args.manifest):
        errors.append("sensitivity result/manifest SHA-256 linkage mismatch")
    payload = {
        "schema_version": VERIFIER_SCHEMA_VERSION,
        "gate": VERIFIER_GATE_NAME,
        "status": "passed" if not errors else "failed",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "error_count": len(errors),
        "errors": errors,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "result_path": str(args.result.resolve()),
        "result_sha256": file_sha256(args.result),
        "dataset_reconstruction_performed": True,
        "all_source_and_weight_files_fully_hashed": True,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
