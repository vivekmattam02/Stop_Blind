#!/usr/bin/env python3
"""Independently reconstruct the disjoint native-Qwen sensitivity-v2 artifact."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.run_spark_native_sensitivity import EXPECTED_DOCKER_IMAGE_ID
from scripts.run_spark_native_sensitivity_v2 import (
    processor_prerequisite_record,
    v1_prerequisite_record,
)
from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
)
from spatialvlm.eval.action_decision import validate_action_decision_record
from spatialvlm.eval.native_sensitivity_v2 import (
    NATIVE_SENSITIVITY_V2_CONDITIONS,
    NATIVE_SENSITIVITY_V2_GATE_NAME,
    NATIVE_SENSITIVITY_V2_SCHEMA_VERSION,
    NATIVE_SENSITIVITY_V2_SOURCE_LOCK_PATHS,
    NativeSensitivityV2Protocol,
    select_native_sensitivity_v2_samples,
    summarize_native_sensitivity_v2,
)
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    file_sha256,
    model_weight_locks,
)

VERIFIER_SCHEMA_VERSION = 2
VERIFIER_GATE_NAME = "independent_native_qwen_c0_visual_sensitivity_verification_v2"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def verify_native_sensitivity_v2_result(
    result: dict[str, Any],
    manifest: dict[str, Any],
    *,
    repo: Path,
    episodes: Path,
    ground_truth: Path,
    model_snapshot: Path,
    processor_result: Path,
    processor_verification: Path,
    v1_manifest: Path,
    v1_result: Path,
    v1_verification: Path,
) -> list[str]:
    """Return every independent prerequisite, data, source, objective, and metric error."""

    errors: list[str] = []
    protocol = NativeSensitivityV2Protocol()
    try:
        expected_processor = processor_prerequisite_record(
            processor_result, processor_verification
        )
    except Exception as exc:
        errors.append(
            f"sensitivity-v2 processor prerequisite failed: {type(exc).__name__}: {exc}"
        )
        expected_processor = None
    try:
        expected_v1, excluded = v1_prerequisite_record(
            v1_manifest, v1_result, v1_verification
        )
    except Exception as exc:
        errors.append(f"sensitivity-v2 v1 prerequisite failed: {type(exc).__name__}: {exc}")
        expected_v1 = None
        excluded = set()

    if manifest.get("schema_version") != NATIVE_SENSITIVITY_V2_SCHEMA_VERSION:
        errors.append("sensitivity-v2 manifest schema mismatch")
    if (
        manifest.get("gate") != NATIVE_SENSITIVITY_V2_GATE_NAME
        or manifest.get("status") != "frozen"
    ):
        errors.append("sensitivity-v2 manifest gate/status mismatch")
    if manifest.get("protocol") != protocol.to_record():
        errors.append("sensitivity-v2 manifest protocol mismatch")
    if manifest.get("processor_prerequisite") != expected_processor:
        errors.append("sensitivity-v2 processor prerequisite record mismatch")
    if manifest.get("v1_discovery_prerequisite") != expected_v1:
        errors.append("sensitivity-v2 v1 prerequisite record mismatch")
    if manifest.get("dataset", {}).get("episodes_sha256") != file_sha256(episodes):
        errors.append("sensitivity-v2 episodes hash mismatch")
    if manifest.get("dataset", {}).get("ground_truth_sha256") != file_sha256(
        ground_truth
    ):
        errors.append("sensitivity-v2 ground-truth hash mismatch")

    reconstructed: list[dict[str, Any]] = []
    if excluded:
        try:
            dataset = R2RCEDataset.from_release(
                episodes, ground_truth, split=protocol.split
            )
            reconstructed = [
                sample.to_record()
                for sample in select_native_sensitivity_v2_samples(
                    dataset,
                    excluded_trajectory_ids=excluded,
                    protocol=protocol,
                )
            ]
        except Exception as exc:
            errors.append(
                f"sensitivity-v2 sample reconstruction failed: {type(exc).__name__}: {exc}"
            )
        else:
            if manifest.get("samples") != reconstructed:
                errors.append("sensitivity-v2 sample selection does not reconstruct")
            selected = {str(row["trajectory_id"]) for row in reconstructed}
            if selected & excluded:
                errors.append("sensitivity-v2 sample selection overlaps v1 trajectories")

    locks = manifest.get("source_locks")
    if not isinstance(locks, dict) or set(locks) != set(
        NATIVE_SENSITIVITY_V2_SOURCE_LOCK_PATHS
    ):
        errors.append("sensitivity-v2 source-lock path set mismatch")
    else:
        for relative in NATIVE_SENSITIVITY_V2_SOURCE_LOCK_PATHS:
            path = repo / relative
            if not path.is_file() or file_sha256(path) != locks.get(relative):
                errors.append(f"sensitivity-v2 source-lock mismatch: {relative}")

    if result.get("schema_version") != NATIVE_SENSITIVITY_V2_SCHEMA_VERSION:
        errors.append("sensitivity-v2 result schema mismatch")
    if result.get("gate") != NATIVE_SENSITIVITY_V2_GATE_NAME or result.get(
        "status"
    ) != "passed":
        errors.append("sensitivity-v2 result gate/status mismatch")
    if result.get("protocol") != protocol.to_record():
        errors.append("sensitivity-v2 result protocol mismatch")
    if result.get("processor_prerequisite") != expected_processor:
        errors.append("sensitivity-v2 result processor prerequisite mismatch")
    if result.get("v1_discovery_prerequisite") != expected_v1:
        errors.append("sensitivity-v2 result v1 prerequisite mismatch")
    if result.get("source_locks") != locks:
        errors.append("sensitivity-v2 result/manifest source-lock mismatch")
    runtime = result.get("runtime", {})
    if runtime.get("architecture") != "aarch64":
        errors.append("sensitivity-v2 result did not run on aarch64")
    if runtime.get("gpu") != "NVIDIA GB10":
        errors.append("sensitivity-v2 result did not run on NVIDIA GB10")
    if runtime.get("docker_image_id") != EXPECTED_DOCKER_IMAGE_ID:
        errors.append("sensitivity-v2 Docker image identity mismatch")

    model = result.get("model", {})
    if (
        model.get("model_id") != NATIVE_C0_MODEL_ID
        or model.get("model_revision") != NATIVE_C0_MODEL_REVISION
        or model.get("optimization_updates") != 0
        or model.get("initialization") != "fresh_seeded_public_pretrained_lora"
    ):
        errors.append("sensitivity-v2 model/initialization contract mismatch")
    try:
        if model.get("weight_locks") != model_weight_locks(model_snapshot):
            errors.append("sensitivity-v2 frozen model-weight locks mismatch")
    except Exception as exc:
        errors.append(
            f"sensitivity-v2 model-weight verification failed: {type(exc).__name__}: {exc}"
        )
    if result.get("supervision") != {
        "schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
        "method": NATIVE_QWEN_SUPERVISION_METHOD,
        "supervised_token_count_per_decision": 1,
    }:
        errors.append("sensitivity-v2 top-level one-token objective mismatch")

    codebook = result.get("action_codebook")
    decisions = result.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != protocol.sample_count:
        errors.append("sensitivity-v2 decision count mismatch")
        return errors
    expected_by_index = {row["sample_index"]: row for row in reconstructed}
    for row in decisions:
        sample_index = row.get("sample_index")
        expected = expected_by_index.get(sample_index)
        if expected is None:
            errors.append("sensitivity-v2 decision has an invalid sample index")
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
                errors.append(f"sensitivity-v2 sample {sample_index} identity mismatch: {key}")
        donor = expected_by_index.get(expected["donor_sample_index"])
        if (
            donor is None
            or row.get("donor_episode_id") != donor.get("episode_id")
            or row.get("donor_target_action") != donor.get("target_action")
        ):
            errors.append(f"sensitivity-v2 sample {sample_index} donor identity mismatch")
        for key in ("clean_rgb_sha256", "donor_rgb_sha256"):
            digest = row.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                errors.append(f"sensitivity-v2 sample {sample_index} invalid {key}")
        conditions = row.get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != set(
            NATIVE_SENSITIVITY_V2_CONDITIONS
        ):
            errors.append(f"sensitivity-v2 sample {sample_index} condition surface mismatch")
            continue
        for condition, decision in conditions.items():
            if not isinstance(decision, dict):
                errors.append(
                    f"sensitivity-v2 sample {sample_index}/{condition} decision missing"
                )
                continue
            errors.extend(
                f"sensitivity-v2 sample {sample_index}/{condition}: {error}"
                for error in validate_action_decision_record(
                    decision, codebook_record=codebook
                )
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
                    f"sensitivity-v2 sample {sample_index}/{condition} one-token mismatch"
                )
            logits = decision.get("action_logits")
            if not isinstance(logits, dict) or set(logits) != {
                action.name for action in R2RCEAction
            }:
                errors.append(
                    f"sensitivity-v2 sample {sample_index}/{condition} logits missing"
                )
            elif not all(math.isfinite(float(value)) for value in logits.values()):
                errors.append(
                    f"sensitivity-v2 sample {sample_index}/{condition} logits non-finite"
                )
    try:
        expected_summary = summarize_native_sensitivity_v2(decisions, protocol)
    except Exception as exc:
        errors.append(
            f"sensitivity-v2 summary reconstruction failed: {type(exc).__name__}: {exc}"
        )
    else:
        if result.get("summary") != expected_summary:
            errors.append("sensitivity-v2 aggregate summary mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--processor-result", type=Path, required=True)
    parser.add_argument("--processor-verification", type=Path, required=True)
    parser.add_argument("--v1-manifest", type=Path, required=True)
    parser.add_argument("--v1-result", type=Path, required=True)
    parser.add_argument("--v1-verification", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    errors = verify_native_sensitivity_v2_result(
        result,
        manifest,
        repo=args.repo.resolve(),
        episodes=args.episodes.resolve(),
        ground_truth=args.ground_truth.resolve(),
        model_snapshot=args.model_snapshot.resolve(),
        processor_result=args.processor_result.resolve(),
        processor_verification=args.processor_verification.resolve(),
        v1_manifest=args.v1_manifest.resolve(),
        v1_result=args.v1_result.resolve(),
        v1_verification=args.v1_verification.resolve(),
    )
    if result.get("manifest_sha256") != file_sha256(args.manifest):
        errors.append("sensitivity-v2 result/manifest SHA-256 linkage mismatch")
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
        "v1_manifest_sha256": file_sha256(args.v1_manifest),
        "v1_result_sha256": file_sha256(args.v1_result),
        "v1_verification_sha256": file_sha256(args.v1_verification),
        "processor_result_sha256": file_sha256(args.processor_result),
        "processor_verification_sha256": file_sha256(args.processor_verification),
        "dataset_reconstruction_performed": True,
        "all_source_and_weight_files_fully_hashed": True,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
