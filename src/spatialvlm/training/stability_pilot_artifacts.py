"""Independent integrity checks for optimization/stability pilot evidence."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.eval.action_decision_artifacts import ACTION_DECISION_SOURCE_LOCK_PATHS
from spatialvlm.training.real_batch_gate import file_sha256
from spatialvlm.training.sft import FREEZE_GROUP_PATTERNS
from spatialvlm.training.stability_pilot import (
    STABILITY_PILOT_GATE_NAME,
    STABILITY_PILOT_SCHEMA_VERSION,
    StabilityPilotProtocol,
    select_pilot_checkpoint,
    select_pilot_samples,
)

STABILITY_PILOT_VERIFIER_SCHEMA_VERSION = 1
STABILITY_PILOT_VERIFIER_GATE_NAME = "independent_stability_pilot_artifact_verification_v1"
STABILITY_PILOT_SOURCE_LOCK_PATHS = tuple(
    dict.fromkeys(
        (
            *ACTION_DECISION_SOURCE_LOCK_PATHS,
            "docs/SPARK_STABILITY_PILOT.md",
            "pyproject.toml",
            "scripts/check_stability_pilot_artifacts.py",
            "scripts/run_spark_stability_pilot.py",
            "scripts/run_spark_stability_pilot.sh",
            "scripts/start_spark_stability_pilot.sh",
            "src/spatialvlm/training/stability_pilot.py",
            "src/spatialvlm/training/stability_pilot_artifacts.py",
        )
    )
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    repo: Path,
    dataset: R2RCEDataset | None,
) -> list[str]:
    errors: list[str] = []
    protocol = StabilityPilotProtocol()
    if manifest.get("schema_version") != STABILITY_PILOT_SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    if manifest.get("gate") != STABILITY_PILOT_GATE_NAME:
        errors.append("manifest gate mismatch")
    if manifest.get("status") != "passed":
        errors.append("manifest status is not passed")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"stability_pilot_v[1-9][0-9]*", run_id) is None:
        errors.append("manifest run_id is missing or invalid")
    if manifest.get("protocol") != protocol.to_record():
        errors.append("manifest protocol does not match the frozen protocol")

    preflights = manifest.get("preflights")
    expected_preflights = {
        "processor": "spark_real_batch_checkpoint_processor_preflight_v1",
        "action": "spark_action_decision_tokenizer_preflight_v1",
    }
    if not isinstance(preflights, dict) or set(preflights) != set(expected_preflights):
        errors.append("manifest preflight index mismatch")
    else:
        evidence_root = manifest_path.resolve().parent
        for label, expected_gate in expected_preflights.items():
            row = preflights[label]
            relative = row.get("relative_path") if isinstance(row, dict) else None
            if not isinstance(relative, str):
                errors.append(f"{label} preflight relative path is missing")
                continue
            path = (evidence_root / relative).resolve()
            if not path.is_relative_to(evidence_root):
                errors.append(f"{label} preflight path escapes the evidence root")
            elif not path.is_file():
                errors.append(f"{label} preflight file is missing")
            else:
                if file_sha256(path) != row.get("sha256"):
                    errors.append(f"{label} preflight SHA-256 mismatch")
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    errors.append(f"{label} preflight is invalid JSON")
                else:
                    if record.get("status") != "passed" or record.get("gate") != expected_gate:
                        errors.append(f"{label} preflight gate/status mismatch")

    locks = manifest.get("source_locks")
    expected_paths = set(STABILITY_PILOT_SOURCE_LOCK_PATHS)
    if not isinstance(locks, dict) or set(locks) != expected_paths:
        errors.append("manifest source-lock path set mismatch")
    else:
        for relative, expected_hash in locks.items():
            path = repo / relative
            if not path.is_file():
                errors.append(f"locked source is missing: {relative}")
            elif file_sha256(path) != expected_hash:
                errors.append(f"locked source hash mismatch: {relative}")

    rows = manifest.get("samples")
    if not isinstance(rows, list):
        errors.append("manifest samples are missing")
        return errors
    expected_count = protocol.train_sample_count + protocol.calibration_sample_count
    if len(rows) != expected_count:
        errors.append("manifest sample count mismatch")
    trajectories = [row.get("trajectory_id") for row in rows if isinstance(row, dict)]
    if len(trajectories) != len(set(trajectories)):
        errors.append("manifest reuses a trajectory across pilot samples")
    for split, per_action in (
        ("train", protocol.train_samples_per_action),
        ("calibration", protocol.calibration_samples_per_action),
    ):
        for action in R2RCEAction:
            count = sum(
                isinstance(row, dict)
                and row.get("split") == split
                and row.get("target_action_id") == int(action)
                and row.get("target_action") == action.name
                for row in rows
            )
            if count != per_action:
                errors.append(f"manifest {split}/{action.name} sample count mismatch")
    if dataset is not None:
        expected_rows = [sample.to_record() for sample in select_pilot_samples(dataset, protocol)]
        if rows != expected_rows:
            errors.append("manifest samples do not match deterministic release reconstruction")
        release = manifest.get("release_manifest")
        observed_release = (
            dataset.release_manifest.to_dict() if dataset.release_manifest is not None else None
        )
        if release != observed_release:
            errors.append("manifest release identity does not match supplied dataset")
    return errors


def _validate_gradient_groups(step: dict[str, Any], errors: list[str], label: str) -> None:
    gradients = step.get("gradients_by_group")
    if not isinstance(gradients, dict):
        errors.append(f"{label}: gradient groups are missing")
        return
    for group in FREEZE_GROUP_PATTERNS:
        row = gradients.get(group)
        if not isinstance(row, dict):
            errors.append(f"{label}: {group} gradient row is missing")
            continue
        if row.get("trainable_tensor_count", 0) <= 0:
            errors.append(f"{label}: {group} has no trainable tensors")
        if row.get("gradient_tensor_count") != row.get("trainable_tensor_count"):
            errors.append(f"{label}: {group} has missing gradients")
        if not row.get("all_gradients_finite") or not _finite(row.get("gradient_l2_norm")):
            errors.append(f"{label}: {group} gradients are non-finite")
        elif float(row["gradient_l2_norm"]) <= 0.0:
            errors.append(f"{label}: {group} gradient norm is non-positive")


def _validate_update_audit(step: dict[str, Any], errors: list[str], label: str) -> None:
    updates = step.get("updates_by_group")
    if not isinstance(updates, dict):
        errors.append(f"{label}: required update audit is missing")
        return
    for group in FREEZE_GROUP_PATTERNS:
        row = updates.get(group)
        if not isinstance(row, dict):
            errors.append(f"{label}: {group} update row is missing")
            continue
        if not row.get("all_updates_finite") or not _finite(row.get("update_l2_norm")):
            errors.append(f"{label}: {group} updates are non-finite")
        elif float(row["update_l2_norm"]) <= 0.0:
            errors.append(f"{label}: {group} update norm is non-positive")


def _validate_arm(
    arm: dict[str, Any],
    *,
    arm_id: str,
    manifest_sha256: str,
    checkpoint_root: Path,
) -> list[str]:
    errors: list[str] = []
    protocol = StabilityPilotProtocol()
    expected_rate = {protocol.arm_id(rate): rate for rate in protocol.learning_rates}.get(arm_id)
    if expected_rate is None:
        return [f"unknown arm ID: {arm_id}"]
    if arm.get("schema_version") != STABILITY_PILOT_SCHEMA_VERSION:
        errors.append(f"{arm_id}: schema_version mismatch")
    if arm.get("gate") != STABILITY_PILOT_GATE_NAME or arm.get("status") != "passed":
        errors.append(f"{arm_id}: gate/status mismatch")
    if arm.get("arm_id") != arm_id or arm.get("learning_rate") != expected_rate:
        errors.append(f"{arm_id}: identity or learning-rate mismatch")
    if arm.get("manifest_sha256") != manifest_sha256:
        errors.append(f"{arm_id}: manifest hash mismatch")
    if arm.get("protocol") != protocol.to_record():
        errors.append(f"{arm_id}: protocol mismatch")
    initial_hash = arm.get("initial_trainable_state_sha256")
    if not isinstance(initial_hash, str) or re.fullmatch(r"[0-9a-f]{64}", initial_hash) is None:
        errors.append(f"{arm_id}: initial trainable-state hash is invalid")

    training = arm.get("training")
    steps = training.get("steps") if isinstance(training, dict) else None
    if not isinstance(steps, list) or len(steps) != protocol.updates_per_arm:
        errors.append(f"{arm_id}: training step count mismatch")
        steps = []
    expected_audits = {1, *protocol.checkpoint_steps}
    for expected_step, row in enumerate(steps, start=1):
        label = f"{arm_id}/step-{expected_step}"
        if not isinstance(row, dict) or row.get("step") != expected_step:
            errors.append(f"{label}: step identity mismatch")
            continue
        if not _finite(row.get("target_token_nll")):
            errors.append(f"{label}: target token NLL is non-finite")
        elif float(row["target_token_nll"]) > protocol.hard_max_token_nll:
            errors.append(f"{label}: target token NLL exceeds hard limit")
        if not _finite(row.get("global_gradient_norm_before_clip")):
            errors.append(f"{label}: global gradient norm is non-finite")
        elif float(row["global_gradient_norm_before_clip"]) > protocol.hard_max_gradient_norm:
            errors.append(f"{label}: global gradient norm exceeds hard limit")
        if row.get("peak_allocated_bytes", 0) > protocol.hard_max_peak_allocated_bytes:
            errors.append(f"{label}: peak allocated memory exceeds hard limit")
        _validate_gradient_groups(row, errors, label)
        if expected_step in expected_audits:
            _validate_update_audit(row, errors, label)
        elif row.get("updates_by_group") is not None:
            errors.append(f"{label}: unexpected update audit outside preregistered steps")

    calibration = arm.get("calibration")
    if not isinstance(calibration, list) or [row.get("step") for row in calibration] != list(
        protocol.calibration_steps
    ):
        errors.append(f"{arm_id}: calibration step schedule mismatch")
        calibration = []
    for row in calibration:
        step = row.get("step")
        if row.get("sample_count") != protocol.calibration_sample_count:
            errors.append(f"{arm_id}/calibration-{step}: sample count mismatch")
        if not _finite(row.get("mean_target_token_nll")):
            errors.append(f"{arm_id}/calibration-{step}: mean target NLL is non-finite")
        samples = row.get("samples")
        if not isinstance(samples, list) or len(samples) != protocol.calibration_sample_count:
            errors.append(f"{arm_id}/calibration-{step}: sample records are incomplete")

    checkpoints = arm.get("checkpoints")
    if not isinstance(checkpoints, list) or [row.get("step") for row in checkpoints] != list(
        protocol.checkpoint_steps
    ):
        errors.append(f"{arm_id}: checkpoint schedule mismatch")
        checkpoints = []
    root = checkpoint_root.resolve()
    for row in checkpoints:
        relative = row.get("relative_path")
        if not isinstance(relative, str):
            errors.append(f"{arm_id}: checkpoint relative path is missing")
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            errors.append(f"{arm_id}: checkpoint path escapes checkpoint root")
        elif not path.is_file():
            errors.append(f"{arm_id}: checkpoint file is missing: {relative}")
        else:
            if path.stat().st_size != row.get("bytes"):
                errors.append(f"{arm_id}: checkpoint byte-size mismatch: {relative}")
            if file_sha256(path) != row.get("sha256"):
                errors.append(f"{arm_id}: checkpoint SHA-256 mismatch: {relative}")
        if not row.get("payload_trainable_hash_verified"):
            errors.append(f"{arm_id}: checkpoint payload hash was not verified: {relative}")
    return errors


def verify_stability_pilot(
    suite: dict[str, Any],
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    arms_root: Path,
    checkpoint_root: Path,
    repo: Path,
    dataset: R2RCEDataset | None = None,
) -> list[str]:
    """Return all integrity errors for a complete optimization/stability pilot."""
    errors = _validate_manifest(
        manifest,
        manifest_path=manifest_path,
        repo=repo,
        dataset=dataset,
    )
    protocol = StabilityPilotProtocol()
    manifest_hash = file_sha256(manifest_path)
    if suite.get("schema_version") != STABILITY_PILOT_SCHEMA_VERSION:
        errors.append("suite schema_version mismatch")
    if suite.get("gate") != STABILITY_PILOT_GATE_NAME or suite.get("status") != "passed":
        errors.append("suite gate/status mismatch")
    if suite.get("run_id") != manifest.get("run_id"):
        errors.append("suite/manifest run_id mismatch")
    if suite.get("manifest_sha256") != manifest_hash:
        errors.append("suite manifest SHA-256 mismatch")
    if suite.get("protocol") != protocol.to_record():
        errors.append("suite protocol mismatch")

    arm_index = suite.get("arms")
    expected_arm_ids = [protocol.arm_id(rate) for rate in protocol.learning_rates]
    if not isinstance(arm_index, dict) or list(arm_index) != expected_arm_ids:
        errors.append("suite arm index mismatch")
        arm_index = {}
    arms: dict[str, dict[str, Any]] = {}
    initial_hashes: set[str] = set()
    checkpoint_candidates: list[dict[str, Any]] = []
    for arm_id in expected_arm_ids:
        path = arms_root / arm_id / "result.json"
        if not path.is_file():
            errors.append(f"arm result is missing: {arm_id}")
            continue
        try:
            arm = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"arm result is invalid JSON: {arm_id}")
            continue
        arms[arm_id] = arm
        index_row = arm_index.get(arm_id)
        if not isinstance(index_row, dict) or index_row.get("result_sha256") != file_sha256(path):
            errors.append(f"suite arm result hash mismatch: {arm_id}")
        errors.extend(
            _validate_arm(
                arm,
                arm_id=arm_id,
                manifest_sha256=manifest_hash,
                checkpoint_root=checkpoint_root,
            )
        )
        initial_hashes.add(str(arm.get("initial_trainable_state_sha256")))
        for row in arm.get("calibration", []):
            if row.get("step") in protocol.checkpoint_steps:
                checkpoint = next(
                    (
                        checkpoint_row
                        for checkpoint_row in arm.get("checkpoints", [])
                        if checkpoint_row.get("step") == row.get("step")
                    ),
                    None,
                )
                checkpoint_candidates.append(
                    {
                        "arm_id": arm_id,
                        "learning_rate": arm.get("learning_rate"),
                        "step": row.get("step"),
                        "mean_target_token_nll": row.get("mean_target_token_nll"),
                        "checkpoint": checkpoint,
                    }
                )
    if len(initial_hashes) != 1:
        errors.append("pilot arms did not start from one identical trainable-state hash")

    try:
        selected = select_pilot_checkpoint(checkpoint_candidates)
    except (TypeError, ValueError) as exc:
        errors.append(f"checkpoint selection cannot be reconstructed: {exc}")
        return errors
    recorded = suite.get("selected_checkpoint")
    if recorded != selected:
        errors.append("recorded selected checkpoint does not match the frozen selection rule")
    selected_arm = arms.get(selected["arm_id"])
    baseline = None
    if selected_arm is not None:
        baseline_row = next(
            (row for row in selected_arm.get("calibration", []) if row.get("step") == 0), None
        )
        if isinstance(baseline_row, dict):
            baseline = baseline_row.get("mean_target_token_nll")
    improvement = (
        float(baseline) - float(selected["mean_target_token_nll"])
        if _finite(baseline)
        else math.nan
    )
    if (
        not math.isfinite(improvement)
        or improvement < protocol.minimum_calibration_improvement_nats
    ):
        errors.append("selected checkpoint does not meet minimum held-out calibration improvement")
    criteria = suite.get("pass_criteria")
    if (
        not isinstance(criteria, dict)
        or criteria.get("calibration_improvement_nats") != improvement
    ):
        errors.append("suite calibration-improvement criterion mismatch")
    return errors
