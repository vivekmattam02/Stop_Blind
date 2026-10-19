"""Independent integrity checks for the bounded development SFT run."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.training.development_sft import (
    DEVELOPMENT_SFT_CHECKPOINT_SCHEMA_VERSION,
    DEVELOPMENT_SFT_GATE_NAME,
    DEVELOPMENT_SFT_SCHEMA_VERSION,
    PILOT_MANIFEST_SHA256,
    PILOT_RUN_ID,
    PILOT_SUITE_SHA256,
    PILOT_VERIFICATION_V2_SHA256,
    WARMSTART_CHECKPOINT_BYTES,
    WARMSTART_CHECKPOINT_SHA256,
    WARMSTART_TRAINABLE_STATE_SHA256,
    DevelopmentSFTProtocol,
    select_development_checkpoint,
    select_development_samples,
    update_early_stopping_state,
)
from spatialvlm.training.real_batch_gate import file_sha256, tensor_state_sha256
from spatialvlm.training.sft import FREEZE_GROUP_PATTERNS
from spatialvlm.training.stability_pilot_artifacts import STABILITY_PILOT_SOURCE_LOCK_PATHS

DEVELOPMENT_SFT_VERIFIER_SCHEMA_VERSION = 1
DEVELOPMENT_SFT_VERIFIER_GATE_NAME = "independent_development_sft_artifact_verification_v1"
DEVELOPMENT_SFT_SOURCE_LOCK_PATHS = tuple(
    dict.fromkeys(
        (
            *STABILITY_PILOT_SOURCE_LOCK_PATHS,
            "docs/SPARK_DEVELOPMENT_SFT.md",
            "scripts/check_development_sft_artifacts.py",
            "scripts/run_spark_development_sft.py",
            "scripts/run_spark_development_sft.sh",
            "scripts/start_spark_development_sft.sh",
            "src/spatialvlm/training/development_sft.py",
            "src/spatialvlm/training/development_sft_artifacts.py",
            "tests/test_development_sft.py",
            "tests/test_development_sft_artifacts.py",
        )
    )
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_source_locks(manifest: dict[str, Any], repo: Path, errors: list[str]) -> None:
    locks = manifest.get("source_locks")
    expected = set(DEVELOPMENT_SFT_SOURCE_LOCK_PATHS)
    if not isinstance(locks, dict) or set(locks) != expected:
        errors.append("manifest source-lock path set mismatch")
        return
    for relative, expected_hash in locks.items():
        path = repo / relative
        if not path.is_file():
            errors.append(f"locked source is missing: {relative}")
        elif file_sha256(path) != expected_hash:
            errors.append(f"locked source hash mismatch: {relative}")


def _validate_preflights(manifest: dict[str, Any], manifest_path: Path, errors: list[str]) -> None:
    expected = {
        "processor": "spark_real_batch_checkpoint_processor_preflight_v1",
        "action": "spark_action_decision_tokenizer_preflight_v1",
    }
    preflights = manifest.get("preflights")
    if not isinstance(preflights, dict) or set(preflights) != set(expected):
        errors.append("manifest preflight index mismatch")
        return
    root = manifest_path.resolve().parent
    for label, gate in expected.items():
        row = preflights[label]
        relative = row.get("relative_path") if isinstance(row, dict) else None
        if not isinstance(relative, str):
            errors.append(f"{label} preflight relative path is missing")
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f"{label} preflight path is invalid")
            continue
        if file_sha256(path) != row.get("sha256"):
            errors.append(f"{label} preflight SHA-256 mismatch")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"{label} preflight is invalid JSON")
        else:
            if payload.get("status") != "passed" or payload.get("gate") != gate:
                errors.append(f"{label} preflight gate/status mismatch")


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    repo: Path,
    dataset: R2RCEDataset | None,
    errors: list[str],
) -> None:
    protocol = DevelopmentSFTProtocol()
    if manifest.get("schema_version") != DEVELOPMENT_SFT_SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    if manifest.get("gate") != DEVELOPMENT_SFT_GATE_NAME or manifest.get("status") != "passed":
        errors.append("manifest gate/status mismatch")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"development_sft_v[1-9][0-9]*", run_id) is None:
        errors.append("manifest run_id is invalid")
    if manifest.get("protocol") != protocol.to_record():
        errors.append("manifest protocol mismatch")
    _validate_source_locks(manifest, repo, errors)
    _validate_preflights(manifest, manifest_path, errors)

    pilot = manifest.get("pilot_link")
    if not isinstance(pilot, dict):
        errors.append("pilot link is missing")
        excluded: set[str] = set()
    else:
        expected_pilot = {
            "run_id": PILOT_RUN_ID,
            "manifest_sha256": PILOT_MANIFEST_SHA256,
            "suite_sha256": PILOT_SUITE_SHA256,
            "verification_v2_sha256": PILOT_VERIFICATION_V2_SHA256,
            "warmstart_checkpoint_bytes": WARMSTART_CHECKPOINT_BYTES,
            "warmstart_checkpoint_sha256": WARMSTART_CHECKPOINT_SHA256,
            "warmstart_trainable_state_sha256": WARMSTART_TRAINABLE_STATE_SHA256,
        }
        for key, expected_value in expected_pilot.items():
            if pilot.get(key) != expected_value:
                errors.append(f"pilot link mismatch: {key}")
        excluded_rows = pilot.get("excluded_trajectory_ids")
        excluded = set(excluded_rows) if isinstance(excluded_rows, list) else set()
        if len(excluded) != 24:
            errors.append("pilot exclusion set does not contain 24 trajectories")

    rows = manifest.get("samples")
    if not isinstance(rows, list):
        errors.append("manifest samples are missing")
        return
    if len(rows) != protocol.train_sample_count + protocol.calibration_sample_count:
        errors.append("manifest sample count mismatch")
    trajectories = [row.get("trajectory_id") for row in rows if isinstance(row, dict)]
    if len(trajectories) != len(set(trajectories)):
        errors.append("manifest trajectories are not globally disjoint")
    if set(trajectories) & excluded:
        errors.append("development samples overlap pilot trajectories")
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
                errors.append(f"manifest {split}/{action.name} count mismatch")
    if dataset is not None:
        expected_rows = [
            sample.to_record()
            for sample in select_development_samples(
                dataset, excluded_trajectory_ids=excluded, protocol=protocol
            )
        ]
        if rows != expected_rows:
            errors.append("manifest samples do not match release reconstruction")
        release = dataset.release_manifest.to_dict() if dataset.release_manifest else None
        if manifest.get("release_manifest") != release:
            errors.append("manifest release identity mismatch")


def _validate_cache(
    cache_manifest: dict[str, Any],
    *,
    cache_manifest_path: Path,
    manifest_sha256: str,
    expected_samples: list[dict[str, Any]],
    verify_cache_files: bool,
    errors: list[str],
) -> None:
    protocol = DevelopmentSFTProtocol()
    if cache_manifest.get("status") != "passed":
        errors.append("cache manifest status mismatch")
    if cache_manifest.get("manifest_sha256") != manifest_sha256:
        errors.append("cache/manifest SHA-256 mismatch")
    expected_count = protocol.train_sample_count + protocol.calibration_sample_count
    entries = cache_manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_count:
        errors.append("cache entry count mismatch")
        return
    sample_keys = {(row["split"], row["sample_index"]) for row in expected_samples}
    entry_keys = {
        (row.get("split"), row.get("sample_index")) for row in entries if isinstance(row, dict)
    }
    if entry_keys != sample_keys:
        errors.append("cache sample index mismatch")
    if not verify_cache_files:
        return
    root = cache_manifest_path.resolve().parent
    for row in entries:
        relative = row.get("relative_path") if isinstance(row, dict) else None
        if not isinstance(relative, str):
            errors.append("cache entry relative path is missing")
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f"cache entry path is invalid: {relative}")
            continue
        if path.stat().st_size != row.get("bytes"):
            errors.append(f"cache entry byte-size mismatch: {relative}")
        elif file_sha256(path) != row.get("sha256"):
            errors.append(f"cache entry SHA-256 mismatch: {relative}")


def _validate_gradient_groups(row: dict[str, Any], label: str, errors: list[str]) -> None:
    gradients = row.get("gradients_by_group")
    if not isinstance(gradients, dict):
        errors.append(f"{label}: gradient groups are missing")
        return
    for group in FREEZE_GROUP_PATTERNS:
        item = gradients.get(group)
        if not isinstance(item, dict):
            errors.append(f"{label}: {group} gradient row is missing")
            continue
        if item.get("trainable_tensor_count", 0) <= 0:
            errors.append(f"{label}: {group} has no trainable tensors")
        if item.get("gradient_tensor_count") != item.get("trainable_tensor_count"):
            errors.append(f"{label}: {group} has missing gradients")
        if not item.get("all_gradients_finite") or not _finite(item.get("gradient_l2_norm")):
            errors.append(f"{label}: {group} has invalid gradients")
        elif float(item["gradient_l2_norm"]) <= 0.0:
            errors.append(f"{label}: {group} gradient norm is non-positive")


def _validate_update_groups(row: dict[str, Any], label: str, errors: list[str]) -> None:
    updates = row.get("updates_by_group")
    if not isinstance(updates, dict):
        errors.append(f"{label}: required update audit is missing")
        return
    for group in FREEZE_GROUP_PATTERNS:
        item = updates.get(group)
        if not isinstance(item, dict):
            errors.append(f"{label}: {group} update row is missing")
            continue
        if not item.get("all_updates_finite") or not _finite(item.get("update_l2_norm")):
            errors.append(f"{label}: {group} has invalid updates")
        elif float(item["update_l2_norm"]) <= 0.0:
            errors.append(f"{label}: {group} update norm is non-positive")


def _validate_checkpoint_payload(path: Path, row: dict[str, Any], errors: list[str]) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        errors.append(f"selected checkpoint cannot be loaded: {type(exc).__name__}")
        return
    if payload.get("schema_version") != DEVELOPMENT_SFT_CHECKPOINT_SCHEMA_VERSION:
        errors.append("selected checkpoint schema mismatch")
    state = payload.get("trainable_state_dict")
    if not isinstance(state, dict) or tensor_state_sha256(state) != payload.get(
        "trainable_state_sha256"
    ):
        errors.append("selected checkpoint tensor payload hash mismatch")
    if payload.get("trainable_state_sha256") != row.get("trainable_state_sha256"):
        errors.append("selected checkpoint/result trainable hash mismatch")


def verify_development_sft(
    result: dict[str, Any],
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    cache_manifest: dict[str, Any],
    cache_manifest_path: Path,
    checkpoint_root: Path,
    warmstart_checkpoint: Path,
    repo: Path,
    dataset: R2RCEDataset | None = None,
    verify_cache_files: bool = True,
    verify_all_checkpoint_files: bool = True,
) -> list[str]:
    """Return every independent integrity error for a completed development run."""
    errors: list[str] = []
    protocol = DevelopmentSFTProtocol()
    _validate_manifest(
        manifest,
        manifest_path=manifest_path,
        repo=repo,
        dataset=dataset,
        errors=errors,
    )
    manifest_hash = file_sha256(manifest_path)
    samples = manifest.get("samples") if isinstance(manifest.get("samples"), list) else []
    _validate_cache(
        cache_manifest,
        cache_manifest_path=cache_manifest_path,
        manifest_sha256=manifest_hash,
        expected_samples=samples,
        verify_cache_files=verify_cache_files,
        errors=errors,
    )

    if result.get("schema_version") != DEVELOPMENT_SFT_SCHEMA_VERSION:
        errors.append("result schema_version mismatch")
    if result.get("gate") != DEVELOPMENT_SFT_GATE_NAME or result.get("status") != "passed":
        errors.append("result gate/status mismatch")
    if result.get("run_id") != manifest.get("run_id"):
        errors.append("result/manifest run_id mismatch")
    if result.get("manifest_sha256") != manifest_hash:
        errors.append("result/manifest SHA-256 mismatch")
    if result.get("protocol") != protocol.to_record():
        errors.append("result protocol mismatch")
    if result.get("source_locks") != manifest.get("source_locks"):
        errors.append("result source locks differ from manifest")
    if result.get("cache_manifest_sha256") != file_sha256(cache_manifest_path):
        errors.append("result/cache-manifest SHA-256 mismatch")
    if warmstart_checkpoint.stat().st_size != WARMSTART_CHECKPOINT_BYTES:
        errors.append("warm-start checkpoint byte-size mismatch")
    elif file_sha256(warmstart_checkpoint) != WARMSTART_CHECKPOINT_SHA256:
        errors.append("warm-start checkpoint SHA-256 mismatch")

    training = result.get("training")
    if not isinstance(training, dict):
        errors.append("training record is missing")
        return errors
    completed = training.get("completed_updates")
    steps = training.get("steps")
    if not isinstance(completed, int) or completed < protocol.minimum_updates_before_early_stop:
        errors.append("completed update count is below the frozen minimum")
        completed = 0
    if not isinstance(steps, list) or len(steps) != completed:
        errors.append("training step count mismatch")
        steps = []
    train_samples = [row for row in samples if row.get("split") == "train"]
    audits = {1, *protocol.checkpoint_steps, *protocol.calibration_steps[1:]}
    for expected_step, row in enumerate(steps, start=1):
        label = f"step-{expected_step}"
        if not isinstance(row, dict) or row.get("step") != expected_step:
            errors.append(f"{label}: identity mismatch")
            continue
        sample = train_samples[expected_step - 1]
        for key in ("sample_index", "episode_id", "trajectory_id", "target_action"):
            if row.get(key) != sample.get(key):
                errors.append(f"{label}: sample {key} mismatch")
        if (
            not _finite(row.get("target_token_nll"))
            or float(row.get("target_token_nll", math.inf)) > protocol.hard_max_token_nll
        ):
            errors.append(f"{label}: target NLL violates hard limit")
        if (
            not _finite(row.get("global_gradient_norm_before_clip"))
            or float(row.get("global_gradient_norm_before_clip", math.inf))
            > protocol.hard_max_gradient_norm
        ):
            errors.append(f"{label}: gradient norm violates hard limit")
        if row.get("peak_allocated_bytes", 0) > protocol.hard_max_peak_allocated_bytes:
            errors.append(f"{label}: allocated memory violates hard limit")
        _validate_gradient_groups(row, label, errors)
        if expected_step in audits:
            _validate_update_groups(row, label, errors)
        elif row.get("updates_by_group") is not None:
            errors.append(f"{label}: unexpected update audit")

    calibrations = training.get("calibrations")
    if not isinstance(calibrations, list) or not calibrations:
        errors.append("calibration records are missing")
        calibrations = []
    calibration_steps = [row.get("step") for row in calibrations if isinstance(row, dict)]
    expected_prefix = list(protocol.calibration_steps[: len(calibration_steps)])
    if calibration_steps != expected_prefix:
        errors.append("calibration schedule is not a frozen prefix")
    for row in calibrations:
        step = row.get("step")
        if row.get("sample_count") != protocol.calibration_sample_count:
            errors.append(f"calibration-{step}: sample count mismatch")
        if not _finite(row.get("mean_target_token_nll")):
            errors.append(f"calibration-{step}: mean NLL is non-finite")
        calibration_samples = row.get("samples")
        if (
            not isinstance(calibration_samples, list)
            or len(calibration_samples) != protocol.calibration_sample_count
        ):
            errors.append(f"calibration-{step}: sample rows are incomplete")

    terminal = result.get("terminal_reason")
    if terminal == "planned_update_budget_completed" and completed != protocol.planned_updates:
        errors.append("planned-budget terminal reason has wrong update count")
    elif terminal == "pre_registered_early_stopping":
        if completed not in protocol.calibration_steps:
            errors.append("early stop did not occur at a calibration step")
        elif calibrations:
            best = float(calibrations[0]["mean_target_token_nll"])
            count = 0
            stop = False
            for row in calibrations[1:]:
                best, count, stop = update_early_stopping_state(
                    previous_best_nll=best,
                    previous_non_improving_count=count,
                    current_nll=float(row["mean_target_token_nll"]),
                    current_step=int(row["step"]),
                    protocol=protocol,
                )
            if not stop:
                errors.append("early-stop terminal reason does not satisfy frozen patience")
    elif terminal == "wall_time_budget_reached":
        if completed < protocol.minimum_updates_before_early_stop:
            errors.append("wall-time stop occurred below the minimum update floor")
    else:
        errors.append("terminal reason is not accepted by the frozen protocol")

    checkpoint_rows = training.get("checkpoints")
    if not isinstance(checkpoint_rows, list) or not checkpoint_rows:
        errors.append("checkpoint index is missing")
        checkpoint_rows = []
    root = checkpoint_root.resolve()
    checkpoint_by_step: dict[int, dict[str, Any]] = {}
    for row in checkpoint_rows:
        relative = row.get("relative_path") if isinstance(row, dict) else None
        step = row.get("update_step") if isinstance(row, dict) else None
        if not isinstance(relative, str) or not isinstance(step, int):
            errors.append("checkpoint row identity is invalid")
            continue
        path = (root / relative).resolve()
        checkpoint_by_step[step] = row
        if verify_all_checkpoint_files:
            if not path.is_relative_to(root) or not path.is_file():
                errors.append(f"checkpoint path is invalid: {relative}")
            elif path.stat().st_size != row.get("bytes"):
                errors.append(f"checkpoint byte-size mismatch: {relative}")
            elif file_sha256(path) != row.get("sha256"):
                errors.append(f"checkpoint SHA-256 mismatch: {relative}")
    for step in calibration_steps[1:]:
        if step not in checkpoint_by_step:
            errors.append(f"calibration candidate has no checkpoint: step {step}")

    selection = result.get("selection")
    if not isinstance(selection, dict) or not calibrations:
        errors.append("selection record is missing")
    else:
        expected_selected = select_development_checkpoint(calibrations)
        if selection.get("selected_calibration") != expected_selected:
            errors.append("selected calibration does not satisfy frozen rule")
        selected_step = expected_selected["step"]
        selected_checkpoint = selection.get("selected_checkpoint")
        if not isinstance(selected_checkpoint, dict):
            errors.append("selected checkpoint is missing")
        elif selected_step == 0:
            if (
                selected_checkpoint.get("sha256") != WARMSTART_CHECKPOINT_SHA256
                or selected_checkpoint.get("bytes") != WARMSTART_CHECKPOINT_BYTES
            ):
                errors.append("step-zero selection does not identify the warm start")
        else:
            expected_checkpoint = checkpoint_by_step.get(selected_step)
            if expected_checkpoint is None:
                errors.append("selected development checkpoint is absent from index")
            elif any(
                selected_checkpoint.get(key) != expected_checkpoint.get(key)
                for key in ("sha256", "bytes", "trainable_state_sha256", "update_step")
            ):
                errors.append("selected checkpoint differs from checkpoint index")
            else:
                selected_path = (root / expected_checkpoint["relative_path"]).resolve()
                if selected_path.is_file():
                    _validate_checkpoint_payload(selected_path, expected_checkpoint, errors)
    return errors
