"""Independent admission checks for the native-Qwen train-only history suite."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
)
from spatialvlm.training.native_c0_gate import (
    file_sha256,
    model_weight_locks,
    tensor_sha256,
    tensor_state_sha256,
)
from spatialvlm.training.native_history_select import (
    HISTORY_ARMS,
    NATIVE_HISTORY_CHECKPOINT_SCHEMA_VERSION,
    NATIVE_HISTORY_GATE_NAME,
    NATIVE_HISTORY_SCHEMA_VERSION,
    NativeHistoryProtocol,
    select_arm_checkpoint,
    select_history_arm,
    select_native_history_samples,
)

NATIVE_HISTORY_VERIFIER_SCHEMA_VERSION = 1
NATIVE_HISTORY_VERIFIER_GATE_NAME = "independent_native_qwen_history_selection_verification_v1"
EXPECTED_DOCKER_IMAGE_ID = "sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf"
NATIVE_HISTORY_SOURCE_LOCK_PATHS: tuple[str, ...] = (
    "scripts/run_spark_native_history_select.py",
    "scripts/check_native_history_select_artifacts.py",
    "scripts/run_spark_native_history_select.sh",
    "scripts/start_spark_native_history_select.sh",
    "scripts/start_spark_native_history_diagnostic.sh",
    "src/spatialvlm/backbone/qwen3_vl.py",
    "src/spatialvlm/backbone/rope_patch.py",
    "src/spatialvlm/data/datasets.py",
    "src/spatialvlm/data/habitat_sim_r2r.py",
    "src/spatialvlm/data/history.py",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/eval/action_decision.py",
    "src/spatialvlm/eval/metrics.py",
    "src/spatialvlm/eval/r2r_ce.py",
    "src/spatialvlm/training/native_c0_gate.py",
    "src/spatialvlm/training/native_history_select.py",
    "src/spatialvlm/training/native_history_select_artifacts.py",
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _protocol(record: Any, errors: list[str]) -> NativeHistoryProtocol | None:
    if not isinstance(record, dict):
        errors.append("manifest protocol is missing")
        return None
    names = {field.name for field in fields(NativeHistoryProtocol)}
    try:
        protocol = NativeHistoryProtocol(
            **{
                name: tuple(record[name]) if name == "calibration_steps" else record[name]
                for name in names
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"manifest protocol cannot be reconstructed: {exc}")
        return None
    if record != protocol.to_record():
        errors.append("manifest protocol record mismatch")
    return protocol


def _source_locks(manifest: dict[str, Any], repo: Path, errors: list[str]) -> None:
    locks = manifest.get("source_locks")
    if not isinstance(locks, dict) or set(locks) != set(NATIVE_HISTORY_SOURCE_LOCK_PATHS):
        errors.append("manifest source-lock path set mismatch")
        return
    for relative, expected in locks.items():
        path = repo / relative
        if not path.is_file():
            errors.append(f"locked source is missing: {relative}")
        elif file_sha256(path) != expected:
            errors.append(f"locked source hash mismatch: {relative}")


def _manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    repo: Path,
    dataset: R2RCEDataset | None,
    episodes_path: Path | None,
    ground_truth_path: Path | None,
    errors: list[str],
) -> NativeHistoryProtocol | None:
    if manifest.get("schema_version") != NATIVE_HISTORY_SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    if manifest.get("gate") != NATIVE_HISTORY_GATE_NAME or manifest.get("status") != "frozen":
        errors.append("manifest gate/status mismatch")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith("native_history_select_v"):
        errors.append("manifest run_id mismatch")
    protocol = _protocol(manifest.get("protocol"), errors)
    _source_locks(manifest, repo, errors)
    model = manifest.get("model")
    expected_model = {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "transformers_version": "4.57.6",
        "dtype": "bf16",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "base_weights_frozen": True,
        "gradient_checkpointing": "non_reentrant",
        "spatial_rope_enabled": False,
        "peft_2880_workaround_enabled": False,
    }
    if model != expected_model:
        errors.append("manifest model contract mismatch")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or (
        runtime.get("docker_image_id") != EXPECTED_DOCKER_IMAGE_ID
        or runtime.get("sequential_arms") is not True
        or runtime.get("fully_offline") is not True
    ):
        errors.append("manifest runtime contract mismatch")
    data = manifest.get("dataset")
    if not isinstance(data, dict):
        errors.append("manifest dataset record is missing")
    else:
        if data.get("split_used") != "train" or data.get("sealed_splits") != [
            "val_seen",
            "val_unseen",
        ]:
            errors.append("manifest development/sealed split contract mismatch")
        for label, path in (("episodes", episodes_path), ("ground_truth", ground_truth_path)):
            if path is not None and (
                not path.is_file() or file_sha256(path) != data.get(f"{label}_sha256")
            ):
                errors.append(f"manifest {label} source hash mismatch")
    rows = manifest.get("samples")
    if not isinstance(rows, list) or protocol is None:
        errors.append("manifest samples are missing")
        return protocol
    if _canonical_sha256(rows) != manifest.get("sample_records_sha256"):
        errors.append("manifest sample-record hash mismatch")
    expected_count = protocol.train_sample_count + protocol.calibration_sample_count
    if len(rows) != expected_count:
        errors.append("manifest sample count mismatch")
    trajectories = [row.get("trajectory_id") for row in rows if isinstance(row, dict)]
    if len(trajectories) != len(set(trajectories)):
        errors.append("manifest reuses a physical trajectory")
    for row in rows:
        if not isinstance(row, dict):
            errors.append("manifest has a non-object sample")
            continue
        text = row.get("instruction_text")
        digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        if not isinstance(text, str) or digest != row.get("instruction_sha256"):
            errors.append("manifest instruction text/hash mismatch")
        prefix = row.get("replay_prefix_action_ids")
        if not isinstance(prefix, list) or len(prefix) < protocol.min_decision_step or 0 in prefix:
            errors.append("manifest replay prefix violates the causal no-padding contract")
    for split, expected in (
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
            if count != expected:
                errors.append(f"manifest {split}/{action.name} count mismatch")
    if dataset is not None:
        rebuilt = [
            sample.to_record()
            for sample in select_native_history_samples(dataset, protocol=protocol)
        ]
        if rebuilt != rows:
            errors.append("manifest samples do not match locked release reconstruction")
    if not manifest_path.is_file():
        errors.append("manifest file is missing")
    return protocol


def _cache(
    cache_manifest: dict[str, Any],
    *,
    cache_manifest_path: Path,
    manifest_sha256: str,
    samples: list[dict[str, Any]],
    verify_files: bool,
    errors: list[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    entries = cache_manifest.get("entries")
    if cache_manifest.get("status") != "passed":
        errors.append("cache manifest status mismatch")
    if cache_manifest.get("manifest_sha256") != manifest_sha256:
        errors.append("cache/manifest SHA-256 mismatch")
    if not isinstance(entries, list) or len(entries) != len(samples):
        errors.append("cache entry count mismatch")
        return {}
    if _canonical_sha256(entries) != cache_manifest.get("entry_index_sha256"):
        errors.append("cache entry-index hash mismatch")
    index = {
        (row.get("split"), row.get("sample_index")): row for row in entries if isinstance(row, dict)
    }
    expected = {(row["split"], row["sample_index"]) for row in samples}
    if set(index) != expected:
        errors.append("cache sample-index mismatch")
    if not verify_files:
        return index
    root = cache_manifest_path.resolve().parent
    sample_index = {(row["split"], row["sample_index"]): row for row in samples}
    for key, row in index.items():
        relative = row.get("relative_path")
        if not isinstance(relative, str):
            errors.append(f"cache entry {key} has no relative path")
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f"cache path is invalid: {relative}")
            continue
        if path.stat().st_size != row.get("bytes") or file_sha256(path) != row.get("sha256"):
            errors.append(f"cache file identity mismatch: {relative}")
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            tensor = payload["rgb_frames"]
            valid = (
                payload.get("sample") == sample_index[key]
                and payload.get("sample_sha256") == row.get("sample_sha256")
                and isinstance(tensor, torch.Tensor)
                and tensor.dtype == torch.uint8
                and tensor.ndim == 4
                and tuple(tensor.shape[:1]) == (4,)
                and tensor.shape[-1] == 3
                and tensor_sha256(tensor) == row.get("rgb_frames_sha256")
                and len(payload.get("history_action_ids", ())) == 3
                and 0 not in payload.get("history_action_ids", ())
            )
        except Exception:
            valid = False
        if not valid:
            errors.append(f"cache payload validation failed: {relative}")
    return index


def _event_rows(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"update event log is missing: {path}")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"update event log has invalid JSON at line {line_number}")
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            errors.append(f"update event log line {line_number} is not an object")
    return rows


def _checkpoint_payload(path: Path, row: dict[str, Any], errors: list[str]) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("trainable_state_dict")
        valid = (
            payload.get("schema_version") == NATIVE_HISTORY_CHECKPOINT_SCHEMA_VERSION
            and payload.get("gate") == NATIVE_HISTORY_GATE_NAME
            and isinstance(state, dict)
            and tensor_state_sha256(state) == payload.get("trainable_state_sha256")
            and payload.get("trainable_state_sha256") == row.get("trainable_state_sha256")
            and payload.get("cursor", {}).get("update_step") == row.get("update_step")
        )
    except Exception:
        valid = False
    if not valid:
        errors.append(f"checkpoint payload validation failed: {path.name}")


def _arm(
    arm_id: str,
    result: dict[str, Any],
    *,
    result_path: Path,
    checkpoint_root: Path,
    protocol: NativeHistoryProtocol,
    manifest_sha256: str,
    cache_manifest_sha256: str,
    train_samples: list[dict[str, Any]],
    calibration_samples: list[dict[str, Any]],
    cache_index: dict[tuple[str, int], dict[str, Any]],
    verify_all_checkpoints: bool,
    errors: list[str],
) -> list[float]:
    label = f"arm {arm_id}"
    if result.get("gate") != NATIVE_HISTORY_GATE_NAME or result.get("status") != "passed":
        errors.append(f"{label} gate/status mismatch")
    if (
        result.get("arm_id") != arm_id
        or result.get("completed_updates") != protocol.planned_updates
    ):
        errors.append(f"{label} identity/update count mismatch")
    if result.get("manifest_sha256") != manifest_sha256:
        errors.append(f"{label} manifest linkage mismatch")
    if result.get("cache_manifest_sha256") != cache_manifest_sha256:
        errors.append(f"{label} cache linkage mismatch")
    if (
        not _finite(result.get("training_seconds"))
        or float(result.get("training_seconds", -1.0)) < 0.0
    ):
        errors.append(f"{label} training duration is invalid")
    if (
        not _finite(result.get("peak_allocated_bytes"))
        or int(result.get("peak_allocated_bytes", protocol.hard_max_peak_allocated_bytes + 1))
        > protocol.hard_max_peak_allocated_bytes
    ):
        errors.append(f"{label} exceeded the hard memory gate")

    calibrations = result.get("calibrations")
    if not isinstance(calibrations, list) or [row.get("step") for row in calibrations] != list(
        protocol.calibration_steps
    ):
        errors.append(f"{label} calibration schedule mismatch")
        return []
    expected_calibration = {
        (row["sample_index"], row["episode_id"], row["trajectory_id"], row["target_action"])
        for row in calibration_samples
    }
    for calibration in calibrations:
        decisions = calibration.get("per_decision")
        if not isinstance(decisions, list) or len(decisions) != protocol.calibration_sample_count:
            errors.append(f"{label} calibration decision count mismatch")
            continue
        observed = {
            (
                row.get("sample_index"),
                row.get("episode_id"),
                row.get("trajectory_id"),
                row.get("target_action"),
            )
            for row in decisions
        }
        if observed != expected_calibration:
            errors.append(f"{label} calibration decision identity mismatch")
        values = [row.get("target_token_nll") for row in decisions]
        if not all(
            _finite(value) and 0.0 <= float(value) <= protocol.hard_max_token_nll
            for value in values
        ):
            errors.append(f"{label} has invalid calibration NLL")
        elif not math.isclose(
            sum(float(value) for value in values) / len(values),
            float(calibration.get("mean_target_token_nll", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(f"{label} calibration mean mismatch")
        for row in decisions:
            cache_row = cache_index.get(("calibration", row.get("sample_index")), {})
            if row.get("cache_sha256") != cache_row.get("sha256"):
                errors.append(f"{label} calibration cache linkage mismatch")
                break
            if (
                row.get("supervision_schema_version")
                != NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION
                or row.get("supervision_method") != NATIVE_QWEN_SUPERVISION_METHOD
                or row.get("supervised_token_count") != 1
                or row.get("action_token_position") is None
                or row.get("supervised_token_id") is None
            ):
                errors.append(f"{label} calibration one-token objective mismatch")
                break
    try:
        selected = select_arm_checkpoint(calibrations)
    except ValueError as exc:
        errors.append(f"{label} checkpoint selection failed: {exc}")
        selected = None
    if selected is not None and result.get("selected_calibration") != selected:
        errors.append(f"{label} selected calibration mismatch")

    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, list):
        errors.append(f"{label} checkpoint index is missing")
        checkpoints = []
    indexed_steps = {row.get("update_step") for row in checkpoints if isinstance(row, dict)}
    if not set(protocol.checkpoint_steps).issubset(indexed_steps):
        errors.append(f"{label} required checkpoints are missing")
    root = checkpoint_root / arm_id
    selected_checkpoint = result.get("selected_checkpoint")
    for row in checkpoints:
        relative = row.get("relative_path") if isinstance(row, dict) else None
        if not isinstance(relative, str):
            errors.append(f"{label} checkpoint relative path is missing")
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f"{label} checkpoint path is invalid: {relative}")
            continue
        if path.stat().st_size != row.get("bytes") or file_sha256(path) != row.get("sha256"):
            errors.append(f"{label} checkpoint identity mismatch: {relative}")
        elif verify_all_checkpoints or row == selected_checkpoint:
            _checkpoint_payload(path, row, errors)
    if isinstance(selected_checkpoint, dict):
        selected_step = result.get("selected_calibration", {}).get("step")
        if selected_checkpoint.get("update_step") != selected_step:
            errors.append(f"{label} selected checkpoint step mismatch")
        if (
            selected_step == 0
            and selected_checkpoint.get("source") != "public_pretrained_seeded_initialization"
        ):
            errors.append(f"{label} step-zero checkpoint provenance mismatch")
        if selected_step != 0 and selected_checkpoint not in checkpoints:
            errors.append(f"{label} selected checkpoint is not indexed")
    else:
        errors.append(f"{label} selected checkpoint is missing")

    events = _event_rows(result_path.parent / "updates.jsonl", errors)
    by_step: dict[int, dict[str, Any]] = {}
    for row in events:
        step = row.get("step")
        if not isinstance(step, int) or not 1 <= step <= protocol.planned_updates:
            errors.append(f"{label} event has invalid step")
            continue
        by_step[step] = row
    if set(by_step) != set(range(1, protocol.planned_updates + 1)):
        errors.append(f"{label} update event coverage mismatch")
    for step, sample in enumerate(train_samples, start=1):
        row = by_step.get(step, {})
        if any(
            row.get(key) != sample[key]
            for key in ("sample_index", "episode_id", "trajectory_id", "target_action")
        ):
            errors.append(f"{label} step-{step} sample identity mismatch")
            continue
        gradient = row.get("gradient_report")
        valid_gradient = (
            isinstance(gradient, dict)
            and gradient.get("trainable_tensor_count", 0) > 0
            and gradient.get("gradient_tensor_count") == gradient.get("trainable_tensor_count")
            and gradient.get("all_gradients_finite") is True
            and gradient.get("nonzero_gradient_tensor_count", 0) > 0
        )
        if (
            not valid_gradient
            or not _finite(row.get("target_token_nll"))
            or not _finite(row.get("global_gradient_norm_before_clip"))
        ):
            errors.append(f"{label} step-{step} numerical/gradient gate mismatch")
        if (
            row.get("supervision_schema_version") != NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION
            or row.get("supervision_method") != NATIVE_QWEN_SUPERVISION_METHOD
            or row.get("supervised_token_count") != 1
            or row.get("action_token_position") is None
            or row.get("supervised_token_id") is None
        ):
            errors.append(f"{label} step-{step} one-token objective mismatch")
    expected_audits = {1, *protocol.checkpoint_steps}
    audits = result.get("audit_updates")
    if not isinstance(audits, list) or {row.get("step") for row in audits} != expected_audits:
        errors.append(f"{label} update-audit schedule mismatch")
    else:
        for row in audits:
            update = row.get("updates")
            if (
                not isinstance(update, dict)
                or update.get("changed_tensor_count", 0) <= 0
                or not update.get("all_updates_finite")
            ):
                errors.append(f"{label} has an invalid audited update")
    selected_rows = result.get("selected_calibration", {}).get("per_decision", [])
    return [float(row["target_token_nll"]) for row in selected_rows]


def verify_native_history_suite(
    suite: dict[str, Any],
    *,
    suite_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    cache_manifest: dict[str, Any],
    cache_manifest_path: Path,
    checkpoint_root: Path,
    model_snapshot: Path,
    repo: Path,
    dataset: R2RCEDataset | None = None,
    episodes_path: Path | None = None,
    ground_truth_path: Path | None = None,
    verify_cache_files: bool = True,
    verify_all_checkpoint_files: bool = True,
) -> list[str]:
    """Return every fail-closed integrity error for a completed three-arm suite."""
    errors: list[str] = []
    protocol = _manifest(
        manifest,
        manifest_path=manifest_path,
        repo=repo,
        dataset=dataset,
        episodes_path=episodes_path,
        ground_truth_path=ground_truth_path,
        errors=errors,
    )
    if protocol is None:
        return errors
    manifest_sha256 = file_sha256(manifest_path)
    cache_manifest_sha256 = file_sha256(cache_manifest_path)
    samples = manifest["samples"]
    cache_index = _cache(
        cache_manifest,
        cache_manifest_path=cache_manifest_path,
        manifest_sha256=manifest_sha256,
        samples=samples,
        verify_files=verify_cache_files,
        errors=errors,
    )
    if suite.get("schema_version") != NATIVE_HISTORY_SCHEMA_VERSION:
        errors.append("suite schema_version mismatch")
    if suite.get("gate") != NATIVE_HISTORY_GATE_NAME or suite.get("status") != "passed":
        errors.append("suite gate/status mismatch")
    if suite.get("manifest_sha256") != manifest_sha256:
        errors.append("suite/manifest SHA-256 mismatch")
    if suite.get("cache_manifest_sha256") != cache_manifest_sha256:
        errors.append("suite/cache SHA-256 mismatch")
    if suite.get("protocol") != protocol.to_record() or suite.get("source_locks") != manifest.get(
        "source_locks"
    ):
        errors.append("suite protocol/source-lock mismatch")
    if suite.get("run_id") != manifest.get("run_id"):
        errors.append("suite/manifest run_id mismatch")
    runtime = suite.get("runtime")
    if not isinstance(runtime, dict) or (
        runtime.get("docker_image_id") != EXPECTED_DOCKER_IMAGE_ID
        or runtime.get("architecture") != "aarch64"
        or runtime.get("transformers") != "4.57.6"
        or runtime.get("gpu") != "NVIDIA GB10"
    ):
        errors.append("suite runtime identity mismatch")
    codebook = suite.get("action_codebook")
    actions = codebook.get("actions") if isinstance(codebook, dict) else None
    if not isinstance(actions, dict) or set(actions) != {action.name for action in R2RCEAction}:
        errors.append("suite action codebook is missing")
    else:
        action_token_ids = []
        for action in R2RCEAction:
            row = actions[action.name]
            if (
                not isinstance(row, dict)
                or row.get("action_id") != int(action)
                or row.get("action_specific_token_count") != 1
                or not isinstance(row.get("action_token_id"), int)
            ):
                errors.append(f"suite action codebook mismatch: {action.name}")
            else:
                action_token_ids.append(row["action_token_id"])
        if len(set(action_token_ids)) != len(R2RCEAction):
            errors.append("suite action codebook token IDs are not distinct")
    if suite.get("supervision") != {
        "schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
        "method": NATIVE_QWEN_SUPERVISION_METHOD,
        "supervised_token_count_per_decision": 1,
    }:
        errors.append("suite one-token supervision contract mismatch")
    try:
        if suite.get("model_weight_locks") != model_weight_locks(model_snapshot):
            errors.append("suite model-weight lock mismatch")
    except Exception as exc:
        errors.append(f"model-weight verification failed: {type(exc).__name__}: {exc}")

    arm_index = suite.get("arm_results")
    if not isinstance(arm_index, dict) or set(arm_index) != set(HISTORY_ARMS):
        errors.append("suite arm-result index mismatch")
        return errors
    train_samples = [row for row in samples if row["split"] == "train"]
    calibration_samples = [row for row in samples if row["split"] == "calibration"]
    nll: dict[str, list[float]] = {}
    initial_hashes: set[str] = set()
    output_root = suite_path.resolve().parent
    for arm_id in HISTORY_ARMS:
        row = arm_index[arm_id]
        relative = row.get("relative_path") if isinstance(row, dict) else None
        if not isinstance(relative, str):
            errors.append(f"suite {arm_id} result path is missing")
            continue
        result_path = (output_root / relative).resolve()
        if not result_path.is_relative_to(output_root) or not result_path.is_file():
            errors.append(f"suite {arm_id} result path is invalid")
            continue
        if file_sha256(result_path) != row.get("sha256"):
            errors.append(f"suite {arm_id} result SHA-256 mismatch")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        initial_hashes.add(str(result.get("initial_trainable_state_sha256")))
        nll[arm_id] = _arm(
            arm_id,
            result,
            result_path=result_path,
            checkpoint_root=checkpoint_root,
            protocol=protocol,
            manifest_sha256=manifest_sha256,
            cache_manifest_sha256=cache_manifest_sha256,
            train_samples=train_samples,
            calibration_samples=calibration_samples,
            cache_index=cache_index,
            verify_all_checkpoints=verify_all_checkpoint_files,
            errors=errors,
        )
    if (
        len(initial_hashes) != 1
        or suite.get("identical_initial_trainable_state_sha256") not in initial_hashes
    ):
        errors.append("history arms do not share one verified initialization")
    if set(nll) == set(HISTORY_ARMS) and all(
        len(values) == protocol.calibration_sample_count for values in nll.values()
    ):
        try:
            expected_selection = select_history_arm(nll, protocol=protocol)
        except ValueError as exc:
            errors.append(f"history selection reconstruction failed: {exc}")
        else:
            if suite.get("history_selection") != expected_selection:
                errors.append("suite history selection mismatch")
    if not suite_path.is_file():
        errors.append("suite file is missing")
    return errors
