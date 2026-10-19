#!/usr/bin/env python3
"""Prepare and run the native-Qwen optimizer-admission stress or main profile."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import signal
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoProcessor

from scripts.run_spark_native_history_select import (
    EXPECTED_DOCKER_IMAGE_ID,
    _atomic_json,
    _cache_index,
    _canonical_sha256,
    _load_cached,
    _load_dataset,
    _load_model,
    _move_batch,
    _native_batch,
    _prepare_cache,
)
from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
)
from spatialvlm.data.native_qwen_artifacts import (
    NATIVE_QWEN_HISTORY_PREFLIGHT_GATE_NAME,
    NATIVE_QWEN_HISTORY_VERIFIER_GATE_NAME,
)
from spatialvlm.eval.action_decision import (
    ActionTokenCodebook,
    build_action_token_codebook,
    score_action_token_logits,
)
from spatialvlm.eval.native_sensitivity_v2 import (
    NATIVE_SENSITIVITY_V2_GATE_NAME,
    NATIVE_SENSITIVITY_V2_VERDICT_ADMIT,
)
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    file_sha256,
    gradient_report,
    model_weight_locks,
    tensor_state_sha256,
    trainable_state_dict,
    update_report,
    validate_lora_only_trainables,
    validate_optimization_reports,
)
from spatialvlm.training.native_history_select import (
    NativeHistoryProtocol,
    capture_rng_state,
    optimizer_tensor_diagnostics,
    select_native_history_samples,
)
from spatialvlm.training.native_optimizer_admission import (
    NATIVE_OPTIMIZER_ADMISSION_GATE,
    NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
    NATIVE_OPTIMIZER_OBJECTIVE,
    NativeOptimizerAdmissionProtocol,
    learning_rate_multiplier,
    select_admitted_rate,
    summarize_calibration,
    training_sample_order,
)

SOURCE_LOCK_PATHS: tuple[str, ...] = (
    "docs/NATIVE_QWEN_SENSITIVITY_DIAGNOSTIC_V2.md",
    "scripts/check_native_optimizer_admission_artifacts.py",
    "scripts/check_native_sensitivity_v2_artifacts.py",
    "scripts/native_optimizer_admission_sync_files.txt",
    "scripts/native_sensitivity_v2_sync_files.txt",
    "scripts/run_spark_native_optimizer_admission.py",
    "scripts/run_spark_native_optimizer_admission.sh",
    "scripts/run_spark_native_optimizer_pipeline.sh",
    "scripts/run_spark_native_sensitivity.py",
    "scripts/run_spark_native_sensitivity_v2.py",
    "scripts/run_spark_native_sensitivity_v2.sh",
    "scripts/spark_launch_helpers.sh",
    "scripts/start_spark_native_optimizer_admission.sh",
    "scripts/start_spark_native_optimizer_pipeline.sh",
    "scripts/start_spark_native_optimizer_stress.sh",
    "scripts/start_spark_native_sensitivity_v2.sh",
    "scripts/run_spark_native_history_select.py",
    "src/spatialvlm/backbone/qwen3_vl.py",
    "src/spatialvlm/backbone/rope_patch.py",
    "src/spatialvlm/data/datasets.py",
    "src/spatialvlm/data/habitat_sim_r2r.py",
    "src/spatialvlm/data/history.py",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/data/native_qwen_artifacts.py",
    "src/spatialvlm/eval/action_decision.py",
    "src/spatialvlm/eval/native_sensitivity.py",
    "src/spatialvlm/eval/native_sensitivity_v2.py",
    "src/spatialvlm/training/native_c0_gate.py",
    "src/spatialvlm/training/native_history_select.py",
    "src/spatialvlm/training/native_optimizer_admission.py",
    "src/spatialvlm/training/unit_ledger.py",
    "tests/test_native_optimizer_admission.py",
    "tests/test_native_optimizer_admission_artifacts.py",
    "tests/test_native_sensitivity_v2.py",
    "tests/test_native_sensitivity_v2_artifacts.py",
    "tests/test_spark_launch_helpers.py",
    "tests/test_unit_ledger.py",
)

_STOP_REQUESTED = False
SENSITIVITY_VERIFIER_GATE = (
    "independent_native_qwen_c0_visual_sensitivity_verification_v2"
)
SENSITIVITY_ADMISSION_VERDICT = NATIVE_SENSITIVITY_V2_VERDICT_ADMIT
OPTIMIZER_VERIFIER_GATE = "independent_native_qwen_optimizer_admission_verification_v1"


def _request_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("Termination requested; checkpointing after the current optimizer update.", flush=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_locks(repo: Path) -> dict[str, str]:
    locks: dict[str, str] = {}
    for relative in SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source-lock path is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _protocol(profile: str) -> NativeOptimizerAdmissionProtocol:
    if profile == "stress":
        return NativeOptimizerAdmissionProtocol.stress()
    if profile == "main":
        return NativeOptimizerAdmissionProtocol()
    raise ValueError("profile must be stress or main")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Required {label} must contain a JSON object: {resolved}")
    return payload


def _prerequisite_records(args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed unless all logically prior, independently verified gates pass."""
    processor_result = _load_json_object(args.processor_result, label="processor result")
    processor_verification = _load_json_object(
        args.processor_verification, label="processor verification"
    )
    if (
        processor_result.get("gate") != NATIVE_QWEN_HISTORY_PREFLIGHT_GATE_NAME
        or processor_result.get("status") != "passed"
    ):
        raise RuntimeError("Corrected one-token processor preflight has not passed.")
    if (
        processor_verification.get("gate") != NATIVE_QWEN_HISTORY_VERIFIER_GATE_NAME
        or processor_verification.get("status") != "passed"
        or processor_verification.get("error_count") != 0
        or processor_verification.get("result_sha256") != file_sha256(args.processor_result)
    ):
        raise RuntimeError("Corrected processor preflight lacks a linked passing verification.")

    sensitivity_result = _load_json_object(
        args.sensitivity_result, label="sensitivity result"
    )
    sensitivity_verification = _load_json_object(
        args.sensitivity_verification, label="sensitivity verification"
    )
    if (
        sensitivity_result.get("gate") != NATIVE_SENSITIVITY_V2_GATE_NAME
        or sensitivity_result.get("status") != "passed"
        or sensitivity_result.get("summary", {}).get("verdict")
        != SENSITIVITY_ADMISSION_VERDICT
    ):
        raise RuntimeError(
            "Visual-sensitivity v2 confirmation did not return the preregistered "
            "logit-sensitive optimizer-admit verdict."
        )
    if (
        sensitivity_verification.get("gate") != SENSITIVITY_VERIFIER_GATE
        or sensitivity_verification.get("status") != "passed"
        or sensitivity_verification.get("error_count") != 0
        or sensitivity_verification.get("result_sha256")
        != file_sha256(args.sensitivity_result)
    ):
        raise RuntimeError("Sensitivity diagnostic lacks a linked passing verification.")

    records: dict[str, Any] = {
        "processor_preflight": {
            "result_path": str(args.processor_result.expanduser().resolve()),
            "result_sha256": file_sha256(args.processor_result),
            "verification_path": str(args.processor_verification.expanduser().resolve()),
            "verification_sha256": file_sha256(args.processor_verification),
        },
        "visual_sensitivity_v2": {
            "verdict": SENSITIVITY_ADMISSION_VERDICT,
            "result_path": str(args.sensitivity_result.expanduser().resolve()),
            "result_sha256": file_sha256(args.sensitivity_result),
            "verification_path": str(args.sensitivity_verification.expanduser().resolve()),
            "verification_sha256": file_sha256(args.sensitivity_verification),
        },
    }
    if args.profile == "main":
        if args.stress_suite is None or args.stress_verification is None:
            raise RuntimeError("Main profile requires the completed stress suite and verification.")
        stress_suite = _load_json_object(args.stress_suite, label="stress suite")
        stress_verification = _load_json_object(
            args.stress_verification, label="stress verification"
        )
        if (
            stress_suite.get("gate") != NATIVE_OPTIMIZER_ADMISSION_GATE
            or stress_suite.get("profile") != "stress"
            or stress_suite.get("status") != "passed"
            or stress_suite.get("protocol")
            != NativeOptimizerAdmissionProtocol.stress().to_record()
        ):
            raise RuntimeError("Optimizer-admission stress suite has not passed.")
        if stress_suite.get("source_locks") != _source_locks(args.repo.expanduser().resolve()):
            raise RuntimeError(
                "Optimizer-admission code changed after the stress suite; rerun stress first."
            )
        if (
            stress_verification.get("gate") != OPTIMIZER_VERIFIER_GATE
            or stress_verification.get("profile") != "stress"
            or stress_verification.get("status") != "passed"
            or stress_verification.get("error_count") != 0
            or stress_verification.get("suite_sha256") != file_sha256(args.stress_suite)
        ):
            raise RuntimeError("Stress suite lacks a linked passing independent verification.")
        records["stress_profile"] = {
            "suite_path": str(args.stress_suite.expanduser().resolve()),
            "suite_sha256": file_sha256(args.stress_suite),
            "verification_path": str(args.stress_verification.expanduser().resolve()),
            "verification_sha256": file_sha256(args.stress_verification),
        }
    return records


def _selector_protocol(protocol: NativeOptimizerAdmissionProtocol) -> NativeHistoryProtocol:
    train_count = protocol.unique_train_decisions
    return replace(
        NativeHistoryProtocol(),
        planned_updates=train_count,
        train_samples_per_action=protocol.train_samples_per_action,
        calibration_samples_per_action=protocol.calibration_samples_per_action,
        calibration_steps=(0, train_count),
        checkpoint_interval=train_count,
        min_decision_step=protocol.min_decision_step,
        max_replay_prefix_actions=protocol.max_replay_prefix_actions,
        max_hours=protocol.max_hours,
    )


def _select_samples(
    dataset: Any, protocol: NativeOptimizerAdmissionProtocol
) -> list[dict[str, Any]]:
    return [
        sample.to_record()
        for sample in select_native_history_samples(
            dataset,
            protocol=_selector_protocol(protocol),
        )
    ]


def _prepare_manifest(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID:
        raise ValueError("Docker image ID does not match the admitted native-Qwen Habitat image.")
    repo = args.repo.expanduser().resolve()
    protocol = _protocol(args.profile)
    prerequisites = _prerequisite_records(args)
    dataset = _load_dataset(args)
    samples = _select_samples(dataset, protocol)
    trajectories = [str(sample["trajectory_id"]) for sample in samples]
    if len(trajectories) != len(set(trajectories)):
        raise RuntimeError("Frozen train/calibration partitions reuse a physical trajectory.")
    counts = {
        split: {
            action.name: sum(
                sample["split"] == split and sample["target_action"] == action.name
                for sample in samples
            )
            for action in R2RCEAction
        }
        for split in ("train", "calibration")
    }
    manifest = {
        "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
        "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
        "run_id": args.run_id,
        "status": "frozen",
        "profile": args.profile,
        "protocol": protocol.to_record(),
        "dataset": {
            "release_manifest": (
                asdict(dataset.release_manifest) if dataset.release_manifest is not None else None
            ),
            "episodes_path": str(args.episodes),
            "episodes_sha256": file_sha256(args.episodes),
            "ground_truth_path": str(args.ground_truth),
            "ground_truth_sha256": file_sha256(args.ground_truth),
            "split_used": "train",
            "sealed_splits": ["val_seen", "val_unseen"],
        },
        "model": {
            "model_id": NATIVE_C0_MODEL_ID,
            "revision": NATIVE_C0_MODEL_REVISION,
            "dtype": "bf16_base_fp32_lora_and_adam",
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "base_weights_frozen": True,
            "fresh_seeded_adapter_per_arm": True,
        },
        "samples": samples,
        "sample_counts": counts,
        "sample_records_sha256": _canonical_sha256(samples),
        "source_locks": _source_locks(repo),
        "prerequisites": prerequisites,
        "runtime": {
            "docker_image_ref": args.docker_image_ref,
            "docker_image_id": args.docker_image_id,
            "fully_offline": True,
            "sequential_arms": True,
        },
        "evidence_boundary": {
            "established_if_passed": (
                "The frozen native-Qwen LoRA learning-rate arms completed the declared numerical "
                "range finder with exact one-action-token supervision, reconciled units, finite "
                "optimizer state, and independently verified held-out train-split calibration."
            ),
            "not_established": (
                "Navigation competence, generalization, val_seen/val_unseen performance, paper "
                "superiority, convergence, or a transferable learning-rate optimum."
            ),
        },
    }
    _atomic_json(output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _validate_manifest(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], NativeOptimizerAdmissionProtocol, Any, list[dict[str, Any]]]:
    path = args.manifest.expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    protocol = _protocol(args.profile)
    if manifest.get("gate") != NATIVE_OPTIMIZER_ADMISSION_GATE:
        raise RuntimeError("Manifest gate mismatch.")
    if manifest.get("run_id") != args.run_id or manifest.get("profile") != args.profile:
        raise RuntimeError("Manifest run/profile mismatch.")
    if manifest.get("protocol") != protocol.to_record():
        raise RuntimeError("Manifest protocol differs from the invocation.")
    if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID or manifest.get("runtime") != {
        "docker_image_ref": args.docker_image_ref,
        "docker_image_id": args.docker_image_id,
        "fully_offline": True,
        "sequential_arms": True,
    }:
        raise RuntimeError("Manifest Docker/runtime identity differs from the invocation.")
    if manifest.get("source_locks") != _source_locks(args.repo.expanduser().resolve()):
        raise RuntimeError("Manifest source locks no longer match the repository.")
    if manifest.get("prerequisites") != _prerequisite_records(args):
        raise RuntimeError("Manifest prerequisites no longer match the verified gate artifacts.")
    dataset = _load_dataset(args)
    samples = _select_samples(dataset, protocol)
    if manifest.get("samples") != samples:
        raise RuntimeError("Manifest samples do not reconstruct from the locked train release.")
    if manifest.get("sample_records_sha256") != _canonical_sha256(samples):
        raise RuntimeError("Manifest sample-record hash mismatch.")
    return manifest, protocol, dataset, samples


def _restricted_loss(
    logits: torch.Tensor,
    *,
    action_position: int,
    codebook: ActionTokenCodebook,
    target_action: R2RCEAction,
) -> torch.Tensor:
    row = logits[0, action_position - 1, list(codebook.action_token_ids)].float()
    target = torch.tensor([int(target_action)], device=row.device, dtype=torch.long)
    return functional.cross_entropy(row.unsqueeze(0), target)


def _decision_record(
    *,
    outputs: Any,
    action_position: int,
    codebook: ActionTokenCodebook,
    sample: Mapping[str, Any],
    cache_sha256: str,
    batch_record: Mapping[str, Any],
) -> dict[str, Any]:
    decision = score_action_token_logits(
        outputs.logits,
        action_token_position=action_position,
        codebook=codebook,
    )
    target = str(sample["target_action"])
    score = decision["scores"][target]
    probability = float(score["restricted_probability"])
    if not math.isfinite(probability) or probability <= 0.0:
        raise RuntimeError("Target restricted probability is invalid.")
    return {
        "sample_index": int(sample["sample_index"]),
        "episode_id": str(sample["episode_id"]),
        "trajectory_id": str(sample["trajectory_id"]),
        "target_action": target,
        "predicted_action": decision["predicted_action"],
        "correct": decision["predicted_action"] == target,
        "target_token_nll": float(score["token_nll"]),
        "target_restricted_probability": probability,
        "restricted_action_ce": -math.log(probability),
        "cache_sha256": cache_sha256,
        "action_token_position": action_position,
        "supervision_schema_version": batch_record["supervision_schema_version"],
        "supervision_method": batch_record["supervision_method"],
        "supervised_token_count": batch_record["supervised_token_count"],
        "supervised_token_id": batch_record["supervised_token_id"],
        "action_decision": decision,
    }


def _calibrate(
    *,
    model: Any,
    processor: Any,
    codebook: ActionTokenCodebook,
    samples: list[dict[str, Any]],
    cache: dict[tuple[str, int], dict[str, Any]],
    update_step: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            entry = cache[("calibration", int(sample["sample_index"]))]
            cached = _load_cached(entry, sample)
            batch, position = _native_batch(
                processor=processor,
                codebook=codebook,
                cached=cached,
                sample=sample,
                arm_id="H0_CURRENT",
            )
            inputs = _move_batch(batch, device)
            inputs.pop("labels")
            outputs = model(**inputs)
            row = _decision_record(
                outputs=outputs,
                action_position=position,
                codebook=codebook,
                sample=sample,
                cache_sha256=entry["sha256"],
                batch_record=batch.record,
            )
            if row["restricted_action_ce"] > 100.0:
                raise RuntimeError("Calibration restricted action CE exceeded the hard ceiling.")
            rows.append(row)
            del outputs, inputs, batch
    model.train()
    return {
        "update_step": update_step,
        "summary": summarize_calibration(rows),
        "per_decision": rows,
    }


def _state_l2_norm(state: Mapping[str, torch.Tensor]) -> float:
    value = sum(float(tensor.detach().float().pow(2).sum().item()) for tensor in state.values())
    return math.sqrt(value)


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, Any],
    cursor: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite optimizer checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(model)
    payload = {
        "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
        "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
        "metadata": dict(metadata),
        "cursor": dict(cursor),
        "trainable_state_dict": state,
        "trainable_state_sha256": tensor_state_sha256(state),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "path": str(path),
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "trainable_state_sha256": payload["trainable_state_sha256"],
        "update_step": int(cursor["update_step"]),
    }


def _arm_id(rate: float) -> str:
    return f"lr_{rate:.0e}".replace("-", "m").replace("+", "p")


def _run_arm(
    *,
    args: argparse.Namespace,
    protocol: NativeOptimizerAdmissionProtocol,
    manifest_hash: str,
    cache_manifest_hash: str,
    samples: list[dict[str, Any]],
    cache: dict[tuple[str, int], dict[str, Any]],
    processor: Any,
    codebook: ActionTokenCodebook,
    learning_rate: float,
    device: torch.device,
    suite_started: float,
    expected_initial_sha: str | None,
) -> tuple[dict[str, Any], str]:
    arm_id = _arm_id(learning_rate)
    output_root = args.output_root.expanduser().resolve()
    arm_root = output_root / "arms" / arm_id
    checkpoint_root = args.checkpoint_root.expanduser().resolve() / arm_id
    arm_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    result_path = arm_root / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to reuse completed arm: {result_path}")

    model = _load_model(device, protocol.model_seed)
    trainable_contract = validate_lora_only_trainables(model)
    initial_state = trainable_state_dict(model)
    initial_sha = tensor_state_sha256(initial_state)
    if expected_initial_sha is not None and initial_sha != expected_initial_sha:
        raise RuntimeError("Learning-rate arms did not start from identical LoRA initialization.")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=protocol.weight_decay,
    )
    metadata = {
        "manifest_sha256": manifest_hash,
        "cache_manifest_sha256": cache_manifest_hash,
        "profile": protocol.profile,
        "arm_id": arm_id,
        "learning_rate": learning_rate,
        "initial_trainable_state_sha256": initial_sha,
    }
    train_samples = [sample for sample in samples if sample["split"] == "train"]
    calibration_samples = [sample for sample in samples if sample["split"] == "calibration"]
    train_by_index = {int(sample["sample_index"]): sample for sample in train_samples}
    exposure_order = training_sample_order(train_samples, protocol)
    calibrations = [
        _calibrate(
            model=model,
            processor=processor,
            codebook=codebook,
            samples=calibration_samples,
            cache=cache,
            update_step=0,
            device=device,
        )
    ]
    events_path = arm_root / "updates.jsonl"
    checkpoints: list[dict[str, Any]] = []
    changed_update_count = 0
    all_updates_finite = True
    arm_started = time.perf_counter()
    completed_updates = 0

    with events_path.open("x", encoding="utf-8") as event_handle:
        for update_step in range(1, protocol.planned_optimizer_updates + 1):
            if _STOP_REQUESTED or time.perf_counter() - suite_started >= protocol.max_hours * 3600:
                break
            start = (update_step - 1) * protocol.accumulation_steps
            indices = exposure_order[start : start + protocol.accumulation_steps]
            if len(indices) != protocol.accumulation_steps:
                raise RuntimeError("Exposure order ended before a complete optimizer update.")
            optimizer.zero_grad(set_to_none=True)
            before = trainable_state_dict(model)
            before_sha = tensor_state_sha256(before)
            before_norm = _state_l2_norm(before)
            micro_rows: list[dict[str, Any]] = []
            update_started = time.perf_counter()
            for microbatch_index, sample_index in enumerate(indices):
                sample = train_by_index[sample_index]
                entry = cache[("train", sample_index)]
                cached = _load_cached(entry, sample)
                batch, position = _native_batch(
                    processor=processor,
                    codebook=codebook,
                    cached=cached,
                    sample=sample,
                    arm_id="H0_CURRENT",
                )
                inputs = _move_batch(batch, device)
                inputs.pop("labels")
                model.train()
                outputs = model(**inputs)
                target = R2RCEAction(int(sample["target_action_id"]))
                loss = _restricted_loss(
                    outputs.logits,
                    action_position=position,
                    codebook=codebook,
                    target_action=target,
                )
                loss_value = float(loss.detach().cpu().item())
                if not math.isfinite(loss_value) or loss_value < 0.0:
                    raise RuntimeError("Training produced an invalid restricted action CE.")
                if loss_value > protocol.hard_max_restricted_ce:
                    raise RuntimeError("Training restricted action CE exceeded the hard ceiling.")
                (loss / protocol.accumulation_steps).backward()
                scored = score_action_token_logits(
                    outputs.logits,
                    action_token_position=position,
                    codebook=codebook,
                )
                target_score = scored["scores"][sample["target_action"]]
                micro_rows.append(
                    {
                        "microbatch_index": microbatch_index,
                        "sample_index": sample_index,
                        "episode_id": sample["episode_id"],
                        "trajectory_id": sample["trajectory_id"],
                        "target_action": sample["target_action"],
                        "restricted_action_ce": loss_value,
                        "target_token_nll": float(target_score["token_nll"]),
                        "target_restricted_probability": float(
                            target_score["restricted_probability"]
                        ),
                        "predicted_action": scored["predicted_action"],
                        "cache_sha256": entry["sha256"],
                        "action_token_position": position,
                        "supervised_token_count": batch.record["supervised_token_count"],
                        "supervised_token_id": batch.record["supervised_token_id"],
                    }
                )
                del outputs, loss, inputs, batch

            gradients = gradient_report(model)
            if (
                gradients["gradient_tensor_count"] != gradients["trainable_tensor_count"]
                or gradients["nonzero_gradient_tensor_count"] <= 0
                or gradients["all_gradients_finite"] is not True
            ):
                raise RuntimeError("Accumulated LoRA gradients are missing, zero, or non-finite.")
            preclip = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    protocol.max_grad_norm,
                )
                .detach()
                .float()
                .cpu()
                .item()
            )
            if (
                not math.isfinite(preclip)
                or preclip <= 0.0
                or preclip > protocol.hard_max_gradient_norm
            ):
                raise RuntimeError("Global pre-clip gradient norm is invalid.")
            multiplier = learning_rate_multiplier(update_step, protocol)
            applied_lr = learning_rate * multiplier
            if not math.isfinite(applied_lr) or applied_lr <= 0.0:
                raise RuntimeError("Scheduled learning rate is invalid.")
            for group in optimizer.param_groups:
                group["lr"] = applied_lr
            optimizer.step()
            updates = update_report(model, before)
            validate_optimization_reports(gradients, updates)
            changed_update_count += 1
            all_updates_finite = all_updates_finite and bool(updates["all_updates_finite"])
            after_sha = tensor_state_sha256(trainable_state_dict(model))
            if after_sha == before_sha:
                raise RuntimeError("Optimizer update changed no trainable bytes.")
            update_norm = float(updates["update_l2_norm"])
            event = {
                "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
                "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
                "profile": protocol.profile,
                "arm_id": arm_id,
                "learning_rate": learning_rate,
                "update_step": update_step,
                "epoch_index": (start // protocol.unique_train_decisions),
                "sample_exposure_start": start,
                "sample_exposure_end_exclusive": start + len(indices),
                "microbatch_count": len(indices),
                "accumulation_steps": protocol.accumulation_steps,
                "base_learning_rate": learning_rate,
                "learning_rate_multiplier": multiplier,
                "applied_learning_rate": applied_lr,
                "objective": NATIVE_OPTIMIZER_OBJECTIVE,
                "mean_restricted_action_ce": sum(
                    row["restricted_action_ce"] for row in micro_rows
                )
                / len(micro_rows),
                "mean_target_token_nll": sum(row["target_token_nll"] for row in micro_rows)
                / len(micro_rows),
                "gradient_report": gradients,
                "global_gradient_norm_before_clip": preclip,
                "clip_max_norm": protocol.max_grad_norm,
                "clip_ratio": min(1.0, protocol.max_grad_norm / preclip),
                "update_report": updates,
                "parameter_l2_norm_before": before_norm,
                "update_to_weight_ratio": update_norm / before_norm,
                "before_trainable_state_sha256": before_sha,
                "after_trainable_state_sha256": after_sha,
                "optimizer_state": optimizer_tensor_diagnostics(optimizer),
                "microbatches": micro_rows,
                "elapsed_seconds": time.perf_counter() - update_started,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
            event_handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            event_handle.flush()
            os.fsync(event_handle.fileno())
            completed_updates = update_step

            if update_step in protocol.calibration_updates:
                calibrations.append(
                    _calibrate(
                        model=model,
                        processor=processor,
                        codebook=codebook,
                        samples=calibration_samples,
                        cache=cache,
                        update_step=update_step,
                        device=device,
                    )
                )
            if update_step in protocol.checkpoint_updates or _STOP_REQUESTED:
                checkpoint = _save_checkpoint(
                    checkpoint_root / f"step_{update_step:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    metadata=metadata,
                    cursor={
                        "update_step": update_step,
                        "sample_exposure_end_exclusive": start + len(indices),
                    },
                )
                checkpoints.append(checkpoint)
                _atomic_json(
                    output_root / "progress.json",
                    {
                        "status": "running",
                        "profile": protocol.profile,
                        "current_arm": arm_id,
                        "current_learning_rate": learning_rate,
                        "completed_update": update_step,
                        "planned_updates_per_arm": protocol.planned_optimizer_updates,
                        "latest_checkpoint": checkpoint,
                        "elapsed_suite_seconds": time.perf_counter() - suite_started,
                        "recorded_at_utc": _utc_now(),
                    },
                )
            del before

    completed = completed_updates == protocol.planned_optimizer_updates
    if not completed and completed_updates > 0 and (
        not checkpoints or checkpoints[-1]["update_step"] != completed_updates
    ):
        checkpoints.append(
            _save_checkpoint(
                checkpoint_root / f"step_{completed_updates:04d}_partial.pt",
                model=model,
                optimizer=optimizer,
                metadata=metadata,
                cursor={
                    "update_step": completed_updates,
                    "sample_exposure_end_exclusive": (
                        completed_updates * protocol.accumulation_steps
                    ),
                },
            )
        )
    result = {
        "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
        "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
        "profile": protocol.profile,
        "arm_id": arm_id,
        "learning_rate": learning_rate,
        "status": "passed" if completed else "partial",
        "terminal_reason": (
            "planned_updates_completed"
            if completed
            else "termination_signal" if _STOP_REQUESTED else "time_budget"
        ),
        "manifest_sha256": manifest_hash,
        "cache_manifest_sha256": cache_manifest_hash,
        "initial_trainable_state_sha256": initial_sha,
        "final_trainable_state_sha256": tensor_state_sha256(trainable_state_dict(model)),
        "trainable_contract": trainable_contract,
        "completed_updates": completed_updates,
        "changed_update_count": changed_update_count,
        "all_updates_finite": all_updates_finite,
        "calibrations": calibrations,
        "checkpoints": checkpoints,
        "events_path": str(events_path),
        "events_sha256": file_sha256(events_path),
        "training_seconds": time.perf_counter() - arm_started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    _atomic_json(result_path, result)
    del optimizer, model, initial_state
    gc.collect()
    torch.cuda.empty_cache()
    return result, initial_sha


def _run(args: argparse.Namespace) -> int:
    suite_started = time.perf_counter()
    manifest, protocol, dataset, samples = _validate_manifest(args)
    manifest_path = args.manifest.expanduser().resolve()
    manifest_hash = file_sha256(manifest_path)
    cache_manifest = _prepare_cache(
        args=args,
        manifest_hash=manifest_hash,
        dataset=dataset,
        samples=samples,
    )
    cache_manifest_path = args.cache_root.expanduser().resolve() / "cache_manifest.json"
    cache_manifest_hash = file_sha256(cache_manifest_path)
    cache = _cache_index(cache_manifest, args.cache_root.expanduser().resolve())

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native optimizer admission requires a BF16-capable CUDA device.")
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats(device)
    processor = AutoProcessor.from_pretrained(
        NATIVE_C0_MODEL_ID,
        revision=NATIVE_C0_MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    codebook = build_action_token_codebook(processor.tokenizer)
    weight_locks = model_weight_locks(args.model_snapshot.expanduser().resolve())
    arm_results: list[dict[str, Any]] = []
    expected_initial_sha: str | None = None
    for learning_rate in protocol.learning_rates:
        result, initial_sha = _run_arm(
            args=args,
            protocol=protocol,
            manifest_hash=manifest_hash,
            cache_manifest_hash=cache_manifest_hash,
            samples=samples,
            cache=cache,
            processor=processor,
            codebook=codebook,
            learning_rate=learning_rate,
            device=device,
            suite_started=suite_started,
            expected_initial_sha=expected_initial_sha,
        )
        if expected_initial_sha is None:
            expected_initial_sha = initial_sha
        arm_results.append(result)
        if result["status"] != "passed":
            _atomic_json(
                args.output_root.expanduser().resolve() / "partial.json",
                {
                    "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
                    "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
                    "status": "partial",
                    "profile": protocol.profile,
                    "completed_arm_count": sum(row["status"] == "passed" for row in arm_results),
                    "current_arm": result["arm_id"],
                    "resume_not_implemented": True,
                    "recorded_at_utc": _utc_now(),
                },
            )
            return 2

    selection = select_admitted_rate(arm_results, protocol)
    output_root = args.output_root.expanduser().resolve()
    arm_index = []
    for result in arm_results:
        result_path = output_root / "arms" / result["arm_id"] / "result.json"
        arm_index.append(
            {
                "arm_id": result["arm_id"],
                "learning_rate": result["learning_rate"],
                "relative_path": str(result_path.relative_to(output_root)),
                "sha256": file_sha256(result_path),
            }
        )
    suite = {
        "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
        "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
        "run_id": args.run_id,
        "status": "passed",
        "profile": protocol.profile,
        "completed_at_utc": _utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "cache_manifest_path": str(cache_manifest_path),
        "cache_manifest_sha256": cache_manifest_hash,
        "protocol": protocol.to_record(),
        "arm_index": arm_index,
        "selection": selection,
        "identical_initial_trainable_state_sha256": expected_initial_sha,
        "action_codebook": codebook.to_record(),
        "supervision": {
            "schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
            "method": NATIVE_QWEN_SUPERVISION_METHOD,
            "supervised_token_count_per_decision": 1,
        },
        "model_weight_locks": weight_locks,
        "source_locks": manifest["source_locks"],
        "prerequisites": manifest["prerequisites"],
        "runtime": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "elapsed_suite_seconds": time.perf_counter() - suite_started,
            "docker_image_id": EXPECTED_DOCKER_IMAGE_ID,
        },
        "evidence_boundary": manifest["evidence_boundary"],
    }
    suite_path = output_root / "suite.json"
    _atomic_json(suite_path, suite)
    _atomic_json(
        output_root / "progress.json",
        {
            "status": "passed",
            "profile": protocol.profile,
            "completed_arms": [row["arm_id"] for row in arm_results],
            "selection_status": selection["status"],
            "selected_learning_rate": selection["selected_learning_rate"],
            "elapsed_suite_seconds": time.perf_counter() - suite_started,
            "recorded_at_utc": _utc_now(),
        },
    )
    print(json.dumps(suite, indent=2, sort_keys=True))
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("stress", "main"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--processor-result", type=Path, required=True)
    parser.add_argument("--processor-verification", type=Path, required=True)
    parser.add_argument("--sensitivity-result", type=Path, required=True)
    parser.add_argument("--sensitivity-verification", type=Path, required=True)
    parser.add_argument("--stress-suite", type=Path)
    parser.add_argument("--stress-verification", type=Path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _common(prepare)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    _common(run)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--scene-root", type=Path, required=True)
    run.add_argument("--model-snapshot", type=Path, required=True)
    run.add_argument("--cache-root", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        if args.command == "prepare":
            return _prepare_manifest(args)
        return _run(args)
    except Exception as exc:
        failure = {
            "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
            "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
            "status": "failed",
            "recorded_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_root = getattr(args, "output_root", None)
        if isinstance(output_root, Path):
            try:
                _atomic_json(output_root.expanduser().resolve() / "failure.json", failure)
            except Exception:
                pass
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
