#!/usr/bin/env python3
"""Build the train inventory and run the bounded native-Qwen causal-H4 benchmark."""

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
from spatialvlm.training.native_development_measurements import (
    NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
    NATIVE_DEVELOPMENT_MEASUREMENTS_SCHEMA_VERSION,
    NATIVE_H4_HISTORY,
    NATIVE_H4_OBJECTIVE,
    NativeDataInventoryProtocol,
    NativeH4BenchmarkProtocol,
    build_data_policy_inventory,
    h4_training_sample_order,
    summarize_h4_timings,
)
from spatialvlm.training.native_history_select import (
    NativeHistoryProtocol,
    capture_rng_state,
    optimizer_tensor_diagnostics,
    select_native_history_samples,
)
from spatialvlm.training.native_optimizer_admission import summarize_calibration

SOURCE_LOCK_PATHS: tuple[str, ...] = (
    "docs/NATIVE_C0_DEVELOPMENT_MEASUREMENTS.md",
    "pyproject.toml",
    "scripts/__init__.py",
    "scripts/check_context_state.py",
    "scripts/check_native_development_measurements_artifacts.py",
    "scripts/native_development_measurements_sync_files.txt",
    "scripts/run_spark_native_development_measurements.py",
    "scripts/run_spark_native_development_measurements.sh",
    "scripts/spark_launch_helpers.sh",
    "scripts/start_spark_native_development_measurements.sh",
    "scripts/run_spark_native_history_select.py",
    "src/spatialvlm/backbone/qwen3_vl.py",
    "src/spatialvlm/backbone/rope_patch.py",
    "src/spatialvlm/data/datasets.py",
    "src/spatialvlm/data/habitat_sim_r2r.py",
    "src/spatialvlm/data/history.py",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/data/native_qwen_artifacts.py",
    "src/spatialvlm/eval/action_decision.py",
    "src/spatialvlm/training/native_c0_gate.py",
    "src/spatialvlm/training/native_development_measurements.py",
    "src/spatialvlm/training/native_history_select.py",
    "src/spatialvlm/training/native_optimizer_admission.py",
    "src/spatialvlm/training/unit_ledger.py",
    "tests/test_native_development_measurements.py",
    "tests/test_native_development_measurements_artifacts.py",
    "tests/test_action_decision.py",
    "tests/test_native_history_select.py",
    "tests/test_native_history_select_artifacts.py",
    "tests/test_native_qwen.py",
    "tests/test_spark_launch_helpers.py",
    "tests/test_unit_ledger.py",
)

OPTIMIZER_GATE = "native_qwen_optimizer_admission_v1"
OPTIMIZER_VERIFIER_GATE = "independent_native_qwen_optimizer_admission_verification_v1"
PRIOR_MEASUREMENTS_GATE = "native_c0_development_measurements_v1"
MEASUREMENTS_VERIFIER_GATE = "independent_native_c0_development_measurements_verification_v2"
_STOP_REQUESTED = False


def _request_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("Termination requested; stopping after the current complete H4 update.", flush=True)


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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Required {label} must be a JSON object: {resolved}")
    return payload


def _prerequisites(args: argparse.Namespace) -> dict[str, Any]:
    prior_manifest = _load_json(args.prior_manifest, "failed v1 measurement manifest")
    prior_failure = _load_json(args.prior_failure, "failed v1 measurement record")
    if (
        prior_manifest.get("gate") != PRIOR_MEASUREMENTS_GATE
        or prior_manifest.get("run_id") != "native_development_measurements_v1"
        or prior_manifest.get("status") != "frozen"
    ):
        raise RuntimeError("The preserved v1 measurement manifest has the wrong identity.")
    if (
        prior_failure.get("gate") != PRIOR_MEASUREMENTS_GATE
        or prior_failure.get("status") != "failed"
        or prior_failure.get("error_type") != "AttributeError"
        or "ActionTokenCodebook" not in str(prior_failure.get("error", ""))
        or "tokens" not in str(prior_failure.get("error", ""))
    ):
        raise RuntimeError("The preserved v1 failure is not the admitted codebook-API failure.")
    processor_result = _load_json(args.processor_result, "processor result")
    processor_verification = _load_json(args.processor_verification, "processor verification")
    if (
        processor_result.get("gate") != NATIVE_QWEN_HISTORY_PREFLIGHT_GATE_NAME
        or processor_result.get("status") != "passed"
    ):
        raise RuntimeError("The corrected one-token processor gate has not passed.")
    if (
        processor_verification.get("gate") != NATIVE_QWEN_HISTORY_VERIFIER_GATE_NAME
        or processor_verification.get("status") != "passed"
        or processor_verification.get("error_count") != 0
        or processor_verification.get("result_sha256") != file_sha256(args.processor_result)
    ):
        raise RuntimeError("Processor evidence lacks a linked passing verification.")

    optimizer_suite = _load_json(args.optimizer_suite, "optimizer-admission suite")
    optimizer_verification = _load_json(
        args.optimizer_verification, "optimizer-admission verification"
    )
    if (
        optimizer_suite.get("gate") != OPTIMIZER_GATE
        or optimizer_suite.get("profile") != "main"
        or optimizer_suite.get("status") != "passed"
    ):
        raise RuntimeError("Main optimizer admission has not passed.")
    if (
        optimizer_verification.get("gate") != OPTIMIZER_VERIFIER_GATE
        or optimizer_verification.get("profile") != "main"
        or optimizer_verification.get("status") != "passed"
        or optimizer_verification.get("error_count") != 0
        or optimizer_verification.get("suite_sha256") != file_sha256(args.optimizer_suite)
    ):
        raise RuntimeError("Optimizer evidence lacks a linked passing verification.")
    return {
        "prior_attempt": {
            "run_id": "native_development_measurements_v1",
            "manifest_path": str(args.prior_manifest.expanduser().resolve()),
            "manifest_sha256": file_sha256(args.prior_manifest),
            "failure_path": str(args.prior_failure.expanduser().resolve()),
            "failure_sha256": file_sha256(args.prior_failure),
            "failure_boundary": (
                "Cache 144/144 completed; exact Qwen loaded; training stopped before update 1 "
                "because the runner referenced a nonexistent ActionTokenCodebook.tokens field."
            ),
        },
        "processor": {
            "result_path": str(args.processor_result.expanduser().resolve()),
            "result_sha256": file_sha256(args.processor_result),
            "verification_path": str(args.processor_verification.expanduser().resolve()),
            "verification_sha256": file_sha256(args.processor_verification),
        },
        "optimizer_admission": {
            "suite_path": str(args.optimizer_suite.expanduser().resolve()),
            "suite_sha256": file_sha256(args.optimizer_suite),
            "verification_path": str(args.optimizer_verification.expanduser().resolve()),
            "verification_sha256": file_sha256(args.optimizer_verification),
        },
    }


def _selector_protocol(protocol: NativeH4BenchmarkProtocol) -> NativeHistoryProtocol:
    return replace(
        NativeHistoryProtocol(),
        planned_updates=protocol.unique_train_decisions,
        train_samples_per_action=protocol.train_samples_per_action,
        calibration_samples_per_action=protocol.calibration_samples_per_action,
        calibration_steps=(0, protocol.unique_train_decisions),
        checkpoint_interval=protocol.unique_train_decisions,
        min_decision_step=protocol.min_decision_step,
        max_replay_prefix_actions=protocol.max_replay_prefix_actions,
        max_hours=protocol.max_hours,
    )


def _select_samples(dataset: Any, protocol: NativeH4BenchmarkProtocol) -> list[dict[str, Any]]:
    return [
        sample.to_record()
        for sample in select_native_history_samples(
            dataset,
            protocol=_selector_protocol(protocol),
        )
    ]


def _prepare(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    inventory_path = output_root / "inventory.json"
    manifest_path = output_root / "manifest.json"
    if inventory_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite development-measurement metadata.")
    if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID:
        raise ValueError("Docker image ID does not match the admitted Habitat/Qwen image.")
    protocol = NativeH4BenchmarkProtocol()
    dataset = _load_dataset(args)
    inventory = build_data_policy_inventory(
        dataset,
        protocol=NativeDataInventoryProtocol(
            min_decision_step=protocol.min_decision_step,
            max_replay_prefix_actions=protocol.max_replay_prefix_actions,
        ),
    )
    samples = _select_samples(dataset, protocol)
    trajectories = [str(sample["trajectory_id"]) for sample in samples]
    if len(trajectories) != len(set(trajectories)):
        raise RuntimeError("H4 benchmark train/calibration partitions reuse a trajectory.")
    prerequisites = _prerequisites(args)
    _atomic_json(inventory_path, inventory)
    manifest = {
        "schema_version": NATIVE_DEVELOPMENT_MEASUREMENTS_SCHEMA_VERSION,
        "gate": NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
        "run_id": args.run_id,
        "status": "frozen",
        "protocol": protocol.to_record(),
        "dataset": {
            "release_manifest": asdict(dataset.release_manifest),
            "episodes_path": str(args.episodes.expanduser().resolve()),
            "episodes_sha256": file_sha256(args.episodes),
            "ground_truth_path": str(args.ground_truth.expanduser().resolve()),
            "ground_truth_sha256": file_sha256(args.ground_truth),
            "split_used": "train",
            "sealed_splits": ["val_seen", "val_unseen"],
        },
        "inventory_path": str(inventory_path),
        "inventory_file_sha256": file_sha256(inventory_path),
        "inventory_content_sha256": inventory["inventory_sha256"],
        "samples": samples,
        "sample_records_sha256": _canonical_sha256(samples),
        "sample_counts": {
            split: {
                action.name: sum(
                    sample["split"] == split and sample["target_action"] == action.name
                    for sample in samples
                )
                for action in R2RCEAction
            }
            for split in ("train", "calibration")
        },
        "model": {
            "model_id": NATIVE_C0_MODEL_ID,
            "revision": NATIVE_C0_MODEL_REVISION,
            "base_dtype": "bf16",
            "adapter_and_optimizer_dtype": "fp32",
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "fresh_seeded_adapter": True,
        },
        "runtime": {
            "docker_image_ref": args.docker_image_ref,
            "docker_image_id": args.docker_image_id,
            "fully_offline": True,
        },
        "source_locks": _source_locks(args.repo.expanduser().resolve()),
        "prerequisites": prerequisites,
        "evidence_boundary": {
            "established_if_passed": (
                "Exact train-release inventory plus measured causal-H4 effective-batch-16 "
                "throughput, calibration/checkpoint cost, and peak accelerator memory."
            ),
            "not_established": (
                "Final development sample size or runtime cap, learning, checkpoint quality, "
                "closed-loop competence, val_seen/val_unseen performance, or paper claims."
            ),
        },
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"inventory": inventory["summary"], "manifest": manifest}, indent=2))
    return 0


def _validate_manifest(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], NativeH4BenchmarkProtocol, Any, list[dict[str, Any]]]:
    output_root = args.output_root.expanduser().resolve()
    manifest = _load_json(output_root / "manifest.json", "measurement manifest")
    inventory = _load_json(output_root / "inventory.json", "data inventory")
    protocol = NativeH4BenchmarkProtocol()
    if (
        manifest.get("gate") != NATIVE_DEVELOPMENT_MEASUREMENTS_GATE
        or manifest.get("run_id") != args.run_id
        or manifest.get("protocol") != protocol.to_record()
    ):
        raise RuntimeError("Development-measurement manifest identity/protocol mismatch.")
    if manifest.get("runtime") != {
        "docker_image_ref": args.docker_image_ref,
        "docker_image_id": args.docker_image_id,
        "fully_offline": True,
    }:
        raise RuntimeError("Development-measurement runtime identity mismatch.")
    if manifest.get("source_locks") != _source_locks(args.repo.expanduser().resolve()):
        raise RuntimeError("Development-measurement source locks changed after freeze.")
    if manifest.get("prerequisites") != _prerequisites(args):
        raise RuntimeError("Development-measurement prerequisites changed after freeze.")
    dataset = _load_dataset(args)
    rebuilt_inventory = build_data_policy_inventory(
        dataset,
        protocol=protocol.inventory_protocol,
    )
    if inventory != rebuilt_inventory:
        raise RuntimeError("Data inventory does not reconstruct from the locked train release.")
    if manifest.get("inventory_file_sha256") != file_sha256(output_root / "inventory.json"):
        raise RuntimeError("Manifest data-inventory file hash mismatch.")
    samples = _select_samples(dataset, protocol)
    if manifest.get("samples") != samples:
        raise RuntimeError("H4 benchmark samples do not reconstruct.")
    if manifest.get("sample_records_sha256") != _canonical_sha256(samples):
        raise RuntimeError("H4 benchmark sample-record hash mismatch.")
    return manifest, protocol, dataset, samples


def _restricted_loss(
    logits: torch.Tensor,
    *,
    action_position: int,
    codebook: ActionTokenCodebook,
    target_action: R2RCEAction,
) -> torch.Tensor:
    action_ids = [codebook.action_token_id(action) for action in R2RCEAction]
    row = logits[0, action_position - 1].float()
    restricted = row[torch.tensor(action_ids, device=row.device)]
    target_index = list(R2RCEAction).index(target_action)
    return functional.cross_entropy(
        restricted.unsqueeze(0),
        torch.tensor([target_index], device=row.device),
    )


def _decision_record(
    *,
    outputs: Any,
    action_position: int,
    codebook: ActionTokenCodebook,
    sample: Mapping[str, Any],
    cache_sha256: str,
    supervised_count: int,
    supervised_id: int,
) -> dict[str, Any]:
    scored = score_action_token_logits(
        outputs.logits,
        action_token_position=action_position,
        codebook=codebook,
    )
    target = str(sample["target_action"])
    target_score = scored["scores"][target]
    return {
        "sample_index": int(sample["sample_index"]),
        "episode_id": str(sample["episode_id"]),
        "trajectory_id": str(sample["trajectory_id"]),
        "target_action": target,
        "predicted_action": scored["predicted_action"],
        "correct": scored["predicted_action"] == target,
        "target_token_nll": float(target_score["token_nll"]),
        "target_restricted_probability": float(target_score["restricted_probability"]),
        "restricted_action_ce": -math.log(float(target_score["restricted_probability"])),
        "action_decision": scored,
        "cache_sha256": cache_sha256,
        "action_token_position": action_position,
        "supervised_token_count": supervised_count,
        "supervised_token_id": supervised_id,
    }


def _calibrate_h4(
    *,
    model: Any,
    processor: Any,
    codebook: ActionTokenCodebook,
    samples: list[dict[str, Any]],
    cache: dict[tuple[str, int], dict[str, Any]],
    device: torch.device,
    update_step: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            entry = cache[("calibration", int(sample["sample_index"]))]
            cached = _load_cached(
                entry,
                sample,
                gate_name=NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
            )
            batch, position = _native_batch(
                processor=processor,
                codebook=codebook,
                cached=cached,
                sample=sample,
                arm_id=NATIVE_H4_HISTORY,
            )
            inputs = _move_batch(batch, device)
            inputs.pop("labels")
            outputs = model(**inputs)
            rows.append(
                _decision_record(
                    outputs=outputs,
                    action_position=position,
                    codebook=codebook,
                    sample=sample,
                    cache_sha256=entry["sha256"],
                    supervised_count=int(batch.record["supervised_token_count"]),
                    supervised_id=int(batch.record["supervised_token_id"]),
                )
            )
            del outputs, inputs, batch
    return {
        "update_step": update_step,
        "summary": summarize_calibration(rows),
        "per_decision": rows,
    }


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    manifest_sha256: str,
    cache_manifest_sha256: str,
    protocol: NativeH4BenchmarkProtocol,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite H4 benchmark checkpoint: {path}")
    state = trainable_state_dict(model)
    payload = {
        "schema_version": NATIVE_DEVELOPMENT_MEASUREMENTS_SCHEMA_VERSION,
        "gate": NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
        "manifest_sha256": manifest_sha256,
        "cache_manifest_sha256": cache_manifest_sha256,
        "protocol": protocol.to_record(),
        "cursor": {
            "update_step": protocol.planned_optimizer_updates,
            "sample_exposure_end_exclusive": protocol.unique_train_decisions,
        },
        "trainable_state_dict": state,
        "trainable_state_sha256": tensor_state_sha256(state),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "path": str(path),
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "trainable_state_sha256": payload["trainable_state_sha256"],
        "update_step": protocol.planned_optimizer_updates,
    }


def _run(args: argparse.Namespace) -> int:
    run_started = time.perf_counter()
    manifest, protocol, dataset, samples = _validate_manifest(args)
    output_root = args.output_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    manifest_hash = file_sha256(manifest_path)

    cache_started = time.perf_counter()
    cache_manifest = _prepare_cache(
        args=args,
        manifest_hash=manifest_hash,
        dataset=dataset,
        samples=samples,
        gate_name=NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
    )
    cache_prepare_seconds = time.perf_counter() - cache_started
    cache_manifest_path = args.cache_root.expanduser().resolve() / "cache_manifest.json"
    cache_manifest_hash = file_sha256(cache_manifest_path)
    cache = _cache_index(cache_manifest, args.cache_root.expanduser().resolve())

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The native H4 benchmark requires a BF16-capable CUDA device.")
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats(device)
    processor = AutoProcessor.from_pretrained(
        NATIVE_C0_MODEL_ID,
        revision=NATIVE_C0_MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    codebook = build_action_token_codebook(processor.tokenizer)
    model_started = time.perf_counter()
    model = _load_model(device, protocol.model_seed)
    model_load_seconds = time.perf_counter() - model_started
    trainable_contract = validate_lora_only_trainables(model)
    initial_hash = tensor_state_sha256(trainable_state_dict(model))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )

    train_samples = [sample for sample in samples if sample["split"] == "train"]
    calibration_samples = [sample for sample in samples if sample["split"] == "calibration"]
    train_by_index = {int(sample["sample_index"]): sample for sample in train_samples}
    order = h4_training_sample_order(train_samples, protocol)
    events_path = output_root / "updates.jsonl"
    previous_hash = initial_hash
    events: list[dict[str, Any]] = []
    with events_path.open("x", encoding="utf-8") as handle:
        for update_step in range(1, protocol.planned_optimizer_updates + 1):
            if _STOP_REQUESTED:
                raise RuntimeError("H4 benchmark was interrupted before completion.")
            if time.perf_counter() - run_started >= protocol.max_hours * 3600:
                raise TimeoutError("H4 benchmark exceeded its one-hour hard cap.")
            start = (update_step - 1) * protocol.accumulation_steps
            indices = order[start : start + protocol.accumulation_steps]
            if len(indices) != protocol.accumulation_steps:
                raise RuntimeError("H4 benchmark exposure order ended mid-update.")
            optimizer.zero_grad(set_to_none=True)
            before = trainable_state_dict(model)
            before_hash = tensor_state_sha256(before)
            if before_hash != previous_hash:
                raise RuntimeError("H4 benchmark trainable hash chain broke before an update.")
            micro_rows: list[dict[str, Any]] = []
            update_started = time.perf_counter()
            for microbatch_index, sample_index in enumerate(indices):
                sample = train_by_index[sample_index]
                entry = cache[("train", sample_index)]
                cached = _load_cached(
                    entry,
                    sample,
                    gate_name=NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
                )
                batch, position = _native_batch(
                    processor=processor,
                    codebook=codebook,
                    cached=cached,
                    sample=sample,
                    arm_id=NATIVE_H4_HISTORY,
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
                value = float(loss.detach().cpu().item())
                if not math.isfinite(value) or not 0.0 <= value <= protocol.hard_max_restricted_ce:
                    raise RuntimeError("H4 benchmark produced an invalid restricted CE.")
                (loss / protocol.accumulation_steps).backward()
                scored = score_action_token_logits(
                    outputs.logits,
                    action_token_position=position,
                    codebook=codebook,
                )
                micro_rows.append(
                    {
                        "microbatch_index": microbatch_index,
                        "sample_index": sample_index,
                        "episode_id": sample["episode_id"],
                        "trajectory_id": sample["trajectory_id"],
                        "target_action": sample["target_action"],
                        "restricted_action_ce": value,
                        "predicted_action": scored["predicted_action"],
                        "cache_sha256": entry["sha256"],
                        "action_token_position": position,
                        "supervised_token_count": int(batch.record["supervised_token_count"]),
                        "supervised_token_id": int(batch.record["supervised_token_id"]),
                    }
                )
                del outputs, loss, inputs, batch

            gradients = gradient_report(model)
            if (
                gradients["gradient_tensor_count"] != gradients["trainable_tensor_count"]
                or gradients["nonzero_gradient_tensor_count"] <= 0
                or gradients["all_gradients_finite"] is not True
            ):
                raise RuntimeError("H4 benchmark gradients are missing, zero, or non-finite.")
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
                raise RuntimeError("H4 benchmark global gradient norm is invalid.")
            optimizer.step()
            updates = update_report(model, before)
            validate_optimization_reports(gradients, updates)
            after_hash = tensor_state_sha256(trainable_state_dict(model))
            if after_hash == before_hash:
                raise RuntimeError("H4 benchmark optimizer changed no trainable bytes.")
            elapsed = time.perf_counter() - update_started
            event = {
                "schema_version": NATIVE_DEVELOPMENT_MEASUREMENTS_SCHEMA_VERSION,
                "gate": NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
                "history_contract": NATIVE_H4_HISTORY,
                "objective": NATIVE_H4_OBJECTIVE,
                "update_step": update_step,
                "sample_exposure_start": start,
                "sample_exposure_end_exclusive": start + len(indices),
                "microbatch_count": len(indices),
                "accumulation_steps": protocol.accumulation_steps,
                "learning_rate": protocol.learning_rate,
                "mean_restricted_action_ce": sum(row["restricted_action_ce"] for row in micro_rows)
                / len(micro_rows),
                "gradient_report": gradients,
                "global_gradient_norm_before_clip": preclip,
                "clip_max_norm": protocol.max_grad_norm,
                "update_report": updates,
                "before_trainable_state_sha256": before_hash,
                "after_trainable_state_sha256": after_hash,
                "optimizer_state": optimizer_tensor_diagnostics(optimizer),
                "microbatches": micro_rows,
                "elapsed_seconds": elapsed,
                "timing_role": (
                    "measured" if update_step in protocol.timed_update_steps else "warmup_excluded"
                ),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            events.append(event)
            previous_hash = after_hash
            _atomic_json(
                output_root / "progress.json",
                {
                    "status": "running",
                    "current_stage": "h4_updates",
                    "completed_update": update_step,
                    "planned_updates": protocol.planned_optimizer_updates,
                    "latest_update_seconds": elapsed,
                    "elapsed_seconds": time.perf_counter() - run_started,
                    "recorded_at_utc": _utc_now(),
                },
            )
            del before

    calibration_started = time.perf_counter()
    calibration = _calibrate_h4(
        model=model,
        processor=processor,
        codebook=codebook,
        samples=calibration_samples,
        cache=cache,
        device=device,
        update_step=protocol.planned_optimizer_updates,
    )
    calibration_seconds = time.perf_counter() - calibration_started

    checkpoint_started = time.perf_counter()
    checkpoint = _save_checkpoint(
        checkpoint_root / "step_0008.pt",
        model=model,
        optimizer=optimizer,
        manifest_sha256=manifest_hash,
        cache_manifest_sha256=cache_manifest_hash,
        protocol=protocol,
    )
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    timing = summarize_h4_timings(
        events,
        protocol=protocol,
        model_load_seconds=model_load_seconds,
        cache_prepare_seconds=cache_prepare_seconds,
        calibration_seconds=calibration_seconds,
        checkpoint_seconds=checkpoint_seconds,
    )
    result = {
        "schema_version": NATIVE_DEVELOPMENT_MEASUREMENTS_SCHEMA_VERSION,
        "gate": NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
        "run_id": args.run_id,
        "status": "passed",
        "completed_at_utc": _utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "inventory_path": str(output_root / "inventory.json"),
        "inventory_sha256": file_sha256(output_root / "inventory.json"),
        "cache_manifest_path": str(cache_manifest_path),
        "cache_manifest_sha256": cache_manifest_hash,
        "events_path": str(events_path),
        "events_sha256": file_sha256(events_path),
        "protocol": protocol.to_record(),
        "timing": timing,
        "calibration": calibration,
        "checkpoint": checkpoint,
        "initial_trainable_state_sha256": initial_hash,
        "final_trainable_state_sha256": previous_hash,
        "trainable_contract": trainable_contract,
        "action_codebook": codebook.to_record(),
        "supervision": {
            "schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
            "method": NATIVE_QWEN_SUPERVISION_METHOD,
            "supervised_token_count_per_decision": 1,
        },
        "model_weight_locks": model_weight_locks(args.model_snapshot),
        "source_locks": manifest["source_locks"],
        "prerequisites": manifest["prerequisites"],
        "runtime": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "docker_image_id": EXPECTED_DOCKER_IMAGE_ID,
            "total_elapsed_seconds": time.perf_counter() - run_started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "evidence_boundary": manifest["evidence_boundary"],
    }
    _atomic_json(output_root / "result.json", result)
    _atomic_json(
        output_root / "progress.json",
        {
            "status": "passed",
            "completed_updates": protocol.planned_optimizer_updates,
            "timed_update_p50_seconds": timing["timed_update_p50_seconds"],
            "timed_update_p95_nearest_rank_seconds": timing[
                "timed_update_p95_nearest_rank_seconds"
            ],
            "calibration_seconds": calibration_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "recorded_at_utc": _utc_now(),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--output-root", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _common(prepare)
    run = commands.add_parser("run")
    _common(run)
    run.add_argument("--scene-root", type=Path, required=True)
    run.add_argument("--model-snapshot", type=Path, required=True)
    run.add_argument("--cache-root", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        return _prepare(args) if args.command == "prepare" else _run(args)
    except Exception as exc:
        failure = {
            "schema_version": NATIVE_DEVELOPMENT_MEASUREMENTS_SCHEMA_VERSION,
            "gate": NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
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
