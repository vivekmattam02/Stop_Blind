#!/usr/bin/env python3
"""Prepare, execute, and summarize the frozen Spark optimization/stability pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer

from spatialvlm.config.model import SpatialVLMConfig
from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.data.habitat_sim_r2r import HabitatSimR2RConfig, HabitatSimR2REnvironment
from spatialvlm.data.preprocessing import preprocess_encoder_rgb, preprocess_metric_depth
from spatialvlm.data.tokenization import SpatialTokenizerConfig, build_input_ids
from spatialvlm.eval.action_decision import (
    ACTION_DECISION_SYSTEM_PROMPT,
    build_action_token_codebook,
    canonical_action_target,
    score_action_token_logits,
)
from spatialvlm.model import SpatialVLM
from spatialvlm.training.real_batch_gate import (
    file_sha256,
    gradient_group_report,
    save_trainable_checkpoint,
    tensor_state_sha256,
    trainable_state_dict,
    update_group_report,
    validate_group_reports,
)
from spatialvlm.training.sft import FREEZE_GROUP_PATTERNS, SFTConfig, SFTTrainer
from spatialvlm.training.stability_pilot import (
    STABILITY_PILOT_GATE_NAME,
    STABILITY_PILOT_SCHEMA_VERSION,
    StabilityPilotProtocol,
    action_only_labels,
    select_pilot_checkpoint,
    select_pilot_samples,
)
from spatialvlm.training.stability_pilot_artifacts import STABILITY_PILOT_SOURCE_LOCK_PATHS
from spatialvlm.utils.camera import CameraIntrinsics


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _source_locks(repo: Path) -> dict[str, str]:
    locks = {}
    for relative in STABILITY_PILOT_SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required pilot source is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _load_dataset(args: argparse.Namespace) -> R2RCEDataset:
    return R2RCEDataset.from_release(args.episodes, args.ground_truth, split="train")


def _protocol_and_manifest(path: Path) -> tuple[StabilityPilotProtocol, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    protocol = StabilityPilotProtocol()
    if manifest.get("protocol") != protocol.to_record():
        raise ValueError("Pilot manifest does not match the frozen protocol.")
    return protocol, manifest


def _prepare_manifest(args: argparse.Namespace) -> int:
    protocol = StabilityPilotProtocol()
    repo = args.repo.expanduser().resolve()
    dataset = _load_dataset(args)
    samples = select_pilot_samples(dataset, protocol)
    output = args.output.expanduser().resolve()
    preflights = {}
    for label, path, expected_gate in (
        (
            "processor",
            args.processor_preflight.expanduser().resolve(),
            "spark_real_batch_checkpoint_processor_preflight_v1",
        ),
        (
            "action",
            args.action_preflight.expanduser().resolve(),
            "spark_action_decision_tokenizer_preflight_v1",
        ),
    ):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "passed" or record.get("gate") != expected_gate:
            raise RuntimeError(f"Required {label} preflight did not pass its expected gate.")
        preflights[label] = {
            "relative_path": str(path.relative_to(output.parent)),
            "sha256": file_sha256(path),
            "gate": expected_gate,
        }
    payload = {
        "schema_version": STABILITY_PILOT_SCHEMA_VERSION,
        "gate": STABILITY_PILOT_GATE_NAME,
        "run_id": args.run_id,
        "status": "passed",
        "created_at_utc": _utc_now(),
        "protocol": protocol.to_record(),
        "release_manifest": (
            dataset.release_manifest.to_dict() if dataset.release_manifest is not None else None
        ),
        "samples": [sample.to_record() for sample in samples],
        "preflights": preflights,
        "source_locks": _source_locks(repo),
        "evidence_boundary": {
            "established": "Frozen inputs and optimization protocol only.",
            "not_established": "Model loading, optimization stability, or policy performance.",
        },
    }
    _atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        )
        for key, value in batch.items()
    }


def _prepare_batches(
    *,
    samples: list[dict[str, Any]],
    dataset: R2RCEDataset,
    scene_root: Path,
    model_config: SpatialVLMConfig,
    siglip_processor: Any,
    dinov2_processor: Any,
    tokenizer: Any,
    codebook: Any,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    habitat_config = HabitatSimR2RConfig(scene_root=scene_root)
    environment = HabitatSimR2REnvironment(habitat_config)
    batches: dict[tuple[str, int], dict[str, Any]] = {}
    runtime: dict[str, Any] = {}
    try:
        for sample in samples:
            episode = dataset.episode_record(sample["episode_id"])
            ground_truth = dataset.ground_truth_record(episode.episode_id)
            step_index = sample["action_step_index"]
            target_action = R2RCEAction(sample["target_action_id"])
            if ground_truth.actions[step_index] != target_action:
                raise RuntimeError("Frozen sample target no longer matches released actions.")
            prefix = tuple(int(action) for action in ground_truth.actions[:step_index])
            if prefix != tuple(sample["replay_prefix_action_ids"]):
                raise RuntimeError("Frozen sample replay prefix no longer matches the release.")
            state = environment.reset(episode)
            collisions = 0
            for action_id in prefix:
                state = environment.step(R2RCEAction(action_id))
                collisions += int(state.collision)
            rgb = state.observation["rgb"]
            depth_meters = state.observation["depth_meters"]
            processed = preprocess_encoder_rgb(
                rgb,
                siglip_processor=siglip_processor,
                dinov2_processor=dinov2_processor,
            )
            source_intrinsics = CameraIntrinsics.from_hfov(
                width=int(depth_meters.shape[1]),
                height=int(depth_meters.shape[0]),
                hfov_degrees=habitat_config.hfov_degrees,
            )
            depth, intrinsics, depth_manifest = preprocess_metric_depth(
                depth_meters,
                intrinsics=source_intrinsics,
                target_height_width=(
                    model_config.geometry.depth_image_size,
                    model_config.geometry.depth_image_size,
                ),
            )
            token_config = SpatialTokenizerConfig(
                num_spatial_tokens=model_config.fusion.sva_num_queries,
                system_prompt=ACTION_DECISION_SYSTEM_PROMPT,
                max_length=2048,
            )
            rendered = build_input_ids(
                tokenizer,
                instruction=episode.instruction_text,
                target=canonical_action_target(target_action),
                config=token_config,
            )
            labels, action_position = action_only_labels(
                rendered,
                action=target_action,
                codebook=codebook,
            )
            spatial_start = rendered["spatial_start_idx"]
            if not isinstance(spatial_start, int):
                raise TypeError("spatial_start_idx must be an integer.")
            batch = {
                "siglip_pixels": processed["siglip_pixel_values"],
                "siglip_attention_mask": processed["siglip_pixel_attention_mask"],
                "siglip_spatial_shapes": processed["siglip_spatial_shapes"],
                "dinov2_pixels": processed["dinov2_pixel_values"],
                "depth": depth.unsqueeze(0),
                "intrinsics": intrinsics,
                "input_ids": rendered["input_ids"].unsqueeze(0),
                "labels": labels.unsqueeze(0),
                "attention_mask": rendered["attention_mask"].unsqueeze(0),
                "spatial_start_idx": torch.tensor([spatial_start], dtype=torch.long),
            }
            batches[(sample["split"], sample["sample_index"])] = {
                "batch": batch,
                "action_token_position": action_position,
                "target_action": target_action,
                "observation": {
                    "rgb_sha256": _array_sha256(rgb),
                    "metric_depth_sha256": _array_sha256(depth_meters),
                    "position": list(state.position),
                    "distance_to_goal_m": state.distance_to_goal,
                    "replay_collision_count": collisions,
                    "rgb_contract": processed["manifest"],
                    "depth_contract": depth_manifest,
                    "sequence_length": int(rendered["input_ids"].numel()),
                    "target_start_idx": rendered["target_start_idx"],
                },
            }
        runtime = environment.runtime_metadata()
    finally:
        environment.close()
    return batches, runtime


def _validate_gradients(gradients: dict[str, dict[str, Any]]) -> None:
    for group in FREEZE_GROUP_PATTERNS:
        row = gradients[group]
        if row["trainable_tensor_count"] <= 0:
            raise RuntimeError(f"{group} has no trainable tensors.")
        if row["gradient_tensor_count"] != row["trainable_tensor_count"]:
            raise RuntimeError(f"{group} has missing gradients.")
        if not row["all_gradients_finite"] or row["gradient_l2_norm"] <= 0.0:
            raise RuntimeError(f"{group} has invalid gradients.")


def _calibrate(
    *,
    model: SpatialVLM,
    batches: dict[tuple[str, int], dict[str, Any]],
    samples: list[dict[str, Any]],
    codebook: Any,
    device: torch.device,
    protocol: StabilityPilotProtocol,
    step: int,
) -> dict[str, Any]:
    rows = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            prepared = batches[("calibration", sample["sample_index"])]
            batch = _move_batch(prepared["batch"], device)
            inputs = {key: value for key, value in batch.items() if key != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            decision = score_action_token_logits(
                logits,
                action_token_position=prepared["action_token_position"],
                codebook=codebook,
            )
            target_name = prepared["target_action"].name
            target_nll = float(decision["scores"][target_name]["token_nll"])
            if not math.isfinite(target_nll) or target_nll > protocol.hard_max_token_nll:
                raise RuntimeError(f"Calibration target NLL violates the hard stop: {target_nll}.")
            rows.append(
                {
                    "sample_index": sample["sample_index"],
                    "episode_id": sample["episode_id"],
                    "trajectory_id": sample["trajectory_id"],
                    "target_action": target_name,
                    "target_token_nll": target_nll,
                    "predicted_action": decision["predicted_action"],
                    "prediction_matches_target": decision["predicted_action"] == target_name,
                    "top_two_margin_nats": decision["top_two_margin_nats"],
                    "restricted_entropy_nats": decision["restricted_entropy_nats"],
                    "action_token_position": prepared["action_token_position"],
                }
            )
            del batch, inputs, outputs, logits
            torch.cuda.empty_cache()
    return {
        "step": step,
        "sample_count": len(rows),
        "mean_target_token_nll": sum(row["target_token_nll"] for row in rows) / len(rows),
        "descriptive_accuracy": sum(row["prediction_matches_target"] for row in rows) / len(rows),
        "accuracy_is_selection_or_pass_condition": False,
        "samples": rows,
    }


def _run_arm(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    protocol, manifest = _protocol_and_manifest(args.manifest)
    result: dict[str, Any] = {
        "schema_version": STABILITY_PILOT_SCHEMA_VERSION,
        "gate": STABILITY_PILOT_GATE_NAME,
        "run_id": args.run_id,
        "arm_id": args.arm_id,
        "learning_rate": args.learning_rate,
        "status": "running",
        "started_at_utc": _utc_now(),
        "manifest_sha256": file_sha256(args.manifest),
        "protocol": protocol.to_record(),
    }
    try:
        if args.run_id != manifest.get("run_id"):
            raise ValueError("Arm run_id does not match its manifest.")
        if args.arm_id != protocol.arm_id(args.learning_rate):
            raise ValueError("Arm ID does not match the frozen learning rate.")
        repo = args.repo.expanduser().resolve()
        if manifest.get("source_locks") != _source_locks(repo):
            raise RuntimeError("Pilot source locks changed after manifest creation.")
        dataset = _load_dataset(args)
        expected_samples = [
            sample.to_record() for sample in select_pilot_samples(dataset, protocol)
        ]
        if manifest.get("samples") != expected_samples:
            raise RuntimeError("Pilot sample manifest does not reconstruct from the release.")
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The Spark stability pilot requires a visible CUDA device.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not report BF16 support.")

        random.seed(protocol.seed)
        np.random.seed(protocol.seed)
        torch.manual_seed(protocol.seed)
        torch.cuda.manual_seed_all(protocol.seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)

        model_config = SpatialVLMConfig()
        siglip_processor = AutoImageProcessor.from_pretrained(model_config.encoder.siglip_model_id)
        dinov2_processor = AutoImageProcessor.from_pretrained(model_config.encoder.dinov2_model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.backbone.model_id, trust_remote_code=True
        )
        codebook = build_action_token_codebook(tokenizer)
        batches, habitat_runtime = _prepare_batches(
            samples=expected_samples,
            dataset=dataset,
            scene_root=args.scene_root,
            model_config=model_config,
            siglip_processor=siglip_processor,
            dinov2_processor=dinov2_processor,
            tokenizer=tokenizer,
            codebook=codebook,
        )
        model = SpatialVLM(
            config=model_config,
            device=device,
            torch_dtype=torch.bfloat16,
            lazy_load_encoders=False,
            lazy_load_backbone=False,
            local_files_only=False,
        )
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype != torch.bfloat16:
                parameter.data = parameter.data.to(dtype=torch.bfloat16)
        backbone = model.backbone.model
        if backbone is None:
            raise RuntimeError("Qwen backbone did not load.")
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        trainer = SFTTrainer(
            model,
            SFTConfig(
                learning_rate=args.learning_rate,
                weight_decay=protocol.weight_decay,
                max_grad_norm=protocol.max_grad_norm,
                trainable_groups=("siglip_proj", "dino_proj", "gatr", "sva", "lora"),
            ),
        )
        initial_state = trainable_state_dict(model)
        initial_hash = tensor_state_sha256(initial_state)
        del initial_state
        train_samples = [row for row in expected_samples if row["split"] == "train"]
        calibration_samples = [row for row in expected_samples if row["split"] == "calibration"]
        calibration = [
            _calibrate(
                model=model,
                batches=batches,
                samples=calibration_samples,
                codebook=codebook,
                device=device,
                protocol=protocol,
                step=0,
            )
        ]
        steps = []
        checkpoints = []
        audit_steps = {1, *protocol.checkpoint_steps}
        checkpoint_root = args.checkpoint_root.expanduser().resolve()
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        for step, sample in enumerate(train_samples, start=1):
            if time.perf_counter() - started > protocol.max_arm_hours * 3600.0:
                raise TimeoutError("Arm exceeded the frozen two-hour wall-time limit.")
            model.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            prepared = batches[("train", sample["sample_index"])]
            batch = _move_batch(prepared["batch"], device)
            before = trainable_state_dict(model) if step in audit_steps else None
            target_nll = trainer.forward_backward(batch)
            if not math.isfinite(target_nll) or target_nll > protocol.hard_max_token_nll:
                raise RuntimeError(f"Training target NLL violates the hard stop: {target_nll}.")
            gradients = gradient_group_report(model)
            _validate_gradients(gradients)
            gradient_norm = trainer.clip_and_step()
            if not math.isfinite(gradient_norm) or gradient_norm > protocol.hard_max_gradient_norm:
                raise RuntimeError(f"Global gradient norm violates the hard stop: {gradient_norm}.")
            updates = update_group_report(model, before) if before is not None else None
            if updates is not None:
                validate_group_reports(gradients, updates)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            if peak_allocated > protocol.hard_max_peak_allocated_bytes:
                raise RuntimeError("Peak allocated accelerator memory exceeded 64 GiB.")
            steps.append(
                {
                    "step": step,
                    "sample_index": sample["sample_index"],
                    "episode_id": sample["episode_id"],
                    "trajectory_id": sample["trajectory_id"],
                    "scene_id": sample["scene_id"],
                    "action_step_index": sample["action_step_index"],
                    "target_action": sample["target_action"],
                    "action_token_position": prepared["action_token_position"],
                    "target_token_nll": target_nll,
                    "global_gradient_norm_before_clip": gradient_norm,
                    "gradient_clip_coefficient": min(
                        1.0, protocol.max_grad_norm / (gradient_norm + 1e-12)
                    ),
                    "gradient_was_clipped": gradient_norm > protocol.max_grad_norm,
                    "gradients_by_group": gradients,
                    "updates_by_group": updates,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "elapsed_seconds": time.perf_counter() - started,
                    "observation": prepared["observation"],
                }
            )
            del batch, before
            torch.cuda.empty_cache()
            if step in protocol.checkpoint_steps:
                checkpoint_path = checkpoint_root / args.arm_id / f"step_{step:04d}.pt"
                if checkpoint_path.exists():
                    raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
                checkpoint = save_trainable_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=trainer.optimizer,
                    metadata={
                        "gate": STABILITY_PILOT_GATE_NAME,
                        "run_id": args.run_id,
                        "arm_id": args.arm_id,
                        "learning_rate": args.learning_rate,
                        "step": step,
                        "manifest_sha256": file_sha256(args.manifest),
                        "initial_trainable_state_sha256": initial_hash,
                    },
                )
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                payload_hash = tensor_state_sha256(payload["trainable_state_dict"])
                payload_verified = payload_hash == payload["trainable_state_sha256"]
                if not payload_verified:
                    raise RuntimeError("Checkpoint payload trainable-state hash mismatch.")
                del payload
                checkpoints.append(
                    {
                        **checkpoint,
                        "step": step,
                        "relative_path": str(checkpoint_path.relative_to(checkpoint_root)),
                        "payload_trainable_hash_verified": payload_verified,
                    }
                )
                calibration.append(
                    _calibrate(
                        model=model,
                        batches=batches,
                        samples=calibration_samples,
                        codebook=codebook,
                        device=device,
                        protocol=protocol,
                        step=step,
                    )
                )

        result.update(
            {
                "status": "passed",
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "protocol": protocol.to_record(),
                "initial_trainable_state_sha256": initial_hash,
                "final_trainable_state_sha256": checkpoints[-1]["trainable_state_sha256"],
                "release_manifest": manifest["release_manifest"],
                "codebook": codebook.to_record(),
                "runtime": {
                    "platform": platform.platform(),
                    "architecture": platform.machine(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "compiled_cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device),
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                    "seed": protocol.seed,
                    "dtype": "bf16_trainable_modules",
                    "gradient_checkpointing": "non_reentrant",
                    "habitat": habitat_runtime,
                },
                "model": {
                    "config": asdict(model_config),
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": trainer.trainable_parameter_count(),
                },
                "training": {
                    "optimizer": "AdamW",
                    "learning_rate": args.learning_rate,
                    "weight_decay": protocol.weight_decay,
                    "max_grad_norm": protocol.max_grad_norm,
                    "steps": steps,
                    "clipped_step_count": sum(row["gradient_was_clipped"] for row in steps),
                },
                "calibration": calibration,
                "checkpoints": checkpoints,
                "accelerator": {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                },
                "evidence_boundary": {
                    "established_if_suite_passes": (
                        "Bounded numerical optimization stability and checkpoint selection on "
                        "disjoint released train trajectories."
                    ),
                    "not_established": (
                        "Navigation competence, convergence, rollout improvement, val_seen or "
                        "val_unseen performance, baseline parity, or ablation effects."
                    ),
                },
            }
        )
        _atomic_json(args.output.expanduser().resolve(), result)
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _atomic_json(args.output.expanduser().resolve(), result)
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True), file=sys.stderr)
        return 1


def _summarize(args: argparse.Namespace) -> int:
    protocol, manifest = _protocol_and_manifest(args.manifest)
    arms = {}
    initial_hashes = set()
    candidates = []
    for rate in protocol.learning_rates:
        arm_id = protocol.arm_id(rate)
        path = args.arms_root / arm_id / "result.json"
        arm = json.loads(path.read_text(encoding="utf-8"))
        arms[arm_id] = {
            "result_path": str(path.resolve()),
            "result_sha256": file_sha256(path),
            "status": arm.get("status"),
            "learning_rate": rate,
            "elapsed_seconds": arm.get("elapsed_seconds"),
        }
        initial_hashes.add(arm.get("initial_trainable_state_sha256"))
        for row in arm.get("calibration", []):
            if row.get("step") in protocol.checkpoint_steps:
                checkpoint = next(
                    item for item in arm["checkpoints"] if item["step"] == row["step"]
                )
                candidates.append(
                    {
                        "arm_id": arm_id,
                        "learning_rate": rate,
                        "step": row["step"],
                        "mean_target_token_nll": row["mean_target_token_nll"],
                        "checkpoint": checkpoint,
                    }
                )
    selected = select_pilot_checkpoint(candidates)
    selected_arm = json.loads(
        (args.arms_root / selected["arm_id"] / "result.json").read_text(encoding="utf-8")
    )
    baseline = next(
        row["mean_target_token_nll"] for row in selected_arm["calibration"] if row["step"] == 0
    )
    improvement = float(baseline) - float(selected["mean_target_token_nll"])
    arm_statuses_pass = all(row["status"] == "passed" for row in arms.values())
    identical_initialization = len(initial_hashes) == 1
    improved = improvement >= protocol.minimum_calibration_improvement_nats
    passed = arm_statuses_pass and identical_initialization and improved
    suite = {
        "schema_version": STABILITY_PILOT_SCHEMA_VERSION,
        "gate": STABILITY_PILOT_GATE_NAME,
        "run_id": manifest["run_id"],
        "status": "passed" if passed else "failed",
        "completed_at_utc": _utc_now(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "protocol": protocol.to_record(),
        "arms": arms,
        "selected_checkpoint": selected,
        "pass_criteria": {
            "all_arms_passed": arm_statuses_pass,
            "identical_initial_trainable_state": identical_initialization,
            "initial_trainable_state_sha256": next(iter(initial_hashes))
            if identical_initialization
            else None,
            "selected_arm_step0_mean_target_token_nll": baseline,
            "selected_checkpoint_mean_target_token_nll": selected["mean_target_token_nll"],
            "calibration_improvement_nats": improvement,
            "minimum_required_improvement_nats": (protocol.minimum_calibration_improvement_nats),
            "minimum_improvement_passed": improved,
        },
        "evidence_boundary": {
            "established_if_passed": (
                "All three bounded real RGB-D optimization arms passed their numerical, gradient, "
                "memory, checkpoint, and held-out train-trajectory improvement gates."
            ),
            "not_established": (
                "Convergence, rollout improvement, navigation competence, val_seen or val_unseen "
                "performance, learned-baseline parity, ablation effects, or superiority."
            ),
        },
    }
    _atomic_json(args.output.expanduser().resolve(), suite)
    print(json.dumps(suite, indent=2, sort_keys=True))
    return 0 if passed else 1


def _common_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo", type=Path, default=Path.cwd())
    prepare.add_argument("--run-id", required=True)
    _common_dataset_arguments(prepare)
    prepare.add_argument("--processor-preflight", type=Path, required=True)
    prepare.add_argument("--action-preflight", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    arm = subparsers.add_parser("arm")
    arm.add_argument("--repo", type=Path, default=Path.cwd())
    arm.add_argument("--run-id", required=True)
    arm.add_argument("--arm-id", required=True)
    arm.add_argument("--learning-rate", type=float, required=True)
    arm.add_argument("--manifest", type=Path, required=True)
    _common_dataset_arguments(arm)
    arm.add_argument("--scene-root", type=Path, required=True)
    arm.add_argument("--checkpoint-root", type=Path, required=True)
    arm.add_argument("--output", type=Path, required=True)
    arm.add_argument("--device", default="cuda")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.add_argument("--arms-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command in {"prepare", "arm"}:
        if re.fullmatch(r"stability_pilot_v[1-9][0-9]*", args.run_id) is None:
            raise SystemExit("--run-id must match stability_pilot_v<positive integer>")
    if args.command == "prepare":
        return _prepare_manifest(args)
    if args.command == "arm":
        return _run_arm(args)
    return _summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
