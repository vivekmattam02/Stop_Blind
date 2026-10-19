#!/usr/bin/env python3
"""Execute one real R2R-CE RGB-D SFT update and write a fail-closed evidence record."""

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
from spatialvlm.model import SpatialVLM
from spatialvlm.training.real_batch_artifacts import REAL_BATCH_SOURCE_LOCK_PATHS
from spatialvlm.training.real_batch_gate import (
    CANONICAL_ACTION_TEXT,
    REAL_BATCH_GATE_SCHEMA_VERSION,
    file_sha256,
    gradient_group_report,
    load_trainable_checkpoint,
    save_trainable_checkpoint,
    select_probe_episode,
    tensor_state_sha256,
    trainable_state_dict,
    update_group_report,
    validate_group_reports,
)
from spatialvlm.training.sft import SFTConfig, SFTTrainer, supervised_loss
from spatialvlm.utils.camera import CameraIntrinsics

GATE_NAME = "spark_r2r_ce_one_real_rgbd_sft_update_v1"


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
    locks: dict[str, str] = {}
    for relative in REAL_BATCH_SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source-lock path is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _processor_record(processor: Any, model_id: str) -> dict[str, Any]:
    config = processor.to_dict() if hasattr(processor, "to_dict") else vars(processor)
    canonical = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return {
        "model_id": model_id,
        "class": type(processor).__name__,
        "config": config,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _tensor_batch(
    *,
    rgb_processed: dict[str, Any],
    depth: torch.Tensor,
    intrinsics: CameraIntrinsics,
    tokens: dict[str, torch.Tensor | int | str],
    device: torch.device,
) -> dict[str, Any]:
    def move(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=device, non_blocking=True)

    spatial_start = tokens["spatial_start_idx"]
    if not isinstance(spatial_start, int):
        raise TypeError("spatial_start_idx must be an integer for the one-sample gate.")
    return {
        "siglip_pixels": move(rgb_processed["siglip_pixel_values"]),
        "siglip_attention_mask": move(rgb_processed["siglip_pixel_attention_mask"]),
        "siglip_spatial_shapes": move(rgb_processed["siglip_spatial_shapes"]),
        "dinov2_pixels": move(rgb_processed["dinov2_pixel_values"]),
        "depth": move(depth.unsqueeze(0)),
        "intrinsics": intrinsics,
        "input_ids": move(tokens["input_ids"].unsqueeze(0)),
        "labels": move(tokens["labels"].unsqueeze(0)),
        "attention_mask": move(tokens["attention_mask"].unsqueeze(0)),
        "spatial_start_idx": torch.tensor([spatial_start], dtype=torch.long, device=device),
    }


def _evaluate_loss(model: SpatialVLM, batch: dict[str, Any]) -> float:
    labels = batch["labels"]
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        loss = supervised_loss(logits, labels)
    value = float(loss.detach().cpu().item())
    del outputs, logits, loss
    if not math.isfinite(value):
        raise RuntimeError("Evaluation loss is non-finite.")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if re.fullmatch(r"real_batch_v[1-9][0-9]*", args.run_id) is None:
        raise SystemExit("--run-id must match real_batch_v<positive integer>")
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": REAL_BATCH_GATE_SCHEMA_VERSION,
        "gate": GATE_NAME,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": _utc_now(),
        "evidence_boundary": {
            "established_if_passed": (
                "One deterministic released train observation completed checkpoint-native "
                "preprocessing, supervised action loss, backward, optimizer update, compact "
                "checkpoint save/reload, and legal-action constrained scoring."
            ),
            "not_established": (
                "Navigation competence, convergence, simulator rollout improvement, learned-"
                "baseline parity, val_seen or val_unseen performance, ablation effects, or a "
                "paper superiority claim."
            ),
        },
    }
    environment: HabitatSimR2REnvironment | None = None
    try:
        repo = args.repo.expanduser().resolve()
        output = args.output.expanduser().resolve()
        checkpoint_path = args.checkpoint.expanduser().resolve()
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The Spark real-batch gate requires a visible CUDA device.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not report BF16 support.")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)

        dataset = R2RCEDataset.from_release(
            args.episodes,
            args.ground_truth,
            split="train",
        )
        episode, target_action = select_probe_episode(dataset)
        target_text = CANONICAL_ACTION_TEXT[target_action]

        habitat_config = HabitatSimR2RConfig(scene_root=args.scene_root)
        environment = HabitatSimR2REnvironment(habitat_config)
        state = environment.reset(episode)
        rgb = state.observation["rgb"]
        depth_meters = state.observation["depth_meters"]

        model_config = SpatialVLMConfig()
        siglip_id = model_config.encoder.siglip_model_id
        dinov2_id = model_config.encoder.dinov2_model_id
        backbone_id = model_config.backbone.model_id
        siglip_processor = AutoImageProcessor.from_pretrained(siglip_id)
        dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_id)
        tokenizer = AutoTokenizer.from_pretrained(backbone_id, trust_remote_code=True)
        rgb_processed = preprocess_encoder_rgb(
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
            max_length=2048,
        )
        tokens = build_input_ids(
            tokenizer,
            instruction=episode.instruction_text,
            target=target_text,
            config=token_config,
        )
        batch = _tensor_batch(
            rgb_processed=rgb_processed,
            depth=depth,
            intrinsics=intrinsics,
            tokens=tokens,
            device=device,
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
                weight_decay=0.01,
                max_grad_norm=1.0,
                trainable_groups=("siglip_proj", "dino_proj", "gatr", "sva", "lora"),
            ),
        )
        model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        before = trainable_state_dict(model)
        before_sha = tensor_state_sha256(before)
        training_loss = trainer.forward_backward(batch)
        gradients = gradient_group_report(model)
        grad_norm = trainer.clip_and_step()
        updates = update_group_report(model, before)
        validate_group_reports(gradients, updates)
        after_sha = tensor_state_sha256(trainable_state_dict(model))
        if after_sha == before_sha:
            raise RuntimeError("Trainable state hash did not change after the optimizer update.")
        del before

        model.eval()
        post_update_target_loss = _evaluate_loss(model, batch)
        checkpoint_info = save_trainable_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=trainer.optimizer,
            metadata={
                "gate": GATE_NAME,
                "run_id": args.run_id,
                "seed": args.seed,
                "episode_id": episode.episode_id,
                "target_action": target_action.name,
                "source_locks": _source_locks(repo),
            },
        )

        first_trainable = next(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        with torch.no_grad():
            first_trainable.view(-1)[0].add_(1.0)
        perturbed_sha = tensor_state_sha256(trainable_state_dict(model))
        if perturbed_sha == after_sha:
            raise RuntimeError("Checkpoint reload probe did not perturb the trainable state.")
        load_trainable_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=trainer.optimizer,
        )
        reloaded_sha = tensor_state_sha256(trainable_state_dict(model))
        if reloaded_sha != checkpoint_info["trainable_state_sha256"] or reloaded_sha != after_sha:
            raise RuntimeError(
                "Reloaded trainable state does not match the saved post-update state."
            )

        candidate_scores: dict[str, dict[str, float | int]] = {}
        for action, action_text in CANONICAL_ACTION_TEXT.items():
            candidate_tokens = build_input_ids(
                tokenizer,
                instruction=episode.instruction_text,
                target=action_text,
                config=token_config,
            )
            candidate_batch = _tensor_batch(
                rgb_processed=rgb_processed,
                depth=depth,
                intrinsics=intrinsics,
                tokens=candidate_tokens,
                device=device,
            )
            mean_nll = _evaluate_loss(model, candidate_batch)
            count = int(candidate_tokens["supervised_token_count"])
            candidate_scores[action.name] = {
                "mean_nll": mean_nll,
                "total_nll": mean_nll * count,
                "supervised_token_count": count,
            }
            del candidate_batch
            torch.cuda.empty_cache()
        predicted_action = min(
            candidate_scores,
            key=lambda name: float(candidate_scores[name]["total_nll"]),
        )
        reload_target_loss = float(candidate_scores[target_action.name]["mean_nll"])
        reload_loss_delta = abs(reload_target_loss - post_update_target_loss)
        if reload_loss_delta > 1e-5:
            raise RuntimeError(
                f"Post-reload deterministic target loss drifted by {reload_loss_delta:.3e}."
            )

        result.update(
            {
                "status": "passed",
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "selection_rule": "lowest_numeric_train_episode_with_non_stop_first_action",
                "source": {
                    "release_manifest": asdict(dataset.release_manifest)
                    if dataset.release_manifest is not None
                    else None,
                    "episode_id": episode.episode_id,
                    "trajectory_id": episode.trajectory_id,
                    "scene_id": episode.scene_id,
                    "instruction_sha256": hashlib.sha256(
                        episode.instruction_text.encode("utf-8")
                    ).hexdigest(),
                    "rgb_sha256": _array_sha256(rgb),
                    "metric_depth_sha256": _array_sha256(depth_meters),
                    "target_action_id": int(target_action),
                    "target_action": target_action.name,
                    "target_text": target_text,
                    "initial_geodesic_distance_m": state.distance_to_goal,
                },
                "runtime": {
                    "platform": platform.platform(),
                    "architecture": platform.machine(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "compiled_cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device),
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                    "seed": args.seed,
                    "dtype": "bf16_trainable_modules",
                    "gradient_checkpointing": "non_reentrant",
                    "habitat": environment.runtime_metadata(),
                },
                "processors": {
                    "siglip": _processor_record(siglip_processor, siglip_id),
                    "dinov2": _processor_record(dinov2_processor, dinov2_id),
                    "rgb_contract": rgb_processed["manifest"],
                    "depth_contract": depth_manifest,
                    "token_contract": {
                        "config": asdict(token_config),
                        "sequence_length": int(tokens["input_ids"].numel()),
                        "spatial_start_idx": tokens["spatial_start_idx"],
                        "target_start_idx": tokens["target_start_idx"],
                        "supervised_token_count": tokens["supervised_token_count"],
                        "conversation_renderer_version": tokens["conversation_renderer_version"],
                    },
                },
                "model": {
                    "config": asdict(model_config),
                    "backbone_model_id": backbone_id,
                    "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "trainable_parameters": trainer.trainable_parameter_count(),
                },
                "optimization": {
                    "optimizer": "AdamW",
                    "learning_rate": args.learning_rate,
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "training_loss_before_update": training_loss,
                    "global_gradient_norm_before_clip": grad_norm,
                    "post_update_target_loss": post_update_target_loss,
                    "gradients_by_group": gradients,
                    "updates_by_group": updates,
                    "trainable_state_sha256_before": before_sha,
                    "trainable_state_sha256_after": after_sha,
                },
                "checkpoint": {
                    **checkpoint_info,
                    "perturbed_state_sha256": perturbed_sha,
                    "reloaded_state_sha256": reloaded_sha,
                    "reload_target_loss": reload_target_loss,
                    "reload_target_loss_absolute_delta": reload_loss_delta,
                },
                "constrained_decode": {
                    "method": "minimum_total_conditional_nll_over_four_canonical_actions",
                    "legal_actions": [action.name for action in R2RCEAction],
                    "scores": candidate_scores,
                    "predicted_action": predicted_action,
                    "prediction_matches_target": predicted_action == target_action.name,
                    "note": (
                        "A one-update mechanics gate does not require target accuracy; legality "
                        "and deterministic scoring are the pass criteria."
                    ),
                },
                "source_locks": _source_locks(repo),
                "accelerator": {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                },
            }
        )
        _atomic_json(output, result)
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        print(f"Real-batch gate PASSED: {output}")
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
        try:
            _atomic_json(args.output.expanduser().resolve(), result)
        except Exception:
            pass
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
