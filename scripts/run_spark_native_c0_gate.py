#!/usr/bin/env python3
"""Run one exact-revision native-Qwen H0 update on a real R2R-CE observation."""

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from spatialvlm.backbone.qwen3_vl import Qwen3VLBackbone
from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.data.habitat_sim_r2r import HabitatSimR2RConfig, HabitatSimR2REnvironment
from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode, build_history_window
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_SYSTEM_PROMPT,
    NativeQwenBatch,
    build_native_qwen_sft_batch,
)
from spatialvlm.eval.action_decision import (
    build_action_token_codebook,
    locate_shared_action_position,
    score_action_token_logits,
)
from spatialvlm.training.native_c0_artifacts import source_locks
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_GATE_NAME,
    NATIVE_C0_GATE_SCHEMA_VERSION,
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    NATIVE_C0_PARENT_RESULT_SHA256,
    NATIVE_C0_PARENT_VERIFICATION_SHA256,
    array_sha256,
    file_sha256,
    gradient_report,
    load_native_c0_checkpoint,
    model_weight_locks,
    save_native_c0_checkpoint,
    select_native_c0_probe,
    tensor_sha256,
    tensor_state_sha256,
    trainable_state_dict,
    update_report,
    validate_lora_only_trainables,
    validate_optimization_reports,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
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


def _move_batch(batch: NativeQwenBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: tensor.to(device=device, non_blocking=True)
        for key, tensor in batch.model_inputs.items()
    }


def _candidate_record(batch: NativeQwenBatch) -> dict[str, torch.Tensor | int | str]:
    return {
        "input_ids": batch.model_inputs["input_ids"][0],
        "labels": batch.model_inputs["labels"][0],
        "target_start_idx": int(batch.record["target_start_idx"]),
    }


def _evaluate(
    model: Qwen3VLBackbone, inputs: dict[str, torch.Tensor]
) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        outputs = model(**inputs)
    loss = float(outputs.loss.detach().float().cpu().item())
    if not math.isfinite(loss) or loss <= 0.0:
        raise RuntimeError("Native-C0 evaluation loss is invalid.")
    return loss, outputs.logits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--processor-result", type=Path, required=True)
    parser.add_argument("--processor-verification", type=Path, required=True)
    parser.add_argument("--processor-reverification", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if re.fullmatch(r"native_c0_real_batch_v[1-9][0-9]*", args.run_id) is None:
        raise SystemExit("--run-id must match native_c0_real_batch_v<positive integer>")
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": NATIVE_C0_GATE_SCHEMA_VERSION,
        "gate": NATIVE_C0_GATE_NAME,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": _utc_now(),
        "evidence_boundary": {
            "established_if_passed": (
                "One deterministic real R2R-CE train observation completed exact-revision native "
                "Qwen H0 RGB processing, LoRA-only forward/backward/update, compact checkpoint "
                "save/reload, and length-fair four-action scoring."
            ),
            "not_established": (
                "History selection, convergence, navigation competence, closed-loop improvement, "
                "val_seen or val_unseen performance, C0-C4 effects, or a paper claim."
            ),
        },
    }
    environment: HabitatSimR2REnvironment | None = None
    try:
        repo = args.repo.expanduser().resolve()
        output = args.output.expanduser().resolve()
        checkpoint_path = args.checkpoint.expanduser().resolve()
        model_snapshot = args.model_snapshot.expanduser().resolve()
        processor_result_path = args.processor_result.expanduser().resolve()
        processor_verification_path = args.processor_verification.expanduser().resolve()
        processor_reverification_path = args.processor_reverification.expanduser().resolve()
        if file_sha256(processor_result_path) != NATIVE_C0_PARENT_RESULT_SHA256:
            raise RuntimeError("Immutable parent processor-result hash mismatch.")
        if file_sha256(processor_verification_path) != NATIVE_C0_PARENT_VERIFICATION_SHA256:
            raise RuntimeError("Immutable parent processor-verification hash mismatch.")
        processor_reverification = json.loads(
            processor_reverification_path.read_text(encoding="utf-8")
        )
        if (
            processor_reverification.get("status") != "passed"
            or processor_reverification.get("error_count") != 0
            or processor_reverification.get("result_sha256") != NATIVE_C0_PARENT_RESULT_SHA256
        ):
            raise RuntimeError("Fresh parent processor reverification did not pass.")

        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The native-C0 gate requires a visible CUDA device.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The native-C0 gate requires BF16 support.")
        if args.learning_rate <= 0.0 or args.max_grad_norm <= 0.0:
            raise ValueError("Optimization hyperparameters must be positive.")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)

        locked_sources = source_locks(repo)
        locked_weights = model_weight_locks(model_snapshot)
        dataset = R2RCEDataset.from_release(
            args.episodes.expanduser().resolve(),
            args.ground_truth.expanduser().resolve(),
            split="train",
        )
        episode, target_action = select_native_c0_probe(dataset)

        habitat_config = HabitatSimR2RConfig(scene_root=args.scene_root.expanduser().resolve())
        environment = HabitatSimR2REnvironment(habitat_config)
        state = environment.reset(episode)
        rgb = state.observation["rgb"]
        rgb_image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        window = build_history_window(
            [rgb_image], [], contract=HISTORY_CONTRACTS[HistoryMode.H0_CURRENT]
        )

        processor = AutoProcessor.from_pretrained(
            NATIVE_C0_MODEL_ID,
            revision=NATIVE_C0_MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=True,
        )
        codebook = build_action_token_codebook(processor.tokenizer)
        candidates = {
            action: build_native_qwen_sft_batch(
                processor,
                window,
                instruction=episode.instruction_text,
                target_action=action,
            )
            for action in R2RCEAction
        }
        action_position = locate_shared_action_position(
            {action: _candidate_record(batch) for action, batch in candidates.items()},
            codebook=codebook,
        )
        target_batch = candidates[target_action]
        model_inputs = _move_batch(target_batch, device)
        expected_keys = {"input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"}
        if set(model_inputs) != expected_keys:
            raise RuntimeError(
                "Native-C0 input-key surface drifted: "
                f"{sorted(model_inputs)} != {sorted(expected_keys)}."
            )

        model = Qwen3VLBackbone(
            model_id=NATIVE_C0_MODEL_ID,
            lora_rank=32,
            lora_alpha=64,
            lora_dropout=0.0,
            enable_lora=True,
            freeze_base_model=True,
            apply_peft_2880_workaround=False,
            apply_spatial_rope=False,
            device=device,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            revision=NATIVE_C0_MODEL_REVISION,
        )
        base_model = model.model
        if base_model is None:
            raise RuntimeError("Exact-revision Qwen model did not load.")
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        if hasattr(base_model, "config"):
            base_model.config.use_cache = False
        trainable_contract = validate_lora_only_trainables(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=0.01,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        before = trainable_state_dict(model)
        before_sha = tensor_state_sha256(before)
        outputs = model(**model_inputs)
        loss = outputs.loss
        training_loss = float(loss.detach().float().cpu().item())
        if not math.isfinite(training_loss) or training_loss <= 0.0:
            raise RuntimeError("Native-C0 training loss is invalid.")
        loss.backward()
        gradients = gradient_report(model)
        global_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
            .detach()
            .float()
            .cpu()
            .item()
        )
        optimizer.step()
        updates = update_report(model, before)
        validate_optimization_reports(gradients, updates)
        after_sha = tensor_state_sha256(trainable_state_dict(model))
        if after_sha == before_sha:
            raise RuntimeError("Native-C0 trainable state did not change after the update.")
        del outputs, loss, before

        model.eval()
        post_update_loss, post_update_logits = _evaluate(model, model_inputs)
        del post_update_logits
        checkpoint_info = save_native_c0_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            metadata={
                "gate": NATIVE_C0_GATE_NAME,
                "run_id": args.run_id,
                "model_id": NATIVE_C0_MODEL_ID,
                "model_revision": NATIVE_C0_MODEL_REVISION,
                "episode_id": episode.episode_id,
                "target_action": target_action.name,
                "source_locks": locked_sources,
                "parent_processor_result_sha256": NATIVE_C0_PARENT_RESULT_SHA256,
                "parent_processor_verification_sha256": NATIVE_C0_PARENT_VERIFICATION_SHA256,
            },
        )

        first_trainable = next(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        with torch.no_grad():
            first_trainable.view(-1)[0].add_(1.0)
        perturbed_sha = tensor_state_sha256(trainable_state_dict(model))
        if perturbed_sha == after_sha:
            raise RuntimeError("Native-C0 checkpoint reload probe did not perturb the state.")
        load_native_c0_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
        reloaded_sha = tensor_state_sha256(trainable_state_dict(model))
        if reloaded_sha != after_sha or reloaded_sha != checkpoint_info["trainable_state_sha256"]:
            raise RuntimeError("Native-C0 reloaded state does not match the saved state.")

        reload_loss, reload_logits = _evaluate(model, model_inputs)
        reload_delta = abs(reload_loss - post_update_loss)
        if reload_delta > 1e-5:
            raise RuntimeError(f"Native-C0 reload loss drifted by {reload_delta:.3e}.")
        decision = score_action_token_logits(
            reload_logits,
            action_token_position=action_position,
            codebook=codebook,
        )

        shapes = {key: list(value.shape) for key, value in model_inputs.items()}
        labels = model_inputs["labels"]
        target_start = int(target_batch.record["target_start_idx"])
        result.update(
            {
                "status": "passed",
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "selection_rule": "lowest_numeric_train_episode_with_non_stop_first_action",
                "parent_processor_gate": {
                    "result_path": str(processor_result_path),
                    "result_sha256": file_sha256(processor_result_path),
                    "verification_path": str(processor_verification_path),
                    "verification_sha256": file_sha256(processor_verification_path),
                    "reverification_path": str(processor_reverification_path),
                    "reverification_sha256": file_sha256(processor_reverification_path),
                },
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
                    "rgb_sha256": array_sha256(rgb),
                    "target_action_id": int(target_action),
                    "target_action": target_action.name,
                    "policy_visual_fields": ["rgb"],
                    "explicitly_excluded_policy_fields": ["depth", "simulator_pose"],
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
                    "dtype": "bf16",
                    "gradient_checkpointing": "non_reentrant",
                    "docker_image": {
                        "reference": args.docker_image_ref,
                        "id": args.docker_image_id,
                    },
                    "habitat": environment.runtime_metadata(),
                },
                "processor": {
                    "class": type(processor).__name__,
                    "tokenizer_class": type(processor.tokenizer).__name__,
                    "model_id": NATIVE_C0_MODEL_ID,
                    "model_revision": NATIVE_C0_MODEL_REVISION,
                    "local_files_only": True,
                },
                "native_batch": {
                    **target_batch.record,
                    "system_prompt": NATIVE_QWEN_SYSTEM_PROMPT,
                    "shared_action_token_position": action_position,
                    "model_input_shapes": shapes,
                    "model_input_sha256": {
                        key: tensor_sha256(value) for key, value in model_inputs.items()
                    },
                    "masked_prefix_token_count": int(
                        (labels[:, :target_start] == -100).sum().item()
                    ),
                    "supervised_label_token_count": int((labels != -100).sum().item()),
                },
                "model": {
                    "model_id": NATIVE_C0_MODEL_ID,
                    "model_revision": NATIVE_C0_MODEL_REVISION,
                    "total_parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "lora_rank": 32,
                    "lora_alpha": 64,
                    "lora_dropout": 0.0,
                    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "spatial_rope_enabled": False,
                    "peft_2880_workaround_enabled": False,
                    "trainable_contract": trainable_contract,
                },
                "optimization": {
                    "optimizer": "AdamW",
                    "completed_updates": 1,
                    "learning_rate": args.learning_rate,
                    "weight_decay": 0.01,
                    "max_grad_norm": args.max_grad_norm,
                    "training_loss_before_update": training_loss,
                    "global_gradient_norm_before_clip": global_grad_norm,
                    "post_update_target_loss": post_update_loss,
                    "gradients": gradients,
                    "updates": updates,
                    "trainable_state_sha256_before": before_sha,
                    "trainable_state_sha256_after": after_sha,
                },
                "checkpoint": {
                    **checkpoint_info,
                    "perturbed_state_sha256": perturbed_sha,
                    "reloaded_state_sha256": reloaded_sha,
                    "reload_target_loss": reload_loss,
                    "reload_target_loss_absolute_delta": reload_delta,
                },
                "action_codebook": {
                    **codebook.to_record(),
                    "system_prompt": NATIVE_QWEN_SYSTEM_PROMPT,
                },
                "action_decision": {
                    **decision,
                    "target_action": target_action.name,
                    "prediction_matches_target": decision["predicted_action"] == target_action.name,
                    "note": (
                        "One-update target accuracy is descriptive only; deterministic legal "
                        "single-position scoring is the mechanics gate."
                    ),
                },
                "model_weight_locks": locked_weights,
                "source_locks": locked_sources,
                "accelerator": {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                },
            }
        )
        _atomic_json(output, result)
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        print(f"Native-Qwen C0 real-batch gate PASSED: {output}")
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
