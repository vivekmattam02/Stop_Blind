#!/usr/bin/env python3
"""Run a read-only, checkpoint-backed length-fair action-decision probe on Spark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import time
import traceback
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
    locate_shared_action_position,
    score_action_token_logits,
)
from spatialvlm.eval.action_decision_artifacts import (
    ACTION_DECISION_CHECKPOINT_RUN_ID,
    ACTION_DECISION_CHECKPOINT_SHA256,
    ACTION_DECISION_GATE_NAME,
    ACTION_DECISION_GATE_SCHEMA_VERSION,
    ACTION_DECISION_SOURCE_LOCK_PATHS,
)
from spatialvlm.model import SpatialVLM
from spatialvlm.training.real_batch_gate import (
    file_sha256,
    load_trainable_checkpoint,
    select_probe_episode,
)
from spatialvlm.training.sft import FREEZE_GROUP_PATTERNS, set_trainable_by_groups
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
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    for relative in ACTION_DECISION_SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source-lock path is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


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


def _forward_decision(
    model: SpatialVLM,
    batch: dict[str, Any],
    *,
    action_token_position: int,
    codebook: Any,
) -> dict[str, Any]:
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        decision = score_action_token_logits(
            logits,
            action_token_position=action_token_position,
            codebook=codebook,
        )
    del outputs, logits
    return decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--repeat-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if re.fullmatch(r"action_decision_v[1-9][0-9]*", args.run_id) is None:
        raise SystemExit("--run-id must match action_decision_v<positive integer>")
    if args.repeat_tolerance < 0.0 or not math.isfinite(args.repeat_tolerance):
        raise SystemExit("--repeat-tolerance must be finite and non-negative")
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": ACTION_DECISION_GATE_SCHEMA_VERSION,
        "gate": ACTION_DECISION_GATE_NAME,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": _utc_now(),
        "evidence_boundary": {
            "established_if_passed": (
                "The exact Qwen tokenizer provided four distinct one-token official action IDs at "
                "one shared causal position, and the preserved real_batch_v2 checkpoint produced "
                "finite deterministic scores on one real frozen observation."
            ),
            "not_established": (
                "Navigation competence, calibration, convergence, rollout improvement, baseline "
                "parity, val_seen or val_unseen performance, or a paper superiority claim."
            ),
        },
    }
    environment: HabitatSimR2REnvironment | None = None
    try:
        repo = args.repo.expanduser().resolve()
        output = args.output.expanduser().resolve()
        checkpoint = args.checkpoint.expanduser().resolve()
        preflight = args.preflight.expanduser().resolve()
        device = torch.device(args.device)
        if args.expected_checkpoint_sha256 != ACTION_DECISION_CHECKPOINT_SHA256:
            raise RuntimeError(
                "Expected checkpoint SHA-256 does not match the frozen protocol identity."
            )
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The Spark action-decision gate requires a visible CUDA device.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not report BF16 support.")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        checkpoint_hash = file_sha256(checkpoint)
        if checkpoint_hash != args.expected_checkpoint_sha256:
            raise RuntimeError(
                "Checkpoint SHA-256 does not match the frozen real_batch_v2 identity."
            )
        if not preflight.is_file():
            raise FileNotFoundError(f"Tokenizer preflight does not exist: {preflight}")
        preflight_record = json.loads(preflight.read_text(encoding="utf-8"))
        if preflight_record.get("status") != "passed":
            raise RuntimeError("Tokenizer preflight has not passed.")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)

        dataset = R2RCEDataset.from_release(args.episodes, args.ground_truth, split="train")
        episode, target_action = select_probe_episode(dataset)
        habitat_config = HabitatSimR2RConfig(scene_root=args.scene_root)
        environment = HabitatSimR2REnvironment(habitat_config)
        state = environment.reset(episode)
        rgb = state.observation["rgb"]
        depth_meters = state.observation["depth_meters"]

        model_config = SpatialVLMConfig()
        siglip_processor = AutoImageProcessor.from_pretrained(model_config.encoder.siglip_model_id)
        dinov2_processor = AutoImageProcessor.from_pretrained(model_config.encoder.dinov2_model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.backbone.model_id,
            trust_remote_code=True,
        )
        codebook = build_action_token_codebook(tokenizer)
        if codebook.to_record() != preflight_record.get("codebook"):
            raise RuntimeError("Runtime action codebook does not match the tokenizer preflight.")

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
            system_prompt=ACTION_DECISION_SYSTEM_PROMPT,
        )
        rendered = {
            action: build_input_ids(
                tokenizer,
                instruction=episode.instruction_text,
                target=canonical_action_target(action),
                config=token_config,
            )
            for action in R2RCEAction
        }
        action_position = locate_shared_action_position(rendered, codebook=codebook)
        batch = _tensor_batch(
            rgb_processed=rgb_processed,
            depth=depth,
            intrinsics=intrinsics,
            tokens=rendered[R2RCEAction.STOP],
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
        selected_names = set_trainable_by_groups(model, FREEZE_GROUP_PATTERNS)
        if not selected_names:
            raise RuntimeError("No trainable parameter names were selected before checkpoint load.")
        checkpoint_payload = load_trainable_checkpoint(checkpoint, model=model)
        checkpoint_metadata = checkpoint_payload.get("metadata", {})
        if checkpoint_metadata.get("run_id") != ACTION_DECISION_CHECKPOINT_RUN_ID:
            raise RuntimeError("Checkpoint metadata does not identify real_batch_v2.")

        model.eval()
        first = _forward_decision(
            model,
            batch,
            action_token_position=action_position,
            codebook=codebook,
        )
        second = _forward_decision(
            model,
            batch,
            action_token_position=action_position,
            codebook=codebook,
        )
        deltas = [
            abs(
                float(first["scores"][action.name]["token_nll"])
                - float(second["scores"][action.name]["token_nll"])
            )
            for action in R2RCEAction
        ]
        max_delta = max(deltas)
        predicted_match = first["predicted_action"] == second["predicted_action"]
        if max_delta > args.repeat_tolerance or not predicted_match:
            raise RuntimeError(
                "Repeated checkpoint-backed action decisions are not deterministic within the "
                "frozen tolerance."
            )

        result.update(
            {
                "status": "passed",
                "completed_at_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "source_locks": _source_locks(repo),
                "preflight_sha256": file_sha256(preflight),
                "codebook": codebook.to_record(),
                "checkpoint": {
                    "path": str(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": checkpoint_hash,
                    "expected_sha256": args.expected_checkpoint_sha256,
                    "metadata_run_id": checkpoint_metadata.get("run_id"),
                    "trainable_state_sha256": checkpoint_payload.get("trainable_state_sha256"),
                },
                "source": {
                    "episode_id": episode.episode_id,
                    "trajectory_id": episode.trajectory_id,
                    "scene_id": episode.scene_id,
                    "instruction_sha256": hashlib.sha256(
                        episode.instruction_text.encode("utf-8")
                    ).hexdigest(),
                    "rgb_sha256": _array_sha256(rgb),
                    "metric_depth_sha256": _array_sha256(depth_meters),
                    "released_target_action_id": int(target_action),
                    "released_target_action": target_action.name,
                },
                "token_contract": {
                    "system_prompt": ACTION_DECISION_SYSTEM_PROMPT,
                    "action_token_position": action_position,
                    "target_start_idx": rendered[R2RCEAction.STOP]["target_start_idx"],
                    "candidate_sequence_lengths": {
                        action.name: int(rendered[action]["input_ids"].numel())
                        for action in R2RCEAction
                    },
                },
                "repeatability": {
                    "first": first,
                    "second": second,
                    "token_nll_tolerance": args.repeat_tolerance,
                    "max_token_nll_absolute_delta": max_delta,
                    "predicted_actions_match": predicted_match,
                    "prediction_matches_released_target": (
                        first["predicted_action"] == target_action.name
                    ),
                    "accuracy_is_pass_condition": False,
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
                    "habitat": environment.runtime_metadata(),
                },
                "processors": {
                    "depth_contract": depth_manifest,
                    "siglip_processor_class": type(siglip_processor).__name__,
                    "dinov2_processor_class": type(dinov2_processor).__name__,
                },
                "model": {
                    "backbone_model_id": model_config.backbone.model_id,
                    "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "selected_trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                },
                "accelerator": {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                },
            }
        )
        _atomic_json(output, result)
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        print(f"Action-decision gate PASSED: {output}")
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
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return 1
    finally:
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
