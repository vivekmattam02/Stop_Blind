#!/usr/bin/env python3
"""Prepare and execute the bounded train-only GridNav development SFT run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from datetime import UTC, datetime
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
from spatialvlm.training.development_sft import (
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
    load_development_checkpoint,
    save_development_checkpoint,
    select_development_checkpoint,
    select_development_samples,
    update_early_stopping_state,
)
from spatialvlm.training.development_sft_artifacts import DEVELOPMENT_SFT_SOURCE_LOCK_PATHS
from spatialvlm.training.real_batch_gate import (
    file_sha256,
    gradient_group_report,
    load_trainable_checkpoint,
    tensor_state_sha256,
    trainable_state_dict,
    update_group_report,
    validate_group_reports,
)
from spatialvlm.training.sft import FREEZE_GROUP_PATTERNS, SFTConfig, SFTTrainer
from spatialvlm.training.stability_pilot import action_only_labels
from spatialvlm.utils.camera import CameraIntrinsics

_STOP_REQUESTED = False


def _request_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("Received a termination signal; checkpointing after the current update.", flush=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _record_sha256(record: dict[str, Any]) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_locks(repo: Path) -> dict[str, str]:
    locks: dict[str, str] = {}
    for relative in DEVELOPMENT_SFT_SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required development-run source is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _load_dataset(args: argparse.Namespace) -> R2RCEDataset:
    return R2RCEDataset.from_release(args.episodes, args.ground_truth, split="train")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _validate_pilot_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], set[str]]:
    expected_files = (
        (args.pilot_manifest, PILOT_MANIFEST_SHA256),
        (args.pilot_suite, PILOT_SUITE_SHA256),
        (args.pilot_verification, PILOT_VERIFICATION_V2_SHA256),
    )
    for path, expected_hash in expected_files:
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"Pilot evidence hash mismatch: {path}")
    if args.warmstart_checkpoint.stat().st_size != WARMSTART_CHECKPOINT_BYTES:
        raise RuntimeError("Warm-start checkpoint byte-size mismatch.")
    if file_sha256(args.warmstart_checkpoint) != WARMSTART_CHECKPOINT_SHA256:
        raise RuntimeError("Warm-start checkpoint SHA-256 mismatch.")

    pilot_manifest = _load_json(args.pilot_manifest)
    suite = _load_json(args.pilot_suite)
    verification = _load_json(args.pilot_verification)
    if pilot_manifest.get("run_id") != PILOT_RUN_ID or pilot_manifest.get("status") != "passed":
        raise RuntimeError("Pilot manifest identity/status mismatch.")
    selected = suite.get("selected_checkpoint")
    if (
        suite.get("status") != "passed"
        or not isinstance(selected, dict)
        or selected.get("arm_id") != "lr_5em05"
        or selected.get("step") != 8
        or selected.get("learning_rate") != 5e-5
        or selected.get("checkpoint", {}).get("sha256") != WARMSTART_CHECKPOINT_SHA256
    ):
        raise RuntimeError("Pilot suite no longer identifies the frozen warm start.")
    if verification.get("status") != "passed" or verification.get("error_count") != 0:
        raise RuntimeError("Corrected pilot verification did not pass cleanly.")
    trajectories = {
        str(row["trajectory_id"])
        for row in pilot_manifest.get("samples", [])
        if isinstance(row, dict) and "trajectory_id" in row
    }
    if len(trajectories) != 24:
        raise RuntimeError("Pilot trajectory exclusion set is incomplete.")
    return suite, trajectories


def _prepare_manifest(args: argparse.Namespace) -> int:
    protocol = DevelopmentSFTProtocol()
    repo = args.repo.expanduser().resolve()
    suite, excluded = _validate_pilot_inputs(args)
    dataset = _load_dataset(args)
    samples = select_development_samples(
        dataset, excluded_trajectory_ids=excluded, protocol=protocol
    )
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
        record = _load_json(path)
        if record.get("status") != "passed" or record.get("gate") != expected_gate:
            raise RuntimeError(f"Required {label} preflight did not pass.")
        preflights[label] = {
            "relative_path": str(path.relative_to(output.parent)),
            "sha256": file_sha256(path),
            "gate": expected_gate,
        }
    payload = {
        "schema_version": DEVELOPMENT_SFT_SCHEMA_VERSION,
        "gate": DEVELOPMENT_SFT_GATE_NAME,
        "run_id": args.run_id,
        "status": "passed",
        "created_at_utc": _utc_now(),
        "protocol": protocol.to_record(),
        "release_manifest": (
            dataset.release_manifest.to_dict() if dataset.release_manifest is not None else None
        ),
        "pilot_link": {
            "run_id": PILOT_RUN_ID,
            "manifest_sha256": PILOT_MANIFEST_SHA256,
            "suite_sha256": PILOT_SUITE_SHA256,
            "verification_v2_sha256": PILOT_VERIFICATION_V2_SHA256,
            "excluded_trajectory_ids": sorted(excluded),
            "selected_checkpoint": suite["selected_checkpoint"],
            "warmstart_checkpoint_bytes": WARMSTART_CHECKPOINT_BYTES,
            "warmstart_checkpoint_sha256": WARMSTART_CHECKPOINT_SHA256,
            "warmstart_trainable_state_sha256": WARMSTART_TRAINABLE_STATE_SHA256,
        },
        "samples": [sample.to_record() for sample in samples],
        "preflights": preflights,
        "source_locks": _source_locks(repo),
        "evidence_boundary": {
            "established": "Frozen train-only inputs, initialization, budget, and stopping rule.",
            "not_established": "Training success, navigation performance, or generalization.",
        },
    }
    _atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validate_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = DevelopmentSFTProtocol()
    manifest = _load_json(args.manifest)
    if manifest.get("protocol") != protocol.to_record():
        raise RuntimeError("Development manifest does not match the frozen protocol.")
    if manifest.get("run_id") != args.run_id or manifest.get("status") != "passed":
        raise RuntimeError("Development manifest identity/status mismatch.")
    repo = args.repo.expanduser().resolve()
    if manifest.get("source_locks") != _source_locks(repo):
        raise RuntimeError("Development source locks changed after manifest creation.")
    _, excluded = _validate_pilot_inputs(args)
    dataset = _load_dataset(args)
    expected = [
        sample.to_record()
        for sample in select_development_samples(
            dataset, excluded_trajectory_ids=excluded, protocol=protocol
        )
    ]
    if manifest.get("samples") != expected:
        raise RuntimeError("Development samples do not reconstruct from the locked release.")
    return manifest, expected


def _render_cache_entry(
    *,
    sample: dict[str, Any],
    dataset: R2RCEDataset,
    environment: HabitatSimR2REnvironment,
    habitat_config: HabitatSimR2RConfig,
    model_config: SpatialVLMConfig,
    siglip_processor: Any,
    dinov2_processor: Any,
    tokenizer: Any,
    codebook: Any,
) -> dict[str, Any]:
    episode = dataset.episode_record(sample["episode_id"])
    ground_truth = dataset.ground_truth_record(episode.episode_id)
    step_index = int(sample["action_step_index"])
    target_action = R2RCEAction(sample["target_action_id"])
    if ground_truth.actions[step_index] != target_action:
        raise RuntimeError("Frozen target no longer matches the release.")
    prefix = tuple(int(action) for action in ground_truth.actions[:step_index])
    if prefix != tuple(sample["replay_prefix_action_ids"]):
        raise RuntimeError("Frozen replay prefix no longer matches the release.")
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
    rendered = build_input_ids(
        tokenizer,
        instruction=episode.instruction_text,
        target=canonical_action_target(target_action),
        config=SpatialTokenizerConfig(
            num_spatial_tokens=model_config.fusion.sva_num_queries,
            system_prompt=ACTION_DECISION_SYSTEM_PROMPT,
            max_length=2048,
        ),
    )
    labels, action_position = action_only_labels(rendered, action=target_action, codebook=codebook)
    spatial_start = rendered["spatial_start_idx"]
    if not isinstance(spatial_start, int):
        raise TypeError("spatial_start_idx must be an integer.")
    return {
        "schema_version": 1,
        "sample": sample,
        "sample_sha256": _record_sha256(sample),
        "batch": {
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
        },
        "action_token_position": action_position,
        "target_action_id": int(target_action),
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


def _cache_filename(sample: dict[str, Any]) -> str:
    return (
        f"{sample['split']}_{int(sample['sample_index']):04d}_"
        f"episode-{sample['episode_id']}_action-{int(sample['action_step_index']):03d}.pt"
    )


def _prepare_cache(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    cache_root = args.cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    complete_path = cache_root / "cache_manifest.json"
    manifest_hash = file_sha256(args.manifest)
    if complete_path.is_file():
        complete = _load_json(complete_path)
        if (
            complete.get("status") == "passed"
            and complete.get("manifest_sha256") == manifest_hash
            and len(complete.get("entries", [])) == len(samples)
        ):
            print(f"Using complete immutable observation cache: {complete_path}", flush=True)
            return complete
        raise RuntimeError("Existing cache manifest does not match this development run.")

    dataset = _load_dataset(args)
    model_config = SpatialVLMConfig()
    siglip_processor = AutoImageProcessor.from_pretrained(model_config.encoder.siglip_model_id)
    dinov2_processor = AutoImageProcessor.from_pretrained(model_config.encoder.dinov2_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.backbone.model_id, trust_remote_code=True
    )
    codebook = build_action_token_codebook(tokenizer)
    habitat_config = HabitatSimR2RConfig(scene_root=args.scene_root)
    environment = HabitatSimR2REnvironment(habitat_config)
    entries = []
    progress_path = cache_root / "cache_progress.json"
    # Rendering order is grouped by scene to avoid reloading the same GLB hundreds of times.
    # Training/calibration order remains exactly the manifest order through the keyed cache.
    render_order = sorted(
        samples,
        key=lambda row: (str(row["scene_id"]), str(row["split"]), int(row["sample_index"])),
    )
    try:
        for ordinal, sample in enumerate(render_order, start=1):
            path = cache_root / _cache_filename(sample)
            if path.is_file():
                cached = torch.load(path, map_location="cpu", weights_only=False)
                if (
                    cached.get("schema_version") != 1
                    or cached.get("sample") != sample
                    or cached.get("sample_sha256") != _record_sha256(sample)
                ):
                    raise RuntimeError(f"Existing cache entry is incompatible: {path}")
            else:
                cached = _render_cache_entry(
                    sample=sample,
                    dataset=dataset,
                    environment=environment,
                    habitat_config=habitat_config,
                    model_config=model_config,
                    siglip_processor=siglip_processor,
                    dinov2_processor=dinov2_processor,
                    tokenizer=tokenizer,
                    codebook=codebook,
                )
                temporary = path.with_suffix(path.suffix + ".tmp")
                torch.save(cached, temporary)
                os.replace(temporary, path)
            entry = {
                "ordinal": ordinal,
                "split": sample["split"],
                "sample_index": sample["sample_index"],
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "sample_sha256": _record_sha256(sample),
            }
            entries.append(entry)
            _atomic_json(
                progress_path,
                {
                    "schema_version": 1,
                    "status": "running",
                    "manifest_sha256": manifest_hash,
                    "completed_entries": ordinal,
                    "target_entries": len(samples),
                    "last_entry": entry,
                    "updated_at_utc": _utc_now(),
                },
            )
            print(
                f"[cache {ordinal}/{len(samples)}] {sample['split']} "
                f"episode={sample['episode_id']} action={sample['target_action']}",
                flush=True,
            )
        runtime = environment.runtime_metadata()
    finally:
        environment.close()
    complete = {
        "schema_version": 1,
        "gate": DEVELOPMENT_SFT_GATE_NAME,
        "status": "passed",
        "completed_at_utc": _utc_now(),
        "manifest_sha256": manifest_hash,
        "entry_count": len(entries),
        "entries": entries,
        "render_order": "scene_id_then_split_then_sample_index",
        "training_order_changed_by_cache": False,
        "habitat_runtime": runtime,
        "licensed_derived_cache": True,
        "redistribution_allowed": False,
    }
    _atomic_json(complete_path, complete)
    return complete


def _cache_index(
    cache_manifest: dict[str, Any], cache_root: Path
) -> dict[tuple[str, int], dict[str, Any]]:
    index = {}
    for row in cache_manifest["entries"]:
        key = (str(row["split"]), int(row["sample_index"]))
        path = (cache_root / row["relative_path"]).resolve()
        if not path.is_relative_to(cache_root.resolve()):
            raise RuntimeError("Cache entry escapes the cache root.")
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"Cache entry is missing or has wrong size: {path}")
        if file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"Cache entry hash mismatch: {path}")
        index[key] = {**row, "path": path}
    return index


def _load_cached(entry: dict[str, Any], expected_sample: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(entry["path"], map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != 1
        or payload.get("sample") != expected_sample
        or payload.get("sample_sha256") != entry["sample_sha256"]
    ):
        raise RuntimeError(f"Cache payload identity mismatch: {entry['path']}")
    return payload


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


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
    calibration_samples: list[dict[str, Any]],
    cache: dict[tuple[str, int], dict[str, Any]],
    codebook: Any,
    device: torch.device,
    protocol: DevelopmentSFTProtocol,
    step: int,
) -> dict[str, Any]:
    rows = []
    model.eval()
    with torch.no_grad():
        for sample in calibration_samples:
            entry = cache[("calibration", int(sample["sample_index"]))]
            prepared = _load_cached(entry, sample)
            batch = _move_batch(prepared["batch"], device)
            inputs = {key: value for key, value in batch.items() if key != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            decision = score_action_token_logits(
                logits,
                action_token_position=int(prepared["action_token_position"]),
                codebook=codebook,
            )
            target_name = R2RCEAction(prepared["target_action_id"]).name
            nll = float(decision["scores"][target_name]["token_nll"])
            if not math.isfinite(nll) or nll > protocol.hard_max_token_nll:
                raise RuntimeError(f"Calibration target NLL violates hard stop: {nll}")
            rows.append(
                {
                    "sample_index": sample["sample_index"],
                    "episode_id": sample["episode_id"],
                    "trajectory_id": sample["trajectory_id"],
                    "target_action": target_name,
                    "target_token_nll": nll,
                    "predicted_action": decision["predicted_action"],
                    "prediction_matches_target": decision["predicted_action"] == target_name,
                    "top_two_margin_nats": decision["top_two_margin_nats"],
                    "restricted_entropy_nats": decision["restricted_entropy_nats"],
                    "action_token_position": prepared["action_token_position"],
                    "cache_sha256": entry["sha256"],
                }
            )
            del batch, inputs, outputs, logits, prepared
            torch.cuda.empty_cache()
    return {
        "step": step,
        "sample_count": len(rows),
        "mean_target_token_nll": sum(row["target_token_nll"] for row in rows) / len(rows),
        "descriptive_accuracy": sum(row["prediction_matches_target"] for row in rows) / len(rows),
        "accuracy_is_selection_or_stop_condition": False,
        "samples": rows,
    }


def _load_checkpoint_index(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = _load_json(path)
    rows = payload.get("checkpoints")
    if not isinstance(rows, list):
        raise RuntimeError("Checkpoint index is malformed.")
    return rows


def _save_checkpoint_index(path: Path, rows: list[dict[str, Any]], manifest_hash: str) -> None:
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "gate": DEVELOPMENT_SFT_GATE_NAME,
            "manifest_sha256": manifest_hash,
            "checkpoints": rows,
            "updated_at_utc": _utc_now(),
        },
    )


def _run_training(args: argparse.Namespace) -> int:
    invocation_started = time.perf_counter()
    protocol = DevelopmentSFTProtocol()
    manifest, samples = _validate_manifest(args)
    manifest_hash = file_sha256(args.manifest)
    cache_manifest = _prepare_cache(args=args, manifest=manifest, samples=samples)
    cache_manifest_path = args.cache_root.expanduser().resolve() / "cache_manifest.json"
    cache = _cache_index(cache_manifest, args.cache_root.expanduser().resolve())
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The Spark development run requires a visible CUDA device.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not report BF16 support.")

    random.seed(20260823)
    np.random.seed(20260823)
    torch.manual_seed(20260823)
    torch.cuda.manual_seed_all(20260823)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats(device)
    model_config = SpatialVLMConfig()
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.backbone.model_id, trust_remote_code=True
    )
    codebook = build_action_token_codebook(tokenizer)
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
            learning_rate=protocol.learning_rate,
            weight_decay=protocol.weight_decay,
            max_grad_norm=protocol.max_grad_norm,
            trainable_groups=("siglip_proj", "dino_proj", "gatr", "sva", "lora"),
        ),
    )
    warm_payload = load_trainable_checkpoint(
        args.warmstart_checkpoint, model=model, optimizer=trainer.optimizer
    )
    if warm_payload.get("trainable_state_sha256") != WARMSTART_TRAINABLE_STATE_SHA256:
        raise RuntimeError("Warm-start payload trainable hash mismatch.")
    if tensor_state_sha256(trainable_state_dict(model)) != WARMSTART_TRAINABLE_STATE_SHA256:
        raise RuntimeError("Loaded warm-start tensors do not reproduce the selected state.")
    for group in trainer.optimizer.param_groups:
        if float(group["lr"]) != protocol.learning_rate:
            raise RuntimeError("Restored optimizer learning rate differs from the frozen protocol.")

    output_root = args.output_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    events_path = output_root / "events.jsonl"
    latest_path = output_root / "latest.json"
    checkpoint_index_path = output_root / "checkpoint_index.json"
    result_path = output_root / "result.json"
    checkpoint_rows = _load_checkpoint_index(checkpoint_index_path)
    train_samples = [row for row in samples if row["split"] == "train"]
    calibration_samples = [row for row in samples if row["split"] == "calibration"]

    if args.resume:
        if not latest_path.is_file():
            raise RuntimeError("Resume requested, but latest.json is absent.")
        latest = _load_json(latest_path)
        path = Path(latest["path"])
        if not path.is_file() or file_sha256(path) != latest["sha256"]:
            raise RuntimeError("Latest resume checkpoint is missing or changed.")
        payload = load_development_checkpoint(
            path,
            model=model,
            optimizer=trainer.optimizer,
            expected_manifest_sha256=manifest_hash,
        )
        state = payload["run_state"]
        if not checkpoint_rows or checkpoint_rows[-1]["sha256"] != latest["sha256"]:
            raise RuntimeError("Latest pointer and checkpoint index disagree.")
        resumed_from = latest
    else:
        if latest_path.exists() or checkpoint_rows or result_path.exists():
            raise RuntimeError("Fresh run refuses existing training state; use --resume.")
        random.seed(protocol.seed)
        np.random.seed(protocol.seed)
        torch.manual_seed(protocol.seed)
        torch.cuda.manual_seed_all(protocol.seed)
        baseline = _calibrate(
            model=model,
            calibration_samples=calibration_samples,
            cache=cache,
            codebook=codebook,
            device=device,
            protocol=protocol,
            step=0,
        )
        state = {
            "update_step": 0,
            "next_train_index": 0,
            "steps": [],
            "calibrations": [baseline],
            "best_nll": baseline["mean_target_token_nll"],
            "non_improving_calibrations": 0,
            "accumulated_elapsed_seconds": 0.0,
        }
        resumed_from = None
    _append_event(
        events_path,
        {
            "event": "training_invocation_started",
            "recorded_at_utc": _utc_now(),
            "resume": bool(args.resume),
            "start_step": state["update_step"],
        },
    )

    audit_steps = {1, *protocol.checkpoint_steps, *protocol.calibration_steps[1:]}
    terminal_reason = "planned_update_budget_completed"
    early_stop = False
    train_clock_started = time.perf_counter()
    model.train()
    try:
        for index in range(int(state["next_train_index"]), protocol.planned_updates):
            elapsed = float(state["accumulated_elapsed_seconds"]) + (
                time.perf_counter() - train_clock_started
            )
            if elapsed >= protocol.max_hours * 3600.0:
                terminal_reason = "wall_time_budget_reached"
                break
            step = index + 1
            sample = train_samples[index]
            prepared = _load_cached(cache[("train", int(sample["sample_index"]))], sample)
            batch = _move_batch(prepared["batch"], device)
            trainer.optimizer.zero_grad(set_to_none=True)
            before = trainable_state_dict(model) if step in audit_steps else None
            step_started = time.perf_counter()
            target_nll = trainer.forward_backward(batch)
            if not math.isfinite(target_nll) or target_nll > protocol.hard_max_token_nll:
                raise RuntimeError(f"Training target NLL violates hard stop: {target_nll}")
            gradients = gradient_group_report(model)
            _validate_gradients(gradients)
            gradient_norm = trainer.clip_and_step()
            if not math.isfinite(gradient_norm) or gradient_norm > protocol.hard_max_gradient_norm:
                raise RuntimeError(f"Global gradient norm violates hard stop: {gradient_norm}")
            updates = update_group_report(model, before) if before is not None else None
            if updates is not None:
                validate_group_reports(gradients, updates)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            if peak_allocated > protocol.hard_max_peak_allocated_bytes:
                raise RuntimeError("Peak allocated accelerator memory exceeded 64 GiB.")
            row = {
                "step": step,
                "sample_index": sample["sample_index"],
                "episode_id": sample["episode_id"],
                "trajectory_id": sample["trajectory_id"],
                "scene_id": sample["scene_id"],
                "action_step_index": sample["action_step_index"],
                "target_action": sample["target_action"],
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
                "optimizer_step_seconds": time.perf_counter() - step_started,
                "cache_sha256": cache[("train", int(sample["sample_index"]))]["sha256"],
            }
            state["steps"].append(row)
            state["update_step"] = step
            state["next_train_index"] = index + 1
            del batch, before, prepared
            torch.cuda.empty_cache()
            model.train()

            if step in protocol.calibration_steps:
                calibration = _calibrate(
                    model=model,
                    calibration_samples=calibration_samples,
                    cache=cache,
                    codebook=codebook,
                    device=device,
                    protocol=protocol,
                    step=step,
                )
                state["calibrations"].append(calibration)
                best, count, early_stop = update_early_stopping_state(
                    previous_best_nll=float(state["best_nll"]),
                    previous_non_improving_count=int(state["non_improving_calibrations"]),
                    current_nll=float(calibration["mean_target_token_nll"]),
                    current_step=step,
                    protocol=protocol,
                )
                state["best_nll"] = best
                state["non_improving_calibrations"] = count
                model.train()

            must_checkpoint = step in protocol.checkpoint_steps or _STOP_REQUESTED
            if must_checkpoint:
                state["accumulated_elapsed_seconds"] = float(
                    state["accumulated_elapsed_seconds"]
                ) + (time.perf_counter() - train_clock_started)
                train_clock_started = time.perf_counter()
                tag = "signal" if _STOP_REQUESTED and step not in protocol.checkpoint_steps else ""
                suffix = f"_{tag}" if tag else ""
                checkpoint_path = checkpoint_root / f"step_{step:04d}{suffix}.pt"
                if checkpoint_path.exists():
                    raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
                artifact = save_development_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=trainer.optimizer,
                    run_state=state,
                    metadata={
                        "gate": DEVELOPMENT_SFT_GATE_NAME,
                        "run_id": args.run_id,
                        "manifest_sha256": manifest_hash,
                        "source_locks": manifest["source_locks"],
                        "warmstart_checkpoint_sha256": WARMSTART_CHECKPOINT_SHA256,
                        "cache_manifest_sha256": file_sha256(cache_manifest_path),
                    },
                )
                artifact["relative_path"] = checkpoint_path.name
                artifact["reason"] = "signal" if _STOP_REQUESTED else "scheduled"
                checkpoint_rows.append(artifact)
                _save_checkpoint_index(checkpoint_index_path, checkpoint_rows, manifest_hash)
                _atomic_json(latest_path, artifact)
            _atomic_json(
                output_root / "progress.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "run_id": args.run_id,
                    "update_step": step,
                    "planned_updates": protocol.planned_updates,
                    "last_target_token_nll": target_nll,
                    "best_calibration_nll": state["best_nll"],
                    "non_improving_calibrations": state["non_improving_calibrations"],
                    "latest_checkpoint": checkpoint_rows[-1] if checkpoint_rows else None,
                    "updated_at_utc": _utc_now(),
                },
            )
            print(
                f"[train {step}/{protocol.planned_updates}] action={sample['target_action']} "
                f"nll={target_nll:.6f} grad={gradient_norm:.3f}",
                flush=True,
            )
            if early_stop:
                terminal_reason = "pre_registered_early_stopping"
                break
            if _STOP_REQUESTED:
                terminal_reason = "termination_signal_checkpointed"
                break

        final_step = int(state["update_step"])
        if final_step and (not checkpoint_rows or checkpoint_rows[-1]["update_step"] != final_step):
            state["accumulated_elapsed_seconds"] = float(state["accumulated_elapsed_seconds"]) + (
                time.perf_counter() - train_clock_started
            )
            train_clock_started = time.perf_counter()
            checkpoint_path = checkpoint_root / f"step_{final_step:04d}_terminal.pt"
            if checkpoint_path.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
            artifact = save_development_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=trainer.optimizer,
                run_state=state,
                metadata={
                    "gate": DEVELOPMENT_SFT_GATE_NAME,
                    "run_id": args.run_id,
                    "manifest_sha256": manifest_hash,
                    "source_locks": manifest["source_locks"],
                    "warmstart_checkpoint_sha256": WARMSTART_CHECKPOINT_SHA256,
                    "cache_manifest_sha256": file_sha256(cache_manifest_path),
                },
            )
            artifact["relative_path"] = checkpoint_path.name
            artifact["reason"] = terminal_reason
            checkpoint_rows.append(artifact)
            _save_checkpoint_index(checkpoint_index_path, checkpoint_rows, manifest_hash)
            _atomic_json(latest_path, artifact)

        selected_calibration = select_development_checkpoint(state["calibrations"])
        selected_step = int(selected_calibration["step"])
        if selected_step == 0:
            selected_checkpoint = {
                "source": "pilot_warmstart",
                "step": 0,
                "path": str(args.warmstart_checkpoint.resolve()),
                "bytes": WARMSTART_CHECKPOINT_BYTES,
                "sha256": WARMSTART_CHECKPOINT_SHA256,
                "trainable_state_sha256": WARMSTART_TRAINABLE_STATE_SHA256,
            }
        else:
            selected_checkpoint = next(
                row for row in checkpoint_rows if row["update_step"] == selected_step
            )
            selected_checkpoint = {**selected_checkpoint, "source": "development_run"}
        baseline_nll = float(state["calibrations"][0]["mean_target_token_nll"])
        selected_nll = float(selected_calibration["mean_target_token_nll"])
        completed_minimum = final_step >= protocol.minimum_updates_before_early_stop
        accepted_terminal = terminal_reason in {
            "planned_update_budget_completed",
            "pre_registered_early_stopping",
            "wall_time_budget_reached",
        }
        passed = completed_minimum and accepted_terminal
        result = {
            "schema_version": DEVELOPMENT_SFT_SCHEMA_VERSION,
            "gate": DEVELOPMENT_SFT_GATE_NAME,
            "run_id": args.run_id,
            "status": "passed" if passed else "stopped",
            "terminal_reason": terminal_reason,
            "completed_at_utc": _utc_now(),
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": manifest_hash,
            "protocol": protocol.to_record(),
            "source_locks": manifest["source_locks"],
            "cache_manifest_path": str(cache_manifest_path),
            "cache_manifest_sha256": file_sha256(cache_manifest_path),
            "warmstart_checkpoint": {
                "path": str(args.warmstart_checkpoint.resolve()),
                "bytes": WARMSTART_CHECKPOINT_BYTES,
                "sha256": WARMSTART_CHECKPOINT_SHA256,
                "trainable_state_sha256": WARMSTART_TRAINABLE_STATE_SHA256,
            },
            "resumed_from": resumed_from,
            "training": {
                "completed_updates": final_step,
                "steps": state["steps"],
                "calibrations": state["calibrations"],
                "checkpoints": checkpoint_rows,
                "accumulated_elapsed_seconds": state["accumulated_elapsed_seconds"],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            },
            "selection": {
                "baseline_mean_target_token_nll": baseline_nll,
                "selected_mean_target_token_nll": selected_nll,
                "improvement_nats": baseline_nll - selected_nll,
                "improved_by_minimum_delta": (
                    baseline_nll - selected_nll >= protocol.minimum_improvement_nats
                ),
                "selected_calibration": selected_calibration,
                "selected_checkpoint": selected_checkpoint,
            },
            "pass_criteria": {
                "completed_minimum_updates": completed_minimum,
                "accepted_terminal_reason": accepted_terminal,
                "val_seen_accessed": False,
                "val_unseen_accessed": False,
            },
            "runtime": {
                "device": str(device),
                "accelerator": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "compiled_cuda": torch.version.cuda,
                "dtype": "bfloat16",
                "gradient_checkpointing": "non_reentrant",
                "trainable_parameter_count": trainer.trainable_parameter_count(),
                "invocation_elapsed_seconds": time.perf_counter() - invocation_started,
            },
            "evidence_boundary": {
                "established_if_passed": (
                    "A bounded train-only full-graph development schedule completed under its "
                    "frozen initialization, sample, calibration, stopping, and checkpoint rules."
                ),
                "not_established": (
                    "Closed-loop navigation improvement, val_seen/val_unseen performance, baseline "
                    "parity, multi-seed stability, ablation effects, or paper superiority."
                ),
            },
        }
        _atomic_json(result_path, result)
        _append_event(
            events_path,
            {
                "event": "training_invocation_finished",
                "recorded_at_utc": _utc_now(),
                "status": result["status"],
                "terminal_reason": terminal_reason,
                "completed_updates": final_step,
            },
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if passed else 2
    except Exception as exc:
        failure = {
            "schema_version": DEVELOPMENT_SFT_SCHEMA_VERSION,
            "gate": DEVELOPMENT_SFT_GATE_NAME,
            "run_id": args.run_id,
            "status": "failed",
            "recorded_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "manifest_sha256": manifest_hash,
            "last_completed_update": state.get("update_step"),
        }
        _atomic_json(output_root / "failure.json", failure)
        _append_event(events_path, {"event": "failed", **failure})
        raise


def _common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--pilot-suite", type=Path, required=True)
    parser.add_argument("--pilot-verification", type=Path, required=True)
    parser.add_argument("--warmstart-checkpoint", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    _common_inputs(prepare)
    prepare.add_argument("--processor-preflight", type=Path, required=True)
    prepare.add_argument("--action-preflight", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train")
    _common_inputs(train)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--scene-root", type=Path, required=True)
    train.add_argument("--cache-root", type=Path, required=True)
    train.add_argument("--checkpoint-root", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "prepare":
        return _prepare_manifest(args)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    return _run_training(args)


if __name__ == "__main__":
    sys.exit(main())
