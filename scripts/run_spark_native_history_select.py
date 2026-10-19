#!/usr/bin/env python3
"""Prepare, cache, and run the matched native-Qwen H0/H1/H4 train-only suite."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import signal
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, replace
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
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
    NativeQwenBatch,
    build_native_qwen_sft_batch,
)
from spatialvlm.eval.action_decision import (
    ActionTokenCodebook,
    build_action_token_codebook,
    locate_action_token,
    score_action_token_logits,
)
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    file_sha256,
    gradient_report,
    model_weight_locks,
    tensor_sha256,
    tensor_state_sha256,
    trainable_state_dict,
    update_report,
    validate_lora_only_trainables,
    validate_optimization_reports,
)
from spatialvlm.training.native_history_select import (
    HISTORY_ARMS,
    NATIVE_HISTORY_GATE_NAME,
    NATIVE_HISTORY_SCHEMA_VERSION,
    NativeHistoryProtocol,
    causal_logit_diagnostics,
    diagnostic_float,
    load_history_checkpoint,
    optimizer_tensor_diagnostics,
    save_history_checkpoint,
    select_arm_checkpoint,
    select_history_arm,
    select_native_history_samples,
    tensor_collection_diagnostics,
)

EXPECTED_DOCKER_IMAGE_ID = "sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf"
SOURCE_LOCK_PATHS: tuple[str, ...] = (
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

ARM_TO_MODE = {
    "H0_CURRENT": HistoryMode.H0_CURRENT,
    "H1_PREVIOUS_ACTION": HistoryMode.H1_PREVIOUS_ACTION,
    "H4_CAUSAL": HistoryMode.H4_SHORT_WINDOW,
}

_STOP_REQUESTED = False


def _request_stop(signum: int, frame: Any) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("Termination requested; checkpointing after the current update.", flush=True)


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


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_locks(repo: Path) -> dict[str, str]:
    locks: dict[str, str] = {}
    for relative in SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source-lock path is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _load_dataset(args: argparse.Namespace) -> R2RCEDataset:
    return R2RCEDataset.from_release(
        args.episodes.expanduser().resolve(),
        args.ground_truth.expanduser().resolve(),
        split="train",
    )


def _protocol(args: argparse.Namespace) -> NativeHistoryProtocol:
    defaults = NativeHistoryProtocol()
    if args.planned_updates == defaults.planned_updates:
        calibration_steps = defaults.calibration_steps
    else:
        checkpoint_steps = set(
            range(args.checkpoint_interval, args.planned_updates + 1, args.checkpoint_interval)
        )
        checkpoint_steps.add(args.planned_updates)
        calibration_steps = tuple(
            sorted(
                {0, args.planned_updates}
                | {
                    max(
                        (step for step in checkpoint_steps if step <= fraction),
                        default=args.planned_updates,
                    )
                    for fraction in (
                        args.planned_updates // 8,
                        args.planned_updates // 4,
                        args.planned_updates // 2,
                        (3 * args.planned_updates) // 4,
                    )
                }
            )
        )
    return replace(
        defaults,
        planned_updates=args.planned_updates,
        train_samples_per_action=args.train_samples_per_action,
        calibration_samples_per_action=args.calibration_samples_per_action,
        calibration_steps=calibration_steps,
        checkpoint_interval=args.checkpoint_interval,
        max_hours=args.max_hours,
    )


def _prepare_manifest(args: argparse.Namespace) -> int:
    repo = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID:
        raise ValueError("Docker image ID does not match the admitted native-C0 image.")
    protocol = _protocol(args)
    dataset = _load_dataset(args)
    samples = select_native_history_samples(dataset, protocol=protocol)
    sample_records = [sample.to_record() for sample in samples]
    counts = {
        split: {
            action.name: sum(
                row["split"] == split and row["target_action_id"] == int(action)
                for row in sample_records
            )
            for action in R2RCEAction
        }
        for split in ("train", "calibration")
    }
    trajectories = [row["trajectory_id"] for row in sample_records]
    if len(trajectories) != len(set(trajectories)):
        raise RuntimeError("Manifest selection reused a physical trajectory.")
    manifest = {
        "schema_version": NATIVE_HISTORY_SCHEMA_VERSION,
        "gate": NATIVE_HISTORY_GATE_NAME,
        "run_id": args.run_id,
        "status": "frozen",
        "protocol": protocol.to_record(),
        "dataset": {
            "release_manifest": asdict(dataset.release_manifest)
            if dataset.release_manifest is not None
            else None,
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
        },
        "runtime": {
            "docker_image_ref": args.docker_image_ref,
            "docker_image_id": args.docker_image_id,
            "sequential_arms": True,
            "fully_offline": True,
        },
        "samples": sample_records,
        "sample_counts": counts,
        "sample_records_sha256": _canonical_sha256(sample_records),
        "source_locks": source_locks(repo),
        "outputs": {
            "cache_manifest": "/checkpoints/cache/cache_manifest.json",
            "progress": "/artifacts/progress.json",
            "suite": "/artifacts/suite.json",
            "verification": "/artifacts/verification.json",
            "checkpoint_root": "/checkpoints/arms",
        },
        "evidence_boundary": {
            "established_if_passed": (
                "A train-only held-out action-token NLL development choice among matched native-"
                "Qwen H0, H1, and causal H4 under the frozen paired selection rule."
            ),
            "not_established": (
                "Closed-loop navigation competence, val_seen or val_unseen performance, C1-C4 "
                "effects, baseline parity, generalization, or any paper-level superiority claim."
            ),
        },
    }
    _atomic_json(output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Frozen native-history manifest written to {output}")
    return 0


def _validate_manifest(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], NativeHistoryProtocol, R2RCEDataset, list[dict[str, Any]]]:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("gate") != NATIVE_HISTORY_GATE_NAME or manifest.get("run_id") != args.run_id:
        raise RuntimeError("Manifest gate or run identity mismatch.")
    if manifest.get("source_locks") != source_locks(args.repo.expanduser().resolve()):
        raise RuntimeError("Manifest source locks no longer match the repository.")
    protocol = _protocol(args)
    if manifest.get("protocol") != protocol.to_record():
        raise RuntimeError("Manifest protocol differs from the invocation.")
    dataset = _load_dataset(args)
    rebuilt = [
        sample.to_record() for sample in select_native_history_samples(dataset, protocol=protocol)
    ]
    if manifest.get("samples") != rebuilt:
        raise RuntimeError("Manifest samples do not reconstruct from the locked train release.")
    if manifest.get("sample_records_sha256") != _canonical_sha256(rebuilt):
        raise RuntimeError("Manifest sample-record hash mismatch.")
    return manifest, protocol, dataset, rebuilt


def _cache_filename(sample: dict[str, Any]) -> str:
    return f"{sample['split']}_{int(sample['sample_index']):04d}.pt"


def _render_cache_entry(
    environment: HabitatSimR2REnvironment,
    episode: Any,
    sample: dict[str, Any],
    *,
    gate_name: str = NATIVE_HISTORY_GATE_NAME,
) -> dict[str, Any]:
    state = environment.reset(episode)
    frames: deque[np.ndarray] = deque(maxlen=4)
    frames.append(np.asarray(state.observation["rgb"], dtype=np.uint8).copy())
    prefix = tuple(R2RCEAction(value) for value in sample["replay_prefix_action_ids"])
    for action in prefix:
        if action == R2RCEAction.STOP:
            raise RuntimeError("A cache replay prefix contains STOP.")
        state = environment.step(action)
        frames.append(np.asarray(state.observation["rgb"], dtype=np.uint8).copy())
    if len(frames) != 4:
        raise RuntimeError("Causal H4 cache entry does not contain exactly four real frames.")
    stacked = np.stack(tuple(frames), axis=0)
    history_actions = tuple(int(action) for action in prefix[-3:])
    if len(history_actions) != 3:
        raise RuntimeError("Causal H4 cache entry does not contain three prior actions.")
    tensor = torch.from_numpy(stacked.copy())
    return {
        "schema_version": 1,
        "gate": gate_name,
        "sample": sample,
        "sample_sha256": _canonical_sha256(sample),
        "rgb_frames": tensor,
        "rgb_frames_sha256": tensor_sha256(tensor),
        "history_action_ids": history_actions,
        "policy_visible_fields": ["rgb", "previous_action_id_when_enabled"],
        "explicitly_excluded_policy_fields": ["depth", "simulator_pose", "goal_coordinate"],
    }


def _validate_cache_entry(
    path: Path,
    expected_sample: dict[str, Any],
    *,
    gate_name: str = NATIVE_HISTORY_GATE_NAME,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("gate") != gate_name:
        raise RuntimeError(f"Cache gate mismatch: {path}")
    tensor = payload.get("rgb_frames")
    if payload.get("sample") != expected_sample:
        raise RuntimeError(f"Cache sample identity mismatch: {path}")
    if payload.get("sample_sha256") != _canonical_sha256(expected_sample):
        raise RuntimeError(f"Cache sample hash mismatch: {path}")
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
        raise RuntimeError(f"Cache RGB tensor is invalid: {path}")
    if tensor.ndim != 4 or tuple(tensor.shape[0:1]) != (4,) or tensor.shape[-1] != 3:
        raise RuntimeError(f"Cache RGB shape is invalid: {path}")
    if payload.get("rgb_frames_sha256") != tensor_sha256(tensor):
        raise RuntimeError(f"Cache RGB content hash mismatch: {path}")
    actions = payload.get("history_action_ids")
    if not isinstance(actions, (tuple, list)) or len(actions) != 3 or 0 in actions:
        raise RuntimeError(f"Cache causal-action window is invalid: {path}")
    return payload


def _prepare_cache(
    *,
    args: argparse.Namespace,
    manifest_hash: str,
    dataset: R2RCEDataset,
    samples: list[dict[str, Any]],
    gate_name: str = NATIVE_HISTORY_GATE_NAME,
) -> dict[str, Any]:
    cache_root = args.cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "cache_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") != manifest_hash or existing.get("status") != "passed":
            raise RuntimeError("Existing cache manifest does not match this frozen suite.")
        for row, sample in zip(existing["entries"], samples):
            path = cache_root / row["relative_path"]
            if file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"Existing cache file hash mismatch: {path}")
            _validate_cache_entry(path, sample, gate_name=gate_name)
        print(f"Using verified immutable shared RGB cache: {manifest_path}", flush=True)
        return existing

    episodes = {episode.episode_id: episode for episode in dataset.official_episodes}
    ordered = sorted(samples, key=lambda row: (row["scene_id"], row["split"], row["sample_index"]))
    environment = HabitatSimR2REnvironment(
        HabitatSimR2RConfig(scene_root=args.scene_root.expanduser().resolve())
    )
    entries_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    try:
        for ordinal, sample in enumerate(ordered, start=1):
            filename = _cache_filename(sample)
            path = cache_root / filename
            if path.exists():
                payload = _validate_cache_entry(path, sample, gate_name=gate_name)
            else:
                payload = _render_cache_entry(
                    environment,
                    episodes[sample["episode_id"]],
                    sample,
                    gate_name=gate_name,
                )
                temporary = path.with_suffix(path.suffix + ".tmp")
                torch.save(payload, temporary)
                os.replace(temporary, path)
            entries_by_key[(sample["split"], int(sample["sample_index"]))] = {
                "split": sample["split"],
                "sample_index": int(sample["sample_index"]),
                "sample_sha256": payload["sample_sha256"],
                "rgb_frames_sha256": payload["rgb_frames_sha256"],
                "relative_path": filename,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            if ordinal % 25 == 0 or ordinal == len(ordered):
                _atomic_json(
                    cache_root / "cache_progress.json",
                    {
                        "status": "running",
                        "manifest_sha256": manifest_hash,
                        "completed": ordinal,
                        "target": len(ordered),
                        "recorded_at_utc": _utc_now(),
                    },
                )
                print(f"[cache {ordinal}/{len(ordered)}] {sample['scene_id']}", flush=True)
    finally:
        environment.close()

    entries = [entries_by_key[(row["split"], int(row["sample_index"]))] for row in samples]
    cache_manifest = {
        "schema_version": 1,
        "gate": gate_name,
        "status": "passed",
        "manifest_sha256": manifest_hash,
        "entry_count": len(entries),
        "entries": entries,
        "entry_index_sha256": _canonical_sha256(entries),
        "licensed_derived_rgb_cache": True,
        "contains_depth": False,
        "contains_simulator_pose": False,
        "training_order_changed_by_cache": False,
    }
    _atomic_json(manifest_path, cache_manifest)
    return cache_manifest


def _cache_index(
    cache_manifest: dict[str, Any], cache_root: Path
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (row["split"], int(row["sample_index"])): {
            **row,
            "path": cache_root / row["relative_path"],
        }
        for row in cache_manifest["entries"]
    }


def _load_cached(
    entry: dict[str, Any],
    sample: dict[str, Any],
    *,
    gate_name: str = NATIVE_HISTORY_GATE_NAME,
) -> dict[str, Any]:
    path = entry["path"]
    if file_sha256(path) != entry["sha256"]:
        raise RuntimeError(f"Cache entry changed after indexing: {path}")
    return _validate_cache_entry(path, sample, gate_name=gate_name)


def _native_batch(
    *,
    processor: Any,
    codebook: ActionTokenCodebook,
    cached: dict[str, Any],
    sample: dict[str, Any],
    arm_id: str,
) -> tuple[NativeQwenBatch, int]:
    frames = [Image.fromarray(frame.numpy()) for frame in cached["rgb_frames"]]
    actions = [R2RCEAction(value) for value in cached["history_action_ids"]]
    mode = ARM_TO_MODE[arm_id]
    window = build_history_window(frames, actions, contract=HISTORY_CONTRACTS[mode])
    if arm_id == "H4_CAUSAL" and any(frame.is_padding for frame in window.frames):
        raise RuntimeError("H4 cache unexpectedly required synthetic padding.")
    target = R2RCEAction(int(sample["target_action_id"]))
    batch = build_native_qwen_sft_batch(
        processor,
        window,
        instruction=sample["instruction_text"],
        target_action=target,
    )
    rendered = {
        "input_ids": batch.model_inputs["input_ids"][0],
        "labels": batch.model_inputs["labels"][0],
        "target_start_idx": int(batch.record["target_start_idx"]),
    }
    action_position = locate_action_token(rendered, action=target, codebook=codebook)
    labels = batch.model_inputs["labels"]
    if int(batch.record.get("supervised_token_count", -1)) != 1:
        raise RuntimeError("Native history batch record must declare one supervised token.")
    if int(batch.record.get("supervised_token_position", -1)) != action_position:
        raise RuntimeError("Native history batch record and action-token locator disagree.")
    if int((labels != -100).sum().item()) != 1:
        raise RuntimeError("Native history objective must supervise exactly one action token.")
    if int(labels[0, action_position].item()) != int(
        batch.model_inputs["input_ids"][0, action_position].item()
    ):
        raise RuntimeError("Native history action-token label does not match input_ids.")
    return batch, action_position


def _move_batch(batch: NativeQwenBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: tensor.to(device=device, non_blocking=True)
        for key, tensor in batch.model_inputs.items()
    }


def _load_model(device: torch.device, seed: int) -> Qwen3VLBackbone:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    base = model.model
    if base is None:
        raise RuntimeError("Exact-revision Qwen model did not load.")
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    if hasattr(base, "config"):
        base.config.use_cache = False
    validate_lora_only_trainables(model)
    return model


def _calibrate(
    *,
    model: Qwen3VLBackbone,
    processor: Any,
    codebook: ActionTokenCodebook,
    samples: list[dict[str, Any]],
    cache: dict[tuple[str, int], dict[str, Any]],
    arm_id: str,
    step: int,
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
                arm_id=arm_id,
            )
            inputs = _move_batch(batch, device)
            outputs = model(**inputs)
            scores = score_action_token_logits(
                outputs.logits,
                action_token_position=position,
                codebook=codebook,
            )
            target = sample["target_action"]
            nll = float(scores["scores"][target]["token_nll"])
            if not math.isfinite(nll) or nll > 100.0:
                raise RuntimeError("Calibration produced an invalid target-token NLL.")
            rows.append(
                {
                    "sample_index": int(sample["sample_index"]),
                    "episode_id": sample["episode_id"],
                    "trajectory_id": sample["trajectory_id"],
                    "target_action": target,
                    "target_token_nll": nll,
                    "predicted_action": scores["predicted_action"],
                    "correct": scores["predicted_action"] == target,
                    "cache_sha256": entry["sha256"],
                    "action_token_position": position,
                    "supervision_schema_version": batch.record[
                        "supervision_schema_version"
                    ],
                    "supervision_method": batch.record["supervision_method"],
                    "supervised_token_count": batch.record["supervised_token_count"],
                    "supervised_token_id": batch.record["supervised_token_id"],
                }
            )
            del outputs, inputs, batch
    model.train()
    return {
        "step": step,
        "decision_count": len(rows),
        "mean_target_token_nll": sum(row["target_token_nll"] for row in rows) / len(rows),
        "descriptive_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "per_decision": rows,
        "accuracy_used_for_selection": False,
    }


def _checkpoint_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("checkpoints")
    if not isinstance(rows, list):
        raise RuntimeError("Checkpoint index is malformed.")
    return rows


def _write_checkpoint_index(
    path: Path, rows: list[dict[str, Any]], *, manifest_hash: str, arm_id: str
) -> None:
    _atomic_json(
        path,
        {
            "gate": NATIVE_HISTORY_GATE_NAME,
            "manifest_sha256": manifest_hash,
            "arm_id": arm_id,
            "checkpoints": rows,
        },
    )


def _run_arm(
    *,
    args: argparse.Namespace,
    protocol: NativeHistoryProtocol,
    manifest_hash: str,
    cache_manifest_hash: str,
    samples: list[dict[str, Any]],
    cache: dict[tuple[str, int], dict[str, Any]],
    processor: Any,
    codebook: ActionTokenCodebook,
    arm_id: str,
    device: torch.device,
    suite_started: float,
    expected_initial_sha: str | None,
) -> tuple[dict[str, Any], str]:
    output_root = args.output_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve() / arm_id
    arm_root = output_root / "arms" / arm_id
    result_path = arm_root / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "passed":
            raise RuntimeError(f"Existing arm result is not passed: {result_path}")
        return result, str(result["initial_trainable_state_sha256"])

    arm_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model = _load_model(device, protocol.seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    initial_sha = tensor_state_sha256(trainable_state_dict(model))
    if expected_initial_sha is not None and initial_sha != expected_initial_sha:
        raise RuntimeError("History arms did not start from identical LoRA initialization.")
    metadata = {
        "manifest_sha256": manifest_hash,
        "cache_manifest_sha256": cache_manifest_hash,
        "arm_id": arm_id,
        "initial_trainable_state_sha256": initial_sha,
    }
    index_path = arm_root / "checkpoint_index.json"
    latest_path = arm_root / "latest.json"
    checkpoint_rows = _checkpoint_index(index_path)
    train_samples = [row for row in samples if row["split"] == "train"]
    calibration_samples = [row for row in samples if row["split"] == "calibration"]
    state: dict[str, Any]
    if args.resume and latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        checkpoint = checkpoint_root / latest["relative_path"]
        if file_sha256(checkpoint) != latest["sha256"]:
            raise RuntimeError("Latest resume checkpoint is missing or changed.")
        payload = load_history_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_metadata=metadata,
        )
        state = dict(payload["cursor"])
        if not checkpoint_rows or checkpoint_rows[-1]["sha256"] != latest["sha256"]:
            raise RuntimeError("Latest pointer and checkpoint index disagree.")
    else:
        if latest_path.exists() or checkpoint_rows:
            raise RuntimeError("Existing arm checkpoint state requires --resume.")
        baseline = _calibrate(
            model=model,
            processor=processor,
            codebook=codebook,
            samples=calibration_samples,
            cache=cache,
            arm_id=arm_id,
            step=0,
            device=device,
        )
        state = {
            "arm_id": arm_id,
            "update_step": 0,
            "next_train_index": 0,
            "calibrations": [baseline],
            "audit_updates": [],
            "started_at_utc": _utc_now(),
            "accumulated_training_seconds": 0.0,
        }

    events_path = arm_root / "updates.jsonl"
    arm_started = time.perf_counter()
    terminal_reason = "planned_updates_completed"
    try:
        for step in range(int(state["update_step"]) + 1, protocol.planned_updates + 1):
            if time.perf_counter() - suite_started >= protocol.max_hours * 3600:
                terminal_reason = "partial_time_budget"
                break
            sample = train_samples[(step - 1) % len(train_samples)]
            entry = cache[("train", int(sample["sample_index"]))]
            cached = _load_cached(entry, sample)
            batch, action_position = _native_batch(
                processor=processor,
                codebook=codebook,
                cached=cached,
                sample=sample,
                arm_id=arm_id,
            )
            inputs = _move_batch(batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            audit = step == 1 or step in protocol.checkpoint_steps
            before = trainable_state_dict(model) if audit else None
            before_sha = tensor_state_sha256(before) if before is not None else None
            update_started = time.perf_counter()
            outputs = model(**inputs)
            loss = outputs.loss
            loss_value = float(loss.detach().float().cpu().item())
            if (
                not math.isfinite(loss_value)
                or loss_value <= 0.0
                or loss_value > protocol.hard_max_token_nll
            ):
                labels = inputs["labels"]
                target_token_id = int(labels[0, action_position].detach().cpu().item())
                if not math.isfinite(loss_value):
                    violation = "nonfinite"
                elif loss_value <= 0.0:
                    violation = "nonpositive"
                else:
                    violation = "above_hard_maximum"
                diagnostic = {
                    "schema_version": 1,
                    "gate": "native_history_invalid_training_nll_diagnostic_v1",
                    "status": "failed",
                    "recorded_at_utc": _utc_now(),
                    "arm_id": arm_id,
                    "step": step,
                    "violation": violation,
                    "observed_training_nll": diagnostic_float(loss_value),
                    "required_training_nll": {
                        "finite": True,
                        "exclusive_minimum": 0.0,
                        "inclusive_maximum": protocol.hard_max_token_nll,
                    },
                    "sample": {
                        "sample_index": int(sample["sample_index"]),
                        "episode_id": sample["episode_id"],
                        "trajectory_id": sample["trajectory_id"],
                        "target_action": sample["target_action"],
                        "target_action_id": int(sample["target_action_id"]),
                        "cache_sha256": entry["sha256"],
                    },
                    "causal_logits": causal_logit_diagnostics(
                        outputs.logits,
                        action_position=action_position,
                        target_token_id=target_token_id,
                    ),
                    "trainable_parameters": tensor_collection_diagnostics(
                        [
                            (name, parameter)
                            for name, parameter in model.named_parameters()
                            if parameter.requires_grad
                        ]
                    ),
                    "optimizer_state": optimizer_tensor_diagnostics(optimizer),
                    "accepted_prior_update_count": int(state["update_step"]),
                    "events_path": str(events_path),
                }
                diagnostic_path = arm_root / f"invalid_training_nll_step_{step:04d}.json"
                _atomic_json(diagnostic_path, diagnostic)
                raise RuntimeError(
                    "Invalid one-token training NLL at "
                    f"{arm_id} step {step}: observed={loss_value!r}, "
                    f"violation={violation}, diagnostic={diagnostic_path}."
                )
            loss.backward()
            gradients = gradient_report(model)
            if (
                gradients.get("trainable_tensor_count", 0) <= 0
                or gradients.get("gradient_tensor_count") != gradients.get("trainable_tensor_count")
                or gradients.get("nonzero_gradient_tensor_count", 0) <= 0
                or gradients.get("all_gradients_finite") is not True
                or not math.isfinite(float(gradients.get("gradient_l2_norm", float("nan"))))
            ):
                raise RuntimeError(f"Invalid LoRA gradient coverage at {arm_id} step {step}.")
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
                raise RuntimeError(f"Invalid global gradient norm at {arm_id} step {step}.")
            optimizer.step()
            audit_update = None
            if before is not None:
                audit_update = update_report(model, before)
                validate_optimization_reports(gradients, audit_update)
                after_sha = tensor_state_sha256(trainable_state_dict(model))
                if after_sha == before_sha:
                    raise RuntimeError(f"No trainable update at {arm_id} step {step}.")
                state["audit_updates"].append(
                    {
                        "step": step,
                        "before_sha256": before_sha,
                        "after_sha256": after_sha,
                        "updates": audit_update,
                    }
                )
            elapsed = time.perf_counter() - update_started
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            if peak_allocated > protocol.hard_max_peak_allocated_bytes:
                raise RuntimeError(f"Hard memory limit exceeded at {arm_id} step {step}.")
            state["update_step"] = step
            state["next_train_index"] = step
            state["accumulated_training_seconds"] = (
                float(state["accumulated_training_seconds"]) + elapsed
            )
            _append_jsonl(
                events_path,
                {
                    "step": step,
                    "sample_index": int(sample["sample_index"]),
                    "episode_id": sample["episode_id"],
                    "trajectory_id": sample["trajectory_id"],
                    "target_action": sample["target_action"],
                    "action_token_position": action_position,
                    "supervision_schema_version": batch.record[
                        "supervision_schema_version"
                    ],
                    "supervision_method": batch.record["supervision_method"],
                    "supervised_token_count": batch.record["supervised_token_count"],
                    "supervised_token_id": batch.record["supervised_token_id"],
                    "target_token_nll": loss_value,
                    "gradient_report": gradients,
                    "global_gradient_norm_before_clip": preclip,
                    "clip_max_norm": protocol.max_grad_norm,
                    "cache_sha256": entry["sha256"],
                    "elapsed_seconds": elapsed,
                    "peak_allocated_bytes": peak_allocated,
                    "audited_update": audit_update is not None,
                },
            )
            del outputs, loss, inputs, batch, before

            if step in protocol.calibration_steps:
                state["calibrations"].append(
                    _calibrate(
                        model=model,
                        processor=processor,
                        codebook=codebook,
                        samples=calibration_samples,
                        cache=cache,
                        arm_id=arm_id,
                        step=step,
                        device=device,
                    )
                )

            must_checkpoint = step in protocol.checkpoint_steps or _STOP_REQUESTED
            if must_checkpoint:
                suffix = (
                    "_signal" if _STOP_REQUESTED and step not in protocol.checkpoint_steps else ""
                )
                path = checkpoint_root / f"step_{step:04d}{suffix}.pt"
                info = save_history_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    metadata=metadata,
                    cursor=state,
                )
                info["relative_path"] = path.name
                checkpoint_rows.append(info)
                _write_checkpoint_index(
                    index_path,
                    checkpoint_rows,
                    manifest_hash=manifest_hash,
                    arm_id=arm_id,
                )
                _atomic_json(latest_path, info)
                _atomic_json(
                    output_root / "progress.json",
                    {
                        "status": "running",
                        "current_arm": arm_id,
                        "completed_update": step,
                        "planned_updates_per_arm": protocol.planned_updates,
                        "completed_arms": [
                            arm
                            for arm in HISTORY_ARMS
                            if (output_root / "arms" / arm / "result.json").exists()
                        ],
                        "latest_checkpoint": info,
                        "elapsed_suite_seconds": time.perf_counter() - suite_started,
                        "recorded_at_utc": _utc_now(),
                    },
                )
            if _STOP_REQUESTED:
                terminal_reason = "termination_signal_checkpointed"
                break
    except Exception:
        raise

    completed = int(state["update_step"]) == protocol.planned_updates
    if not completed:
        step = int(state["update_step"])
        if step > 0 and (not checkpoint_rows or checkpoint_rows[-1]["update_step"] != step):
            path = checkpoint_root / f"step_{step:04d}_partial.pt"
            info = save_history_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                metadata=metadata,
                cursor=state,
            )
            info["relative_path"] = path.name
            checkpoint_rows.append(info)
            _write_checkpoint_index(
                index_path, checkpoint_rows, manifest_hash=manifest_hash, arm_id=arm_id
            )
            _atomic_json(latest_path, info)
        partial = {
            "schema_version": NATIVE_HISTORY_SCHEMA_VERSION,
            "gate": NATIVE_HISTORY_GATE_NAME,
            "arm_id": arm_id,
            "status": "partial_time_budget"
            if terminal_reason == "partial_time_budget"
            else "partial_signal",
            "terminal_reason": terminal_reason,
            "initial_trainable_state_sha256": initial_sha,
            "completed_updates": int(state["update_step"]),
            "calibrations": state["calibrations"],
            "checkpoints": checkpoint_rows,
            "resume_required": True,
        }
        _atomic_json(arm_root / "partial.json", partial)
        return partial, initial_sha

    selected_calibration = dict(select_arm_checkpoint(state["calibrations"]))
    selected_step = int(selected_calibration["step"])
    if selected_step == 0:
        selected_checkpoint = {
            "source": "public_pretrained_seeded_initialization",
            "update_step": 0,
            "trainable_state_sha256": initial_sha,
            "path": None,
            "relative_path": None,
            "bytes": 0,
            "sha256": None,
        }
    else:
        selected_checkpoint = next(
            row for row in checkpoint_rows if int(row["update_step"]) == selected_step
        )
    result = {
        "schema_version": NATIVE_HISTORY_SCHEMA_VERSION,
        "gate": NATIVE_HISTORY_GATE_NAME,
        "arm_id": arm_id,
        "status": "passed",
        "started_at_utc": state["started_at_utc"],
        "completed_at_utc": _utc_now(),
        "terminal_reason": terminal_reason,
        "manifest_sha256": manifest_hash,
        "cache_manifest_sha256": cache_manifest_hash,
        "initial_trainable_state_sha256": initial_sha,
        "completed_updates": int(state["update_step"]),
        "calibrations": state["calibrations"],
        "audit_updates": state["audit_updates"],
        "checkpoints": checkpoint_rows,
        "selected_calibration": selected_calibration,
        "selected_checkpoint": selected_checkpoint,
        "training_seconds": state["accumulated_training_seconds"],
        "arm_invocation_seconds": time.perf_counter() - arm_started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    _atomic_json(result_path, result)
    return result, initial_sha


def _run_suite(args: argparse.Namespace) -> int:
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
        raise RuntimeError("Native history selection requires a BF16-capable CUDA device.")
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

    arm_results: dict[str, dict[str, Any]] = {}
    expected_initial_sha: str | None = None
    for arm_id in HISTORY_ARMS:
        result, initial_sha = _run_arm(
            args=args,
            protocol=protocol,
            manifest_hash=manifest_hash,
            cache_manifest_hash=cache_manifest_hash,
            samples=samples,
            cache=cache,
            processor=processor,
            codebook=codebook,
            arm_id=arm_id,
            device=device,
            suite_started=suite_started,
            expected_initial_sha=expected_initial_sha,
        )
        if expected_initial_sha is None:
            expected_initial_sha = initial_sha
        arm_results[arm_id] = result
        if result.get("status") != "passed":
            _atomic_json(
                args.output_root.expanduser().resolve() / "progress.json",
                {
                    "status": result["status"],
                    "current_arm": arm_id,
                    "completed_arms": [
                        arm for arm, row in arm_results.items() if row.get("status") == "passed"
                    ],
                    "elapsed_suite_seconds": time.perf_counter() - suite_started,
                    "resume_required": True,
                    "recorded_at_utc": _utc_now(),
                },
            )
            print(
                f"Suite stopped safely with {result['status']} in {arm_id}; rerun with --resume.",
                flush=True,
            )
            return 2
        gc.collect()
        torch.cuda.empty_cache()

    per_arm_nll = {
        arm_id: [
            float(row["target_token_nll"])
            for row in arm_results[arm_id]["selected_calibration"]["per_decision"]
        ]
        for arm_id in HISTORY_ARMS
    }
    history_selection = select_history_arm(per_arm_nll, protocol=protocol)
    suite = {
        "schema_version": NATIVE_HISTORY_SCHEMA_VERSION,
        "gate": NATIVE_HISTORY_GATE_NAME,
        "run_id": args.run_id,
        "status": "passed",
        "completed_at_utc": _utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "cache_manifest_path": str(cache_manifest_path),
        "cache_manifest_sha256": cache_manifest_hash,
        "protocol": protocol.to_record(),
        "arm_results": {
            arm_id: {
                "relative_path": f"arms/{arm_id}/result.json",
                "sha256": file_sha256(
                    args.output_root.expanduser().resolve() / "arms" / arm_id / "result.json"
                ),
                "selected_step": arm_results[arm_id]["selected_calibration"]["step"],
                "selected_mean_target_token_nll": arm_results[arm_id]["selected_calibration"][
                    "mean_target_token_nll"
                ],
                "selected_checkpoint": arm_results[arm_id]["selected_checkpoint"],
            }
            for arm_id in HISTORY_ARMS
        },
        "history_selection": history_selection,
        "identical_initial_trainable_state_sha256": expected_initial_sha,
        "action_codebook": codebook.to_record(),
        "supervision": {
            "schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
            "method": NATIVE_QWEN_SUPERVISION_METHOD,
            "supervised_token_count_per_decision": 1,
        },
        "model_weight_locks": weight_locks,
        "source_locks": manifest["source_locks"],
        "runtime": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "compiled_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "elapsed_suite_seconds": time.perf_counter() - suite_started,
            "max_hours": protocol.max_hours,
            "docker_image_id": EXPECTED_DOCKER_IMAGE_ID,
        },
        "evidence_boundary": manifest["evidence_boundary"],
    }
    suite_path = args.output_root.expanduser().resolve() / "suite.json"
    _atomic_json(suite_path, suite)
    _atomic_json(
        args.output_root.expanduser().resolve() / "progress.json",
        {
            "status": "passed",
            "completed_arms": list(HISTORY_ARMS),
            "selected_arm": history_selection["selected_arm"],
            "elapsed_suite_seconds": time.perf_counter() - suite_started,
            "recorded_at_utc": _utc_now(),
        },
    )
    print(json.dumps(suite, indent=2, sort_keys=True))
    print(f"Native history selection suite PASSED: {suite_path}")
    return 0


def _common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--planned-updates", type=int, default=2_048)
    parser.add_argument("--train-samples-per-action", type=int, default=512)
    parser.add_argument("--calibration-samples-per-action", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=128)
    parser.add_argument("--max-hours", type=float, default=10.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Freeze the deterministic train manifest")
    _common_inputs(prepare)
    prepare.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="Build/reuse cache and run the three matched arms")
    _common_inputs(run)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--scene-root", type=Path, required=True)
    run.add_argument("--model-snapshot", type=Path, required=True)
    run.add_argument("--cache-root", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        if args.command == "prepare":
            return _prepare_manifest(args)
        return _run_suite(args)
    except Exception as exc:
        failure = {
            "schema_version": NATIVE_HISTORY_SCHEMA_VERSION,
            "gate": NATIVE_HISTORY_GATE_NAME,
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
