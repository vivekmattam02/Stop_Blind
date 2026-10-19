#!/usr/bin/env python3
"""Run the forward-only, action-balanced CR-025 native-Qwen sensitivity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
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
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
    build_native_qwen_sft_batch,
)
from spatialvlm.eval.action_decision import (
    build_action_token_codebook,
    locate_action_token,
    score_action_token_logits,
)
from spatialvlm.eval.native_sensitivity import (
    NATIVE_SENSITIVITY_GATE_NAME,
    NATIVE_SENSITIVITY_SCHEMA_VERSION,
    NATIVE_SENSITIVITY_SOURCE_LOCK_PATHS,
    NativeSensitivityProtocol,
    select_native_sensitivity_samples,
    summarize_native_sensitivity,
)
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    array_sha256,
    file_sha256,
    model_weight_locks,
    tensor_state_sha256,
    trainable_state_dict,
    validate_lora_only_trainables,
)

EXPECTED_DOCKER_IMAGE_ID = "sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf"
def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
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


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_locks(repo: Path) -> dict[str, str]:
    locks = {}
    for relative in NATIVE_SENSITIVITY_SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Sensitivity source-lock file is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _constant_image_like(rgb: np.ndarray) -> Image.Image:
    constant = np.full_like(rgb, 127)
    if constant.ndim == 3 and constant.shape[-1] == 4:
        constant[..., 3] = 255
    return Image.fromarray(constant)


def _score_condition(
    *,
    model: Qwen3VLBackbone,
    processor: Any,
    codebook: Any,
    device: torch.device,
    image: Image.Image,
    instruction: str,
    target_action: R2RCEAction,
) -> dict[str, Any]:
    window = build_history_window(
        [image], [], contract=HISTORY_CONTRACTS[HistoryMode.H0_CURRENT]
    )
    batch = build_native_qwen_sft_batch(
        processor,
        window,
        instruction=instruction,
        target_action=target_action,
    )
    labels = batch.model_inputs["labels"]
    if int((labels != -100).sum().item()) != 1:
        raise RuntimeError("Sensitivity batch did not contain exactly one supervised token.")
    rendered = {
        "input_ids": batch.model_inputs["input_ids"][0],
        "labels": labels[0],
        "target_start_idx": int(batch.record["target_start_idx"]),
    }
    action_position = locate_action_token(rendered, action=target_action, codebook=codebook)
    if action_position != batch.record["supervised_token_position"]:
        raise RuntimeError("Sensitivity action locator and batch supervision position disagree.")
    inputs = {
        key: tensor.to(device=device, non_blocking=True)
        for key, tensor in batch.model_inputs.items()
        if key != "labels"
    }
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits
    decision = score_action_token_logits(
        logits, action_token_position=action_position, codebook=codebook
    )
    row = logits[0, action_position - 1].detach().float()
    action_logits = {
        action.name: float(row[codebook.action_token_id(action)].cpu().item())
        for action in R2RCEAction
    }
    return {
        **decision,
        "action_logits": action_logits,
        "sequence_length": int(batch.record["sequence_length"]),
        "image_count": int(batch.record["image_count"]),
        "total_visual_tokens": int(batch.record["total_visual_tokens"]),
        "supervision_schema_version": batch.record["supervision_schema_version"],
        "supervision_method": batch.record["supervision_method"],
        "supervised_token_count": batch.record["supervised_token_count"],
        "supervised_token_position": batch.record["supervised_token_position"],
        "supervised_token_id": batch.record["supervised_token_id"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.output_root.expanduser().resolve()
    result_path = root / "result.json"
    progress_path = root / "progress.json"
    failure_path = root / "failure.json"
    if result_path.exists() or failure_path.exists():
        raise SystemExit(
            "Sensitivity output is already terminal; preserve it and use a new run ID."
        )
    started = time.perf_counter()
    failure: dict[str, Any] = {
        "schema_version": NATIVE_SENSITIVITY_SCHEMA_VERSION,
        "gate": NATIVE_SENSITIVITY_GATE_NAME,
        "run_id": args.run_id,
        "status": "running",
        "recorded_at_utc": _utc_now(),
    }
    try:
        if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID:
            raise ValueError("Sensitivity diagnostic Docker image identity mismatch.")
        repo = args.repo.expanduser().resolve()
        protocol = NativeSensitivityProtocol()
        dataset = R2RCEDataset.from_release(
            args.episodes.expanduser().resolve(),
            args.ground_truth.expanduser().resolve(),
            split=protocol.split,
        )
        samples = select_native_sensitivity_samples(dataset, protocol)
        sample_records = [sample.to_record() for sample in samples]
        source_locks = _source_locks(repo)
        manifest = {
            "schema_version": NATIVE_SENSITIVITY_SCHEMA_VERSION,
            "gate": NATIVE_SENSITIVITY_GATE_NAME,
            "run_id": args.run_id,
            "status": "frozen",
            "recorded_at_utc": _utc_now(),
            "protocol": protocol.to_record(),
            "dataset": {
                "split_used": protocol.split,
                "sealed_split": "val_unseen",
                "episodes_path": str(args.episodes),
                "episodes_sha256": file_sha256(args.episodes),
                "ground_truth_path": str(args.ground_truth),
                "ground_truth_sha256": file_sha256(args.ground_truth),
                "release_manifest": asdict(dataset.release_manifest)
                if dataset.release_manifest is not None
                else None,
            },
            "samples": sample_records,
            "sample_records_sha256": _canonical_sha256(sample_records),
            "source_locks": source_locks,
            "evidence_boundary": {
                "established_if_passed": (
                    "Whether fresh native Qwen four-action probabilities are measurably sensitive "
                    "to paired real-image swaps on 16 source-locked, action-balanced val_seen "
                    "H0 decisions under the preregistered three-region rule."
                ),
                "not_established": (
                    "Navigation competence, causal visual grounding, post-training behavior, "
                    "closed-loop success, val_unseen performance, or any C0-C4 paper claim."
                ),
            },
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)

        episode_index = {episode.episode_id: episode for episode in dataset.official_episodes}
        observations: dict[int, np.ndarray] = {}
        habitat = HabitatSimR2REnvironment(
            HabitatSimR2RConfig(scene_root=args.scene_root.expanduser().resolve())
        )
        try:
            for ordinal, sample in enumerate(samples, start=1):
                state = habitat.reset(episode_index[sample.episode_id])
                for action_id in sample.replay_prefix_action_ids:
                    state = habitat.step(R2RCEAction(action_id))
                observations[sample.sample_index] = np.asarray(
                    state.observation["rgb"], dtype=np.uint8
                ).copy()
                _atomic_json(
                    progress_path,
                    {
                        "status": "caching_observations",
                        "completed": ordinal,
                        "total": protocol.sample_count,
                        "recorded_at_utc": _utc_now(),
                    },
                )
            habitat_runtime = habitat.runtime_metadata()
        finally:
            habitat.close()

        random.seed(protocol.seed)
        np.random.seed(protocol.seed)
        torch.manual_seed(protocol.seed)
        torch.cuda.manual_seed_all(protocol.seed)
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Sensitivity diagnostic requires CUDA.")
        processor = AutoProcessor.from_pretrained(
            NATIVE_C0_MODEL_ID,
            revision=NATIVE_C0_MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=True,
        )
        codebook = build_action_token_codebook(processor.tokenizer)
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
        trainable_contract = validate_lora_only_trainables(model)
        initial_trainable_state_sha256 = tensor_state_sha256(trainable_state_dict(model))
        model.eval()
        decisions: list[dict[str, Any]] = []
        for ordinal, sample in enumerate(samples, start=1):
            donor = samples[sample.donor_sample_index]
            clean_rgb = observations[sample.sample_index]
            donor_rgb = observations[donor.sample_index]
            condition_inputs = {
                "clean": (Image.fromarray(clean_rgb), sample.instruction_text),
                "paired_image_swap": (Image.fromarray(donor_rgb), sample.instruction_text),
                "constant_image": (_constant_image_like(clean_rgb), sample.instruction_text),
                "instruction_swap": (Image.fromarray(clean_rgb), donor.instruction_text),
            }
            target = R2RCEAction(sample.target_action_id)
            conditions = {
                name: _score_condition(
                    model=model,
                    processor=processor,
                    codebook=codebook,
                    device=device,
                    image=image,
                    instruction=instruction,
                    target_action=target,
                )
                for name, (image, instruction) in condition_inputs.items()
            }
            decisions.append(
                {
                    "sample_index": sample.sample_index,
                    "episode_id": sample.episode_id,
                    "trajectory_id": sample.trajectory_id,
                    "scene_id": sample.scene_id,
                    "target_action_id": sample.target_action_id,
                    "target_action": sample.target_action,
                    "donor_sample_index": donor.sample_index,
                    "donor_episode_id": donor.episode_id,
                    "donor_target_action": donor.target_action,
                    "clean_rgb_sha256": array_sha256(clean_rgb),
                    "donor_rgb_sha256": array_sha256(donor_rgb),
                    "conditions": conditions,
                }
            )
            _atomic_json(
                progress_path,
                {
                    "status": "scoring_conditions",
                    "completed": ordinal,
                    "total": protocol.sample_count,
                    "forward_passes_completed": ordinal * 4,
                    "forward_passes_total": protocol.sample_count * 4,
                    "recorded_at_utc": _utc_now(),
                },
            )

        summary = summarize_native_sensitivity(decisions, protocol)
        result = {
            "schema_version": NATIVE_SENSITIVITY_SCHEMA_VERSION,
            "gate": NATIVE_SENSITIVITY_GATE_NAME,
            "run_id": args.run_id,
            "status": "passed",
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "protocol": protocol.to_record(),
            "source_locks": source_locks,
            "runtime": {
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "compiled_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "docker_image_ref": args.docker_image_ref,
                "docker_image_id": args.docker_image_id,
                "habitat": habitat_runtime,
            },
            "model": {
                "model_id": NATIVE_C0_MODEL_ID,
                "model_revision": NATIVE_C0_MODEL_REVISION,
                "initialization": "fresh_seeded_public_pretrained_lora",
                "optimization_updates": 0,
                "trainable_contract": trainable_contract,
                "initial_trainable_state_sha256": initial_trainable_state_sha256,
                "weight_locks": model_weight_locks(args.model_snapshot),
            },
            "supervision": {
                "schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
                "method": NATIVE_QWEN_SUPERVISION_METHOD,
                "supervised_token_count_per_decision": 1,
            },
            "action_codebook": codebook.to_record(),
            "decisions": decisions,
            "summary": summary,
        }
        _atomic_json(result_path, result)
        _atomic_json(
            progress_path,
            {
                "status": "passed",
                "completed": protocol.sample_count,
                "total": protocol.sample_count,
                "verdict": summary["verdict"],
                "recorded_at_utc": _utc_now(),
            },
        )
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure.update(
            {
                "status": "failed",
                "recorded_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _atomic_json(failure_path, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
