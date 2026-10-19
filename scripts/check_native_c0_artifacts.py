#!/usr/bin/env python3
"""Independently reconstruct and verify a native-Qwen C0 real-batch gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoProcessor

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
)
from spatialvlm.training.native_c0_artifacts import (
    NATIVE_C0_VERIFIER_GATE_NAME,
    NATIVE_C0_VERIFIER_SCHEMA_VERSION,
    verify_native_c0_result,
)
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    array_sha256,
    file_sha256,
    select_native_c0_probe,
    tensor_sha256,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _candidate_record(batch: NativeQwenBatch) -> dict[str, Any]:
    return {
        "input_ids": batch.model_inputs["input_ids"][0],
        "labels": batch.model_inputs["labels"][0],
        "target_start_idx": int(batch.record["target_start_idx"]),
    }


def _reconstruct_expected_probe(args: argparse.Namespace) -> dict[str, Any]:
    dataset = R2RCEDataset.from_release(args.episodes, args.ground_truth, split="train")
    episode, target_action = select_native_c0_probe(dataset)
    environment = HabitatSimR2REnvironment(HabitatSimR2RConfig(scene_root=args.scene_root))
    try:
        state = environment.reset(episode)
        rgb = np.asarray(state.observation["rgb"], dtype=np.uint8).copy()
        habitat_runtime = environment.runtime_metadata()
    finally:
        environment.close()

    window = build_history_window(
        [Image.fromarray(rgb)],
        [],
        contract=HISTORY_CONTRACTS[HistoryMode.H0_CURRENT],
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
    labels = target_batch.model_inputs["labels"]
    target_start = int(target_batch.record["target_start_idx"])
    native_batch = {
        **target_batch.record,
        "system_prompt": NATIVE_QWEN_SYSTEM_PROMPT,
        "shared_action_token_position": action_position,
        "model_input_shapes": {
            key: list(value.shape) for key, value in target_batch.model_inputs.items()
        },
        "model_input_sha256": {
            key: tensor_sha256(value) for key, value in target_batch.model_inputs.items()
        },
        "masked_prefix_token_count": int((labels[:, :target_start] == -100).sum().item()),
        "supervised_label_token_count": int((labels != -100).sum().item()),
    }
    return {
        "episode_id": episode.episode_id,
        "trajectory_id": episode.trajectory_id,
        "scene_id": episode.scene_id,
        "instruction_sha256": hashlib.sha256(episode.instruction_text.encode("utf-8")).hexdigest(),
        "rgb_sha256": array_sha256(rgb),
        "target_action_id": int(target_action),
        "target_action": target_action.name,
        "release_manifest": asdict(dataset.release_manifest)
        if dataset.release_manifest is not None
        else None,
        "habitat_runtime": habitat_runtime,
        "action_codebook": {
            **codebook.to_record(),
            "system_prompt": NATIVE_QWEN_SYSTEM_PROMPT,
        },
        "native_batch": native_batch,
    }


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
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError("Native-C0 result must be a JSON object.")

    reconstruction_errors: list[str] = []
    try:
        expected_probe = _reconstruct_expected_probe(args)
    except Exception as exc:
        expected_probe = {}
        reconstruction_errors.append(
            f"independent native-C0 probe reconstruction failed: {type(exc).__name__}: {exc}"
        )
    errors = [
        *reconstruction_errors,
        *verify_native_c0_result(
            result,
            repo=args.repo,
            checkpoint=args.checkpoint,
            model_snapshot=args.model_snapshot,
            processor_result=args.processor_result,
            processor_verification=args.processor_verification,
            processor_reverification=args.processor_reverification,
            expected_probe=expected_probe,
        ),
    ]
    payload = {
        "schema_version": NATIVE_C0_VERIFIER_SCHEMA_VERSION,
        "gate": NATIVE_C0_VERIFIER_GATE_NAME,
        "status": "passed" if not errors else "failed",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "error_count": len(errors),
        "errors": errors,
        "dataset_reconstruction_performed": not reconstruction_errors,
        "result_path": str(args.result.resolve()),
        "result_sha256": file_sha256(args.result),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint.is_file() else None,
        "processor_result_sha256": file_sha256(args.processor_result),
        "processor_verification_sha256": file_sha256(args.processor_verification),
        "processor_reverification_sha256": file_sha256(args.processor_reverification),
        "model_snapshot": str(args.model_snapshot.resolve()),
        "repo": str(args.repo.resolve()),
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
