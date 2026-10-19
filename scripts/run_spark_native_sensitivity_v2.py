#!/usr/bin/env python3
"""Run the disjoint saturation-robust native-Qwen visual-sensitivity confirmation."""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from scripts.run_spark_native_sensitivity import (
    EXPECTED_DOCKER_IMAGE_ID,
    _atomic_json,
    _canonical_sha256,
    _jsonable,
    _score_condition,
    _utc_now,
)
from spatialvlm.backbone.qwen3_vl import Qwen3VLBackbone
from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.data.habitat_sim_r2r import HabitatSimR2RConfig, HabitatSimR2REnvironment
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
)
from spatialvlm.data.native_qwen_artifacts import (
    NATIVE_QWEN_HISTORY_PREFLIGHT_GATE_NAME,
    NATIVE_QWEN_HISTORY_VERIFIER_GATE_NAME,
)
from spatialvlm.eval.action_decision import build_action_token_codebook
from spatialvlm.eval.native_sensitivity import NATIVE_SENSITIVITY_GATE_NAME
from spatialvlm.eval.native_sensitivity_v2 import (
    NATIVE_SENSITIVITY_V2_GATE_NAME,
    NATIVE_SENSITIVITY_V2_SCHEMA_VERSION,
    NATIVE_SENSITIVITY_V2_SOURCE_LOCK_PATHS,
    NativeSensitivityV2Protocol,
    select_native_sensitivity_v2_samples,
    summarize_native_sensitivity_v2,
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

V1_VERIFIER_GATE = "independent_native_qwen_c0_visual_sensitivity_verification_v1"
V1_REQUIRED_VERDICT = "inconclusive_expand_before_optimizer_diagnostic"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--processor-result", type=Path, required=True)
    parser.add_argument("--processor-verification", type=Path, required=True)
    parser.add_argument("--v1-manifest", type=Path, required=True)
    parser.add_argument("--v1-result", type=Path, required=True)
    parser.add_argument("--v1-verification", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _source_locks(repo: Path) -> dict[str, str]:
    locks: dict[str, str] = {}
    for relative in NATIVE_SENSITIVITY_V2_SOURCE_LOCK_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Sensitivity-v2 source-lock file is missing: {path}")
        locks[relative] = file_sha256(path)
    return locks


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Required {label} must be a JSON object.")
    return payload


def v1_prerequisite_record(
    manifest_path: Path, result_path: Path, verification_path: Path
) -> tuple[dict[str, Any], set[str]]:
    """Validate and hash-link the immutable discovery artifact used for exclusions."""

    manifest = _load_json(manifest_path, "v1 sensitivity manifest")
    result = _load_json(result_path, "v1 sensitivity result")
    verification = _load_json(verification_path, "v1 sensitivity verification")
    if (
        manifest.get("gate") != NATIVE_SENSITIVITY_GATE_NAME
        or manifest.get("status") != "frozen"
    ):
        raise RuntimeError("V1 sensitivity manifest gate/status mismatch.")
    if (
        result.get("gate") != NATIVE_SENSITIVITY_GATE_NAME
        or result.get("status") != "passed"
        or result.get("summary", {}).get("verdict") != V1_REQUIRED_VERDICT
    ):
        raise RuntimeError("V1 sensitivity discovery result is not the preserved inconclusive run.")
    if result.get("manifest_sha256") != file_sha256(manifest_path):
        raise RuntimeError("V1 sensitivity result is not linked to its manifest.")
    if (
        verification.get("gate") != V1_VERIFIER_GATE
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("manifest_sha256") != file_sha256(manifest_path)
        or verification.get("result_sha256") != file_sha256(result_path)
    ):
        raise RuntimeError("V1 sensitivity artifact lacks a linked passing verification.")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("V1 sensitivity manifest has no frozen sample list.")
    excluded = {str(sample["trajectory_id"]) for sample in samples}
    if len(excluded) != len(samples):
        raise RuntimeError("V1 sensitivity sample trajectories are not unique.")
    return (
        {
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "result_path": str(result_path),
            "result_sha256": file_sha256(result_path),
            "verification_path": str(verification_path),
            "verification_sha256": file_sha256(verification_path),
            "verdict": V1_REQUIRED_VERDICT,
            "excluded_trajectory_count": len(excluded),
            "excluded_trajectory_ids": sorted(excluded),
        },
        excluded,
    )


def processor_prerequisite_record(
    result_path: Path, verification_path: Path
) -> dict[str, Any]:
    """Validate and bind the corrected one-token processor preflight."""

    result = _load_json(result_path, "processor result")
    verification = _load_json(verification_path, "processor verification")
    if (
        result.get("gate") != NATIVE_QWEN_HISTORY_PREFLIGHT_GATE_NAME
        or result.get("status") != "passed"
    ):
        raise RuntimeError("Corrected one-token processor result has not passed.")
    if (
        verification.get("gate") != NATIVE_QWEN_HISTORY_VERIFIER_GATE_NAME
        or verification.get("status") != "passed"
        or verification.get("error_count") != 0
        or verification.get("result_sha256") != file_sha256(result_path)
    ):
        raise RuntimeError("Corrected processor result lacks linked passing verification.")
    return {
        "result_path": str(result_path),
        "result_sha256": file_sha256(result_path),
        "verification_path": str(verification_path),
        "verification_sha256": file_sha256(verification_path),
    }


def main() -> int:
    args = _parse_args()
    root = args.output_root.expanduser().resolve()
    result_path = root / "result.json"
    progress_path = root / "progress.json"
    failure_path = root / "failure.json"
    if result_path.exists() or failure_path.exists():
        raise SystemExit(
            "Sensitivity-v2 output is already terminal; preserve it and use a new run ID."
        )
    started = time.perf_counter()
    failure: dict[str, Any] = {
        "schema_version": NATIVE_SENSITIVITY_V2_SCHEMA_VERSION,
        "gate": NATIVE_SENSITIVITY_V2_GATE_NAME,
        "run_id": args.run_id,
        "status": "running",
        "recorded_at_utc": _utc_now(),
    }
    try:
        if args.docker_image_id != EXPECTED_DOCKER_IMAGE_ID:
            raise ValueError("Sensitivity-v2 Docker image identity mismatch.")
        repo = args.repo.expanduser().resolve()
        protocol = NativeSensitivityV2Protocol()
        processor_record = processor_prerequisite_record(
            args.processor_result.expanduser().resolve(),
            args.processor_verification.expanduser().resolve(),
        )
        v1_record, excluded = v1_prerequisite_record(
            args.v1_manifest.expanduser().resolve(),
            args.v1_result.expanduser().resolve(),
            args.v1_verification.expanduser().resolve(),
        )
        dataset = R2RCEDataset.from_release(
            args.episodes.expanduser().resolve(),
            args.ground_truth.expanduser().resolve(),
            split=protocol.split,
        )
        samples = select_native_sensitivity_v2_samples(
            dataset, excluded_trajectory_ids=excluded, protocol=protocol
        )
        sample_records = [sample.to_record() for sample in samples]
        source_locks = _source_locks(repo)
        manifest = {
            "schema_version": NATIVE_SENSITIVITY_V2_SCHEMA_VERSION,
            "gate": NATIVE_SENSITIVITY_V2_GATE_NAME,
            "run_id": args.run_id,
            "status": "frozen",
            "recorded_at_utc": _utc_now(),
            "protocol": protocol.to_record(),
            "processor_prerequisite": processor_record,
            "v1_discovery_prerequisite": v1_record,
            "dataset": {
                "split_used": protocol.split,
                "sealed_split": "val_unseen",
                "episodes_path": str(args.episodes),
                "episodes_sha256": file_sha256(args.episodes),
                "ground_truth_path": str(args.ground_truth),
                "ground_truth_sha256": file_sha256(args.ground_truth),
                "release_manifest": (
                    asdict(dataset.release_manifest)
                    if dataset.release_manifest is not None
                    else None
                ),
            },
            "samples": sample_records,
            "sample_records_sha256": _canonical_sha256(sample_records),
            "source_locks": source_locks,
            "evidence_boundary": {
                "established_if_passed": (
                    "Whether fresh native Qwen relative four-action logits respond consistently "
                    "to paired real-image swaps on a new v1-disjoint, action-balanced 64-decision "
                    "val_seen cohort while identical-input repeats remain deterministic."
                ),
                "not_established": (
                    "Semantic visual grounding, navigation competence, calibration, learned "
                    "behavior, closed-loop success, val_unseen performance, or a C0-C4 claim."
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
            raise RuntimeError("Sensitivity v2 requires CUDA.")
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
            clean_image = Image.fromarray(clean_rgb)
            condition_inputs = {
                "clean": (clean_image, sample.instruction_text),
                "clean_repeat": (Image.fromarray(clean_rgb), sample.instruction_text),
                "paired_image_swap": (Image.fromarray(donor_rgb), sample.instruction_text),
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
                    "forward_passes_completed": ordinal
                    * len(protocol.to_record()["conditions"]),
                    "forward_passes_total": protocol.forward_pass_count,
                    "recorded_at_utc": _utc_now(),
                },
            )

        summary = summarize_native_sensitivity_v2(decisions, protocol)
        result = {
            "schema_version": NATIVE_SENSITIVITY_V2_SCHEMA_VERSION,
            "gate": NATIVE_SENSITIVITY_V2_GATE_NAME,
            "run_id": args.run_id,
            "status": "passed",
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "protocol": protocol.to_record(),
            "processor_prerequisite": processor_record,
            "v1_discovery_prerequisite": v1_record,
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
