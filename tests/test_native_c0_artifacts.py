"""Adversarial tests for independent native-Qwen C0 artifact admission."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

import spatialvlm.training.native_c0_artifacts as artifacts_module
from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode
from spatialvlm.data.native_qwen import NATIVE_QWEN_RENDERER_VERSION, NATIVE_QWEN_SYSTEM_PROMPT
from spatialvlm.data.native_qwen_artifacts import (
    NATIVE_QWEN_HISTORY_PREFLIGHT_V3_GATE_NAME,
    NATIVE_QWEN_HISTORY_VERIFIER_V3_GATE_NAME,
)
from spatialvlm.eval.action_decision import (
    build_action_token_codebook,
    score_action_token_logits,
)
from spatialvlm.training.native_c0_artifacts import (
    NATIVE_C0_SOURCE_LOCK_PATHS,
    verify_native_c0_result,
)
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_CHECKPOINT_SCHEMA_VERSION,
    NATIVE_C0_GATE_NAME,
    NATIVE_C0_GATE_SCHEMA_VERSION,
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    file_sha256,
    model_weight_locks,
    tensor_state_sha256,
    trainable_name_sha256,
)


class _Tokenizer:
    name_or_path = "fixture/tokenizer"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [100, 101, 10 + int(text[-1])]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch) -> tuple[dict, dict, dict[str, Path]]:
    source_locks: dict[str, str] = {}
    for relative in NATIVE_C0_SOURCE_LOCK_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")
        source_locks[relative] = file_sha256(path)

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"frozen weights")
    _write_json(
        snapshot / "model.safetensors.index.json",
        {"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}},
    )

    processor_result = tmp_path / "processor_result.json"
    _write_json(
        processor_result,
        {
            "status": "passed",
            "gate": NATIVE_QWEN_HISTORY_PREFLIGHT_V3_GATE_NAME,
            "model_revision": NATIVE_C0_MODEL_REVISION,
        },
    )
    processor_result_sha = file_sha256(processor_result)
    processor_verification = tmp_path / "processor_verification.json"
    _write_json(
        processor_verification,
        {
            "status": "passed",
            "error_count": 0,
            "gate": NATIVE_QWEN_HISTORY_VERIFIER_V3_GATE_NAME,
            "result_sha256": processor_result_sha,
        },
    )
    processor_verification_sha = file_sha256(processor_verification)
    processor_reverification = tmp_path / "processor_reverification.json"
    _write_json(
        processor_reverification,
        {
            "status": "passed",
            "error_count": 0,
            "gate": NATIVE_QWEN_HISTORY_VERIFIER_V3_GATE_NAME,
            "result_sha256": processor_result_sha,
        },
    )
    monkeypatch.setattr(artifacts_module, "NATIVE_C0_PARENT_RESULT_SHA256", processor_result_sha)
    monkeypatch.setattr(
        artifacts_module,
        "NATIVE_C0_PARENT_VERIFICATION_SHA256",
        processor_verification_sha,
    )

    state = {"base_model.layer.lora_A.weight": torch.arange(6).reshape(2, 3).float()}
    state_sha = tensor_state_sha256(state)
    name_sha = trainable_name_sha256(list(state))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint_payload = {
        "schema_version": NATIVE_C0_CHECKPOINT_SCHEMA_VERSION,
        "gate": NATIVE_C0_GATE_NAME,
        "trainable_state_dict": state,
        "trainable_state_sha256": state_sha,
        "trainable_name_sha256": name_sha,
        "optimizer_state_dict": {},
        "metadata": {
            "run_id": "native_c0_real_batch_v1",
            "model_revision": NATIVE_C0_MODEL_REVISION,
            "episode_id": "11",
            "source_locks": source_locks,
            "parent_processor_result_sha256": processor_result_sha,
            "parent_processor_verification_sha256": processor_verification_sha,
        },
    }
    torch.save(checkpoint_payload, checkpoint)

    codebook = build_action_token_codebook(_Tokenizer())
    decision = score_action_token_logits(
        torch.zeros(1, 8, 32), action_token_position=5, codebook=codebook
    )
    decision.update(
        {
            "target_action": "MOVE_FORWARD",
            "prediction_matches_target": False,
            "note": "descriptive only",
        }
    )
    native_batch = {
        "renderer_version": NATIVE_QWEN_RENDERER_VERSION,
        "system_prompt": NATIVE_QWEN_SYSTEM_PROMPT,
        "pathway": "native_qwen3_vl_processor_and_visual_encoder",
        "policy_visual_fields": ["rgb"],
        "explicitly_excluded_native_fields": ["depth", "simulator_pose"],
        "history": {
            "contract": HISTORY_CONTRACTS[HistoryMode.H0_CURRENT].to_record(),
            "decision_step_index": 0,
            "frames": [{"relative_timestep": 0}],
        },
        "processor_class": "Qwen3VLProcessor",
        "tokenizer_class": "Qwen2TokenizerFast",
        "rendered_prefix_length": 3,
        "sequence_length": 8,
        "target_start_idx": 3,
        "supervised_token_count": 5,
        "image_count": 1,
        "visual_tokens_per_image": [1],
        "total_visual_tokens": 1,
        "model_input_keys": [
            "attention_mask",
            "image_grid_thw",
            "input_ids",
            "labels",
            "pixel_values",
        ],
        "target_surface": "Action ID: 1",
        "shared_action_token_position": 5,
        "model_input_shapes": {
            "input_ids": [1, 8],
            "attention_mask": [1, 8],
            "labels": [1, 8],
            "pixel_values": [4, 1176],
            "image_grid_thw": [1, 3],
        },
        "model_input_sha256": {
            key: key[0] * 64
            for key in (
                "input_ids",
                "attention_mask",
                "labels",
                "pixel_values",
                "image_grid_thw",
            )
        },
        "masked_prefix_token_count": 3,
        "supervised_label_token_count": 5,
    }
    expected_probe = {
        "episode_id": "11",
        "trajectory_id": "22",
        "scene_id": "mp3d/scene/scene.glb",
        "instruction_sha256": "a" * 64,
        "rgb_sha256": "b" * 64,
        "target_action_id": 1,
        "target_action": "MOVE_FORWARD",
        "release_manifest": {"split": "train", "episode_count": 1},
        "habitat_runtime": {"habitat_sim_version": "0.3.3"},
        "action_codebook": {
            **codebook.to_record(),
            "system_prompt": NATIVE_QWEN_SYSTEM_PROMPT,
        },
        "native_batch": copy.deepcopy(native_batch),
    }
    result = {
        "schema_version": NATIVE_C0_GATE_SCHEMA_VERSION,
        "gate": NATIVE_C0_GATE_NAME,
        "run_id": "native_c0_real_batch_v1",
        "status": "passed",
        "selection_rule": "lowest_numeric_train_episode_with_non_stop_first_action",
        "source_locks": source_locks,
        "model_weight_locks": model_weight_locks(snapshot),
        "parent_processor_gate": {
            "result_sha256": processor_result_sha,
            "verification_sha256": processor_verification_sha,
            "reverification_sha256": file_sha256(processor_reverification),
        },
        "source": {
            **{
                key: value
                for key, value in expected_probe.items()
                if key not in {"native_batch", "habitat_runtime", "action_codebook"}
            },
            "policy_visual_fields": ["rgb"],
            "explicitly_excluded_policy_fields": ["depth", "simulator_pose"],
        },
        "runtime": {
            "architecture": "aarch64",
            "transformers": "4.57.6",
            "bf16_supported": True,
            "dtype": "bf16",
            "gradient_checkpointing": "non_reentrant",
            "docker_image": {
                "reference": "gridnav-spark-habitat:hsim0.3.3",
                "id": "sha256:" + "c" * 64,
            },
            "habitat": copy.deepcopy(expected_probe["habitat_runtime"]),
        },
        "processor": {
            "model_id": NATIVE_C0_MODEL_ID,
            "model_revision": NATIVE_C0_MODEL_REVISION,
            "local_files_only": True,
        },
        "native_batch": native_batch,
        "model": {
            "model_id": NATIVE_C0_MODEL_ID,
            "model_revision": NATIVE_C0_MODEL_REVISION,
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "spatial_rope_enabled": False,
            "peft_2880_workaround_enabled": False,
            "trainable_contract": {
                "all_trainable_names_are_lora": True,
                "trainable_tensor_count": 1,
                "trainable_parameter_count": 6,
            },
        },
        "optimization": {
            "optimizer": "AdamW",
            "completed_updates": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "training_loss_before_update": 2.0,
            "global_gradient_norm_before_clip": 3.0,
            "post_update_target_loss": 1.5,
            "gradients": {
                "trainable_tensor_count": 1,
                "gradient_tensor_count": 1,
                "nonzero_gradient_tensor_count": 1,
                "all_gradients_finite": True,
                "gradient_l2_norm": 3.0,
            },
            "updates": {
                "trainable_tensor_count": 1,
                "changed_tensor_count": 1,
                "all_updates_finite": True,
                "update_l2_norm": 0.1,
            },
            "trainable_state_sha256_before": "d" * 64,
            "trainable_state_sha256_after": state_sha,
        },
        "checkpoint": {
            "bytes": checkpoint.stat().st_size,
            "sha256": file_sha256(checkpoint),
            "trainable_state_sha256": state_sha,
            "trainable_name_sha256": name_sha,
            "trainable_tensor_count": 1,
            "trainable_parameter_count": 6,
            "perturbed_state_sha256": "e" * 64,
            "reloaded_state_sha256": state_sha,
            "reload_target_loss": 1.5,
            "reload_target_loss_absolute_delta": 0.0,
        },
        "action_codebook": copy.deepcopy(expected_probe["action_codebook"]),
        "action_decision": decision,
    }
    paths = {
        "checkpoint": checkpoint,
        "snapshot": snapshot,
        "processor_result": processor_result,
        "processor_verification": processor_verification,
        "processor_reverification": processor_reverification,
    }
    return result, expected_probe, paths


def _verify(
    result: dict, expected_probe: dict, paths: dict[str, Path], tmp_path: Path
) -> list[str]:
    return verify_native_c0_result(
        result,
        repo=tmp_path,
        checkpoint=paths["checkpoint"],
        model_snapshot=paths["snapshot"],
        processor_result=paths["processor_result"],
        processor_verification=paths["processor_verification"],
        processor_reverification=paths["processor_reverification"],
        expected_probe=expected_probe,
    )


def test_native_c0_artifact_checker_accepts_complete_fixture(tmp_path: Path, monkeypatch) -> None:
    result, expected_probe, paths = _fixture(tmp_path, monkeypatch)
    assert _verify(result, expected_probe, paths, tmp_path) == []


def test_checker_detects_source_weight_and_checkpoint_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    result, expected_probe, paths = _fixture(tmp_path, monkeypatch)
    (tmp_path / NATIVE_C0_SOURCE_LOCK_PATHS[0]).write_text("changed\n", encoding="utf-8")
    (paths["snapshot"] / "model-00001-of-00001.safetensors").write_bytes(b"changed")
    paths["checkpoint"].write_bytes(b"changed")
    errors = _verify(result, expected_probe, paths, tmp_path)
    assert any("source hash mismatch" in error for error in errors)
    assert "frozen model-weight lock mapping mismatch" in errors
    assert any("checkpoint" in error for error in errors)


def test_checker_detects_batch_and_optimization_tampering(tmp_path: Path, monkeypatch) -> None:
    result, expected_probe, paths = _fixture(tmp_path, monkeypatch)
    result["native_batch"]["model_input_sha256"]["pixel_values"] = "0" * 64
    result["optimization"]["completed_updates"] = 2
    result["action_decision"]["scores"]["STOP"]["scored_token_position"] = 6
    errors = _verify(result, expected_probe, paths, tmp_path)
    assert any("independent native-processor reconstruction" in error for error in errors)
    assert "native-C0 gate did not complete exactly one update" in errors
    assert any("scored-token position mismatch" in error for error in errors)

