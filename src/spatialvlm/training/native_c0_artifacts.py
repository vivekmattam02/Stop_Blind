"""Independent admission checks for the native-Qwen C0 real-batch gate."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode
from spatialvlm.data.native_qwen import NATIVE_QWEN_RENDERER_VERSION, NATIVE_QWEN_SYSTEM_PROMPT
from spatialvlm.data.native_qwen_artifacts import (
    NATIVE_QWEN_HISTORY_PREFLIGHT_V3_GATE_NAME,
    NATIVE_QWEN_HISTORY_VERIFIER_V3_GATE_NAME,
)
from spatialvlm.eval.action_decision import validate_action_decision_record
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_CHECKPOINT_SCHEMA_VERSION,
    NATIVE_C0_DOCKER_IMAGE_REF,
    NATIVE_C0_GATE_NAME,
    NATIVE_C0_GATE_SCHEMA_VERSION,
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    NATIVE_C0_PARENT_RESULT_SHA256,
    NATIVE_C0_PARENT_VERIFICATION_SHA256,
    NATIVE_C0_TRANSFORMERS_VERSION,
    file_sha256,
    model_weight_locks,
    tensor_state_sha256,
    trainable_name_sha256,
)

NATIVE_C0_VERIFIER_SCHEMA_VERSION = 1
NATIVE_C0_VERIFIER_GATE_NAME = "independent_native_qwen_c0_real_batch_verification_v1"
NATIVE_C0_SOURCE_LOCK_PATHS = (
    "docs/NATIVE_QWEN_C0_GATE.md",
    "docker/spark/Dockerfile",
    "docker/spark/habitat.Dockerfile",
    "docker/spark/requirements.txt",
    "pyproject.toml",
    "scripts/check_native_c0_artifacts.py",
    "scripts/check_native_qwen_history_artifacts.py",
    "scripts/native_c0_gate_sync_files.txt",
    "scripts/run_spark_native_c0_gate.py",
    "scripts/run_spark_native_c0_gate.sh",
    "scripts/start_spark_native_c0_gate.sh",
    "src/spatialvlm/backbone/qwen3_vl.py",
    "src/spatialvlm/backbone/rope_patch.py",
    "src/spatialvlm/data/datasets.py",
    "src/spatialvlm/data/habitat_sim_r2r.py",
    "src/spatialvlm/data/history.py",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/data/native_qwen_artifacts.py",
    "src/spatialvlm/eval/action_decision.py",
    "src/spatialvlm/eval/r2r_ce.py",
    "src/spatialvlm/training/native_c0_artifacts.py",
    "src/spatialvlm/training/native_c0_gate.py",
    "tests/test_action_decision.py",
    "tests/test_datasets.py",
    "tests/test_habitat_sim_r2r.py",
    "tests/test_history.py",
    "tests/test_native_c0_artifacts.py",
    "tests/test_native_c0_gate.py",
    "tests/test_native_qwen.py",
    "tests/test_native_qwen_artifacts.py",
    "tests/test_qwen3_vl.py",
)


def source_locks(repo: str | Path) -> dict[str, str]:
    """Hash the complete reviewed native-C0 gate source surface."""
    root = Path(repo).resolve()
    return {relative: file_sha256(root / relative) for relative in NATIVE_C0_SOURCE_LOCK_PATHS}


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _validate_source_locks(result: Mapping[str, Any], repo: Path, errors: list[str]) -> None:
    locks = result.get("source_locks")
    expected = set(NATIVE_C0_SOURCE_LOCK_PATHS)
    if not isinstance(locks, dict) or set(locks) != expected:
        errors.append("native-C0 source-lock path set mismatch")
        return
    for relative, expected_hash in locks.items():
        path = repo / relative
        if not path.is_file():
            errors.append(f"locked native-C0 source is missing: {relative}")
        elif file_sha256(path) != expected_hash:
            errors.append(f"locked native-C0 source hash mismatch: {relative}")


def _validate_parent(
    result: Mapping[str, Any],
    processor_result_path: Path,
    processor_verification_path: Path,
    processor_reverification_path: Path,
    errors: list[str],
) -> None:
    parent = result.get("parent_processor_gate")
    if not isinstance(parent, dict):
        errors.append("parent processor-gate record is missing")
        return
    if (
        not processor_result_path.is_file()
        or not processor_verification_path.is_file()
        or not processor_reverification_path.is_file()
    ):
        errors.append("parent processor-gate artifacts are missing")
        return
    observed_result_hash = file_sha256(processor_result_path)
    observed_verification_hash = file_sha256(processor_verification_path)
    if observed_result_hash != NATIVE_C0_PARENT_RESULT_SHA256:
        errors.append("parent processor result has unexpected immutable hash")
    if observed_verification_hash != NATIVE_C0_PARENT_VERIFICATION_SHA256:
        errors.append("parent processor verification has unexpected immutable hash")
    if parent.get("result_sha256") != observed_result_hash:
        errors.append("parent processor result linkage mismatch")
    if parent.get("verification_sha256") != observed_verification_hash:
        errors.append("parent processor verification linkage mismatch")

    processor_result = json.loads(processor_result_path.read_text(encoding="utf-8"))
    processor_verification = json.loads(processor_verification_path.read_text(encoding="utf-8"))
    processor_reverification = json.loads(processor_reverification_path.read_text(encoding="utf-8"))
    if (
        processor_result.get("status") != "passed"
        or processor_result.get("gate") != NATIVE_QWEN_HISTORY_PREFLIGHT_V3_GATE_NAME
    ):
        errors.append("parent processor result did not pass the v3 gate")
    if (
        processor_verification.get("status") != "passed"
        or processor_verification.get("error_count") != 0
        or processor_verification.get("gate") != NATIVE_QWEN_HISTORY_VERIFIER_V3_GATE_NAME
    ):
        errors.append("parent processor verification did not pass the v3 gate")
    if processor_verification.get("result_sha256") != observed_result_hash:
        errors.append("parent processor verifier/result linkage mismatch")
    if (
        processor_reverification.get("status") != "passed"
        or processor_reverification.get("error_count") != 0
        or processor_reverification.get("gate") != NATIVE_QWEN_HISTORY_VERIFIER_V3_GATE_NAME
    ):
        errors.append("fresh parent processor reverification did not pass")
    if processor_reverification.get("result_sha256") != observed_result_hash:
        errors.append("fresh parent processor reverification/result linkage mismatch")
    if parent.get("reverification_sha256") != file_sha256(processor_reverification_path):
        errors.append("fresh parent processor reverification linkage mismatch")
    if processor_result.get("model_revision") != NATIVE_C0_MODEL_REVISION:
        errors.append("parent processor model revision mismatch")


def _validate_weight_locks(
    result: Mapping[str, Any], model_snapshot: Path, errors: list[str]
) -> None:
    recorded = result.get("model_weight_locks")
    try:
        observed = model_weight_locks(model_snapshot)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        errors.append(f"could not reconstruct frozen model-weight locks: {exc}")
        return
    if recorded != observed:
        errors.append("frozen model-weight lock mapping mismatch")


def _validate_checkpoint(result: Mapping[str, Any], checkpoint: Path, errors: list[str]) -> None:
    record = result.get("checkpoint")
    if not isinstance(record, dict):
        errors.append("native-C0 checkpoint record is missing")
        return
    if not checkpoint.is_file():
        errors.append("native-C0 checkpoint file is missing")
        return
    if checkpoint.stat().st_size != record.get("bytes"):
        errors.append("native-C0 checkpoint byte-size mismatch")
    if file_sha256(checkpoint) != record.get("sha256"):
        errors.append("native-C0 checkpoint SHA-256 mismatch")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        errors.append(f"native-C0 checkpoint could not be loaded: {exc}")
        return
    if payload.get("schema_version") != NATIVE_C0_CHECKPOINT_SCHEMA_VERSION:
        errors.append("native-C0 checkpoint schema mismatch")
    if payload.get("gate") != NATIVE_C0_GATE_NAME:
        errors.append("native-C0 checkpoint gate mismatch")
    state = payload.get("trainable_state_dict")
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(value, torch.Tensor) for value in state.values())
    ):
        errors.append("native-C0 checkpoint trainable state is invalid")
        return
    state_hash = tensor_state_sha256(state)
    name_hash = trainable_name_sha256(list(state))
    if payload.get("trainable_state_sha256") != state_hash:
        errors.append("native-C0 checkpoint payload state hash mismatch")
    if payload.get("trainable_name_sha256") != name_hash:
        errors.append("native-C0 checkpoint payload name hash mismatch")
    if record.get("trainable_state_sha256") != state_hash:
        errors.append("native-C0 result/checkpoint state linkage mismatch")
    if record.get("trainable_name_sha256") != name_hash:
        errors.append("native-C0 result/checkpoint name linkage mismatch")
    if record.get("trainable_tensor_count") != len(state):
        errors.append("native-C0 checkpoint tensor count mismatch")
    if record.get("trainable_parameter_count") != sum(tensor.numel() for tensor in state.values()):
        errors.append("native-C0 checkpoint parameter count mismatch")
    if any("lora_" not in name.lower() for name in state):
        errors.append("native-C0 checkpoint contains non-LoRA trainable tensors")
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        errors.append("native-C0 checkpoint optimizer state is missing")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("native-C0 checkpoint metadata is missing")
    else:
        if metadata.get("run_id") != result.get("run_id"):
            errors.append("native-C0 checkpoint run linkage mismatch")
        if metadata.get("model_revision") != NATIVE_C0_MODEL_REVISION:
            errors.append("native-C0 checkpoint model revision mismatch")
        source = result.get("source", {})
        if metadata.get("episode_id") != source.get("episode_id"):
            errors.append("native-C0 checkpoint episode linkage mismatch")
        if metadata.get("source_locks") != result.get("source_locks"):
            errors.append("native-C0 checkpoint source-lock linkage mismatch")
        if metadata.get("parent_processor_result_sha256") != NATIVE_C0_PARENT_RESULT_SHA256:
            errors.append("native-C0 checkpoint parent-result linkage mismatch")
        if (
            metadata.get("parent_processor_verification_sha256")
            != NATIVE_C0_PARENT_VERIFICATION_SHA256
        ):
            errors.append("native-C0 checkpoint parent-verification linkage mismatch")


def _validate_batch(result: Mapping[str, Any], errors: list[str]) -> None:
    batch = result.get("native_batch")
    if not isinstance(batch, dict):
        errors.append("native-C0 batch record is missing")
        return
    if batch.get("renderer_version") != NATIVE_QWEN_RENDERER_VERSION:
        errors.append("native-C0 renderer version mismatch")
    if batch.get("system_prompt") != NATIVE_QWEN_SYSTEM_PROMPT:
        errors.append("native-C0 system-prompt contract mismatch")
    if (
        batch.get("history", {}).get("contract")
        != HISTORY_CONTRACTS[HistoryMode.H0_CURRENT].to_record()
    ):
        errors.append("native-C0 gate did not use the frozen H0 mechanics contract")
    if batch.get("history", {}).get("decision_step_index") != 0:
        errors.append("native-C0 H0 mechanics probe is not at decision step zero")
    if batch.get("policy_visual_fields") != ["rgb"]:
        errors.append("native-C0 policy visual-field contract mismatch")
    if batch.get("explicitly_excluded_native_fields") != ["depth", "simulator_pose"]:
        errors.append("native-C0 excluded-field contract mismatch")
    shapes = batch.get("model_input_shapes")
    required = {"input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"}
    if not isinstance(shapes, dict) or set(shapes) != required:
        errors.append("native-C0 model-input shape surface mismatch")
        return
    sequence_length = batch.get("sequence_length")
    if not isinstance(sequence_length, int) or sequence_length <= 0 or sequence_length > 4096:
        errors.append("native-C0 sequence length is invalid")
        return
    for key in ("input_ids", "attention_mask", "labels"):
        if shapes.get(key) != [1, sequence_length]:
            errors.append(f"native-C0 {key} shape mismatch")
    if shapes.get("image_grid_thw") != [1, 3]:
        errors.append("native-C0 image-grid shape mismatch")
    pixel_shape = shapes.get("pixel_values")
    if not isinstance(pixel_shape, list) or not pixel_shape or pixel_shape[0] <= 0:
        errors.append("native-C0 pixel-values shape is invalid")
    target_start = batch.get("target_start_idx")
    supervised = batch.get("supervised_token_count")
    if (
        not isinstance(target_start, int)
        or not isinstance(supervised, int)
        or target_start <= 0
        or supervised <= 0
        or target_start + supervised != sequence_length
    ):
        errors.append("native-C0 suffix supervision boundary mismatch")
    if batch.get("masked_prefix_token_count") != target_start:
        errors.append("native-C0 masked-prefix count mismatch")
    if batch.get("supervised_label_token_count") != supervised:
        errors.append("native-C0 supervised-label count mismatch")
    if batch.get("image_count") != 1 or batch.get("total_visual_tokens", 0) <= 0:
        errors.append("native-C0 visual-token contract mismatch")
    action_position = batch.get("shared_action_token_position")
    if (
        not isinstance(action_position, int)
        or action_position < target_start
        or action_position >= sequence_length
    ):
        errors.append("native-C0 shared action-token position is invalid")
    input_hashes = batch.get("model_input_sha256")
    if not isinstance(input_hashes, dict) or set(input_hashes) != required:
        errors.append("native-C0 model-input hash surface mismatch")


def _validate_optimization(result: Mapping[str, Any], errors: list[str]) -> None:
    model = result.get("model")
    if not isinstance(model, dict):
        errors.append("native-C0 model record is missing")
        return
    if model.get("model_id") != NATIVE_C0_MODEL_ID:
        errors.append("native-C0 model ID mismatch")
    if model.get("model_revision") != NATIVE_C0_MODEL_REVISION:
        errors.append("native-C0 model revision mismatch")
    if model.get("lora_rank") != 32 or model.get("lora_alpha") != 64:
        errors.append("native-C0 LoRA rank/alpha contract mismatch")
    if model.get("lora_dropout") != 0.0:
        errors.append("native-C0 LoRA dropout contract mismatch")
    if model.get("lora_target_modules") != ["q_proj", "k_proj", "v_proj", "o_proj"]:
        errors.append("native-C0 LoRA target-module contract mismatch")
    if model.get("spatial_rope_enabled") is not False:
        errors.append("native C0 unexpectedly enabled the proposed spatial RoPE")
    if model.get("peft_2880_workaround_enabled") is not False:
        errors.append("native C0 unexpectedly enabled external-branch vision training")
    trainable = model.get("trainable_contract", {})
    if trainable.get("all_trainable_names_are_lora") is not True:
        errors.append("native C0 is not LoRA-only")
    if trainable.get("trainable_tensor_count", 0) <= 0:
        errors.append("native C0 has no trainable tensors")
    if trainable.get("trainable_parameter_count", 0) <= 0:
        errors.append("native C0 has no trainable parameters")

    optimization = result.get("optimization")
    if not isinstance(optimization, dict):
        errors.append("native-C0 optimization record is missing")
        return
    if optimization.get("completed_updates") != 1:
        errors.append("native-C0 gate did not complete exactly one update")
    if optimization.get("optimizer") != "AdamW":
        errors.append("native-C0 optimizer contract mismatch")
    if optimization.get("learning_rate") != 5e-5:
        errors.append("native-C0 learning-rate contract mismatch")
    if optimization.get("weight_decay") != 0.01:
        errors.append("native-C0 weight-decay contract mismatch")
    if optimization.get("max_grad_norm") != 1.0:
        errors.append("native-C0 gradient-clip contract mismatch")
    if not _finite_positive(optimization.get("training_loss_before_update")):
        errors.append("native-C0 training loss is invalid")
    if not _finite_positive(optimization.get("global_gradient_norm_before_clip")):
        errors.append("native-C0 global gradient norm is invalid")
    if not _finite_positive(optimization.get("post_update_target_loss")):
        errors.append("native-C0 post-update target loss is invalid")
    gradients = optimization.get("gradients", {})
    updates = optimization.get("updates", {})
    expected_tensors = trainable.get("trainable_tensor_count")
    if gradients.get("trainable_tensor_count") != expected_tensors:
        errors.append("native-C0 gradient trainable count mismatch")
    if gradients.get("gradient_tensor_count") != expected_tensors:
        errors.append("native-C0 gradient coverage mismatch")
    if (
        not gradients.get("all_gradients_finite")
        or gradients.get("nonzero_gradient_tensor_count", 0) <= 0
    ):
        errors.append("native-C0 gradient validity mismatch")
    if not _finite_positive(gradients.get("gradient_l2_norm")):
        errors.append("native-C0 gradient L2 norm is invalid")
    if updates.get("trainable_tensor_count") != expected_tensors:
        errors.append("native-C0 update trainable count mismatch")
    if not updates.get("all_updates_finite") or updates.get("changed_tensor_count", 0) <= 0:
        errors.append("native-C0 update validity mismatch")
    if not _finite_positive(updates.get("update_l2_norm")):
        errors.append("native-C0 update L2 norm is invalid")
    before = optimization.get("trainable_state_sha256_before")
    after = optimization.get("trainable_state_sha256_after")
    if not isinstance(before, str) or not isinstance(after, str) or before == after:
        errors.append("native-C0 trainable state did not change")

    checkpoint = result.get("checkpoint", {})
    if checkpoint.get("trainable_state_sha256") != after:
        errors.append("native-C0 checkpoint/post-update state mismatch")
    if checkpoint.get("reloaded_state_sha256") != after:
        errors.append("native-C0 reload state mismatch")
    if checkpoint.get("perturbed_state_sha256") in {None, after}:
        errors.append("native-C0 reload perturbation probe is invalid")
    try:
        reload_delta = float(checkpoint.get("reload_target_loss_absolute_delta", float("inf")))
    except (TypeError, ValueError):
        reload_delta = float("inf")
    if not math.isfinite(reload_delta) or reload_delta > 1e-5:
        errors.append("native-C0 reload target-loss drift exceeds tolerance")
    try:
        recomputed_delta = abs(
            float(checkpoint.get("reload_target_loss"))
            - float(optimization.get("post_update_target_loss"))
        )
    except (TypeError, ValueError):
        recomputed_delta = float("inf")
    if not math.isfinite(recomputed_delta) or abs(recomputed_delta - reload_delta) > 1e-8:
        errors.append("native-C0 reload target-loss delta is inconsistent")


def verify_native_c0_result(
    result: dict[str, Any],
    *,
    repo: str | Path,
    checkpoint: str | Path,
    model_snapshot: str | Path,
    processor_result: str | Path,
    processor_verification: str | Path,
    processor_reverification: str | Path,
    expected_probe: Mapping[str, Any],
) -> list[str]:
    """Return every independent native-C0 artifact error."""
    errors: list[str] = []
    if result.get("schema_version") != NATIVE_C0_GATE_SCHEMA_VERSION:
        errors.append("native-C0 result schema mismatch")
    if result.get("gate") != NATIVE_C0_GATE_NAME:
        errors.append("native-C0 result gate mismatch")
    if result.get("status") != "passed":
        errors.append("native-C0 result status is not passed")
    if result.get("selection_rule") != "lowest_numeric_train_episode_with_non_stop_first_action":
        errors.append("native-C0 probe-selection rule mismatch")
    run_id = result.get("run_id")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"native_c0_real_batch_v[1-9][0-9]*", run_id) is None
    ):
        errors.append("native-C0 run ID is invalid")
    runtime = result.get("runtime", {})
    if runtime.get("architecture") != "aarch64":
        errors.append("native-C0 runtime architecture mismatch")
    if runtime.get("transformers") != NATIVE_C0_TRANSFORMERS_VERSION:
        errors.append("native-C0 Transformers version mismatch")
    if runtime.get("bf16_supported") is not True:
        errors.append("native-C0 runtime does not report BF16 support")
    if runtime.get("dtype") != "bf16":
        errors.append("native-C0 runtime dtype mismatch")
    if runtime.get("gradient_checkpointing") != "non_reentrant":
        errors.append("native-C0 gradient-checkpointing contract mismatch")
    if runtime.get("docker_image", {}).get("reference") != NATIVE_C0_DOCKER_IMAGE_REF:
        errors.append("native-C0 Docker image reference mismatch")
    image_id = runtime.get("docker_image", {}).get("id")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        errors.append("native-C0 Docker image ID is invalid")

    source = result.get("source", {})
    for key in (
        "episode_id",
        "trajectory_id",
        "scene_id",
        "instruction_sha256",
        "rgb_sha256",
        "target_action_id",
        "target_action",
    ):
        if source.get(key) != expected_probe.get(key):
            errors.append(f"native-C0 selected probe mismatch: {key}")
    if source.get("policy_visual_fields") != ["rgb"]:
        errors.append("native-C0 source exposed a non-RGB policy field")
    if source.get("explicitly_excluded_policy_fields") != ["depth", "simulator_pose"]:
        errors.append("native-C0 source exclusion record mismatch")
    if source.get("release_manifest") != expected_probe.get("release_manifest"):
        errors.append("native-C0 release-manifest reconstruction mismatch")
    if runtime.get("habitat") != expected_probe.get("habitat_runtime"):
        errors.append("native-C0 Habitat runtime reconstruction mismatch")
    processor = result.get("processor", {})
    if (
        processor.get("model_id") != NATIVE_C0_MODEL_ID
        or processor.get("model_revision") != NATIVE_C0_MODEL_REVISION
        or processor.get("local_files_only") is not True
    ):
        errors.append("native-C0 processor identity/offline contract mismatch")

    _validate_source_locks(result, Path(repo).resolve(), errors)
    _validate_parent(
        result,
        Path(processor_result).resolve(),
        Path(processor_verification).resolve(),
        Path(processor_reverification).resolve(),
        errors,
    )
    _validate_weight_locks(result, Path(model_snapshot).resolve(), errors)
    _validate_batch(result, errors)
    expected_batch = expected_probe.get("native_batch")
    if result.get("native_batch") != expected_batch:
        errors.append("native-C0 batch does not match independent native-processor reconstruction")
    if result.get("action_codebook") != expected_probe.get("action_codebook"):
        errors.append(
            "native-C0 action codebook does not match independent tokenizer reconstruction"
        )
    _validate_optimization(result, errors)
    _validate_checkpoint(result, Path(checkpoint).resolve(), errors)

    decision_errors = validate_action_decision_record(
        result.get("action_decision", {}), codebook_record=result.get("action_codebook")
    )
    errors.extend(f"action decision: {error}" for error in decision_errors)
    decision = result.get("action_decision", {})
    predicted_action = decision.get("predicted_action")
    target_action = result.get("source", {}).get("target_action")
    if decision.get("target_action") != target_action:
        errors.append("action decision: target-action linkage mismatch")
    if decision.get("prediction_matches_target") is not (predicted_action == target_action):
        errors.append("action decision: target-match diagnostic is inconsistent")
    action_position = result.get("native_batch", {}).get("shared_action_token_position")
    scores = result.get("action_decision", {}).get("scores", {})
    if isinstance(scores, dict):
        for action, row in scores.items():
            if not isinstance(row, dict) or row.get("scored_token_position") != action_position:
                errors.append(f"action decision: {action} scored-token position mismatch")
    return errors
