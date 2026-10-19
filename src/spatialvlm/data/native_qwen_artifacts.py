"""Independent admission checks for the native-Qwen C0 history preflight."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_RENDERER_VERSION,
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
)

NATIVE_QWEN_HISTORY_PREFLIGHT_SCHEMA_VERSION = 4
NATIVE_QWEN_HISTORY_PREFLIGHT_GATE_NAME = "native_qwen_c0_history_processor_preflight_v4"
NATIVE_QWEN_HISTORY_PREFLIGHT_V3_GATE_NAME = "native_qwen_c0_history_processor_preflight_v3"
NATIVE_QWEN_METADATA_HYDRATION_SCHEMA_VERSION = 1
NATIVE_QWEN_METADATA_HYDRATION_GATE_NAME = "qwen_processor_metadata_hydration_v1"
NATIVE_QWEN_HISTORY_VERIFIER_SCHEMA_VERSION = 4
NATIVE_QWEN_HISTORY_VERIFIER_GATE_NAME = (
    "independent_native_qwen_c0_history_artifact_verification_v4"
)
NATIVE_QWEN_HISTORY_VERIFIER_V3_GATE_NAME = (
    "independent_native_qwen_c0_history_artifact_verification_v3"
)
NATIVE_QWEN_HISTORY_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
NATIVE_QWEN_HISTORY_MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
NATIVE_QWEN_HISTORY_TRANSFORMERS_VERSION = "4.57.6"
NATIVE_QWEN_PROCESSOR_METADATA_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
NATIVE_QWEN_HISTORY_SOURCE_LOCK_PATHS = (
    "docs/NATIVE_QWEN_HISTORY_GATE.md",
    "scripts/check_native_qwen_history_artifacts.py",
    "scripts/check_native_qwen_history_preflight.py",
    "scripts/hydrate_spark_qwen_metadata.py",
    "scripts/native_qwen_history_sync_files.txt",
    "scripts/run_spark_native_qwen_history_preflight.sh",
    "scripts/spark_launch_helpers.sh",
    "scripts/start_spark_qwen_metadata_hydration.sh",
    "scripts/start_spark_native_qwen_history_preflight.sh",
    "src/spatialvlm/config/__init__.py",
    "src/spatialvlm/config/workshop.py",
    "src/spatialvlm/data/__init__.py",
    "src/spatialvlm/data/history.py",
    "src/spatialvlm/data/native_qwen.py",
    "src/spatialvlm/data/native_qwen_artifacts.py",
    "tests/test_history.py",
    "tests/test_native_qwen.py",
    "tests/test_native_qwen_artifacts.py",
    "tests/test_qwen_metadata_hydration.py",
    "tests/test_spark_launch_helpers.py",
    "tests/test_workshop_config.py",
)


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_locks(repo: str | Path) -> dict[str, str]:
    """Hash the complete reviewed source surface for this gate."""
    root = Path(repo).resolve()
    return {
        relative: file_sha256(root / relative) for relative in NATIVE_QWEN_HISTORY_SOURCE_LOCK_PATHS
    }


def processor_metadata_locks(snapshot: str | Path) -> dict[str, dict[str, int | str]]:
    """Hash every small processor/tokenizer file required by the frozen Qwen revision."""
    root = Path(snapshot).resolve()
    locks: dict[str, dict[str, int | str]] = {}
    for relative in NATIVE_QWEN_PROCESSOR_METADATA_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required Qwen processor metadata is missing: {path}")
        locks[relative] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return locks


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_source_locks(result: dict[str, Any], repo: Path, errors: list[str]) -> None:
    locks = result.get("source_locks")
    expected = set(NATIVE_QWEN_HISTORY_SOURCE_LOCK_PATHS)
    if not isinstance(locks, dict) or set(locks) != expected:
        errors.append("result source-lock path set mismatch")
        return
    for relative, expected_hash in locks.items():
        path = repo / relative
        if not path.is_file():
            errors.append(f"locked source is missing: {relative}")
        elif file_sha256(path) != expected_hash:
            errors.append(f"locked source hash mismatch: {relative}")


def _validate_processor_metadata_locks(
    result: dict[str, Any], snapshot: Path, errors: list[str]
) -> None:
    locks = result.get("processor_metadata_locks")
    expected = set(NATIVE_QWEN_PROCESSOR_METADATA_FILES)
    if not isinstance(locks, dict) or set(locks) != expected:
        errors.append("processor metadata-lock path set mismatch")
        return
    for relative, record in locks.items():
        path = snapshot / relative
        if not path.is_file():
            errors.append(f"processor metadata is missing: {relative}")
            continue
        if not isinstance(record, dict):
            errors.append(f"processor metadata record is invalid: {relative}")
            continue
        if record.get("bytes") != path.stat().st_size:
            errors.append(f"processor metadata byte-size mismatch: {relative}")
        if record.get("sha256") != file_sha256(path):
            errors.append(f"processor metadata hash mismatch: {relative}")


def _validate_shape_map(row: dict[str, Any], label: str, errors: list[str]) -> None:
    shapes = row.get("model_input_shapes")
    required = {
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "image_grid_thw",
    }
    if not isinstance(shapes, dict) or not required <= set(shapes):
        errors.append(f"{label}: required native model-input shapes are missing")
        return
    model_input_keys = row.get("model_input_keys")
    if not isinstance(model_input_keys, list) or set(model_input_keys) != set(shapes):
        errors.append(f"{label}: model-input key and shape indexes disagree")
    if "token_type_ids" in shapes:
        errors.append(f"{label}: generic token_type_ids leaked into native model inputs")
    sequence_length = row.get("sequence_length")
    expected_sequence_shape = [1, sequence_length]
    for key in ("input_ids", "attention_mask", "labels"):
        if shapes.get(key) != expected_sequence_shape:
            errors.append(f"{label}: {key} shape does not match sequence length")
    mm_types_present = "mm_token_type_ids" in shapes
    optional_inputs = row.get("optional_model_inputs")
    if optional_inputs != {"mm_token_type_ids_present": mm_types_present}:
        errors.append(f"{label}: optional model-input presence record mismatch")
    if mm_types_present:
        if shapes.get("mm_token_type_ids") != expected_sequence_shape:
            errors.append(f"{label}: mm_token_type_ids shape does not match sequence length")
        errors.append(f"{label}: pinned processor unexpectedly emitted mm_token_type_ids")
    image_count = row.get("image_count")
    image_grid_shape = shapes.get("image_grid_thw")
    if image_grid_shape != [image_count, 3]:
        errors.append(f"{label}: image_grid_thw shape does not match image count")
    pixel_shape = shapes.get("pixel_values")
    if not isinstance(pixel_shape, list) or not pixel_shape or not _is_positive_int(pixel_shape[0]):
        errors.append(f"{label}: pixel_values shape is invalid")


def _validate_mode(mode: HistoryMode, row: Any, errors: list[str]) -> None:
    label = mode.value
    if not isinstance(row, dict):
        errors.append(f"{label}: mode record is missing")
        return
    contract = HISTORY_CONTRACTS[mode]
    history = row.get("history")
    if not isinstance(history, dict) or history.get("contract") != contract.to_record():
        errors.append(f"{label}: history contract mismatch")
        return
    frames = history.get("frames")
    if not isinstance(frames, list) or len(frames) != contract.frame_count:
        errors.append(f"{label}: history frame count mismatch")
        return
    if [frame.get("relative_timestep") for frame in frames] != list(
        range(1 - contract.frame_count, 1)
    ):
        errors.append(f"{label}: relative timestep sequence mismatch")
    if any(frame.get("is_padding") for frame in frames):
        errors.append(f"{label}: deterministic step-3 preflight must not contain padding")
    expected_sources = {
        HistoryMode.H0_CURRENT: [3],
        HistoryMode.H1_PREVIOUS_ACTION: [3],
        HistoryMode.H4_SHORT_WINDOW: [0, 1, 2, 3],
    }[mode]
    expected_action_ids = {
        HistoryMode.H0_CURRENT: [None],
        HistoryMode.H1_PREVIOUS_ACTION: [3],
        HistoryMode.H4_SHORT_WINDOW: [None, 1, 2, 3],
    }[mode]
    expected_action_names = {
        HistoryMode.H0_CURRENT: [None],
        HistoryMode.H1_PREVIOUS_ACTION: ["TURN_RIGHT"],
        HistoryMode.H4_SHORT_WINDOW: [None, "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
    }[mode]
    if history.get("decision_step_index") != 3:
        errors.append(f"{label}: deterministic decision step mismatch")
    if [frame.get("source_step_index") for frame in frames] != expected_sources:
        errors.append(f"{label}: chronological source-step sequence mismatch")
    if [frame.get("previous_action_id") for frame in frames] != expected_action_ids:
        errors.append(f"{label}: causal previous-action ID sequence mismatch")
    if [frame.get("previous_action") for frame in frames] != expected_action_names:
        errors.append(f"{label}: causal previous-action name sequence mismatch")
    if row.get("renderer_version") != NATIVE_QWEN_RENDERER_VERSION:
        errors.append(f"{label}: renderer version mismatch")
    if row.get("supervision_schema_version") != NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION:
        errors.append(f"{label}: supervision schema version mismatch")
    if row.get("supervision_method") != NATIVE_QWEN_SUPERVISION_METHOD:
        errors.append(f"{label}: supervision method mismatch")
    if row.get("pathway") != "native_qwen3_vl_processor_and_visual_encoder":
        errors.append(f"{label}: native pathway marker mismatch")
    if row.get("policy_visual_fields") != ["rgb"]:
        errors.append(f"{label}: native policy visual fields mismatch")
    if row.get("explicitly_excluded_native_fields") != ["depth", "simulator_pose"]:
        errors.append(f"{label}: native excluded-field contract mismatch")
    if row.get("target_surface") != "Action ID: 2":
        errors.append(f"{label}: canonical target surface mismatch")
    if row.get("image_count") != contract.frame_count:
        errors.append(f"{label}: image count mismatch")
    counts = row.get("visual_tokens_per_image")
    if (
        not isinstance(counts, list)
        or len(counts) != contract.frame_count
        or not all(_is_positive_int(value) for value in counts)
    ):
        errors.append(f"{label}: visual token counts are invalid")
    elif row.get("total_visual_tokens") != sum(counts):
        errors.append(f"{label}: total visual token count mismatch")
    sequence_length = row.get("sequence_length")
    target_start = row.get("target_start_idx")
    target_token_count = row.get("target_token_count")
    action_token_offset = row.get("action_token_offset")
    supervised_position = row.get("supervised_token_position")
    supervised_token_id = row.get("supervised_token_id")
    supervised = row.get("supervised_token_count")
    sequence_valid = _is_positive_int(sequence_length) and sequence_length <= 4096
    if not sequence_valid:
        errors.append(f"{label}: sequence length is invalid")
    target_start_valid = (
        sequence_valid
        and isinstance(target_start, int)
        and not isinstance(target_start, bool)
        and 0 < target_start < sequence_length
    )
    if not target_start_valid:
        errors.append(f"{label}: target start is invalid")
    target_count_valid = (
        _is_positive_int(target_token_count)
        and target_start_valid
        and target_start + target_token_count == sequence_length
    )
    if not target_count_valid:
        errors.append(f"{label}: canonical target boundary mismatch")
    action_offset_valid = (
        target_count_valid
        and isinstance(action_token_offset, int)
        and not isinstance(action_token_offset, bool)
        and 0 <= action_token_offset < target_token_count
    )
    if not action_offset_valid:
        errors.append(f"{label}: action-token offset is invalid")
    if (
        not action_offset_valid
        or supervised_position != target_start + action_token_offset
        or not isinstance(supervised_position, int)
        or isinstance(supervised_position, bool)
    ):
        errors.append(f"{label}: supervised action-token position mismatch")
    if not _is_nonnegative_int(supervised_token_id):
        errors.append(f"{label}: supervised action-token ID is invalid")
    if supervised != 1:
        errors.append(f"{label}: objective must supervise exactly one action token")
    _validate_shape_map(row, label, errors)


def verify_native_qwen_history_preflight(
    result: dict[str, Any], *, repo: str | Path, model_snapshot: str | Path
) -> list[str]:
    """Return every independent integrity/contract error in one preflight artifact."""
    errors: list[str] = []
    root = Path(repo).resolve()
    if result.get("schema_version") != NATIVE_QWEN_HISTORY_PREFLIGHT_SCHEMA_VERSION:
        errors.append("result schema_version mismatch")
    if result.get("gate") != NATIVE_QWEN_HISTORY_PREFLIGHT_GATE_NAME:
        errors.append("result gate mismatch")
    if result.get("status") != "passed":
        errors.append("result status is not passed")
    run_id = result.get("run_id")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"native_qwen_history_preflight_v[1-9][0-9]*", run_id) is None
    ):
        errors.append("result run_id is invalid")
    if result.get("model_id") != NATIVE_QWEN_HISTORY_MODEL_ID:
        errors.append("result model identity mismatch")
    if result.get("model_revision") != NATIVE_QWEN_HISTORY_MODEL_REVISION:
        errors.append("result model revision mismatch")
    if result.get("local_files_only") is not True:
        errors.append("preflight did not use the frozen local processor cache")
    if result.get("transformers") != NATIVE_QWEN_HISTORY_TRANSFORMERS_VERSION:
        errors.append("Transformers version mismatch")
    docker_image = result.get("docker_image")
    if not isinstance(docker_image, dict):
        errors.append("Docker image identity is missing")
    else:
        if docker_image.get("reference") != "gridnav-spark:cuda13-py25.10":
            errors.append("Docker image reference mismatch")
        image_id = docker_image.get("id")
        if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
            errors.append("Docker image ID is invalid")
    for key in ("python", "torch", "platform", "processor_class", "tokenizer_class"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            errors.append(f"runtime field is missing: {key}")
    if isinstance(result.get("platform"), str) and "aarch64" not in result["platform"].lower():
        errors.append("runtime platform is not aarch64")
    _validate_source_locks(result, root, errors)
    _validate_processor_metadata_locks(result, Path(model_snapshot).resolve(), errors)

    modes = result.get("history_modes")
    if not isinstance(modes, dict) or set(modes) != {mode.value for mode in HistoryMode}:
        errors.append("history mode index mismatch")
        return errors
    for mode in HistoryMode:
        _validate_mode(mode, modes[mode.value], errors)
        row = modes[mode.value]
        if isinstance(row, dict):
            if row.get("processor_class") != result.get("processor_class"):
                errors.append(f"{mode.value}: processor class differs from runtime identity")
            if row.get("tokenizer_class") != result.get("tokenizer_class"):
                errors.append(f"{mode.value}: tokenizer class differs from runtime identity")
    h0 = modes[HistoryMode.H0_CURRENT.value]
    h1 = modes[HistoryMode.H1_PREVIOUS_ACTION.value]
    if isinstance(h0, dict) and isinstance(h1, dict):
        if h0.get("visual_tokens_per_image") != h1.get("visual_tokens_per_image"):
            errors.append("H0/H1 current-image visual tokenization mismatch")
    return errors
