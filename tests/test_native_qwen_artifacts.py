"""Adversarial tests for native-Qwen C0 history artifact admission."""

from __future__ import annotations

from pathlib import Path

from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_RENDERER_VERSION,
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
)
from spatialvlm.data.native_qwen_artifacts import (
    NATIVE_QWEN_HISTORY_MODEL_ID,
    NATIVE_QWEN_HISTORY_MODEL_REVISION,
    NATIVE_QWEN_HISTORY_PREFLIGHT_GATE_NAME,
    NATIVE_QWEN_HISTORY_PREFLIGHT_SCHEMA_VERSION,
    NATIVE_QWEN_HISTORY_SOURCE_LOCK_PATHS,
    NATIVE_QWEN_HISTORY_TRANSFORMERS_VERSION,
    NATIVE_QWEN_PROCESSOR_METADATA_FILES,
    file_sha256,
    processor_metadata_locks,
    verify_native_qwen_history_preflight,
)


def _mode_record(mode: HistoryMode) -> dict:
    contract = HISTORY_CONTRACTS[mode]
    image_count = contract.frame_count
    prefix = 32 + image_count
    target_token_count = 4
    action_token_offset = 2
    source_indices = {
        HistoryMode.H0_CURRENT: [3],
        HistoryMode.H1_PREVIOUS_ACTION: [3],
        HistoryMode.H4_SHORT_WINDOW: [0, 1, 2, 3],
    }[mode]
    action_ids = {
        HistoryMode.H0_CURRENT: [None],
        HistoryMode.H1_PREVIOUS_ACTION: [3],
        HistoryMode.H4_SHORT_WINDOW: [None, 1, 2, 3],
    }[mode]
    action_names = {
        HistoryMode.H0_CURRENT: [None],
        HistoryMode.H1_PREVIOUS_ACTION: ["TURN_RIGHT"],
        HistoryMode.H4_SHORT_WINDOW: [None, "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
    }[mode]
    frames = []
    for index, (source_index, action_id, action_name) in enumerate(
        zip(source_indices, action_ids, action_names, strict=True)
    ):
        frames.append(
            {
                "source_step_index": source_index,
                "relative_timestep": index - image_count + 1,
                "previous_action_id": action_id,
                "previous_action": action_name,
                "is_padding": False,
            }
        )
    visual_counts = [49] * image_count
    return {
        "renderer_version": NATIVE_QWEN_RENDERER_VERSION,
        "supervision_schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
        "supervision_method": NATIVE_QWEN_SUPERVISION_METHOD,
        "pathway": "native_qwen3_vl_processor_and_visual_encoder",
        "policy_visual_fields": ["rgb"],
        "explicitly_excluded_native_fields": ["depth", "simulator_pose"],
        "history": {
            "contract": contract.to_record(),
            "decision_step_index": 3,
            "frames": frames,
        },
        "processor_class": "Qwen3VLProcessor",
        "tokenizer_class": "Qwen2TokenizerFast",
        "rendered_prefix_length": prefix,
        "sequence_length": prefix + target_token_count,
        "target_start_idx": prefix,
        "target_token_count": target_token_count,
        "action_token_offset": action_token_offset,
        "supervised_token_position": prefix + action_token_offset,
        "supervised_token_id": 12,
        "supervised_token_count": 1,
        "image_count": image_count,
        "visual_tokens_per_image": visual_counts,
        "total_visual_tokens": sum(visual_counts),
        "model_input_keys": [
            "attention_mask",
            "image_grid_thw",
            "input_ids",
            "labels",
            "pixel_values",
        ],
        "model_input_shapes": {
            "input_ids": [1, prefix + target_token_count],
            "attention_mask": [1, prefix + target_token_count],
            "labels": [1, prefix + target_token_count],
            "pixel_values": [image_count * 196, 768],
            "image_grid_thw": [image_count, 3],
        },
        "optional_model_inputs": {"mm_token_type_ids_present": False},
        "target_surface": "Action ID: 2",
    }


def _result(repo: Path) -> dict:
    locks = {}
    for relative in NATIVE_QWEN_HISTORY_SOURCE_LOCK_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {relative}\n", encoding="utf-8")
        locks[relative] = file_sha256(path)
    snapshot = repo / "model_snapshot"
    snapshot.mkdir()
    for relative in NATIVE_QWEN_PROCESSOR_METADATA_FILES:
        (snapshot / relative).write_text(f"metadata: {relative}\n", encoding="utf-8")
    return {
        "schema_version": NATIVE_QWEN_HISTORY_PREFLIGHT_SCHEMA_VERSION,
        "gate": NATIVE_QWEN_HISTORY_PREFLIGHT_GATE_NAME,
        "run_id": "native_qwen_history_preflight_v4",
        "status": "passed",
        "model_id": NATIVE_QWEN_HISTORY_MODEL_ID,
        "model_revision": NATIVE_QWEN_HISTORY_MODEL_REVISION,
        "docker_image": {
            "reference": "gridnav-spark:cuda13-py25.10",
            "id": f"sha256:{'a' * 64}",
        },
        "local_files_only": True,
        "platform": "Linux-aarch64",
        "python": "3.12.3",
        "torch": "2.9.0a0+nv25.10",
        "transformers": NATIVE_QWEN_HISTORY_TRANSFORMERS_VERSION,
        "processor_class": "Qwen3VLProcessor",
        "tokenizer_class": "Qwen2TokenizerFast",
        "history_modes": {mode.value: _mode_record(mode) for mode in HistoryMode},
        "processor_metadata_locks": processor_metadata_locks(snapshot),
        "source_locks": locks,
    }


def _verify(result: dict, repo: Path) -> list[str]:
    return verify_native_qwen_history_preflight(
        result, repo=repo, model_snapshot=repo / "model_snapshot"
    )


def test_verifier_accepts_complete_processor_preflight(tmp_path: Path) -> None:
    result = _result(tmp_path)
    assert _verify(result, tmp_path) == []


def test_verifier_rejects_mm_token_type_ids_from_the_pinned_default_processor(
    tmp_path: Path,
) -> None:
    result = _result(tmp_path)
    for row in result["history_modes"].values():
        row["model_input_keys"].append("mm_token_type_ids")
        row["model_input_keys"].sort()
        row["model_input_shapes"]["mm_token_type_ids"] = [1, row["sequence_length"]]
        row["optional_model_inputs"]["mm_token_type_ids_present"] = True

    errors = _verify(result, tmp_path)
    assert errors == [
        f"{mode.value}: pinned processor unexpectedly emitted mm_token_type_ids"
        for mode in HistoryMode
    ]


def test_verifier_rejects_optional_input_presence_or_shape_drift(tmp_path: Path) -> None:
    result = _result(tmp_path)
    result["history_modes"]["H0"]["optional_model_inputs"]["mm_token_type_ids_present"] = True
    h1 = result["history_modes"]["H1"]
    h1["model_input_keys"].append("mm_token_type_ids")
    h1["model_input_shapes"]["mm_token_type_ids"] = [1, 999]
    h1["optional_model_inputs"]["mm_token_type_ids_present"] = True

    errors = _verify(result, tmp_path)
    assert "H0: optional model-input presence record mismatch" in errors
    assert "H1: mm_token_type_ids shape does not match sequence length" in errors
    assert "H1: pinned processor unexpectedly emitted mm_token_type_ids" in errors


def test_verifier_rejects_source_and_visual_count_tamper(tmp_path: Path) -> None:
    result = _result(tmp_path)
    (tmp_path / NATIVE_QWEN_HISTORY_SOURCE_LOCK_PATHS[0]).write_text("tampered\n", encoding="utf-8")
    result["history_modes"]["H4"]["total_visual_tokens"] += 1

    errors = _verify(result, tmp_path)
    assert any("locked source hash mismatch" in error for error in errors)
    assert "H4: total visual token count mismatch" in errors


def test_verifier_rejects_multi_token_or_misaligned_supervision(tmp_path: Path) -> None:
    result = _result(tmp_path)
    result["history_modes"]["H0"]["supervised_token_count"] = 4
    result["history_modes"]["H1"]["supervised_token_position"] += 1
    result["history_modes"]["H4"]["target_token_count"] += 1

    errors = _verify(result, tmp_path)
    assert "H0: objective must supervise exactly one action token" in errors
    assert "H1: supervised action-token position mismatch" in errors
    assert "H4: canonical target boundary mismatch" in errors


def test_verifier_rejects_runtime_and_sequence_contract_drift(tmp_path: Path) -> None:
    result = _result(tmp_path)
    result["transformers"] = "4.58.0"
    result["history_modes"]["H1"]["model_input_shapes"]["labels"] = [1, 999]

    errors = _verify(result, tmp_path)
    assert "Transformers version mismatch" in errors
    assert "H1: labels shape does not match sequence length" in errors


def test_verifier_rejects_causal_history_and_image_identity_drift(tmp_path: Path) -> None:
    result = _result(tmp_path)
    result["docker_image"]["id"] = "mutable-tag-only"
    result["history_modes"]["H4"]["history"]["frames"][2]["previous_action_id"] = 3

    errors = _verify(result, tmp_path)
    assert "Docker image ID is invalid" in errors
    assert "H4: causal previous-action ID sequence mismatch" in errors


def test_verifier_rejects_processor_metadata_tamper(tmp_path: Path) -> None:
    result = _result(tmp_path)
    (tmp_path / "model_snapshot" / "preprocessor_config.json").write_text(
        "tampered\n", encoding="utf-8"
    )

    errors = _verify(result, tmp_path)
    assert "processor metadata byte-size mismatch: preprocessor_config.json" in errors
    assert "processor metadata hash mismatch: preprocessor_config.json" in errors

