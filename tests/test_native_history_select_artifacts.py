"""Adversarial CPU tests for native-history evidence admission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
)
from spatialvlm.training.native_c0_gate import file_sha256, model_weight_locks
from spatialvlm.training.native_history_select import (
    HISTORY_ARMS,
    NATIVE_HISTORY_GATE_NAME,
    NativeHistoryProtocol,
    select_history_arm,
)
from spatialvlm.training.native_history_select_artifacts import (
    NATIVE_HISTORY_SOURCE_LOCK_PATHS,
    verify_native_history_suite,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(root: Path) -> dict:
    protocol = replace(
        NativeHistoryProtocol(),
        planned_updates=4,
        train_samples_per_action=1,
        calibration_samples_per_action=1,
        calibration_steps=(0, 2, 4),
        checkpoint_interval=2,
        bootstrap_samples=100,
    )
    repo = root / "repo"
    locks = {}
    for relative in NATIVE_HISTORY_SOURCE_LOCK_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {relative}\n", encoding="utf-8")
        locks[relative] = file_sha256(path)

    samples = []
    for split, offset in (("train", 0), ("calibration", 10)):
        for index, action in enumerate(R2RCEAction):
            instruction = f"instruction-{split}-{action.name}"
            samples.append(
                {
                    "split": split,
                    "sample_index": index,
                    "episode_id": str(offset + index),
                    "trajectory_id": f"trajectory-{offset + index}",
                    "scene_id": "mp3d/scene/scene.glb",
                    "instruction_text": instruction,
                    "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                    "action_step_index": 3,
                    "target_action_id": int(action),
                    "target_action": action.name,
                    "replay_prefix_action_ids": [1, 2, 3],
                    "replay_prefix_sha256": "a" * 64,
                    "selection_rank_sha256": "b" * 64,
                    "order_rank_sha256": "c" * 64,
                }
            )
    canonical = hashlib.sha256(
        json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "gate": NATIVE_HISTORY_GATE_NAME,
        "run_id": "native_history_select_v1",
        "status": "frozen",
        "protocol": protocol.to_record(),
        "dataset": {
            "split_used": "train",
            "sealed_splits": ["val_seen", "val_unseen"],
        },
        "samples": samples,
        "sample_records_sha256": canonical,
        "source_locks": locks,
        "model": {
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
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
            "docker_image_ref": "gridnav-spark-habitat:hsim0.3.3",
            "docker_image_id": (
                "sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf"
            ),
            "sequential_arms": True,
            "fully_offline": True,
        },
    }
    manifest_path = root / "artifacts" / "manifest.json"
    _write(manifest_path, manifest)
    cache_entries = [
        {
            "split": row["split"],
            "sample_index": row["sample_index"],
            "relative_path": f"{row['split']}_{row['sample_index']}.pt",
            "bytes": 1,
            "sha256": "d" * 64,
            "sample_sha256": "e" * 64,
            "rgb_frames_sha256": "f" * 64,
        }
        for row in samples
    ]
    cache_manifest = {
        "status": "passed",
        "manifest_sha256": file_sha256(manifest_path),
        "entries": cache_entries,
        "entry_index_sha256": hashlib.sha256(
            json.dumps(cache_entries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    cache_path = root / "checkpoints" / "cache" / "cache_manifest.json"
    _write(cache_path, cache_manifest)

    snapshot = root / "snapshot"
    snapshot.mkdir()
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    _write(
        snapshot / "model.safetensors.index.json",
        {"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}},
    )
    calibration_samples = [row for row in samples if row["split"] == "calibration"]
    train_samples = [row for row in samples if row["split"] == "train"]
    per_arm_nll = {}
    arm_index = {}
    initialization = "1" * 64
    for arm_ordinal, arm_id in enumerate(HISTORY_ARMS):
        arm_root = manifest_path.parent / "arms" / arm_id
        arm_root.mkdir(parents=True)
        calibrations = []
        for step, penalty in ((0, 0.0), (2, 0.5), (4, 1.0)):
            rows = [
                {
                    "sample_index": sample["sample_index"],
                    "episode_id": sample["episode_id"],
                    "trajectory_id": sample["trajectory_id"],
                    "target_action": sample["target_action"],
                    "target_token_nll": 1.0 + arm_ordinal * 0.1 + penalty,
                    "predicted_action": sample["target_action"],
                    "correct": True,
                    "cache_sha256": cache_entries[4 + sample["sample_index"]]["sha256"],
                    "action_token_position": 100,
                    "supervision_schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
                    "supervision_method": NATIVE_QWEN_SUPERVISION_METHOD,
                    "supervised_token_count": 1,
                    "supervised_token_id": 100 + int(sample["target_action_id"]),
                }
                for sample in calibration_samples
            ]
            calibrations.append(
                {
                    "step": step,
                    "decision_count": 4,
                    "mean_target_token_nll": rows[0]["target_token_nll"],
                    "descriptive_accuracy": 1.0,
                    "per_decision": rows,
                    "accuracy_used_for_selection": False,
                }
            )
        checkpoints = []
        for step in (2, 4):
            path = root / "checkpoints" / "arms" / arm_id / f"step_{step:04d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{arm_id}-{step}".encode())
            checkpoints.append(
                {
                    "path": f"/checkpoints/arms/{arm_id}/{path.name}",
                    "relative_path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "trainable_state_sha256": str(step) * 64,
                    "update_step": step,
                }
            )
        events = []
        for step, sample in enumerate(train_samples, start=1):
            events.append(
                {
                    "step": step,
                    "sample_index": sample["sample_index"],
                    "episode_id": sample["episode_id"],
                    "trajectory_id": sample["trajectory_id"],
                    "target_action": sample["target_action"],
                    "action_token_position": 100,
                    "supervision_schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
                    "supervision_method": NATIVE_QWEN_SUPERVISION_METHOD,
                    "supervised_token_count": 1,
                    "supervised_token_id": 100 + int(sample["target_action_id"]),
                    "target_token_nll": 1.0,
                    "global_gradient_norm_before_clip": 2.0,
                    "gradient_report": {
                        "trainable_tensor_count": 2,
                        "gradient_tensor_count": 2,
                        "nonzero_gradient_tensor_count": 2,
                        "all_gradients_finite": True,
                    },
                }
            )
        (arm_root / "updates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
        )
        result = {
            "schema_version": 1,
            "gate": NATIVE_HISTORY_GATE_NAME,
            "arm_id": arm_id,
            "status": "passed",
            "terminal_reason": "planned_updates_completed",
            "manifest_sha256": file_sha256(manifest_path),
            "cache_manifest_sha256": file_sha256(cache_path),
            "initial_trainable_state_sha256": initialization,
            "completed_updates": 4,
            "calibrations": calibrations,
            "audit_updates": [
                {
                    "step": step,
                    "updates": {"changed_tensor_count": 2, "all_updates_finite": True},
                }
                for step in (1, 2, 4)
            ],
            "checkpoints": checkpoints,
            "selected_calibration": calibrations[0],
            "selected_checkpoint": {
                "source": "public_pretrained_seeded_initialization",
                "update_step": 0,
                "trainable_state_sha256": initialization,
                "path": None,
                "relative_path": None,
                "bytes": 0,
                "sha256": None,
            },
            "peak_allocated_bytes": 100,
            "training_seconds": 1.0,
        }
        result_path = arm_root / "result.json"
        _write(result_path, result)
        arm_index[arm_id] = {
            "relative_path": f"arms/{arm_id}/result.json",
            "sha256": file_sha256(result_path),
        }
        per_arm_nll[arm_id] = [1.0 + arm_ordinal * 0.1] * 4
    suite = {
        "schema_version": 1,
        "gate": NATIVE_HISTORY_GATE_NAME,
        "run_id": "native_history_select_v1",
        "status": "passed",
        "manifest_sha256": file_sha256(manifest_path),
        "cache_manifest_sha256": file_sha256(cache_path),
        "protocol": protocol.to_record(),
        "source_locks": locks,
        "arm_results": arm_index,
        "history_selection": select_history_arm(per_arm_nll, protocol=protocol),
        "identical_initial_trainable_state_sha256": initialization,
        "model_weight_locks": model_weight_locks(snapshot),
        "action_codebook": {
            "actions": {
                action.name: {
                    "action_id": int(action),
                    "action_token_id": 100 + int(action),
                    "action_specific_token_count": 1,
                }
                for action in R2RCEAction
            }
        },
        "supervision": {
            "schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
            "method": NATIVE_QWEN_SUPERVISION_METHOD,
            "supervised_token_count_per_decision": 1,
        },
        "runtime": {
            "docker_image_id": (
                "sha256:6d4f86090bcd1bf4d75709858194865353ebf7243e3564b359106518fcbecabf"
            ),
            "architecture": "aarch64",
            "transformers": "4.57.6",
            "gpu": "NVIDIA GB10",
        },
    }
    suite_path = manifest_path.parent / "suite.json"
    _write(suite_path, suite)
    return {
        "suite": suite,
        "suite_path": suite_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "cache": cache_manifest,
        "cache_path": cache_path,
        "checkpoint_root": root / "checkpoints" / "arms",
        "snapshot": snapshot,
        "repo": repo,
    }


def _verify(fixture: dict) -> list[str]:
    return verify_native_history_suite(
        fixture["suite"],
        suite_path=fixture["suite_path"],
        manifest=fixture["manifest"],
        manifest_path=fixture["manifest_path"],
        cache_manifest=fixture["cache"],
        cache_manifest_path=fixture["cache_path"],
        checkpoint_root=fixture["checkpoint_root"],
        model_snapshot=fixture["snapshot"],
        repo=fixture["repo"],
        verify_cache_files=False,
        verify_all_checkpoint_files=False,
    )


def test_verifier_accepts_complete_matched_fixture(tmp_path: Path) -> None:
    assert _verify(_fixture(tmp_path)) == []


def test_verifier_rejects_event_and_source_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    event_path = fixture["suite_path"].parent / "arms" / "H0_CURRENT" / "updates.jsonl"
    rows = event_path.read_text(encoding="utf-8").splitlines()
    event_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    locked = fixture["repo"] / NATIVE_HISTORY_SOURCE_LOCK_PATHS[0]
    locked.write_text("tampered\n", encoding="utf-8")
    errors = _verify(fixture)
    assert "locked source hash mismatch" in "\n".join(errors)
    assert "arm H0_CURRENT update event coverage mismatch" in errors


def test_runner_and_verifier_source_lock_surfaces_match() -> None:
    runner = Path("scripts/run_spark_native_history_select.py").read_text(encoding="utf-8")
    for relative in NATIVE_HISTORY_SOURCE_LOCK_PATHS:
        assert f'"{relative}"' in runner
