"""Adversarial tests for development SFT artifact admission."""

from __future__ import annotations

import json
from pathlib import Path

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.training import development_sft_artifacts as artifacts
from spatialvlm.training.development_sft import (
    DEVELOPMENT_SFT_GATE_NAME,
    DevelopmentSFTProtocol,
)
from spatialvlm.training.real_batch_gate import file_sha256
from spatialvlm.training.sft import FREEZE_GROUP_PATTERNS


def _gradient_rows() -> dict:
    return {
        group: {
            "trainable_tensor_count": 1,
            "gradient_tensor_count": 1,
            "all_gradients_finite": True,
            "gradient_l2_norm": 1.0,
        }
        for group in FREEZE_GROUP_PATTERNS
    }


def _update_rows() -> dict:
    return {
        group: {
            "trainable_tensor_count": 1,
            "changed_tensor_count": 1,
            "all_updates_finite": True,
            "update_l2_norm": 0.1,
        }
        for group in FREEZE_GROUP_PATTERNS
    }


def _fixture(root: Path, monkeypatch) -> tuple:
    protocol = DevelopmentSFTProtocol()
    repo = root / "repo"
    repo.mkdir()
    locks = {}
    for relative in artifacts.DEVELOPMENT_SFT_SOURCE_LOCK_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {relative}\n", encoding="utf-8")
        locks[relative] = file_sha256(path)

    samples = []
    trajectory = 100
    for split, count in (
        ("train", protocol.train_samples_per_action),
        ("calibration", protocol.calibration_samples_per_action),
    ):
        sample_index = 0
        for _ in range(count):
            for action in R2RCEAction:
                samples.append(
                    {
                        "split": split,
                        "sample_index": sample_index,
                        "episode_id": str(trajectory),
                        "trajectory_id": str(trajectory),
                        "scene_id": f"scene-{trajectory % 61}",
                        "instruction_sha256": "a" * 64,
                        "action_step_index": int(action),
                        "target_action_id": int(action),
                        "target_action": action.name,
                        "replay_prefix_action_ids": [],
                        "replay_prefix_sha256": "b" * 64,
                    }
                )
                trajectory += 1
                sample_index += 1

    pilot_excluded = [str(index) for index in range(24)]
    manifest = {
        "schema_version": 1,
        "gate": DEVELOPMENT_SFT_GATE_NAME,
        "run_id": "development_sft_v1",
        "status": "passed",
        "protocol": protocol.to_record(),
        "release_manifest": None,
        "samples": samples,
        "source_locks": locks,
        "pilot_link": {
            "run_id": artifacts.PILOT_RUN_ID,
            "manifest_sha256": artifacts.PILOT_MANIFEST_SHA256,
            "suite_sha256": artifacts.PILOT_SUITE_SHA256,
            "verification_v2_sha256": artifacts.PILOT_VERIFICATION_V2_SHA256,
            "warmstart_checkpoint_bytes": 4,
            "warmstart_checkpoint_sha256": "pending",
            "warmstart_trainable_state_sha256": artifacts.WARMSTART_TRAINABLE_STATE_SHA256,
            "excluded_trajectory_ids": pilot_excluded,
        },
        "preflights": {},
    }
    manifest_path = root / "manifest.json"
    for label, gate in (
        ("processor", "spark_real_batch_checkpoint_processor_preflight_v1"),
        ("action", "spark_action_decision_tokenizer_preflight_v1"),
    ):
        path = root / f"{label}.json"
        path.write_text(json.dumps({"status": "passed", "gate": gate}), encoding="utf-8")
        manifest["preflights"][label] = {
            "relative_path": path.name,
            "sha256": file_sha256(path),
            "gate": gate,
        }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    cache_entries = [
        {
            "split": row["split"],
            "sample_index": row["sample_index"],
            "relative_path": f"{row['split']}-{row['sample_index']}.pt",
            "bytes": 1,
            "sha256": "c" * 64,
        }
        for row in samples
    ]
    cache_manifest = {
        "status": "passed",
        "manifest_sha256": file_sha256(manifest_path),
        "entries": cache_entries,
    }
    cache_path = root / "cache_manifest.json"
    cache_path.write_text(json.dumps(cache_manifest), encoding="utf-8")

    audits = {1, *protocol.checkpoint_steps, *protocol.calibration_steps[1:]}
    train_samples = [row for row in samples if row["split"] == "train"]
    steps = []
    for step in range(1, 129):
        sample = train_samples[step - 1]
        steps.append(
            {
                "step": step,
                "sample_index": sample["sample_index"],
                "episode_id": sample["episode_id"],
                "trajectory_id": sample["trajectory_id"],
                "target_action": sample["target_action"],
                "target_token_nll": 1.0,
                "global_gradient_norm_before_clip": 2.0,
                "peak_allocated_bytes": 100,
                "gradients_by_group": _gradient_rows(),
                "updates_by_group": _update_rows() if step in audits else None,
            }
        )
    calibrations = [
        {
            "step": step,
            "sample_count": protocol.calibration_sample_count,
            "mean_target_token_nll": 1.0 + ordinal,
            "samples": [{} for _ in range(protocol.calibration_sample_count)],
        }
        for ordinal, step in enumerate((0, 16, 32, 64, 128))
    ]
    checkpoints = [
        {
            "update_step": step,
            "relative_path": f"step_{step:04d}.pt",
            "bytes": 1,
            "sha256": "d" * 64,
            "trainable_state_sha256": "e" * 64,
        }
        for step in (16, 32, 64, 128)
    ]
    warmstart = root / "warm.pt"
    warmstart.write_bytes(b"warm")
    warm_hash = file_sha256(warmstart)
    monkeypatch.setattr(artifacts, "WARMSTART_CHECKPOINT_BYTES", 4)
    monkeypatch.setattr(artifacts, "WARMSTART_CHECKPOINT_SHA256", warm_hash)
    manifest["pilot_link"]["warmstart_checkpoint_sha256"] = warm_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache_manifest["manifest_sha256"] = file_sha256(manifest_path)
    cache_path.write_text(json.dumps(cache_manifest), encoding="utf-8")

    result = {
        "schema_version": 1,
        "gate": DEVELOPMENT_SFT_GATE_NAME,
        "run_id": "development_sft_v1",
        "status": "passed",
        "terminal_reason": "pre_registered_early_stopping",
        "manifest_sha256": file_sha256(manifest_path),
        "protocol": protocol.to_record(),
        "source_locks": locks,
        "cache_manifest_sha256": file_sha256(cache_path),
        "training": {
            "completed_updates": 128,
            "steps": steps,
            "calibrations": calibrations,
            "checkpoints": checkpoints,
        },
        "selection": {
            "selected_calibration": calibrations[0],
            "selected_checkpoint": {
                "source": "pilot_warmstart",
                "step": 0,
                "bytes": 4,
                "sha256": warm_hash,
            },
        },
    }
    return result, manifest, manifest_path, cache_manifest, cache_path, warmstart, repo


def test_verifier_accepts_complete_early_stopped_fixture(tmp_path: Path, monkeypatch) -> None:
    result, manifest, manifest_path, cache, cache_path, warmstart, repo = _fixture(
        tmp_path, monkeypatch
    )
    errors = artifacts.verify_development_sft(
        result,
        manifest=manifest,
        manifest_path=manifest_path,
        cache_manifest=cache,
        cache_manifest_path=cache_path,
        checkpoint_root=tmp_path / "checkpoints",
        warmstart_checkpoint=warmstart,
        repo=repo,
        verify_cache_files=False,
        verify_all_checkpoint_files=False,
    )
    assert errors == []


def test_verifier_rejects_step_sample_tamper(tmp_path: Path, monkeypatch) -> None:
    result, manifest, manifest_path, cache, cache_path, warmstart, repo = _fixture(
        tmp_path, monkeypatch
    )
    result["training"]["steps"][0]["trajectory_id"] = "tampered"
    errors = artifacts.verify_development_sft(
        result,
        manifest=manifest,
        manifest_path=manifest_path,
        cache_manifest=cache,
        cache_manifest_path=cache_path,
        checkpoint_root=tmp_path / "checkpoints",
        warmstart_checkpoint=warmstart,
        repo=repo,
        verify_cache_files=False,
        verify_all_checkpoint_files=False,
    )
    assert "step-1: sample trajectory_id mismatch" in errors
