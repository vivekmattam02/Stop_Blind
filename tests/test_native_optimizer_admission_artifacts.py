"""Independent-verifier tests for native optimizer admission."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import scripts.check_native_optimizer_admission_artifacts as verifier_module
from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.training.native_c0_gate import file_sha256
from spatialvlm.training.native_optimizer_admission import (
    NATIVE_OPTIMIZER_ADMISSION_GATE,
    NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
    NATIVE_OPTIMIZER_OBJECTIVE,
    NativeOptimizerAdmissionProtocol,
    learning_rate_multiplier,
    select_admitted_rate,
    summarize_calibration,
    training_sample_order,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _decision(sample_index: int, action: R2RCEAction) -> dict:
    return {
        "sample_index": sample_index,
        "episode_id": f"episode-{action.name}-{sample_index}",
        "trajectory_id": f"trajectory-{action.name}-{sample_index}",
        "target_action": action.name,
        "predicted_action": action.name,
        "correct": True,
        "target_token_nll": 2.0,
        "target_restricted_probability": 0.5,
        "restricted_action_ce": 0.6931471805599453,
        "cache_sha256": "cache",
        "action_token_position": 10,
        "supervision_schema_version": 1,
        "supervision_method": "single_shared_position_official_action_id_token_v1",
        "supervised_token_count": 1,
        "supervised_token_id": 10 + int(action),
        "action_decision": {"stub": True},
    }


def _fixture(tmp_path: Path, monkeypatch) -> tuple[argparse.Namespace, dict, dict]:
    protocol = NativeOptimizerAdmissionProtocol.stress()
    result_root = tmp_path / "artifacts"
    cache_root = tmp_path / "cache"
    checkpoint_root = tmp_path / "checkpoints"
    model_snapshot = tmp_path / "model"
    model_snapshot.mkdir()
    episodes = tmp_path / "train.json.gz"
    ground_truth = tmp_path / "train_gt.json.gz"
    episodes.write_bytes(b"episodes")
    ground_truth.write_bytes(b"ground-truth")
    samples = [
        {
            "split": split,
            "sample_index": index,
            "episode_id": f"{split}-{index}",
            "trajectory_id": f"{split}-trajectory-{index}",
            "target_action": R2RCEAction(index % 4).name,
            "target_action_id": index % 4,
        }
        for split in ("train", "calibration")
        for index in range(16)
    ]
    prerequisites = {"processor_preflight": {"result_sha256": "processor"}}
    manifest = {
        "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
        "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
        "run_id": "native_optimizer_stress_v1",
        "profile": "stress",
        "protocol": protocol.to_record(),
        "source_locks": {"source": "hash"},
        "prerequisites": prerequisites,
        "runtime": {"docker_image_ref": "image", "docker_image_id": "sha256:image"},
        "dataset": {
            "episodes_sha256": file_sha256(episodes),
            "ground_truth_sha256": file_sha256(ground_truth),
        },
        "samples": samples,
        "sample_records_sha256": verifier_module._canonical_sha256(samples),
    }
    _write_json(result_root / "manifest.json", manifest)
    cache_entries = []
    for index, sample in enumerate(samples):
        path = cache_root / f"entry-{index}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"cache-{index}".encode())
        cache_entries.append({"relative_path": path.name, "sha256": file_sha256(path)})
    cache_manifest = {
        "status": "passed",
        "manifest_sha256": file_sha256(result_root / "manifest.json"),
        "entry_count": len(samples),
        "entries": cache_entries,
    }
    _write_json(cache_root / "cache_manifest.json", cache_manifest)

    arm_results = []
    arm_index = []
    for arm_number, rate in enumerate(protocol.learning_rates):
        arm_id = f"arm-{arm_number}"
        arm_root = result_root / "arms" / arm_id
        decisions = [
            _decision(index, R2RCEAction(index % 4))
            for index in range(protocol.unique_calibration_decisions)
        ]
        summary = summarize_calibration(decisions)
        calibrations = [
            {"update_step": step, "summary": summary, "per_decision": decisions}
            for step in protocol.calibration_updates
        ]
        checkpoint = checkpoint_root / arm_id / "step_0001.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{arm_number}".encode())
        initial = "a" * 64
        final = f"{arm_number + 1:064x}"
        exposure_order = training_sample_order(
            [row for row in samples if row["split"] == "train"], protocol
        )
        event = {
            "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
            "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
            "objective": NATIVE_OPTIMIZER_OBJECTIVE,
            "update_step": 1,
            "microbatch_count": 16,
            "epoch_index": 0,
            "sample_exposure_start": 0,
            "sample_exposure_end_exclusive": 16,
            "learning_rate_multiplier": learning_rate_multiplier(1, protocol),
            "applied_learning_rate": rate * learning_rate_multiplier(1, protocol),
            "mean_restricted_action_ce": 1.0,
            "update_to_weight_ratio": 1e-4,
            "before_trainable_state_sha256": initial,
            "after_trainable_state_sha256": final,
            "gradient_report": {
                "all_gradients_finite": True,
                "gradient_tensor_count": 2,
                "trainable_tensor_count": 2,
                "nonzero_gradient_tensor_count": 2,
                "gradient_l2_norm": 1.0,
            },
            "update_report": {
                "all_updates_finite": True,
                "changed_tensor_count": 2,
                "update_l2_norm": 1e-3,
            },
            "optimizer_state": {
                "exp_avg": {
                    "all_finite": True,
                    "nonfinite_element_count": 0,
                    "tensor_count": 2,
                }
            },
            "microbatches": [
                {"supervised_token_count": 1, "sample_index": sample_index}
                for sample_index in exposure_order
            ],
        }
        events_path = arm_root / "updates.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
        result = {
            "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
            "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
            "profile": "stress",
            "arm_id": arm_id,
            "learning_rate": rate,
            "status": "passed",
            "initial_trainable_state_sha256": initial,
            "final_trainable_state_sha256": final,
            "completed_updates": 1,
            "changed_update_count": 1,
            "all_updates_finite": True,
            "calibrations": calibrations,
            "checkpoints": [
                {
                    "path": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                    "update_step": 1,
                }
            ],
            "events_sha256": file_sha256(events_path),
        }
        result_path = arm_root / "result.json"
        _write_json(result_path, result)
        arm_results.append(result)
        arm_index.append(
            {
                "arm_id": arm_id,
                "learning_rate": rate,
                "relative_path": str(result_path.relative_to(result_root)),
                "sha256": file_sha256(result_path),
            }
        )
    suite = {
        "schema_version": NATIVE_OPTIMIZER_ADMISSION_SCHEMA_VERSION,
        "gate": NATIVE_OPTIMIZER_ADMISSION_GATE,
        "run_id": "native_optimizer_stress_v1",
        "status": "passed",
        "profile": "stress",
        "manifest_sha256": file_sha256(result_root / "manifest.json"),
        "cache_manifest_sha256": file_sha256(cache_root / "cache_manifest.json"),
        "protocol": protocol.to_record(),
        "arm_index": arm_index,
        "selection": select_admitted_rate(arm_results, protocol),
        "identical_initial_trainable_state_sha256": "a" * 64,
        "supervision": {"supervised_token_count_per_decision": 1},
        "model_weight_locks": {"model": {"bytes": 1, "sha256": "weight"}},
        "source_locks": {"source": "hash"},
        "prerequisites": prerequisites,
        "runtime": {"docker_image_id": "sha256:image"},
    }
    _write_json(result_root / "suite.json", suite)
    monkeypatch.setattr(verifier_module, "_source_locks", lambda repo: {"source": "hash"})
    monkeypatch.setattr(
        verifier_module, "_prerequisite_records", lambda args: prerequisites
    )
    monkeypatch.setattr(verifier_module, "_load_dataset", lambda args: object())
    monkeypatch.setattr(verifier_module, "_select_samples", lambda dataset, p: samples)
    monkeypatch.setattr(
        verifier_module,
        "model_weight_locks",
        lambda path: {"model": {"bytes": 1, "sha256": "weight"}},
    )
    monkeypatch.setattr(verifier_module, "validate_action_decision_record", lambda row: [])
    args = argparse.Namespace(
        run_id="native_optimizer_stress_v1",
        profile="stress",
        repo=tmp_path,
        episodes=episodes,
        ground_truth=ground_truth,
        model_snapshot=model_snapshot,
        result_root=result_root,
        cache_root=cache_root,
        docker_image_ref="image",
        docker_image_id="sha256:image",
    )
    return args, manifest, suite


def test_complete_fixture_passes(tmp_path: Path, monkeypatch) -> None:
    args, manifest, suite = _fixture(tmp_path, monkeypatch)
    assert verifier_module.verify_native_optimizer_admission(
        args=args, manifest=manifest, suite=suite
    ) == []


def test_unit_ledger_tamper_fails(tmp_path: Path, monkeypatch) -> None:
    args, manifest, suite = _fixture(tmp_path, monkeypatch)
    manifest = deepcopy(manifest)
    manifest["protocol"]["unit_ledger"]["derived"]["reconciled_optimizer_updates"] = 64
    errors = verifier_module.verify_native_optimizer_admission(
        args=args, manifest=manifest, suite=suite
    )
    assert any("ledger" in error or "protocol" in error for error in errors)


def test_event_supervision_tamper_fails(tmp_path: Path, monkeypatch) -> None:
    args, manifest, suite = _fixture(tmp_path, monkeypatch)
    path = args.result_root / "arms" / "arm-0" / "updates.jsonl"
    event = json.loads(path.read_text())
    event["microbatches"][0]["supervised_token_count"] = 6
    path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    errors = verifier_module.verify_native_optimizer_admission(
        args=args, manifest=manifest, suite=suite
    )
    assert any("supervision" in error for error in errors)


def test_selection_tamper_fails(tmp_path: Path, monkeypatch) -> None:
    args, manifest, suite = _fixture(tmp_path, monkeypatch)
    suite = deepcopy(suite)
    suite["selection"]["selected_learning_rate"] = 9e-4
    errors = verifier_module.verify_native_optimizer_admission(
        args=args, manifest=manifest, suite=suite
    )
    assert any("selection" in error for error in errors)
