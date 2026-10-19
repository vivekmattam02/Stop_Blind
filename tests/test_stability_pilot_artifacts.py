"""Adversarial tests for optimization/stability pilot artifact verification."""

from __future__ import annotations

import json
from pathlib import Path

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.training.real_batch_gate import file_sha256
from spatialvlm.training.sft import FREEZE_GROUP_PATTERNS
from spatialvlm.training.stability_pilot import (
    STABILITY_PILOT_GATE_NAME,
    StabilityPilotProtocol,
)
from spatialvlm.training.stability_pilot_artifacts import (
    STABILITY_PILOT_SOURCE_LOCK_PATHS,
    verify_stability_pilot,
)
from spatialvlm.training.stability_pilot_artifacts_v2 import (
    EXPECTED_V1_ORDER_FALSE_NEGATIVE_ERRORS,
    STABILITY_PILOT_VERIFIER_V2_SOURCE_PATHS,
    verify_stability_pilot_v2,
    verify_v1_order_false_negative,
)


def _group_rows(kind: str) -> dict:
    rows = {}
    for group in FREEZE_GROUP_PATTERNS:
        if kind == "gradient":
            rows[group] = {
                "trainable_tensor_count": 1,
                "gradient_tensor_count": 1,
                "all_gradients_finite": True,
                "gradient_l2_norm": 1.0,
            }
        else:
            rows[group] = {
                "trainable_tensor_count": 1,
                "changed_tensor_count": 1,
                "all_updates_finite": True,
                "update_l2_norm": 0.1,
            }
    return rows


def _fixture(root: Path) -> tuple[dict, dict, Path, Path, Path, Path]:
    protocol = StabilityPilotProtocol()
    repo = root / "repo"
    arms_root = root / "arms"
    checkpoint_root = root / "checkpoints"
    repo.mkdir()
    arms_root.mkdir()
    checkpoint_root.mkdir()
    locks = {}
    for relative in STABILITY_PILOT_SOURCE_LOCK_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {relative}\n", encoding="utf-8")
        locks[relative] = file_sha256(path)
    samples = []
    trajectory = 0
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
                        "scene_id": f"scene-{trajectory}",
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
    manifest = {
        "schema_version": 1,
        "gate": STABILITY_PILOT_GATE_NAME,
        "run_id": "stability_pilot_v1",
        "status": "passed",
        "protocol": protocol.to_record(),
        "release_manifest": None,
        "samples": samples,
        "preflights": {},
        "source_locks": locks,
    }
    manifest_path = root / "manifest.json"
    for label, gate in (
        ("processor", "spark_real_batch_checkpoint_processor_preflight_v1"),
        ("action", "spark_action_decision_tokenizer_preflight_v1"),
    ):
        path = root / f"{label}_preflight.json"
        path.write_text(json.dumps({"status": "passed", "gate": gate}), encoding="utf-8")
        manifest["preflights"][label] = {
            "relative_path": path.name,
            "sha256": file_sha256(path),
            "gate": gate,
        }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    initial_hash = "c" * 64
    arm_index = {}
    for arm_number, rate in enumerate(protocol.learning_rates):
        arm_id = protocol.arm_id(rate)
        arm_dir = arms_root / arm_id
        arm_dir.mkdir()
        training_steps = []
        for step in range(1, protocol.updates_per_arm + 1):
            training_steps.append(
                {
                    "step": step,
                    "target_token_nll": 10.0 - step * 0.1,
                    "global_gradient_norm_before_clip": 10.0,
                    "peak_allocated_bytes": 100,
                    "gradients_by_group": _group_rows("gradient"),
                    "updates_by_group": (
                        _group_rows("update") if step in {1, *protocol.checkpoint_steps} else None
                    ),
                }
            )
        calibration = []
        checkpoints = []
        for step in protocol.calibration_steps:
            mean = 10.0 - step * 0.1 + arm_number * 0.01
            calibration.append(
                {
                    "step": step,
                    "sample_count": protocol.calibration_sample_count,
                    "mean_target_token_nll": mean,
                    "samples": [{} for _ in range(protocol.calibration_sample_count)],
                }
            )
            if step:
                relative = f"{arm_id}/step_{step:04d}.pt"
                checkpoint = checkpoint_root / relative
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(f"{arm_id}:{step}".encode())
                checkpoints.append(
                    {
                        "step": step,
                        "relative_path": relative,
                        "bytes": checkpoint.stat().st_size,
                        "sha256": file_sha256(checkpoint),
                        "payload_trainable_hash_verified": True,
                    }
                )
        arm = {
            "schema_version": 1,
            "gate": STABILITY_PILOT_GATE_NAME,
            "run_id": "stability_pilot_v1",
            "arm_id": arm_id,
            "learning_rate": rate,
            "status": "passed",
            "manifest_sha256": file_sha256(manifest_path),
            "protocol": protocol.to_record(),
            "initial_trainable_state_sha256": initial_hash,
            "training": {"steps": training_steps},
            "calibration": calibration,
            "checkpoints": checkpoints,
        }
        path = arm_dir / "result.json"
        path.write_text(json.dumps(arm), encoding="utf-8")
        arm_index[arm_id] = {"result_sha256": file_sha256(path)}
    selected_arm = protocol.arm_id(protocol.learning_rates[0])
    selected = {
        "arm_id": selected_arm,
        "learning_rate": protocol.learning_rates[0],
        "step": 16,
        "mean_target_token_nll": 8.4,
        "checkpoint": next(
            row
            for row in json.loads((arms_root / selected_arm / "result.json").read_text())[
                "checkpoints"
            ]
            if row["step"] == 16
        ),
    }
    suite = {
        "schema_version": 1,
        "gate": STABILITY_PILOT_GATE_NAME,
        "run_id": "stability_pilot_v1",
        "status": "passed",
        "manifest_sha256": file_sha256(manifest_path),
        "protocol": protocol.to_record(),
        "arms": arm_index,
        "selected_checkpoint": selected,
        "pass_criteria": {"calibration_improvement_nats": 10.0 - 8.4},
    }
    return suite, manifest, manifest_path, arms_root, checkpoint_root, repo


def test_verifier_accepts_complete_fixture(tmp_path: Path) -> None:
    suite, manifest, manifest_path, arms_root, checkpoint_root, repo = _fixture(tmp_path)
    assert (
        verify_stability_pilot(
            suite,
            manifest=manifest,
            manifest_path=manifest_path,
            arms_root=arms_root,
            checkpoint_root=checkpoint_root,
            repo=repo,
        )
        == []
    )


def test_v2_verifier_accepts_sorted_json_arm_mapping_without_weakening_hash_checks(
    tmp_path: Path,
) -> None:
    suite, manifest, manifest_path, arms_root, checkpoint_root, repo = _fixture(tmp_path)
    serialized_suite = json.loads(json.dumps(suite, sort_keys=True))

    v1_errors = verify_stability_pilot(
        serialized_suite,
        manifest=manifest,
        manifest_path=manifest_path,
        arms_root=arms_root,
        checkpoint_root=checkpoint_root,
        repo=repo,
    )
    assert v1_errors[:4] == list(EXPECTED_V1_ORDER_FALSE_NEGATIVE_ERRORS)
    assert (
        verify_stability_pilot_v2(
            serialized_suite,
            manifest=manifest,
            manifest_path=manifest_path,
            arms_root=arms_root,
            checkpoint_root=checkpoint_root,
            repo=repo,
        )
        == []
    )

    first_arm = StabilityPilotProtocol().arm_id(5e-6)
    serialized_suite["arms"][first_arm]["result_sha256"] = "0" * 64
    errors = verify_stability_pilot_v2(
        serialized_suite,
        manifest=manifest,
        manifest_path=manifest_path,
        arms_root=arms_root,
        checkpoint_root=checkpoint_root,
        repo=repo,
    )
    assert f"suite arm result hash mismatch: {first_arm}" in errors


def test_v2_verifier_requires_exact_known_v1_failure_linkage() -> None:
    suite_hash = "a" * 64
    manifest_hash = "b" * 64
    prior = {
        "schema_version": 1,
        "gate": "independent_stability_pilot_artifact_verification_v1",
        "status": "failed",
        "suite_sha256": suite_hash,
        "manifest_sha256": manifest_hash,
        "dataset_reconstruction_performed": True,
        "error_count": 4,
        "errors": list(EXPECTED_V1_ORDER_FALSE_NEGATIVE_ERRORS),
    }
    assert (
        verify_v1_order_false_negative(
            prior,
            suite_sha256=suite_hash,
            manifest_sha256=manifest_hash,
        )
        == []
    )
    prior["errors"] = ["some unrelated failure"]
    assert verify_v1_order_false_negative(
        prior,
        suite_sha256=suite_hash,
        manifest_sha256=manifest_hash,
    )


def test_verifier_detects_source_checkpoint_and_selection_tampering(tmp_path: Path) -> None:
    suite, manifest, manifest_path, arms_root, checkpoint_root, repo = _fixture(tmp_path)
    (repo / STABILITY_PILOT_SOURCE_LOCK_PATHS[0]).write_text("tampered\n", encoding="utf-8")
    first_checkpoint = next(checkpoint_root.rglob("*.pt"))
    first_checkpoint.write_bytes(b"tampered")
    suite["selected_checkpoint"]["step"] = 8

    errors = verify_stability_pilot(
        suite,
        manifest=manifest,
        manifest_path=manifest_path,
        arms_root=arms_root,
        checkpoint_root=checkpoint_root,
        repo=repo,
    )

    assert any("source hash mismatch" in error for error in errors)
    assert any("checkpoint" in error and "mismatch" in error for error in errors)
    assert any("selection rule" in error for error in errors)


def test_verifier_detects_nonfinite_training_and_initialization_drift(tmp_path: Path) -> None:
    suite, manifest, manifest_path, arms_root, checkpoint_root, repo = _fixture(tmp_path)
    protocol = StabilityPilotProtocol()
    first_arm = arms_root / protocol.arm_id(protocol.learning_rates[0]) / "result.json"
    arm = json.loads(first_arm.read_text())
    arm["training"]["steps"][0]["target_token_nll"] = float("nan")
    arm["initial_trainable_state_sha256"] = "d" * 64
    first_arm.write_text(json.dumps(arm), encoding="utf-8")
    suite["arms"][protocol.arm_id(protocol.learning_rates[0])]["result_sha256"] = file_sha256(
        first_arm
    )

    errors = verify_stability_pilot(
        suite,
        manifest=manifest,
        manifest_path=manifest_path,
        arms_root=arms_root,
        checkpoint_root=checkpoint_root,
        repo=repo,
    )

    assert any("target token NLL is non-finite" in error for error in errors)
    assert any("identical trainable-state hash" in error for error in errors)

