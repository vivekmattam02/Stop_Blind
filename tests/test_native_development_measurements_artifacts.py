"""Adversarial tests for native-C0 development-measurement verification."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from scripts.check_native_development_measurements_artifacts import (
    verify_h4_checkpoint,
    verify_h4_event_records,
)
from spatialvlm.training.native_c0_gate import file_sha256, tensor_state_sha256
from spatialvlm.training.native_development_measurements import (
    NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
    NATIVE_H4_HISTORY,
    NATIVE_H4_OBJECTIVE,
    NativeH4BenchmarkProtocol,
)

ACTION_NAMES = ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")


def _samples(protocol: NativeH4BenchmarkProtocol) -> list[dict[str, object]]:
    rows = [
        {
            "split": "train",
            "sample_index": index,
            "episode_id": f"train-{index}",
            "trajectory_id": f"train-trajectory-{index}",
            "target_action": ACTION_NAMES[index % len(ACTION_NAMES)],
        }
        for index in range(protocol.unique_train_decisions)
    ]
    rows.extend(
        {
            "split": "calibration",
            "sample_index": index,
            "episode_id": f"calibration-{index}",
            "trajectory_id": f"calibration-trajectory-{index}",
            "target_action": ACTION_NAMES[index % len(ACTION_NAMES)],
        }
        for index in range(protocol.unique_calibration_decisions)
    )
    return rows


def _events(protocol: NativeH4BenchmarkProtocol) -> list[dict[str, object]]:
    from spatialvlm.training.native_development_measurements import h4_training_sample_order

    samples = _samples(protocol)
    order = h4_training_sample_order([row for row in samples if row["split"] == "train"], protocol)
    events = []
    before = "f" * 64
    for step in range(1, protocol.planned_optimizer_updates + 1):
        start = (step - 1) * protocol.accumulation_steps
        after = f"{step:064x}"
        sample_by_index = {
            int(row["sample_index"]): row for row in samples if row["split"] == "train"
        }
        events.append(
            {
                "gate": NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
                "history_contract": NATIVE_H4_HISTORY,
                "objective": NATIVE_H4_OBJECTIVE,
                "update_step": step,
                "sample_exposure_start": start,
                "sample_exposure_end_exclusive": start + protocol.accumulation_steps,
                "microbatch_count": protocol.accumulation_steps,
                "accumulation_steps": protocol.accumulation_steps,
                "learning_rate": protocol.learning_rate,
                "mean_restricted_action_ce": 1.0,
                "before_trainable_state_sha256": before,
                "after_trainable_state_sha256": after,
                "elapsed_seconds": float(step),
                "global_gradient_norm_before_clip": 2.0,
                "timing_role": (
                    "measured" if step in protocol.timed_update_steps else "warmup_excluded"
                ),
                "gradient_report": {
                    "gradient_tensor_count": 2,
                    "trainable_tensor_count": 2,
                    "nonzero_gradient_tensor_count": 2,
                    "all_gradients_finite": True,
                },
                "update_report": {"changed_tensor_count": 2, "all_updates_finite": True},
                "optimizer_state": {"exp_avg": {"all_finite": True}},
                "microbatches": [
                    {
                        "sample_index": index,
                        "episode_id": sample_by_index[index]["episode_id"],
                        "trajectory_id": sample_by_index[index]["trajectory_id"],
                        "target_action": sample_by_index[index]["target_action"],
                        "restricted_action_ce": 1.0,
                        "action_token_position": 10,
                        "supervised_token_count": 1,
                        "supervised_token_id": 20,
                    }
                    for index in order[start : start + protocol.accumulation_steps]
                ],
            }
        )
        before = after
    return events


def test_event_verifier_accepts_exact_hash_and_unit_chain() -> None:
    protocol = NativeH4BenchmarkProtocol()
    assert (
        verify_h4_event_records(
            _events(protocol),
            protocol=protocol,
            samples=_samples(protocol),
            initial_trainable_sha256="f" * 64,
            final_trainable_sha256=f"{8:064x}",
        )
        == []
    )


def test_event_verifier_rejects_reordered_sample() -> None:
    protocol = NativeH4BenchmarkProtocol()
    events = deepcopy(_events(protocol))
    events[2]["microbatches"][0]["sample_index"] = -1
    errors = verify_h4_event_records(
        events,
        protocol=protocol,
        samples=_samples(protocol),
        initial_trainable_sha256="f" * 64,
        final_trainable_sha256=f"{8:064x}",
    )
    assert "update 3 sample order mismatch" in errors


def test_event_verifier_rejects_warmup_used_as_measured() -> None:
    protocol = NativeH4BenchmarkProtocol()
    events = deepcopy(_events(protocol))
    events[0]["timing_role"] = "measured"
    errors = verify_h4_event_records(
        events,
        protocol=protocol,
        samples=_samples(protocol),
        initial_trainable_sha256="f" * 64,
        final_trainable_sha256=f"{8:064x}",
    )
    assert "update 1 timing role mismatch" in errors


def test_checkpoint_verifier_rejects_tampered_trainable_hash(tmp_path: Path) -> None:
    protocol = NativeH4BenchmarkProtocol()
    state = {"lora_A": torch.tensor([1.0])}
    payload = {
        "gate": NATIVE_DEVELOPMENT_MEASUREMENTS_GATE,
        "manifest_sha256": "manifest",
        "cache_manifest_sha256": "cache",
        "protocol": protocol.to_record(),
        "cursor": {
            "update_step": protocol.planned_optimizer_updates,
            "sample_exposure_end_exclusive": protocol.unique_train_decisions,
        },
        "trainable_state_dict": state,
        "trainable_state_sha256": "tampered",
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(1.0)}}},
        "rng_state": {},
    }
    path = tmp_path / "step_0008.pt"
    torch.save(payload, path)
    record = {
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "trainable_state_sha256": tensor_state_sha256(state),
    }
    errors = verify_h4_checkpoint(
        path,
        record=record,
        manifest_sha256="manifest",
        cache_manifest_sha256="cache",
        protocol=protocol,
    )
    assert "H4 benchmark checkpoint internal trainable hash mismatch" in errors
