"""Tests for the inventory and causal-H4 benchmark contracts."""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from scripts.run_spark_native_development_measurements import _prerequisites, _restricted_loss
from spatialvlm.data.datasets import (
    R2RCEAction,
    R2RCEDataset,
    R2RCEEpisode,
    R2RCEGoal,
    R2RCEGroundTruth,
    R2RCEReleaseManifest,
)
from spatialvlm.eval.action_decision import ActionTokenCodebook
from spatialvlm.training.native_c0_gate import file_sha256
from spatialvlm.training.native_development_measurements import (
    NativeDataInventoryProtocol,
    NativeH4BenchmarkProtocol,
    build_data_policy_inventory,
    h4_training_sample_order,
    summarize_h4_timings,
    validate_h4_protocol_record,
    validate_inventory_record,
)


def test_restricted_loss_uses_public_action_codebook_api() -> None:
    codebook = ActionTokenCodebook(
        shared_prefix_token_ids=(1, 2),
        action_token_ids=(10, 11, 12, 13),
        target_token_ids=((1, 2, 10), (1, 2, 11), (1, 2, 12), (1, 2, 13)),
        tokenizer_class="StubTokenizer",
        tokenizer_name_or_path="stub",
    )
    logits = torch.full((1, 3, 32), -4.0)
    logits[0, 1, 10:14] = torch.tensor([-1.0, 0.0, 2.0, 1.0])
    observed = _restricted_loss(
        logits,
        action_position=2,
        codebook=codebook,
        target_action=R2RCEAction.TURN_LEFT,
    )
    expected = functional.cross_entropy(
        torch.tensor([[-1.0, 0.0, 2.0, 1.0]]),
        torch.tensor([2]),
    )
    assert torch.equal(observed, expected)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _prerequisite_args(tmp_path: Path) -> Namespace:
    prior_manifest = _write_json(
        tmp_path / "prior_manifest.json",
        {
            "gate": "native_c0_development_measurements_v1",
            "run_id": "native_development_measurements_v1",
            "status": "frozen",
        },
    )
    prior_failure = _write_json(
        tmp_path / "prior_failure.json",
        {
            "gate": "native_c0_development_measurements_v1",
            "status": "failed",
            "error_type": "AttributeError",
            "error": "'ActionTokenCodebook' object has no attribute 'tokens'",
        },
    )
    processor_result = _write_json(
        tmp_path / "processor_result.json",
        {"gate": "native_qwen_c0_history_processor_preflight_v4", "status": "passed"},
    )
    processor_verification = _write_json(
        tmp_path / "processor_verification.json",
        {
            "gate": "independent_native_qwen_c0_history_artifact_verification_v4",
            "status": "passed",
            "error_count": 0,
            "result_sha256": file_sha256(processor_result),
        },
    )
    optimizer_suite = _write_json(
        tmp_path / "optimizer_suite.json",
        {
            "gate": "native_qwen_optimizer_admission_v1",
            "profile": "main",
            "status": "passed",
        },
    )
    optimizer_verification = _write_json(
        tmp_path / "optimizer_verification.json",
        {
            "gate": "independent_native_qwen_optimizer_admission_verification_v1",
            "profile": "main",
            "status": "passed",
            "error_count": 0,
            "suite_sha256": file_sha256(optimizer_suite),
        },
    )
    return Namespace(
        prior_manifest=prior_manifest,
        prior_failure=prior_failure,
        processor_result=processor_result,
        processor_verification=processor_verification,
        optimizer_suite=optimizer_suite,
        optimizer_verification=optimizer_verification,
    )


def test_v2_prerequisites_hash_link_the_exact_v1_failure(tmp_path: Path) -> None:
    args = _prerequisite_args(tmp_path)
    record = _prerequisites(args)
    assert record["prior_attempt"]["failure_sha256"] == file_sha256(args.prior_failure)
    assert record["prior_attempt"]["manifest_sha256"] == file_sha256(args.prior_manifest)


def test_v2_prerequisites_reject_a_different_v1_failure(tmp_path: Path) -> None:
    args = _prerequisite_args(tmp_path)
    _write_json(
        args.prior_failure,
        {
            "gate": "native_c0_development_measurements_v1",
            "status": "failed",
            "error_type": "RuntimeError",
            "error": "some other failure",
        },
    )
    with pytest.raises(RuntimeError, match="codebook-API failure"):
        _prerequisites(args)


def _dataset(trajectory_count: int = 8, instructions_per_trajectory: int = 2) -> R2RCEDataset:
    episodes = []
    ground_truth = {}
    actions = (
        R2RCEAction.MOVE_FORWARD,
        R2RCEAction.MOVE_FORWARD,
        R2RCEAction.MOVE_FORWARD,
        R2RCEAction.TURN_LEFT,
        R2RCEAction.TURN_RIGHT,
        R2RCEAction.MOVE_FORWARD,
        R2RCEAction.STOP,
    )
    for trajectory in range(trajectory_count):
        for instruction in range(instructions_per_trajectory):
            episode_id = f"{trajectory}-{instruction}"
            episode = R2RCEEpisode(
                episode_id=episode_id,
                trajectory_id=f"trajectory-{trajectory}",
                scene_id=f"scene-{trajectory % 2}",
                instruction_text=f"instruction {trajectory} {instruction}",
                instruction_tokens=(),
                start_position=(0.0, 0.0, 0.0),
                start_rotation=(0.0, 0.0, 0.0, 1.0),
                goals=(R2RCEGoal(position=(1.0, 0.0, 0.0), radius=3.0),),
                reference_path=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                geodesic_distance=1.0,
                raw={"episode_id": episode_id},
            )
            episodes.append(episode)
            ground_truth[episode_id] = R2RCEGroundTruth(
                episode_id=episode_id,
                actions=actions,
                forward_steps=4,
                locations=tuple((float(index), 0.0, 0.0) for index in range(7)),
            )
    manifest = R2RCEReleaseManifest(
        release_id="fixture",
        split="train",
        episodes_path="/fixture/train.json.gz",
        episodes_sha256="episodes",
        source_episode_count=len(episodes),
        selected_episode_count=len(episodes),
        ground_truth_path="/fixture/train_gt.json.gz",
        ground_truth_sha256="ground-truth",
        ground_truth_count=len(episodes),
    )
    return R2RCEDataset(
        records=[],
        official_episodes=episodes,
        ground_truth=ground_truth,
        release_manifest=manifest,
    )


def test_inventory_counts_physical_and_instruction_units_separately() -> None:
    inventory = build_data_policy_inventory(_dataset())
    summary = inventory["summary"]
    assert summary["scene_count"] == 2
    assert summary["physical_trajectory_count"] == 8
    assert summary["instruction_episode_count"] == 16
    assert summary["eligible_physical_action_counts"] == {
        "STOP": 8,
        "MOVE_FORWARD": 8,
        "TURN_LEFT": 8,
        "TURN_RIGHT": 8,
    }
    assert summary["eligible_instruction_action_counts"] == {
        "STOP": 16,
        "MOVE_FORWARD": 16,
        "TURN_LEFT": 16,
        "TURN_RIGHT": 16,
    }
    assert summary["maximum_balanced_globally_disjoint_trajectories_per_action"] == 2
    assert inventory["baselines"]["uniform_balanced_ce_nats"] == pytest.approx(1.38629436112)
    assert validate_inventory_record(inventory, _dataset()) == []


def test_inventory_rejects_instruction_episodes_with_different_actions() -> None:
    dataset = _dataset(trajectory_count=1)
    corrupted = dataset.ground_truth
    row = corrupted["0-1"]
    corrupted["0-1"] = replace(
        row,
        actions=(
            R2RCEAction.MOVE_FORWARD,
            R2RCEAction.MOVE_FORWARD,
            R2RCEAction.MOVE_FORWARD,
            R2RCEAction.STOP,
        ),
    )
    bad = R2RCEDataset(
        records=[],
        official_episodes=dataset.official_episodes,
        ground_truth=corrupted,
        release_manifest=dataset.release_manifest,
    )
    with pytest.raises(ValueError, match="disagree on actions"):
        build_data_policy_inventory(bad)


def test_h4_protocol_reconciles_exact_single_pass_units() -> None:
    protocol = NativeH4BenchmarkProtocol()
    assert protocol.unique_train_decisions == 128
    assert protocol.unique_calibration_decisions == 16
    assert protocol.effective_batch == 16
    assert protocol.planned_optimizer_updates == 8
    assert protocol.timed_update_steps == (3, 4, 5, 6, 7, 8)
    assert protocol.unit_ledger.supports_learning_quality_claim is True
    assert validate_h4_protocol_record(protocol.to_record()) == []


def test_h4_protocol_rejects_inconsistent_timing_units() -> None:
    with pytest.raises(ValueError, match="warmup plus measured"):
        NativeH4BenchmarkProtocol(measured_updates=5)


def test_h4_training_order_is_deterministic_and_single_pass() -> None:
    protocol = NativeH4BenchmarkProtocol()
    samples = [{"sample_index": index} for index in range(protocol.unique_train_decisions)]
    first = h4_training_sample_order(samples, protocol)
    second = h4_training_sample_order(samples, protocol)
    assert first == second
    assert sorted(first) == list(range(protocol.unique_train_decisions))


def test_h4_timing_summary_excludes_declared_warmup() -> None:
    protocol = NativeH4BenchmarkProtocol()
    events = [
        {"update_step": step, "elapsed_seconds": float(step)}
        for step in range(1, protocol.planned_optimizer_updates + 1)
    ]
    summary = summarize_h4_timings(
        events,
        protocol=protocol,
        model_load_seconds=10.0,
        cache_prepare_seconds=20.0,
        calibration_seconds=8.0,
        checkpoint_seconds=4.0,
    )
    assert summary["timed_update_steps"] == [3, 4, 5, 6, 7, 8]
    assert summary["timed_update_p50_seconds"] == 5.5
    assert summary["timed_update_p95_nearest_rank_seconds"] == 8.0
    assert summary["calibration_seconds_per_decision"] == 0.5


def test_inventory_protocol_requires_three_real_history_actions() -> None:
    with pytest.raises(ValueError, match="three preceding actions"):
        NativeDataInventoryProtocol(min_decision_step=2)
