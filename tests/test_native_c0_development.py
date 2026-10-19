from __future__ import annotations

import math
import pytest

from spatialvlm.data.datasets import (
    R2RCEAction,
    R2RCEDataset,
    R2RCEEpisode,
    R2RCEGoal,
    R2RCEGroundTruth,
    R2RCEReleaseManifest,
)
from spatialvlm.training.native_c0_development import (
    NativeC0DevelopmentProtocol,
    cache_object_key,
    evaluate_existence_gate,
    scene_cluster_bootstrap_upper_bound,
    select_checkpoint,
    select_native_c0_partitions,
    training_sample_order,
    validate_partition_records,
    validate_protocol_record,
)


def _dataset() -> R2RCEDataset:
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
    for index in range(3_603):
        episode_id = str(index)
        trajectory_id = f"trajectory-{index:04d}"
        episodes.append(
            R2RCEEpisode(
                episode_id=episode_id,
                trajectory_id=trajectory_id,
                scene_id=f"mp3d/scene-{index % 61}/scene-{index % 61}.glb",
                instruction_text=f"instruction {index}",
                instruction_tokens=(),
                start_position=(0.0, 0.0, 0.0),
                start_rotation=(0.0, 0.0, 0.0, 1.0),
                goals=(R2RCEGoal(position=(1.0, 0.0, 0.0), radius=3.0),),
                reference_path=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                geodesic_distance=1.0,
                raw={"episode_id": episode_id},
            )
        )
        ground_truth[episode_id] = R2RCEGroundTruth(
            episode_id=episode_id,
            actions=actions,
            forward_steps=4,
            locations=tuple((float(step), 0.0, 0.0) for step in range(len(actions))),
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


def test_frozen_protocol_reconciles_all_units_and_thresholds() -> None:
    protocol = NativeC0DevelopmentProtocol()
    assert protocol.cached_decisions == 4_224
    assert protocol.reserved_trajectories == 1_152
    assert protocol.effective_batch == 16
    assert protocol.planned_optimizer_updates == 192
    assert protocol.unit_ledger.supports_learning_quality_claim is True
    assert protocol.natural_ce_threshold == pytest.approx(0.87470)
    assert protocol.balanced_ce_threshold == pytest.approx(math.log(4.0) - 0.05)
    assert validate_protocol_record(protocol.to_record()) == []


def test_frozen_protocol_refuses_threshold_or_unit_drift() -> None:
    with pytest.raises(ValueError, match="STOP recall"):
        NativeC0DevelopmentProtocol(stop_recall_floor=1.1)
    with pytest.raises(ValueError, match="single-pass"):
        NativeC0DevelopmentProtocol(accumulation_steps=8)
    with pytest.raises(ValueError, match="end-to-end hard cap"):
        NativeC0DevelopmentProtocol(hard_total_hours=5.0)


def test_partition_builder_matches_counts_and_trajectory_boundary() -> None:
    protocol = NativeC0DevelopmentProtocol()
    records = [row.to_record() for row in select_native_c0_partitions(_dataset())]
    summary = validate_partition_records(records, protocol=protocol)
    assert summary["decision_counts"] == {
        "train": 3_072,
        "selection": 256,
        "primary_evaluation": 640,
        "balanced_diagnostic": 256,
    }
    assert summary["trajectory_counts"]["train"] == 2_451
    assert summary["unique_trajectory_count"] == 3_603
    assert summary["diagnostic_action_counts"] == {
        "STOP": 64,
        "MOVE_FORWARD": 64,
        "TURN_LEFT": 64,
        "TURN_RIGHT": 64,
    }
    order = training_sample_order(records, protocol)
    assert len(order) == 3_072
    assert len(set(order)) == 3_072


def test_cache_key_ignores_run_partition_and_host_but_binds_decision() -> None:
    sample = {
        "split": "train",
        "sample_index": 7,
        "scene_id": "mp3d/scene/scene.glb",
        "trajectory_id": "trajectory",
        "action_step_index": 11,
        "replay_prefix_action_ids": [1, 2, 3],
        "replay_prefix_sha256": "a" * 64,
    }
    key = cache_object_key(
        sample,
        episodes_sha256="b" * 64,
        ground_truth_sha256="c" * 64,
        scene_archive_sha256="d" * 64,
    )
    same_decision = {**sample, "split": "primary_evaluation", "sample_index": 99}
    assert cache_object_key(
        same_decision,
        episodes_sha256="b" * 64,
        ground_truth_sha256="c" * 64,
        scene_archive_sha256="d" * 64,
    ) == key
    changed = {**sample, "action_step_index": 12}
    assert cache_object_key(
        changed,
        episodes_sha256="b" * 64,
        ground_truth_sha256="c" * 64,
        scene_archive_sha256="d" * 64,
    ) != key
    assert cache_object_key(
        sample,
        episodes_sha256="b" * 64,
        ground_truth_sha256="c" * 64,
        scene_archive_sha256="e" * 64,
    ) != key


def test_checkpoint_selection_uses_minimum_ce_and_earlier_tie() -> None:
    rows = [
        {"update_step": 0, "summary": {"mean_restricted_action_ce": 2.0}},
        {"update_step": 32, "summary": {"mean_restricted_action_ce": 1.0}},
        {"update_step": 64, "summary": {"mean_restricted_action_ce": 1.0}},
    ]
    assert select_checkpoint(rows)["selected_update"] == 32


def _decisions(count: int, ce: float, *, stop_correct: int | None = None):
    actions = list(R2RCEAction)
    rows = []
    for index in range(count):
        action = actions[index % len(actions)].name
        predicted = action
        if action == "STOP" and stop_correct is not None:
            stop_ordinal = sum(row["target_action"] == "STOP" for row in rows)
            predicted = "STOP" if stop_ordinal < stop_correct else "MOVE_FORWARD"
        rows.append(
            {
                "scene_id": f"scene-{index % 8}",
                "target_action": action,
                "predicted_action": predicted,
                "restricted_action_ce": ce,
            }
        )
    return rows


def test_scene_bootstrap_is_deterministic_and_gate_is_conjunctive() -> None:
    primary = _decisions(640, 0.8)
    diagnostic = _decisions(256, 1.0, stop_correct=16)
    first = scene_cluster_bootstrap_upper_bound(primary, samples=200, seed=17)
    second = scene_cluster_bootstrap_upper_bound(primary, samples=200, seed=17)
    assert first == second
    result = evaluate_existence_gate(
        primary_decisions=primary,
        diagnostic_decisions=diagnostic,
    )
    assert result["status"] == "passed"
    assert result["balanced_diagnostic"]["stop_recall"] == 0.25
    failed = evaluate_existence_gate(
        primary_decisions=primary,
        diagnostic_decisions=_decisions(256, 1.0, stop_correct=15),
    )
    assert failed["status"] == "not_demonstrated_at_frozen_budget"
    assert failed["conditions"]["stop_recall_at_or_above_non_mute_floor"] is False


def test_partition_validator_rejects_cross_split_trajectory_leakage() -> None:
    records = [row.to_record() for row in select_native_c0_partitions(_dataset())]
    selection = next(row for row in records if row["split"] == "selection")
    train_index = next(index for index, row in enumerate(records) if row["split"] == "train")
    records[train_index] = {
        **records[train_index],
        "trajectory_id": selection["trajectory_id"],
    }
    with pytest.raises(ValueError, match="cross partitions"):
        validate_partition_records(records)
