"""Adversarial CPU tests for NATIVE-HISTORY-SELECT-v1 primitives."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from spatialvlm.data.datasets import (
    R2RCEAction,
    R2RCEDataset,
    R2RCEEpisode,
    R2RCEGoal,
    R2RCEGroundTruth,
)
from spatialvlm.training.native_history_select import (
    HISTORY_ARMS,
    NativeHistoryProtocol,
    causal_logit_diagnostics,
    diagnostic_float,
    load_history_checkpoint,
    optimizer_tensor_diagnostics,
    paired_bootstrap_improvement,
    save_history_checkpoint,
    select_arm_checkpoint,
    select_history_arm,
    select_native_history_samples,
    tensor_collection_diagnostics,
)


def _dataset(trajectories: int = 80) -> R2RCEDataset:
    episodes: list[R2RCEEpisode] = []
    ground_truth: dict[str, R2RCEGroundTruth] = {}
    actions = (
        R2RCEAction.MOVE_FORWARD,
        R2RCEAction.TURN_LEFT,
        R2RCEAction.TURN_RIGHT,
        R2RCEAction.MOVE_FORWARD,
        R2RCEAction.TURN_LEFT,
        R2RCEAction.TURN_RIGHT,
        R2RCEAction.STOP,
    )
    for trajectory in range(trajectories):
        for instruction_index in range(3):
            episode_id = str(trajectory * 3 + instruction_index)
            episode = R2RCEEpisode(
                episode_id=episode_id,
                trajectory_id=str(trajectory),
                scene_id=f"mp3d/scene{trajectory % 10}/scene{trajectory % 10}.glb",
                instruction_text=f"Instruction {instruction_index} for trajectory {trajectory}",
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
                forward_steps=2,
                locations=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            )
    return R2RCEDataset(
        records=[episode.raw for episode in episodes],
        official_episodes=episodes,
        ground_truth=ground_truth,
    )


def _tiny_protocol() -> NativeHistoryProtocol:
    return replace(
        NativeHistoryProtocol(),
        planned_updates=32,
        train_samples_per_action=8,
        calibration_samples_per_action=2,
        calibration_steps=(0, 16, 32),
        checkpoint_interval=16,
        bootstrap_samples=200,
    )


def test_protocol_freezes_matched_arms_budget_and_sealed_splits() -> None:
    protocol = NativeHistoryProtocol()
    record = protocol.to_record()

    assert protocol.planned_updates == protocol.train_sample_count == 2_048
    assert protocol.calibration_sample_count == 128
    assert protocol.checkpoint_steps[0] == 128
    assert protocol.checkpoint_steps[-1] == 2_048
    assert record["arms"] == list(HISTORY_ARMS)
    assert record["development_splits_used"] == ["train"]
    assert record["sealed_splits"] == ["val_seen", "val_unseen"]


def test_protocol_rejects_padding_and_budget_drift() -> None:
    with pytest.raises(ValueError, match="one distinct train decision"):
        replace(NativeHistoryProtocol(), planned_updates=2_047)
    with pytest.raises(ValueError, match="decision step"):
        replace(NativeHistoryProtocol(), min_decision_step=2)


def test_invalid_nll_diagnostics_preserve_nonfinite_values_and_causal_row() -> None:
    logits = torch.zeros((1, 3, 5), dtype=torch.bfloat16)
    logits[0, 1, 3] = 2.0
    record = causal_logit_diagnostics(logits, action_position=2, target_token_id=3)

    expected_nll = torch.logsumexp(logits[0, 1].float(), dim=-1).item() - 2.0
    assert record["valid_index_contract"] is True
    assert record["finite_logit_count"] == 5
    assert record["nonfinite_logit_count"] == 0
    assert record["target_logit"] == 2.0
    assert record["reconstructed_target_nll"] == pytest.approx(expected_nll)
    assert diagnostic_float(float("nan")) == "nan"
    assert diagnostic_float(float("inf")) == "inf"


def test_tensor_and_optimizer_diagnostics_detect_nonfinite_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.grad = torch.tensor([0.5, -0.5])
    optimizer.step()

    parameter_record = tensor_collection_diagnostics([("weight", parameter)])
    optimizer_record = optimizer_tensor_diagnostics(optimizer)
    assert parameter_record["all_finite"] is True
    assert optimizer_record["exp_avg"]["all_finite"] is True
    assert optimizer_record["exp_avg_sq"]["all_finite"] is True

    with torch.no_grad():
        parameter[0] = float("nan")
    parameter_record = tensor_collection_diagnostics([("weight", parameter)])
    assert parameter_record["all_finite"] is False
    assert parameter_record["nonfinite_element_count"] == 1


def test_selection_is_deterministic_balanced_disjoint_and_unpadded() -> None:
    protocol = _tiny_protocol()
    first = select_native_history_samples(_dataset(), protocol=protocol)
    second = select_native_history_samples(_dataset(), protocol=protocol)

    assert first == second
    assert len(first) == protocol.train_sample_count + protocol.calibration_sample_count
    assert len({sample.trajectory_id for sample in first}) == len(first)
    assert all(sample.action_step_index >= 3 for sample in first)
    assert all(len(sample.replay_prefix_action_ids) >= 3 for sample in first)
    assert all(0 not in sample.replay_prefix_action_ids for sample in first)
    assert all(
        sample.instruction_sha256
        == __import__("hashlib").sha256(sample.instruction_text.encode("utf-8")).hexdigest()
        for sample in first
    )
    for split, expected in (("train", 8), ("calibration", 2)):
        for action in R2RCEAction:
            assert (
                sum(
                    sample.split == split and sample.target_action_id == int(action)
                    for sample in first
                )
                == expected
            )


def test_selection_fails_closed_when_unique_trajectory_budget_is_impossible() -> None:
    with pytest.raises(ValueError, match="Not enough unique trajectories"):
        select_native_history_samples(_dataset(20), protocol=_tiny_protocol())


def test_arm_checkpoint_and_paired_selection_are_deterministic() -> None:
    rows = [
        {"step": 0, "mean_target_token_nll": 2.0},
        {"step": 16, "mean_target_token_nll": 1.0},
        {"step": 32, "mean_target_token_nll": 1.0},
    ]
    assert select_arm_checkpoint(rows) == rows[1]

    baseline = [2.0] * 8
    improved = [1.9] * 8
    bootstrap = paired_bootstrap_improvement(baseline, improved, samples=200, seed=7)
    assert bootstrap["mean_improvement_nats"] == pytest.approx(0.1)
    assert bootstrap["ci95_lower_nats"] > 0.0

    protocol = replace(_tiny_protocol(), calibration_samples_per_action=2)
    decision = select_history_arm(
        {
            "H0_CURRENT": [2.0] * 8,
            "H1_PREVIOUS_ACTION": [1.9] * 8,
            "H4_CAUSAL": [1.8] * 8,
        },
        protocol=protocol,
    )
    assert decision["selected_arm"] == "H4_CAUSAL"
    assert all(row["accepted"] for row in decision["comparisons"])


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)


def test_checkpoint_restores_trainables_optimizer_rng_and_cursor(tmp_path) -> None:
    model = _Tiny()
    optimizer = torch.optim.AdamW([model.weight], lr=0.01)
    metadata = {"manifest_sha256": "a" * 64, "cache_manifest_sha256": "b" * 64}
    checkpoint = tmp_path / "step_0016.pt"
    info = save_history_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        metadata=metadata,
        cursor={"arm_id": "H0_CURRENT", "update_step": 16, "next_train_index": 16},
    )
    with torch.no_grad():
        model.weight.zero_()
        model.frozen.fill_(9.0)
    payload = load_history_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_metadata=metadata,
    )

    assert model.weight.tolist() == [1.0, 2.0]
    assert model.frozen.tolist() == [9.0]
    assert payload["cursor"]["update_step"] == 16
    assert info["update_step"] == 16


def test_checkpoint_rejects_identity_tamper(tmp_path) -> None:
    model = _Tiny()
    optimizer = torch.optim.AdamW([model.weight], lr=0.01)
    checkpoint = tmp_path / "step.pt"
    save_history_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        metadata={"manifest_sha256": "a" * 64},
        cursor={"update_step": 1},
    )
    with pytest.raises(ValueError, match="metadata identity"):
        load_history_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_metadata={"manifest_sha256": "b" * 64},
        )
