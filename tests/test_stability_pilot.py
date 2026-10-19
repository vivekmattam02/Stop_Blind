"""Tests for the frozen optimization/stability pilot design."""

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
from spatialvlm.eval.action_decision import build_action_token_codebook
from spatialvlm.training.stability_pilot import (
    StabilityPilotProtocol,
    action_only_labels,
    select_pilot_checkpoint,
    select_pilot_samples,
)


class _Tokenizer:
    name_or_path = "fixture/tokenizer"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [100, 101, 10 + int(text[-1])]


def _dataset(trajectories: int = 40) -> R2RCEDataset:
    episodes = []
    ground_truth = {}
    actions = (
        R2RCEAction.MOVE_FORWARD,
        R2RCEAction.TURN_LEFT,
        R2RCEAction.TURN_RIGHT,
        R2RCEAction.STOP,
    )
    for index in range(trajectories):
        episode_id = str(index * 3)
        episode = R2RCEEpisode(
            episode_id=episode_id,
            trajectory_id=str(index),
            scene_id=f"mp3d/scene{index % 30}/scene{index % 30}.glb",
            instruction_text=f"Instruction for trajectory {index}",
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
            forward_steps=1,
            locations=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        )
    return R2RCEDataset(
        records=[episode.raw for episode in episodes],
        official_episodes=episodes,
        ground_truth=ground_truth,
    )


def test_default_protocol_is_bounded_and_seals_benchmark_splits() -> None:
    protocol = StabilityPilotProtocol()
    record = protocol.to_record()

    assert protocol.train_sample_count == protocol.updates_per_arm == 16
    assert protocol.calibration_sample_count == 8
    assert protocol.checkpoint_steps == (8, 16)
    assert record["development_splits_used"] == ["train"]
    assert record["sealed_splits"] == ["val_seen", "val_unseen"]
    assert record["accuracy_is_selection_or_pass_condition"] is False


def test_protocol_rejects_budget_and_calibration_drift() -> None:
    with pytest.raises(ValueError, match="final update"):
        replace(StabilityPilotProtocol(), updates_per_arm=15)
    with pytest.raises(ValueError, match="Calibration must run"):
        replace(StabilityPilotProtocol(), calibration_steps=(0, 16))


def test_selection_is_deterministic_balanced_and_trajectory_disjoint() -> None:
    dataset = _dataset()
    first = select_pilot_samples(dataset)
    second = select_pilot_samples(dataset)

    assert first == second
    assert len(first) == 24
    assert len({sample.trajectory_id for sample in first}) == len(first)
    for split, expected in (("train", 4), ("calibration", 2)):
        for action in R2RCEAction:
            assert (
                sum(
                    sample.split == split and sample.target_action_id == int(action)
                    for sample in first
                )
                == expected
            )


def test_selection_fails_when_unique_trajectory_budget_is_impossible() -> None:
    with pytest.raises(ValueError, match="Not enough unique trajectories"):
        select_pilot_samples(_dataset(trajectories=4))


def test_action_only_labels_supervise_exactly_the_decision_token() -> None:
    codebook = build_action_token_codebook(_Tokenizer())
    rendered = {
        "input_ids": torch.tensor([1, 2, 100, 101, 12, 99]),
        "labels": torch.tensor([-100, -100, 100, 101, 12, 99]),
        "target_start_idx": 2,
    }

    labels, position = action_only_labels(
        rendered,
        action=R2RCEAction.TURN_LEFT,
        codebook=codebook,
    )

    assert position == 4
    assert labels.tolist() == [-100, -100, -100, -100, 12, -100]


def test_checkpoint_selection_uses_nll_then_lower_lr_then_earlier_step() -> None:
    candidates = [
        {"mean_target_token_nll": 3.0, "learning_rate": 1e-5, "step": 16},
        {"mean_target_token_nll": 2.0, "learning_rate": 5e-5, "step": 8},
        {"mean_target_token_nll": 2.0, "learning_rate": 1e-5, "step": 16},
        {"mean_target_token_nll": 2.0, "learning_rate": 1e-5, "step": 8},
    ]
    assert select_pilot_checkpoint(candidates) == candidates[-1]
