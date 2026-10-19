"""Tests for the frozen train-only development SFT protocol."""

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
from spatialvlm.training.development_sft import (
    DevelopmentSFTProtocol,
    load_development_checkpoint,
    save_development_checkpoint,
    select_development_checkpoint,
    select_development_samples,
    update_early_stopping_state,
)


def _dataset(trajectories: int = 1_120) -> R2RCEDataset:
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
            scene_id=f"mp3d/scene{index % 61}/scene{index % 61}.glb",
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


def test_protocol_freezes_budget_calibration_and_sealed_splits() -> None:
    protocol = DevelopmentSFTProtocol()
    record = protocol.to_record()

    assert protocol.train_sample_count == protocol.planned_updates == 1_024
    assert protocol.calibration_sample_count == 64
    assert protocol.checkpoint_steps[:3] == (16, 32, 64)
    assert protocol.checkpoint_steps[-1] == 1_024
    assert record["development_splits_used"] == ["train"]
    assert record["sealed_splits"] == ["val_seen", "val_unseen"]
    assert record["accuracy_is_selection_or_stop_condition"] is False


def test_protocol_rejects_budget_and_calibration_drift() -> None:
    with pytest.raises(ValueError, match="one distinct sample"):
        replace(DevelopmentSFTProtocol(), planned_updates=1_023)
    with pytest.raises(ValueError, match="final planned update"):
        replace(DevelopmentSFTProtocol(), calibration_steps=(0, 16, 32))


def test_selection_is_deterministic_balanced_and_excludes_pilot_trajectories() -> None:
    dataset = _dataset()
    excluded = frozenset(str(index) for index in range(24))
    first = select_development_samples(dataset, excluded_trajectory_ids=excluded)
    second = select_development_samples(dataset, excluded_trajectory_ids=excluded)

    assert first == second
    assert len(first) == 1_088
    trajectories = {sample.trajectory_id for sample in first}
    assert len(trajectories) == len(first)
    assert trajectories.isdisjoint(excluded)
    for split, expected in (("train", 256), ("calibration", 16)):
        for action in R2RCEAction:
            assert (
                sum(
                    sample.split == split and sample.target_action_id == int(action)
                    for sample in first
                )
                == expected
            )


def test_selection_fails_closed_when_unique_budget_is_impossible() -> None:
    with pytest.raises(ValueError, match="Not enough unique trajectories"):
        select_development_samples(
            _dataset(100), excluded_trajectory_ids=frozenset(str(index) for index in range(24))
        )


def test_checkpoint_selection_includes_step_zero_and_breaks_ties_earlier() -> None:
    rows = [
        {"step": 0, "mean_target_token_nll": 2.0},
        {"step": 16, "mean_target_token_nll": 1.0},
        {"step": 32, "mean_target_token_nll": 1.0},
    ]
    assert select_development_checkpoint(rows) == rows[1]


def test_early_stopping_obeys_minimum_delta_floor_and_patience() -> None:
    protocol = DevelopmentSFTProtocol()
    best = 2.0
    count = 0
    stops = []
    for step, nll in ((16, 1.98), (32, 1.975), (64, 1.974), (128, 1.973)):
        best, count, stop = update_early_stopping_state(
            previous_best_nll=best,
            previous_non_improving_count=count,
            current_nll=nll,
            current_step=step,
            protocol=protocol,
        )
        stops.append(stop)
    assert stops == [False, False, False, True]


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)


def test_development_checkpoint_strictly_restores_cursor_optimizer_and_tensors(tmp_path) -> None:
    model = _Tiny()
    optimizer = torch.optim.AdamW([model.weight], lr=0.01)
    checkpoint = tmp_path / "step_0016.pt"
    manifest_hash = "a" * 64
    info = save_development_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        run_state={"update_step": 16, "next_train_index": 16},
        metadata={"manifest_sha256": manifest_hash},
    )
    with torch.no_grad():
        model.weight.zero_()
        model.frozen.fill_(9.0)
    payload = load_development_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_manifest_sha256=manifest_hash,
    )

    assert model.weight.tolist() == [1.0, 2.0]
    assert model.frozen.tolist() == [9.0]
    assert payload["run_state"]["update_step"] == 16
    assert info["update_step"] == 16


def test_development_checkpoint_rejects_manifest_mismatch(tmp_path) -> None:
    model = _Tiny()
    optimizer = torch.optim.AdamW([model.weight], lr=0.01)
    checkpoint = tmp_path / "step.pt"
    save_development_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        run_state={"update_step": 1},
        metadata={"manifest_sha256": "a" * 64},
    )
    with pytest.raises(ValueError, match="manifest identity"):
        load_development_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_manifest_sha256="b" * 64,
        )
