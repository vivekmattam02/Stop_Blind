"""Fast integrity tests for the one-real-batch gate helpers."""

from __future__ import annotations

import torch

from spatialvlm.data.datasets import (
    R2RCEAction,
    R2RCEDataset,
    R2RCEEpisode,
    R2RCEGoal,
    R2RCEGroundTruth,
)
from spatialvlm.training.real_batch_gate import (
    load_trainable_checkpoint,
    save_trainable_checkpoint,
    select_probe_episode,
    tensor_state_sha256,
    trainable_state_dict,
)


def _episode(episode_id: str) -> R2RCEEpisode:
    return R2RCEEpisode(
        episode_id=episode_id,
        trajectory_id=episode_id,
        scene_id="scene",
        instruction_text="go",
        instruction_tokens=(1,),
        goals=(R2RCEGoal(position=(1.0, 0.0, 0.0), radius=3.0),),
        start_position=(0.0, 0.0, 0.0),
        start_rotation=(0.0, 0.0, 0.0, 1.0),
        reference_path=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        geodesic_distance=1.0,
        raw={"episode_id": episode_id},
    )


def _ground_truth(episode_id: str, action: R2RCEAction) -> R2RCEGroundTruth:
    return R2RCEGroundTruth(
        episode_id=episode_id,
        actions=(action, R2RCEAction.STOP),
        forward_steps=int(action == R2RCEAction.MOVE_FORWARD),
        locations=((0.0, 0.0, 0.0),),
    )


def test_probe_selection_is_lowest_non_stop_episode() -> None:
    episodes = (_episode("10"), _episode("2"), _episode("1"))
    ground_truth = {
        "10": _ground_truth("10", R2RCEAction.MOVE_FORWARD),
        "2": _ground_truth("2", R2RCEAction.TURN_LEFT),
        "1": _ground_truth("1", R2RCEAction.STOP),
    }
    dataset = R2RCEDataset(
        records=[episode.raw for episode in episodes],
        official_episodes=episodes,
        ground_truth=ground_truth,
    )

    episode, action = select_probe_episode(dataset)
    assert episode.episode_id == "2"
    assert action == R2RCEAction.TURN_LEFT


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)


def test_compact_checkpoint_roundtrip_restores_only_trainable_state(tmp_path) -> None:
    model = _Tiny()
    optimizer = torch.optim.AdamW([model.weight], lr=0.1)
    state = trainable_state_dict(model)
    original_hash = tensor_state_sha256(state)
    checkpoint = tmp_path / "step1.pt"
    info = save_trainable_checkpoint(
        checkpoint, model=model, optimizer=optimizer, metadata={"gate": "test"}
    )

    with torch.no_grad():
        model.weight.zero_()
        model.frozen.fill_(9.0)
    payload = load_trainable_checkpoint(checkpoint, model=model, optimizer=optimizer)

    assert model.weight.tolist() == [1.0, 2.0]
    assert model.frozen.tolist() == [9.0]
    assert payload["trainable_state_sha256"] == original_hash
    assert info["trainable_tensor_count"] == 1
