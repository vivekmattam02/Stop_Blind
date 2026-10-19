"""Frozen design primitives for the real RGB-D optimization/stability pilot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset, R2RCEEpisode
from spatialvlm.eval.action_decision import ActionTokenCodebook, locate_action_token

STABILITY_PILOT_PROTOCOL_VERSION = 1
STABILITY_PILOT_SCHEMA_VERSION = 1
STABILITY_PILOT_GATE_NAME = "spark_r2r_ce_optimization_stability_pilot_v1"


@dataclass(frozen=True)
class StabilityPilotProtocol:
    """Pre-registered optimization pilot budget and stopping rules."""

    learning_rates: tuple[float, ...] = (5e-6, 1e-5, 5e-5)
    updates_per_arm: int = 16
    checkpoint_steps: tuple[int, ...] = (8, 16)
    calibration_steps: tuple[int, ...] = (0, 8, 16)
    train_samples_per_action: int = 4
    calibration_samples_per_action: int = 2
    max_replay_prefix_actions: int = 96
    seed: int = 20260823
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    hard_max_gradient_norm: float = 1_000_000.0
    hard_max_token_nll: float = 100.0
    hard_max_peak_allocated_bytes: int = 64 * 1024**3
    max_arm_hours: float = 2.0
    minimum_calibration_improvement_nats: float = 0.01

    def __post_init__(self) -> None:
        if not self.learning_rates or any(rate <= 0.0 for rate in self.learning_rates):
            raise ValueError("Pilot learning rates must be positive and non-empty.")
        if len(set(self.learning_rates)) != len(self.learning_rates):
            raise ValueError("Pilot learning rates must be unique.")
        if self.updates_per_arm <= 0:
            raise ValueError("updates_per_arm must be positive.")
        if self.checkpoint_steps != tuple(sorted(set(self.checkpoint_steps))):
            raise ValueError("checkpoint_steps must be sorted and unique.")
        if not self.checkpoint_steps or self.checkpoint_steps[-1] != self.updates_per_arm:
            raise ValueError("The final update must be a checkpoint step.")
        if any(step <= 0 or step > self.updates_per_arm for step in self.checkpoint_steps):
            raise ValueError("Checkpoint steps must lie within the update budget.")
        expected_calibration = (0, *self.checkpoint_steps)
        if self.calibration_steps != expected_calibration:
            raise ValueError("Calibration must run at step 0 and every checkpoint step.")
        if self.train_samples_per_action <= 0 or self.calibration_samples_per_action <= 0:
            raise ValueError("Per-action sample counts must be positive.")
        if self.train_sample_count != self.updates_per_arm:
            raise ValueError("The pilot requires exactly one distinct train sample per update.")
        if self.max_replay_prefix_actions < 0:
            raise ValueError("max_replay_prefix_actions must be non-negative.")
        if self.max_grad_norm <= 0.0 or self.hard_max_gradient_norm <= self.max_grad_norm:
            raise ValueError("Gradient thresholds are invalid.")
        if self.hard_max_token_nll <= 0.0 or self.hard_max_peak_allocated_bytes <= 0:
            raise ValueError("Hard loss and memory limits must be positive.")
        if self.max_arm_hours <= 0.0:
            raise ValueError("max_arm_hours must be positive.")
        if self.minimum_calibration_improvement_nats < 0.0:
            raise ValueError("Minimum calibration improvement cannot be negative.")

    @property
    def train_sample_count(self) -> int:
        return self.train_samples_per_action * len(R2RCEAction)

    @property
    def calibration_sample_count(self) -> int:
        return self.calibration_samples_per_action * len(R2RCEAction)

    def arm_id(self, learning_rate: float) -> str:
        if learning_rate not in self.learning_rates:
            raise ValueError(f"Learning rate {learning_rate} is not in the frozen pilot.")
        scientific = f"{learning_rate:.0e}".replace("-", "m").replace("+", "p")
        return f"lr_{scientific}"

    def to_record(self) -> dict[str, Any]:
        record = json.loads(json.dumps(asdict(self)))
        record.update(
            {
                "protocol_version": STABILITY_PILOT_PROTOCOL_VERSION,
                "gate": STABILITY_PILOT_GATE_NAME,
                "train_sample_count": self.train_sample_count,
                "calibration_sample_count": self.calibration_sample_count,
                "arm_ids": [self.arm_id(rate) for rate in self.learning_rates],
                "objective": "single_shared_position_official_action_id_token_v1",
                "checkpoint_selection_rule": (
                    "minimum held-out train-trajectory mean target action-token NLL; exact ties "
                    "choose lower learning rate, then earlier checkpoint"
                ),
                "accuracy_is_selection_or_pass_condition": False,
                "development_splits_used": ["train"],
                "sealed_splits": ["val_seen", "val_unseen"],
            }
        )
        return record


@dataclass(frozen=True)
class PilotSample:
    """One deterministic observation/action decision used by the pilot."""

    split: str
    sample_index: int
    episode_id: str
    trajectory_id: str
    scene_id: str
    instruction_sha256: str
    action_step_index: int
    target_action_id: int
    target_action: str
    replay_prefix_action_ids: tuple[int, ...]
    replay_prefix_sha256: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["replay_prefix_action_ids"] = list(self.replay_prefix_action_ids)
        return record


@dataclass(frozen=True)
class _Candidate:
    episode: R2RCEEpisode
    action: R2RCEAction
    step_index: int
    replay_prefix: tuple[R2RCEAction, ...]
    rank: str


def _episode_sort_key(episode: R2RCEEpisode) -> tuple[int, int | str]:
    try:
        return (0, int(episode.episode_id))
    except ValueError:
        return (1, episode.episode_id)


def _rank(seed: int, episode: R2RCEEpisode, step_index: int, action: R2RCEAction) -> str:
    identity = (
        f"stability-pilot-v{STABILITY_PILOT_PROTOCOL_VERSION}|{seed}|"
        f"{episode.trajectory_id}|{episode.episode_id}|{step_index}|{int(action)}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_pilot_samples(
    dataset: R2RCEDataset,
    protocol: StabilityPilotProtocol | None = None,
) -> tuple[PilotSample, ...]:
    """Select action-balanced train/calibration samples without trajectory overlap.

    One canonical instruction episode is retained for every physical trajectory. Candidate
    decisions are ranked by a versioned SHA-256 rule. Selection prefers previously unseen scenes,
    while trajectory IDs are globally disjoint across the train and calibration partitions.
    """
    protocol = protocol or StabilityPilotProtocol()
    canonical_by_trajectory: dict[str, R2RCEEpisode] = {}
    for episode in sorted(dataset.official_episodes, key=_episode_sort_key):
        canonical_by_trajectory.setdefault(episode.trajectory_id, episode)

    buckets: dict[R2RCEAction, list[_Candidate]] = {action: [] for action in R2RCEAction}
    for episode in canonical_by_trajectory.values():
        actions = dataset.ground_truth_record(episode.episode_id).actions
        for step_index, action in enumerate(actions):
            if step_index > protocol.max_replay_prefix_actions:
                continue
            prefix = actions[:step_index]
            if R2RCEAction.STOP in prefix:
                continue
            buckets[action].append(
                _Candidate(
                    episode=episode,
                    action=action,
                    step_index=step_index,
                    replay_prefix=prefix,
                    rank=_rank(protocol.seed, episode, step_index, action),
                )
            )
    for candidates in buckets.values():
        candidates.sort(key=lambda candidate: candidate.rank)

    selected: list[PilotSample] = []
    used_trajectories: set[str] = set()
    used_scenes: set[str] = set()
    requests = (
        ("train", protocol.train_samples_per_action),
        ("calibration", protocol.calibration_samples_per_action),
    )
    split_indices = {"train": 0, "calibration": 0}
    for split, count in requests:
        for _ in range(count):
            for action in R2RCEAction:
                candidates = buckets[action]
                eligible = [
                    candidate
                    for candidate in candidates
                    if candidate.episode.trajectory_id not in used_trajectories
                ]
                if not eligible:
                    raise ValueError(
                        f"Not enough unique trajectories for {split}/{action.name} under the "
                        f"{protocol.max_replay_prefix_actions}-action replay-prefix ceiling."
                    )
                novel_scene = [
                    candidate
                    for candidate in eligible
                    if candidate.episode.scene_id not in used_scenes
                ]
                chosen = (novel_scene or eligible)[0]
                episode = chosen.episode
                prefix_ids = tuple(int(item) for item in chosen.replay_prefix)
                prefix_bytes = json.dumps(prefix_ids, separators=(",", ":")).encode("ascii")
                selected.append(
                    PilotSample(
                        split=split,
                        sample_index=split_indices[split],
                        episode_id=episode.episode_id,
                        trajectory_id=episode.trajectory_id,
                        scene_id=episode.scene_id,
                        instruction_sha256=hashlib.sha256(
                            episode.instruction_text.encode("utf-8")
                        ).hexdigest(),
                        action_step_index=chosen.step_index,
                        target_action_id=int(action),
                        target_action=action.name,
                        replay_prefix_action_ids=prefix_ids,
                        replay_prefix_sha256=hashlib.sha256(prefix_bytes).hexdigest(),
                    )
                )
                split_indices[split] += 1
                used_trajectories.add(episode.trajectory_id)
                used_scenes.add(episode.scene_id)

    return tuple(selected)


def action_only_labels(
    rendered: dict[str, torch.Tensor | int | str],
    *,
    action: R2RCEAction,
    codebook: ActionTokenCodebook,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, int]:
    """Return labels supervising only the one frozen action-ID token."""
    position = locate_action_token(rendered, action=action, codebook=codebook)
    input_ids = rendered["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("Rendered input_ids must be a tensor.")
    labels = torch.full_like(input_ids, ignore_index)
    labels[position] = input_ids[position]
    if int((labels != ignore_index).sum().item()) != 1:
        raise RuntimeError("Action-only objective must supervise exactly one token.")
    return labels, position


def select_pilot_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen checkpoint rule to trained-checkpoint calibration rows."""
    if not candidates:
        raise ValueError("No trained checkpoint candidates were provided.")
    normalized: list[tuple[float, float, int, dict[str, Any]]] = []
    for candidate in candidates:
        mean_nll = float(candidate.get("mean_target_token_nll", math.nan))
        learning_rate = float(candidate.get("learning_rate", math.nan))
        step = candidate.get("step")
        if not math.isfinite(mean_nll) or not math.isfinite(learning_rate):
            raise ValueError("Checkpoint candidates must contain finite NLL and learning rate.")
        if not isinstance(step, int) or step <= 0:
            raise ValueError("Checkpoint candidates must identify a positive update step.")
        normalized.append((mean_nll, learning_rate, step, candidate))
    return min(normalized, key=lambda item: item[:3])[3]
