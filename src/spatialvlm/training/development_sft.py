"""Frozen primitives for the first bounded train-only development run.

This module deliberately does not contain Habitat or model-construction code.  It owns the
scientific protocol, deterministic sample selection, early stopping, and compact checkpoint
integrity so those contracts can be tested without a GPU or licensed scene assets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset, R2RCEEpisode
from spatialvlm.training.real_batch_gate import (
    file_sha256,
    tensor_state_sha256,
    trainable_state_dict,
)

DEVELOPMENT_SFT_PROTOCOL_VERSION = 1
DEVELOPMENT_SFT_SCHEMA_VERSION = 1
DEVELOPMENT_SFT_CHECKPOINT_SCHEMA_VERSION = 1
DEVELOPMENT_SFT_GATE_NAME = "spark_r2r_ce_train_only_development_sft_v1"

PILOT_RUN_ID = "stability_pilot_v1"
PILOT_MANIFEST_SHA256 = "231b3531bb31f1af79385fae9c8ae2ec33f06bf5484fd8b3c64603e81a315f14"
PILOT_SUITE_SHA256 = "e56ba45425382d1219a98383f141c7cbc38c545f6ecb45fcf02179d9937b1249"
PILOT_VERIFICATION_V2_SHA256 = "8cc5e126752c16d8f2b2458add2f18f517e4f88c24e5030660e1c542ffe787ee"
WARMSTART_CHECKPOINT_BYTES = 1_490_534_181
WARMSTART_CHECKPOINT_SHA256 = "7abfc32e335695a5fc3c463641e56088296ef0e1ef8ad4bd86a2d9f3ff1a2a17"
WARMSTART_TRAINABLE_STATE_SHA256 = (
    "72e33d9e3eb9e9338a8b0dd9b29fe8e7aa09bfb8a3587f93a65a268bad23639a"
)


@dataclass(frozen=True)
class DevelopmentSFTProtocol:
    """Pre-registered budget, selection rule, and stop conditions."""

    learning_rate: float = 5e-5
    planned_updates: int = 1_024
    train_samples_per_action: int = 256
    calibration_samples_per_action: int = 16
    calibration_steps: tuple[int, ...] = (
        0,
        16,
        32,
        64,
        128,
        256,
        384,
        512,
        640,
        768,
        896,
        1_024,
    )
    checkpoint_interval: int = 64
    early_checkpoint_steps: tuple[int, ...] = (16, 32)
    minimum_updates_before_early_stop: int = 128
    early_stopping_patience: int = 3
    minimum_improvement_nats: float = 0.01
    max_replay_prefix_actions: int = 96
    seed: int = 20260824
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    hard_max_gradient_norm: float = 1_000_000.0
    hard_max_token_nll: float = 100.0
    hard_max_peak_allocated_bytes: int = 64 * 1024**3
    max_hours: float = 8.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.planned_updates <= 0:
            raise ValueError("Learning rate and update budget must be positive.")
        if self.train_sample_count != self.planned_updates:
            raise ValueError("The protocol requires exactly one distinct sample per update.")
        if self.calibration_samples_per_action <= 0:
            raise ValueError("Calibration samples per action must be positive.")
        if self.calibration_steps != tuple(sorted(set(self.calibration_steps))):
            raise ValueError("Calibration steps must be sorted and unique.")
        if not self.calibration_steps or self.calibration_steps[0] != 0:
            raise ValueError("Calibration must start at warm-start step zero.")
        if self.calibration_steps[-1] != self.planned_updates:
            raise ValueError("The final planned update must be calibrated.")
        if any(step < 0 or step > self.planned_updates for step in self.calibration_steps):
            raise ValueError("Calibration steps must lie inside the update budget.")
        if self.checkpoint_interval <= 0:
            raise ValueError("Checkpoint interval must be positive.")
        if any(step <= 0 for step in self.early_checkpoint_steps):
            raise ValueError("Early checkpoint steps must be positive.")
        if not set(self.calibration_steps[1:]).issubset(set(self.checkpoint_steps)):
            raise ValueError("Every trained calibration candidate must be checkpointed.")
        if self.minimum_updates_before_early_stop not in self.calibration_steps:
            raise ValueError("The early-stop floor must be a calibration step.")
        if self.early_stopping_patience <= 0 or self.minimum_improvement_nats < 0.0:
            raise ValueError("Early-stopping settings are invalid.")
        if self.max_grad_norm <= 0.0 or self.hard_max_gradient_norm <= self.max_grad_norm:
            raise ValueError("Gradient thresholds are invalid.")
        if self.max_hours <= 0.0:
            raise ValueError("Wall-time budget must be positive.")

    @property
    def train_sample_count(self) -> int:
        return self.train_samples_per_action * len(R2RCEAction)

    @property
    def calibration_sample_count(self) -> int:
        return self.calibration_samples_per_action * len(R2RCEAction)

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        regular = range(
            self.checkpoint_interval, self.planned_updates + 1, self.checkpoint_interval
        )
        return tuple(sorted({*self.early_checkpoint_steps, *regular, self.planned_updates}))

    def to_record(self) -> dict[str, Any]:
        record = json.loads(json.dumps(asdict(self)))
        record.update(
            {
                "protocol_version": DEVELOPMENT_SFT_PROTOCOL_VERSION,
                "gate": DEVELOPMENT_SFT_GATE_NAME,
                "train_sample_count": self.train_sample_count,
                "calibration_sample_count": self.calibration_sample_count,
                "checkpoint_steps": list(self.checkpoint_steps),
                "objective": "single_shared_position_official_action_id_token_v1",
                "initialization": "OPT-STABILITY-v1 selected lr_5em05 step_0008",
                "optimizer_schedule": "constant learning rate; restored pilot AdamW state",
                "checkpoint_selection_rule": (
                    "minimum fixed held-out train-trajectory mean action-token NLL across step 0 "
                    "and trained calibration checkpoints; exact ties choose earlier step"
                ),
                "early_stopping_rule": (
                    "at or after the update floor, stop after the configured number of consecutive "
                    "calibrations that fail to improve the incumbent by minimum_improvement_nats"
                ),
                "development_splits_used": ["train"],
                "sealed_splits": ["val_seen", "val_unseen"],
                "accuracy_is_selection_or_stop_condition": False,
            }
        )
        return record


@dataclass(frozen=True)
class DevelopmentSample:
    """One immutable train or calibration observation/action decision."""

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
        f"development-sft-v{DEVELOPMENT_SFT_PROTOCOL_VERSION}|{seed}|"
        f"{episode.trajectory_id}|{episode.episode_id}|{step_index}|{int(action)}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_development_samples(
    dataset: R2RCEDataset,
    *,
    excluded_trajectory_ids: set[str] | frozenset[str],
    protocol: DevelopmentSFTProtocol | None = None,
) -> tuple[DevelopmentSample, ...]:
    """Select balanced, globally trajectory-disjoint train-only decisions.

    Pilot train and calibration trajectories are mandatory exclusions. One canonical instruction
    episode is retained for every remaining physical trajectory. Selection is versioned by SHA-256
    rank and prefers scene diversity without sacrificing action balance.
    """
    protocol = protocol or DevelopmentSFTProtocol()
    canonical: dict[str, R2RCEEpisode] = {}
    for episode in sorted(dataset.official_episodes, key=_episode_sort_key):
        if episode.trajectory_id not in excluded_trajectory_ids:
            canonical.setdefault(episode.trajectory_id, episode)

    buckets: dict[R2RCEAction, list[_Candidate]] = {action: [] for action in R2RCEAction}
    for episode in canonical.values():
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
        candidates.sort(key=lambda item: item.rank)

    selected: list[DevelopmentSample] = []
    used_trajectories = set(excluded_trajectory_ids)
    used_scenes: set[str] = set()
    split_indices = {"train": 0, "calibration": 0}
    requests = (
        ("train", protocol.train_samples_per_action),
        ("calibration", protocol.calibration_samples_per_action),
    )
    for split, count in requests:
        for _ in range(count):
            for action in R2RCEAction:
                eligible = [
                    candidate
                    for candidate in buckets[action]
                    if candidate.episode.trajectory_id not in used_trajectories
                ]
                if not eligible:
                    raise ValueError(
                        f"Not enough unique trajectories for {split}/{action.name} under the "
                        f"{protocol.max_replay_prefix_actions}-action replay ceiling."
                    )
                novel_scene = [
                    item for item in eligible if item.episode.scene_id not in used_scenes
                ]
                chosen = (novel_scene or eligible)[0]
                episode = chosen.episode
                prefix_ids = tuple(int(item) for item in chosen.replay_prefix)
                prefix_bytes = json.dumps(prefix_ids, separators=(",", ":")).encode("ascii")
                selected.append(
                    DevelopmentSample(
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


def select_development_checkpoint(calibrations: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the minimum finite calibration NLL, breaking exact ties by earlier step."""
    if not calibrations:
        raise ValueError("At least one calibration row is required.")
    candidates = []
    for row in calibrations:
        step = row.get("step")
        nll = float(row.get("mean_target_token_nll", math.nan))
        if not isinstance(step, int) or step < 0 or not math.isfinite(nll):
            raise ValueError("Calibration candidates require a nonnegative step and finite NLL.")
        candidates.append((nll, step, row))
    return min(candidates, key=lambda item: item[:2])[2]


def update_early_stopping_state(
    *,
    previous_best_nll: float,
    previous_non_improving_count: int,
    current_nll: float,
    current_step: int,
    protocol: DevelopmentSFTProtocol,
) -> tuple[float, int, bool]:
    """Update the frozen minimum-delta/patience state."""
    if not all(math.isfinite(value) for value in (previous_best_nll, current_nll)):
        raise ValueError("Early-stopping NLL values must be finite.")
    if current_nll <= previous_best_nll - protocol.minimum_improvement_nats:
        best = current_nll
        non_improving = 0
    else:
        best = previous_best_nll
        non_improving = previous_non_improving_count + 1
    should_stop = (
        current_step >= protocol.minimum_updates_before_early_stop
        and non_improving >= protocol.early_stopping_patience
    )
    return best, non_improving, should_stop


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG used by the single-process Spark development runner."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("Checkpoint RNG state is incomplete or contains unknown keys.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_development_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically save strict trainable/optimizer/RNG/cursor state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(model)
    payload = {
        "schema_version": DEVELOPMENT_SFT_CHECKPOINT_SCHEMA_VERSION,
        "trainable_state_dict": state,
        "trainable_state_sha256": tensor_state_sha256(state),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "run_state": dict(run_state),
        "metadata": dict(metadata),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return {
        "path": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "trainable_state_sha256": payload["trainable_state_sha256"],
        "trainable_tensor_count": len(state),
        "update_step": run_state.get("update_step"),
    }


def load_development_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_manifest_sha256: str,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Strictly restore a development checkpoint and its exact data cursor."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != DEVELOPMENT_SFT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported development checkpoint schema version.")
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("Development checkpoint manifest identity mismatch.")
    saved = payload.get("trainable_state_dict")
    if not isinstance(saved, dict) or not all(
        isinstance(value, torch.Tensor) for value in saved.values()
    ):
        raise ValueError("Development checkpoint has no valid trainable state.")
    if tensor_state_sha256(saved) != payload.get("trainable_state_sha256"):
        raise ValueError("Development checkpoint trainable-state hash mismatch.")
    named = dict(model.named_parameters())
    expected = {name for name, parameter in named.items() if parameter.requires_grad}
    if set(saved) != expected:
        raise ValueError("Development checkpoint trainable-name set mismatch.")
    with torch.no_grad():
        for name, tensor in saved.items():
            parameter = named[name]
            if tensor.shape != parameter.shape:
                raise ValueError(f"Development checkpoint shape mismatch for {name}.")
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    run_state = payload.get("run_state")
    if not isinstance(run_state, dict) or not isinstance(run_state.get("update_step"), int):
        raise ValueError("Development checkpoint has no valid run cursor.")
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload
