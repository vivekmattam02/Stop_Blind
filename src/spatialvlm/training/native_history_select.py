"""Frozen primitives for the train-only native-Qwen history selection suite.

The suite compares H0, H1, and causal H4 while holding target decisions, ordering,
initialization, optimization, and calibration fixed.  This module deliberately contains no
Habitat or model-loading code so the scientific contract can be tested on CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset, R2RCEEpisode
from spatialvlm.training.native_c0_gate import (
    file_sha256,
    tensor_state_sha256,
    trainable_state_dict,
)

NATIVE_HISTORY_PROTOCOL_VERSION = 1
NATIVE_HISTORY_SCHEMA_VERSION = 1
NATIVE_HISTORY_CHECKPOINT_SCHEMA_VERSION = 1
NATIVE_HISTORY_GATE_NAME = "spark_native_qwen_train_only_history_selection_v1"

HISTORY_ARMS: tuple[str, ...] = (
    "H0_CURRENT",
    "H1_PREVIOUS_ACTION",
    "H4_CAUSAL",
)


def diagnostic_float(value: float) -> float | str:
    """Keep finite diagnostics numeric and represent non-finite values explicitly."""
    return value if math.isfinite(value) else repr(value)


def causal_logit_diagnostics(
    logits: torch.Tensor,
    *,
    action_position: int,
    target_token_id: int,
) -> dict[str, Any]:
    """Summarize only the causal vocabulary row needed to explain an invalid token NLL."""
    record: dict[str, Any] = {
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype).removeprefix("torch."),
        "action_token_position": action_position,
        "causal_logit_position": action_position - 1,
        "target_token_id": target_token_id,
    }
    if (
        logits.ndim != 3
        or logits.shape[0] != 1
        or action_position < 1
        or action_position >= logits.shape[1]
        or target_token_id < 0
        or target_token_id >= logits.shape[2]
    ):
        record["valid_index_contract"] = False
        return record

    row = logits[0, action_position - 1].detach().float()
    finite_mask = torch.isfinite(row)
    finite_count = int(finite_mask.sum().item())
    finite_values = row[finite_mask]
    target_logit = float(row[target_token_id].cpu().item())
    logsumexp = float(torch.logsumexp(row, dim=-1).cpu().item())
    reconstructed_nll = logsumexp - target_logit
    record.update(
        {
            "valid_index_contract": True,
            "vocabulary_size": int(row.numel()),
            "finite_logit_count": finite_count,
            "nonfinite_logit_count": int(row.numel()) - finite_count,
            "finite_logit_min": (float(finite_values.min().cpu().item()) if finite_count else None),
            "finite_logit_max": (float(finite_values.max().cpu().item()) if finite_count else None),
            "target_logit": diagnostic_float(target_logit),
            "logsumexp": diagnostic_float(logsumexp),
            "reconstructed_target_nll": diagnostic_float(reconstructed_nll),
        }
    )
    return record


def tensor_collection_diagnostics(
    tensors: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, Any]:
    """Return compact finiteness and scale telemetry for parameters or optimizer tensors."""
    tensor_count = 0
    element_count = 0
    nonfinite_element_count = 0
    maximum_finite_absolute_value = 0.0
    dtypes: dict[str, int] = {}
    for _, tensor in tensors:
        tensor_count += 1
        element_count += tensor.numel()
        dtype = str(tensor.dtype).removeprefix("torch.")
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
        values = tensor.detach().float()
        finite_mask = torch.isfinite(values)
        finite_count = int(finite_mask.sum().item())
        nonfinite_element_count += tensor.numel() - finite_count
        if finite_count:
            maximum_finite_absolute_value = max(
                maximum_finite_absolute_value,
                float(values[finite_mask].abs().max().cpu().item()),
            )
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "nonfinite_element_count": nonfinite_element_count,
        "all_finite": nonfinite_element_count == 0,
        "maximum_finite_absolute_value": maximum_finite_absolute_value,
        "tensor_dtypes": dtypes,
    }


def optimizer_tensor_diagnostics(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Summarize Adam state by field so BF16/state corruption can be distinguished."""
    by_field: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for parameter_ordinal, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                by_field.setdefault(str(name), []).append(
                    (f"parameter_{parameter_ordinal}.{name}", value)
                )
    return {
        name: tensor_collection_diagnostics(tensors) for name, tensors in sorted(by_field.items())
    }


@dataclass(frozen=True)
class NativeHistoryProtocol:
    """Pre-registered matched-arm budget, numerical gates, and selection rule."""

    learning_rate: float = 5e-5
    planned_updates: int = 2_048
    train_samples_per_action: int = 512
    calibration_samples_per_action: int = 32
    calibration_steps: tuple[int, ...] = (0, 256, 512, 1_024, 1_536, 2_048)
    checkpoint_interval: int = 128
    minimum_history_improvement_nats: float = 0.02
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260825
    min_decision_step: int = 3
    max_replay_prefix_actions: int = 96
    seed: int = 20260824
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    hard_max_gradient_norm: float = 1_000_000.0
    hard_max_token_nll: float = 100.0
    hard_max_peak_allocated_bytes: int = 100 * 1024**3
    max_hours: float = 10.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.planned_updates <= 0:
            raise ValueError("Learning rate and update budget must be positive.")
        if self.train_sample_count != self.planned_updates:
            raise ValueError("Exactly one distinct train decision is required per update.")
        if self.calibration_samples_per_action <= 0:
            raise ValueError("Calibration samples per action must be positive.")
        if self.calibration_steps != tuple(sorted(set(self.calibration_steps))):
            raise ValueError("Calibration steps must be sorted and unique.")
        if not self.calibration_steps or self.calibration_steps[0] != 0:
            raise ValueError("Calibration must include step zero.")
        if self.calibration_steps[-1] != self.planned_updates:
            raise ValueError("The final update must be calibrated.")
        if not set(self.calibration_steps[1:]).issubset(self.checkpoint_steps):
            raise ValueError("Every trained calibration candidate must be checkpointed.")
        if self.checkpoint_interval <= 0:
            raise ValueError("Checkpoint interval must be positive.")
        if self.minimum_history_improvement_nats < 0.0 or self.bootstrap_samples <= 0:
            raise ValueError("History-selection thresholds are invalid.")
        if self.min_decision_step < 3:
            raise ValueError("Causal H4 requires decision step >= 3 without left padding.")
        if self.max_replay_prefix_actions < self.min_decision_step:
            raise ValueError("Replay-prefix ceiling is smaller than the decision-step floor.")
        if self.max_grad_norm <= 0.0 or self.hard_max_gradient_norm <= self.max_grad_norm:
            raise ValueError("Gradient thresholds are invalid.")
        if self.hard_max_peak_allocated_bytes <= 0 or self.max_hours <= 0.0:
            raise ValueError("Memory and wall-time limits must be positive.")

    @property
    def train_sample_count(self) -> int:
        return self.train_samples_per_action * len(R2RCEAction)

    @property
    def calibration_sample_count(self) -> int:
        return self.calibration_samples_per_action * len(R2RCEAction)

    @property
    def checkpoint_steps(self) -> tuple[int, ...]:
        steps = set(
            range(self.checkpoint_interval, self.planned_updates + 1, self.checkpoint_interval)
        )
        steps.add(self.planned_updates)
        return tuple(sorted(steps))

    def to_record(self) -> dict[str, Any]:
        record = json.loads(json.dumps(asdict(self)))
        record.update(
            {
                "protocol_version": NATIVE_HISTORY_PROTOCOL_VERSION,
                "gate": NATIVE_HISTORY_GATE_NAME,
                "arms": list(HISTORY_ARMS),
                "train_sample_count": self.train_sample_count,
                "calibration_sample_count": self.calibration_sample_count,
                "checkpoint_steps": list(self.checkpoint_steps),
                "objective": "single_shared_position_official_action_id_token_v1",
                "initialization": "independent identical seeded public-pretrained Qwen LoRA",
                "development_splits_used": ["train"],
                "sealed_splits": ["val_seen", "val_unseen"],
                "accuracy_is_selection_condition": False,
                "history_selection_rule": (
                    "start at H0; accept each next-complexity arm only when paired mean target-"
                    "token NLL improvement is at least the configured threshold and the fixed-"
                    "seed paired-bootstrap 95% CI lower bound is greater than zero"
                ),
            }
        )
        return record


@dataclass(frozen=True)
class NativeHistorySample:
    """One immutable train or calibration observation/action decision."""

    split: str
    sample_index: int
    episode_id: str
    trajectory_id: str
    scene_id: str
    instruction_text: str
    instruction_sha256: str
    action_step_index: int
    target_action_id: int
    target_action: str
    replay_prefix_action_ids: tuple[int, ...]
    replay_prefix_sha256: str
    selection_rank_sha256: str
    order_rank_sha256: str

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


def _stable_hash(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _canonical_episodes(dataset: R2RCEDataset, seed: int) -> dict[str, R2RCEEpisode]:
    """Choose exactly one instruction episode for every physical trajectory."""
    grouped: dict[str, list[R2RCEEpisode]] = {}
    for episode in dataset.official_episodes:
        grouped.setdefault(episode.trajectory_id, []).append(episode)
    return {
        trajectory_id: min(
            episodes,
            key=lambda episode: _stable_hash(
                "native-history-canonical-instruction-v1",
                seed,
                trajectory_id,
                episode.episode_id,
                episode.instruction_text,
            ),
        )
        for trajectory_id, episodes in grouped.items()
    }


def _candidate_buckets(
    dataset: R2RCEDataset,
    canonical: Mapping[str, R2RCEEpisode],
    protocol: NativeHistoryProtocol,
) -> dict[R2RCEAction, list[_Candidate]]:
    """Choose at most one deterministic eligible step per trajectory and target action."""
    buckets: dict[R2RCEAction, list[_Candidate]] = {action: [] for action in R2RCEAction}
    for trajectory_id, episode in canonical.items():
        actions = dataset.ground_truth_record(episode.episode_id).actions
        per_action: dict[R2RCEAction, list[_Candidate]] = {action: [] for action in R2RCEAction}
        for step_index, action in enumerate(actions):
            if step_index < protocol.min_decision_step:
                continue
            if step_index > protocol.max_replay_prefix_actions:
                continue
            prefix = actions[:step_index]
            if R2RCEAction.STOP in prefix:
                continue
            per_action[action].append(
                _Candidate(
                    episode=episode,
                    action=action,
                    step_index=step_index,
                    replay_prefix=prefix,
                    rank=_stable_hash(
                        "native-history-candidate-v1",
                        protocol.seed,
                        trajectory_id,
                        episode.episode_id,
                        int(action),
                        step_index,
                    ),
                )
            )
        for action, candidates in per_action.items():
            if candidates:
                buckets[action].append(min(candidates, key=lambda item: item.rank))
    for candidates in buckets.values():
        candidates.sort(key=lambda item: item.rank)
    return buckets


def select_native_history_samples(
    dataset: R2RCEDataset,
    *,
    protocol: NativeHistoryProtocol | None = None,
) -> tuple[NativeHistorySample, ...]:
    """Select balanced train-only decisions with global trajectory disjointness.

    Calibration is allocated first and held out from all updates.  Within each split, the rarest
    eligible action bucket is allocated first; ties use official action-ID order.  A physical
    trajectory can appear exactly once across both splits and all target actions.
    """
    protocol = protocol or NativeHistoryProtocol()
    canonical = _canonical_episodes(dataset, protocol.seed)
    buckets = _candidate_buckets(dataset, canonical, protocol)
    action_order = tuple(
        sorted(R2RCEAction, key=lambda action: (len(buckets[action]), int(action)))
    )
    used_trajectories: set[str] = set()
    selected: list[tuple[str, _Candidate]] = []
    requests = (
        ("calibration", protocol.calibration_samples_per_action),
        ("train", protocol.train_samples_per_action),
    )
    for split, per_action_count in requests:
        for _ in range(per_action_count):
            for action in action_order:
                chosen = next(
                    (
                        candidate
                        for candidate in buckets[action]
                        if candidate.episode.trajectory_id not in used_trajectories
                    ),
                    None,
                )
                if chosen is None:
                    raise ValueError(
                        f"Not enough unique trajectories for {split}/{action.name} under the "
                        f"{protocol.max_replay_prefix_actions}-action replay ceiling."
                    )
                used_trajectories.add(chosen.episode.trajectory_id)
                selected.append((split, chosen))

    records: list[NativeHistorySample] = []
    for split in ("train", "calibration"):
        split_rows = [
            (row_split, candidate) for row_split, candidate in selected if row_split == split
        ]
        split_rows.sort(
            key=lambda item: _stable_hash(
                "native-history-order-v1",
                protocol.seed,
                split,
                item[1].episode.trajectory_id,
                item[1].step_index,
                int(item[1].action),
            )
        )
        for sample_index, (_, candidate) in enumerate(split_rows):
            episode = candidate.episode
            prefix_ids = tuple(int(action) for action in candidate.replay_prefix)
            prefix_bytes = json.dumps(prefix_ids, separators=(",", ":")).encode("ascii")
            records.append(
                NativeHistorySample(
                    split=split,
                    sample_index=sample_index,
                    episode_id=episode.episode_id,
                    trajectory_id=episode.trajectory_id,
                    scene_id=episode.scene_id,
                    instruction_text=episode.instruction_text,
                    instruction_sha256=hashlib.sha256(
                        episode.instruction_text.encode("utf-8")
                    ).hexdigest(),
                    action_step_index=candidate.step_index,
                    target_action_id=int(candidate.action),
                    target_action=candidate.action.name,
                    replay_prefix_action_ids=prefix_ids,
                    replay_prefix_sha256=hashlib.sha256(prefix_bytes).hexdigest(),
                    selection_rank_sha256=candidate.rank,
                    order_rank_sha256=_stable_hash(
                        "native-history-order-v1",
                        protocol.seed,
                        split,
                        episode.trajectory_id,
                        candidate.step_index,
                        int(candidate.action),
                    ),
                )
            )
    return tuple(records)


def select_arm_checkpoint(calibrations: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select minimum finite calibration NLL, breaking exact ties by earlier step."""
    candidates: list[tuple[float, int, Mapping[str, Any]]] = []
    for row in calibrations:
        step = row.get("step")
        nll = float(row.get("mean_target_token_nll", math.nan))
        if not isinstance(step, int) or step < 0 or not math.isfinite(nll):
            raise ValueError("Calibration candidates require a nonnegative step and finite NLL.")
        candidates.append((nll, step, row))
    if not candidates:
        raise ValueError("At least one calibration row is required.")
    return min(candidates, key=lambda item: item[:2])[2]


def paired_bootstrap_improvement(
    baseline_nll: Sequence[float],
    candidate_nll: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Return a deterministic percentile CI for paired baseline-minus-candidate NLL."""
    if len(baseline_nll) != len(candidate_nll) or not baseline_nll:
        raise ValueError("Paired NLL vectors must be non-empty and have equal length.")
    improvements = [float(left) - float(right) for left, right in zip(baseline_nll, candidate_nll)]
    if not all(math.isfinite(value) for value in improvements):
        raise ValueError("Paired NLL improvements must be finite.")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    rng = random.Random(seed)
    count = len(improvements)
    means = sorted(
        sum(improvements[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    lower_index = max(0, math.floor(0.025 * (samples - 1)))
    upper_index = min(samples - 1, math.ceil(0.975 * (samples - 1)))
    return {
        "paired_count": count,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "mean_improvement_nats": sum(improvements) / count,
        "ci95_lower_nats": means[lower_index],
        "ci95_upper_nats": means[upper_index],
    }


def select_history_arm(
    per_arm_nll: Mapping[str, Sequence[float]],
    *,
    protocol: NativeHistoryProtocol,
) -> dict[str, Any]:
    """Apply the frozen sequential complexity rule H0 -> H1 -> H4."""
    if set(per_arm_nll) != set(HISTORY_ARMS):
        raise ValueError("History selection requires exactly H0, H1, and H4 NLL vectors.")
    lengths = {len(values) for values in per_arm_nll.values()}
    if lengths != {protocol.calibration_sample_count}:
        raise ValueError("History arms do not share the frozen calibration decision count.")
    selected = HISTORY_ARMS[0]
    comparisons: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(HISTORY_ARMS[1:], start=1):
        baseline = selected
        bootstrap = paired_bootstrap_improvement(
            per_arm_nll[baseline],
            per_arm_nll[candidate],
            samples=protocol.bootstrap_samples,
            seed=protocol.bootstrap_seed + ordinal,
        )
        accepted = bool(
            float(bootstrap["mean_improvement_nats"]) >= protocol.minimum_history_improvement_nats
            and float(bootstrap["ci95_lower_nats"]) > 0.0
        )
        comparisons.append(
            {
                "baseline_arm": baseline,
                "candidate_arm": candidate,
                **bootstrap,
                "minimum_required_improvement_nats": (protocol.minimum_history_improvement_nats),
                "accepted": accepted,
            }
        )
        if accepted:
            selected = candidate
    return {
        "selected_arm": selected,
        "comparisons": comparisons,
        "accuracy_used_for_selection": False,
    }


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG used by the single-process Spark suite."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore an RNG payload produced by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_history_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, Any],
    cursor: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically save LoRA, optimizer, RNG, and exact manifest/cache cursor identity."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite history checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(model)
    payload = {
        "schema_version": NATIVE_HISTORY_CHECKPOINT_SCHEMA_VERSION,
        "gate": NATIVE_HISTORY_GATE_NAME,
        "metadata": dict(metadata),
        "cursor": dict(cursor),
        "trainable_state_dict": state,
        "trainable_state_sha256": tensor_state_sha256(state),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "trainable_state_sha256": payload["trainable_state_sha256"],
        "update_step": cursor.get("update_step"),
    }


def load_history_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly restore a history checkpoint and reject source/cache/manifest drift."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != NATIVE_HISTORY_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported history checkpoint schema version.")
    if payload.get("gate") != NATIVE_HISTORY_GATE_NAME:
        raise ValueError("History checkpoint gate identity mismatch.")
    if payload.get("metadata") != dict(expected_metadata):
        raise ValueError("History checkpoint metadata identity mismatch.")
    state = payload.get("trainable_state_dict")
    if not isinstance(state, dict) or tensor_state_sha256(state) != payload.get(
        "trainable_state_sha256"
    ):
        raise ValueError("History checkpoint trainable-state hash mismatch.")
    current = trainable_state_dict(model)
    if set(current) != set(state):
        raise ValueError("History checkpoint trainable-name set mismatch.")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if tuple(parameters[name].shape) != tuple(value.shape):
                raise ValueError(f"History checkpoint shape mismatch for {name}.")
            parameters[name].copy_(value.to(device=parameters[name].device))
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    restore_rng_state(payload["rng_state"])
    return payload
