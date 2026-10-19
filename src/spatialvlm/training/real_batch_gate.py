"""Reusable integrity primitives for the one-real-batch training gate."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset, R2RCEEpisode
from spatialvlm.training.sft import FREEZE_GROUP_PATTERNS

REAL_BATCH_GATE_SCHEMA_VERSION = 1
CANONICAL_ACTION_TEXT: dict[R2RCEAction, str] = {
    R2RCEAction.STOP: "Action: STOP",
    R2RCEAction.MOVE_FORWARD: "Action: MOVE_FORWARD",
    R2RCEAction.TURN_LEFT: "Action: TURN_LEFT",
    R2RCEAction.TURN_RIGHT: "Action: TURN_RIGHT",
}


def select_probe_episode(dataset: R2RCEDataset) -> tuple[R2RCEEpisode, R2RCEAction]:
    """Choose the lowest-ID train episode whose first released action is non-STOP."""
    candidates: list[tuple[tuple[int, int | str], R2RCEEpisode, R2RCEAction]] = []
    for episode in dataset.official_episodes:
        ground_truth = dataset.ground_truth_record(episode.episode_id)
        first_action = ground_truth.actions[0]
        if first_action == R2RCEAction.STOP:
            continue
        try:
            key: tuple[int, int | str] = (0, int(episode.episode_id))
        except ValueError:
            key = (1, episode.episode_id)
        candidates.append((key, episode, first_action))
    if not candidates:
        raise ValueError("No train episode has a non-STOP first released action.")
    _, episode, action = min(candidates, key=lambda item: item[0])
    return episode, action


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return CPU clones of exactly the currently trainable parameters."""
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise ValueError("Model has no trainable parameters.")
    return state


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, shapes, dtypes, and raw tensor bytes in canonical name order."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _names_for_group(named_parameters: Mapping[str, torch.nn.Parameter], group: str) -> list[str]:
    patterns = FREEZE_GROUP_PATTERNS[group]
    return [
        name
        for name, parameter in named_parameters.items()
        if parameter.requires_grad and any(pattern in name.lower() for pattern in patterns)
    ]


def gradient_group_report(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    """Summarize gradient presence, finiteness, and norm for every declared trainable group."""
    named = dict(model.named_parameters())
    report: dict[str, dict[str, Any]] = {}
    for group in FREEZE_GROUP_PATTERNS:
        names = _names_for_group(named, group)
        grads = [named[name].grad for name in names if named[name].grad is not None]
        finite = all(bool(torch.isfinite(grad).all().item()) for grad in grads)
        norm_sq = sum(float(grad.detach().float().pow(2).sum().item()) for grad in grads)
        report[group] = {
            "trainable_tensor_count": len(names),
            "gradient_tensor_count": len(grads),
            "all_gradients_finite": finite,
            "gradient_l2_norm": math.sqrt(norm_sq),
        }
    return report


def update_group_report(
    model: torch.nn.Module,
    before: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    """Summarize parameter deltas after one optimizer update."""
    named = dict(model.named_parameters())
    if set(before) != {name for name, parameter in named.items() if parameter.requires_grad}:
        raise ValueError("Before-state keys do not match the current trainable parameter set.")
    report: dict[str, dict[str, Any]] = {}
    for group in FREEZE_GROUP_PATTERNS:
        names = _names_for_group(named, group)
        norm_sq = 0.0
        changed = 0
        finite = True
        for name in names:
            delta = named[name].detach().float().cpu() - before[name].detach().float()
            finite = finite and bool(torch.isfinite(delta).all().item())
            value = float(delta.pow(2).sum().item())
            norm_sq += value
            changed += int(value > 0.0)
        report[group] = {
            "trainable_tensor_count": len(names),
            "changed_tensor_count": changed,
            "all_updates_finite": finite,
            "update_l2_norm": math.sqrt(norm_sq),
        }
    return report


def validate_group_reports(
    gradients: Mapping[str, Mapping[str, Any]],
    updates: Mapping[str, Mapping[str, Any]],
) -> None:
    """Fail unless every required architectural group receives a finite, nonzero update."""
    for group in FREEZE_GROUP_PATTERNS:
        grad = gradients[group]
        update = updates[group]
        if grad["trainable_tensor_count"] <= 0:
            raise RuntimeError(f"Required trainable group {group!r} contains no tensors.")
        if grad["gradient_tensor_count"] != grad["trainable_tensor_count"]:
            raise RuntimeError(f"Required trainable group {group!r} has missing gradients.")
        if not grad["all_gradients_finite"] or grad["gradient_l2_norm"] <= 0.0:
            raise RuntimeError(f"Required trainable group {group!r} has invalid gradients.")
        if not update["all_updates_finite"] or update["update_l2_norm"] <= 0.0:
            raise RuntimeError(f"Required trainable group {group!r} did not update.")


def save_trainable_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically save only trainable tensors plus optimizer state and provenance."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(model)
    payload = {
        "schema_version": REAL_BATCH_GATE_SCHEMA_VERSION,
        "trainable_state_dict": state,
        "trainable_state_sha256": tensor_state_sha256(state),
        "optimizer_state_dict": optimizer.state_dict(),
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
    }


def load_trainable_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a compact checkpoint and require exact trainable-name/shape parity."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != REAL_BATCH_GATE_SCHEMA_VERSION:
        raise ValueError("Unsupported real-batch checkpoint schema version.")
    saved = payload.get("trainable_state_dict")
    if not isinstance(saved, dict) or not all(
        isinstance(value, torch.Tensor) for value in saved.values()
    ):
        raise ValueError("Checkpoint has no valid trainable_state_dict.")
    if tensor_state_sha256(saved) != payload.get("trainable_state_sha256"):
        raise ValueError("Checkpoint trainable-state hash does not match its payload.")

    named = dict(model.named_parameters())
    expected = {name for name, parameter in named.items() if parameter.requires_grad}
    if set(saved) != expected:
        missing = sorted(expected - set(saved))[:5]
        extra = sorted(set(saved) - expected)[:5]
        raise ValueError(f"Checkpoint trainable-name mismatch: missing={missing}, extra={extra}.")
    with torch.no_grad():
        for name, tensor in saved.items():
            parameter = named[name]
            if tensor.shape != parameter.shape:
                raise ValueError(f"Checkpoint shape mismatch for {name}.")
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
