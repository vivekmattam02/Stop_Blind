"""Integrity primitives for the exact-revision native-Qwen C0 mechanics gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset, R2RCEEpisode

NATIVE_C0_GATE_SCHEMA_VERSION = 1
NATIVE_C0_CHECKPOINT_SCHEMA_VERSION = 1
NATIVE_C0_GATE_NAME = "spark_native_qwen_c0_one_real_h0_update_v1"
NATIVE_C0_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
NATIVE_C0_MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
NATIVE_C0_TRANSFORMERS_VERSION = "4.57.6"
NATIVE_C0_DOCKER_IMAGE_REF = "gridnav-spark-habitat:hsim0.3.3"
NATIVE_C0_PARENT_RESULT_SHA256 = "bf7555659c6e8cbfdb7ed0b703440c7b5c8739ed420a83f0ec3751d82e939bba"
NATIVE_C0_PARENT_VERIFICATION_SHA256 = (
    "e56b4dfcc85e1bca9dbe9b96f2d326a8d42ce462b95aa06d0c42c21cd69fb7e5"
)


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone exactly the trainable parameter state onto CPU."""
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise ValueError("Native C0 model has no trainable parameters.")
    return state


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and bytes in canonical order."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash one tensor with the same dtype/shape/byte framing as checkpoint state."""
    return tensor_state_sha256({"tensor": tensor})


def array_sha256(array: Any) -> str:
    """Hash one NumPy-compatible array with explicit dtype and shape framing."""
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def select_native_c0_probe(dataset: R2RCEDataset) -> tuple[R2RCEEpisode, R2RCEAction]:
    """Select the lowest-ID train episode whose first released action is non-STOP."""
    candidates: list[tuple[tuple[int, int | str], R2RCEEpisode, R2RCEAction]] = []
    for episode in dataset.official_episodes:
        first_action = dataset.ground_truth_record(episode.episode_id).actions[0]
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


def trainable_name_sha256(names: list[str] | tuple[str, ...]) -> str:
    """Hash an exact sorted trainable-parameter name surface."""
    canonical = "\n".join(sorted(names)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_lora_only_trainables(model: torch.nn.Module) -> dict[str, Any]:
    """Fail unless every trainable tensor belongs to a LoRA adapter."""
    named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not named:
        raise RuntimeError("Native C0 has no trainable LoRA tensors.")
    unexpected = [name for name, _ in named if "lora_" not in name.lower()]
    if unexpected:
        raise RuntimeError(f"Native C0 exposed non-LoRA trainables: {unexpected[:5]}.")
    names = [name for name, _ in named]
    return {
        "trainable_tensor_count": len(named),
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in named),
        "trainable_name_sha256": trainable_name_sha256(names),
        "all_trainable_names_are_lora": True,
    }


def gradient_report(model: torch.nn.Module) -> dict[str, Any]:
    """Summarize gradient coverage and finiteness across every LoRA tensor."""
    named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    gradients = [(name, parameter.grad) for name, parameter in named if parameter.grad is not None]
    finite = all(bool(torch.isfinite(gradient).all().item()) for _, gradient in gradients)
    norm_sq = sum(float(gradient.detach().float().pow(2).sum().item()) for _, gradient in gradients)
    nonzero = sum(int(bool((gradient.detach() != 0).any().item())) for _, gradient in gradients)
    return {
        "trainable_tensor_count": len(named),
        "gradient_tensor_count": len(gradients),
        "nonzero_gradient_tensor_count": nonzero,
        "all_gradients_finite": finite,
        "gradient_l2_norm": math.sqrt(norm_sq),
    }


def update_report(
    model: torch.nn.Module,
    before: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Summarize the exact LoRA parameter delta after one optimizer step."""
    named = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(named) != set(before):
        raise ValueError("Before-state keys do not match the current trainable parameter set.")
    norm_sq = 0.0
    changed = 0
    finite = True
    for name, parameter in named.items():
        delta = parameter.detach().float().cpu() - before[name].detach().float()
        finite = finite and bool(torch.isfinite(delta).all().item())
        value = float(delta.pow(2).sum().item())
        norm_sq += value
        changed += int(value > 0.0)
    return {
        "trainable_tensor_count": len(named),
        "changed_tensor_count": changed,
        "all_updates_finite": finite,
        "update_l2_norm": math.sqrt(norm_sq),
    }


def validate_optimization_reports(gradients: Mapping[str, Any], updates: Mapping[str, Any]) -> None:
    """Fail unless the one-step LoRA optimization is finite, covered, and nonzero."""
    if gradients.get("trainable_tensor_count", 0) <= 0:
        raise RuntimeError("Native C0 has no trainable tensors.")
    if gradients.get("gradient_tensor_count") != gradients.get("trainable_tensor_count"):
        raise RuntimeError("Native C0 has missing LoRA gradients.")
    if not gradients.get("all_gradients_finite"):
        raise RuntimeError("Native C0 has non-finite LoRA gradients.")
    if gradients.get("nonzero_gradient_tensor_count", 0) <= 0:
        raise RuntimeError("Native C0 has no nonzero LoRA gradient tensors.")
    if not math.isfinite(float(gradients.get("gradient_l2_norm", float("nan")))):
        raise RuntimeError("Native C0 gradient norm is non-finite.")
    if not updates.get("all_updates_finite"):
        raise RuntimeError("Native C0 has non-finite LoRA updates.")
    if updates.get("changed_tensor_count", 0) <= 0:
        raise RuntimeError("Native C0 optimizer step changed no LoRA tensors.")
    if (
        not math.isfinite(float(updates.get("update_l2_norm", float("nan"))))
        or float(updates.get("update_l2_norm", 0.0)) <= 0.0
    ):
        raise RuntimeError("Native C0 update norm is invalid.")


def model_weight_locks(snapshot: str | Path) -> dict[str, dict[str, int | str]]:
    """Fully hash the frozen safetensors index and every referenced weight shard."""
    root = Path(snapshot).resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Frozen Qwen weight index is missing: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Frozen Qwen weight index has no non-empty weight_map.")
    shards = sorted(set(weight_map.values()))
    if not shards or not all(
        isinstance(name, str) and name.endswith(".safetensors") for name in shards
    ):
        raise ValueError("Frozen Qwen weight index contains invalid shard names.")
    locks: dict[str, dict[str, int | str]] = {}
    for relative in ["model.safetensors.index.json", *shards]:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Qwen weight file is missing: {path}")
        locks[relative] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return locks


def save_native_c0_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically save LoRA tensors, optimizer state, and immutable provenance."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(model)
    names = sorted(state)
    payload = {
        "schema_version": NATIVE_C0_CHECKPOINT_SCHEMA_VERSION,
        "gate": NATIVE_C0_GATE_NAME,
        "trainable_state_dict": state,
        "trainable_state_sha256": tensor_state_sha256(state),
        "trainable_name_sha256": trainable_name_sha256(names),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": dict(metadata),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "trainable_state_sha256": payload["trainable_state_sha256"],
        "trainable_name_sha256": payload["trainable_name_sha256"],
        "trainable_tensor_count": len(state),
        "trainable_parameter_count": sum(tensor.numel() for tensor in state.values()),
    }


def load_native_c0_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a native-C0 checkpoint and require exact trainable name/shape parity."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != NATIVE_C0_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported native-C0 checkpoint schema version.")
    if payload.get("gate") != NATIVE_C0_GATE_NAME:
        raise ValueError("Native-C0 checkpoint gate identity mismatch.")
    saved = payload.get("trainable_state_dict")
    if (
        not isinstance(saved, dict)
        or not saved
        or not all(isinstance(value, torch.Tensor) for value in saved.values())
    ):
        raise ValueError("Native-C0 checkpoint has no valid trainable_state_dict.")
    if tensor_state_sha256(saved) != payload.get("trainable_state_sha256"):
        raise ValueError("Native-C0 checkpoint trainable-state hash mismatch.")
    if trainable_name_sha256(list(saved)) != payload.get("trainable_name_sha256"):
        raise ValueError("Native-C0 checkpoint trainable-name hash mismatch.")

    named = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(named) != set(saved):
        missing = sorted(set(named) - set(saved))[:5]
        extra = sorted(set(saved) - set(named))[:5]
        raise ValueError(f"Native-C0 trainable-name mismatch: missing={missing}, extra={extra}.")
    with torch.no_grad():
        for name, tensor in saved.items():
            parameter = named[name]
            if parameter.shape != tensor.shape:
                raise ValueError(f"Native-C0 checkpoint shape mismatch for {name}.")
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict):
            raise ValueError("Native-C0 checkpoint optimizer state is missing.")
        optimizer.load_state_dict(optimizer_state)
    return payload


def load_native_c0_selected_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    expected_checkpoint_sha256: str,
    expected_gate: str,
    expected_manifest_sha256: str,
    expected_trainable_state_sha256: str,
    expected_update_step: int,
) -> dict[str, Any]:
    """Load one exact selected development checkpoint for inference.

    Development and action-balance checkpoints intentionally use their own producing gate names
    and omit the mechanics gate's redundant trainable-name digest.  This loader therefore binds
    the complete file hash supplied by the independently verified result, then reconstructs the
    shared tensor contract without pretending the file came from the older one-update gate.
    """
    checkpoint = Path(path).expanduser().resolve()
    if file_sha256(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("Selected native-C0 checkpoint file hash mismatch.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != NATIVE_C0_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported selected native-C0 checkpoint schema version.")
    if payload.get("gate") != expected_gate:
        raise ValueError("Selected native-C0 checkpoint producing gate mismatch.")
    if payload.get("status") != "candidate":
        raise ValueError("Selected native-C0 checkpoint is not a complete candidate.")
    if payload.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Selected native-C0 checkpoint manifest hash mismatch.")
    cursor = payload.get("cursor")
    if not isinstance(cursor, Mapping) or cursor.get("update_step") != expected_update_step:
        raise ValueError("Selected native-C0 checkpoint update cursor mismatch.")

    saved = payload.get("trainable_state_dict")
    if (
        not isinstance(saved, dict)
        or not saved
        or not all(isinstance(value, torch.Tensor) for value in saved.values())
    ):
        raise ValueError("Selected native-C0 checkpoint has no valid trainable_state_dict.")
    observed_state_sha256 = tensor_state_sha256(saved)
    if observed_state_sha256 != payload.get("trainable_state_sha256"):
        raise ValueError("Selected native-C0 checkpoint trainable-state hash mismatch.")
    if observed_state_sha256 != expected_trainable_state_sha256:
        raise ValueError("Selected native-C0 checkpoint differs from the selected result state.")
    saved_name_sha256 = payload.get("trainable_name_sha256")
    if saved_name_sha256 is not None and saved_name_sha256 != trainable_name_sha256(list(saved)):
        raise ValueError("Selected native-C0 checkpoint trainable-name hash mismatch.")

    named = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(named) != set(saved):
        missing = sorted(set(named) - set(saved))[:5]
        extra = sorted(set(saved) - set(named))[:5]
        raise ValueError(
            f"Selected native-C0 trainable-name mismatch: missing={missing}, extra={extra}."
        )
    with torch.no_grad():
        for name, tensor in saved.items():
            parameter = named[name]
            if parameter.shape != tensor.shape:
                raise ValueError(f"Selected native-C0 checkpoint shape mismatch for {name}.")
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return payload
