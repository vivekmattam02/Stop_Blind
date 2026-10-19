"""Unit tests for native-Qwen C0 integrity primitives."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.training.native_c0_gate import (
    array_sha256,
    file_sha256,
    gradient_report,
    load_native_c0_checkpoint,
    load_native_c0_selected_checkpoint,
    model_weight_locks,
    save_native_c0_checkpoint,
    select_native_c0_probe,
    tensor_sha256,
    tensor_state_sha256,
    trainable_state_dict,
    update_report,
    validate_lora_only_trainables,
    validate_optimization_reports,
)


class _TinyLoRAModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = torch.nn.Linear(3, 2)
        self.frozen.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.randn(3, 2))
        self.lora_B = torch.nn.Parameter(torch.zeros(2, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.frozen(value) + value @ self.lora_A @ self.lora_B


def test_lora_update_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = _TinyLoRAModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    contract = validate_lora_only_trainables(model)
    assert contract["trainable_tensor_count"] == 2

    before = trainable_state_dict(model)
    loss = model(torch.randn(4, 3)).pow(2).mean()
    loss.backward()
    gradients = gradient_report(model)
    optimizer.step()
    updates = update_report(model, before)
    validate_optimization_reports(gradients, updates)

    checkpoint = tmp_path / "native_c0.pt"
    record = save_native_c0_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        metadata={"run_id": "native_c0_real_batch_v1"},
    )
    saved_hash = tensor_state_sha256(trainable_state_dict(model))
    with torch.no_grad():
        model.lora_A.add_(1.0)
    assert tensor_state_sha256(trainable_state_dict(model)) != saved_hash
    load_native_c0_checkpoint(checkpoint, model=model, optimizer=optimizer)
    assert tensor_state_sha256(trainable_state_dict(model)) == saved_hash
    assert record["trainable_state_sha256"] == saved_hash


def _selected_checkpoint(tmp_path: Path, model: _TinyLoRAModel) -> tuple[Path, dict[str, str]]:
    state = trainable_state_dict(model)
    state_sha256 = tensor_state_sha256(state)
    manifest_sha256 = "b" * 64
    checkpoint = tmp_path / "selected.pt"
    torch.save(
        {
            "schema_version": 1,
            "gate": "native_qwen_c0_development_v1",
            "status": "candidate",
            "manifest_sha256": manifest_sha256,
            "cursor": {"update_step": 160, "sample_exposure_end_exclusive": 2560},
            "trainable_state_dict": state,
            "trainable_state_sha256": state_sha256,
            "optimizer_state_dict": {},
        },
        checkpoint,
    )
    return checkpoint, {
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": manifest_sha256,
        "state_sha256": state_sha256,
    }


def test_selected_development_checkpoint_loads_under_its_exact_gate(tmp_path: Path) -> None:
    torch.manual_seed(8)
    model = _TinyLoRAModel()
    checkpoint, locks = _selected_checkpoint(tmp_path, model)
    expected_state = locks["state_sha256"]
    with torch.no_grad():
        model.lora_A.add_(1.0)

    payload = load_native_c0_selected_checkpoint(
        checkpoint,
        model=model,
        expected_checkpoint_sha256=locks["checkpoint_sha256"],
        expected_gate="native_qwen_c0_development_v1",
        expected_manifest_sha256=locks["manifest_sha256"],
        expected_trainable_state_sha256=expected_state,
        expected_update_step=160,
    )

    assert payload["gate"] == "native_qwen_c0_development_v1"
    assert tensor_state_sha256(trainable_state_dict(model)) == expected_state


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_checkpoint_sha256": "0" * 64}, "file hash"),
        ({"expected_gate": "wrong_gate"}, "producing gate"),
        ({"expected_manifest_sha256": "0" * 64}, "manifest hash"),
        ({"expected_trainable_state_sha256": "0" * 64}, "selected result state"),
        ({"expected_update_step": 192}, "update cursor"),
    ],
)
def test_selected_checkpoint_rejects_every_external_identity_mismatch(
    tmp_path: Path, overrides: dict[str, str | int], message: str
) -> None:
    model = _TinyLoRAModel()
    checkpoint, locks = _selected_checkpoint(tmp_path, model)
    arguments: dict[str, str | int] = {
        "expected_checkpoint_sha256": locks["checkpoint_sha256"],
        "expected_gate": "native_qwen_c0_development_v1",
        "expected_manifest_sha256": locks["manifest_sha256"],
        "expected_trainable_state_sha256": locks["state_sha256"],
        "expected_update_step": 160,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        load_native_c0_selected_checkpoint(checkpoint, model=model, **arguments)


def test_lora_contract_rejects_non_lora_trainable() -> None:
    model = _TinyLoRAModel()
    model.frozen.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="non-LoRA"):
        validate_lora_only_trainables(model)


def test_optimization_report_rejects_missing_or_nonfinite_gradients() -> None:
    with pytest.raises(RuntimeError, match="missing"):
        validate_optimization_reports(
            {
                "trainable_tensor_count": 2,
                "gradient_tensor_count": 1,
                "nonzero_gradient_tensor_count": 1,
                "all_gradients_finite": True,
                "gradient_l2_norm": 1.0,
            },
            {
                "changed_tensor_count": 1,
                "all_updates_finite": True,
                "update_l2_norm": 1.0,
            },
        )


def test_weight_locks_cover_index_and_every_referenced_shard(tmp_path: Path) -> None:
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    locks = model_weight_locks(tmp_path)
    assert set(locks) == {
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert all(record["bytes"] > 0 for record in locks.values())


def test_typed_tensor_and_array_hashes_detect_metadata_changes() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert tensor_sha256(tensor) != tensor_sha256(tensor.to(torch.float64))
    assert array_sha256(tensor.numpy()) != array_sha256(tensor.numpy().reshape(3, 2))


def test_native_c0_probe_selection_is_numeric_and_skips_stop() -> None:
    episodes = [
        SimpleNamespace(episode_id="10"),
        SimpleNamespace(episode_id="2"),
        SimpleNamespace(episode_id="1"),
    ]
    actions = {
        "10": (R2RCEAction.TURN_LEFT,),
        "2": (R2RCEAction.MOVE_FORWARD,),
        "1": (R2RCEAction.STOP,),
    }
    dataset = SimpleNamespace(
        official_episodes=episodes,
        ground_truth_record=lambda episode_id: SimpleNamespace(actions=actions[episode_id]),
    )
    episode, action = select_native_c0_probe(dataset)
    assert episode.episode_id == "2"
    assert action is R2RCEAction.MOVE_FORWARD
