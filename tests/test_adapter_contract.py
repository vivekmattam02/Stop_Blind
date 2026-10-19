"""Regressions for the train/deploy adapter mismatch that aborted rollout selection.

The rescue campaign trained rank-64 alpha-128 adapters on seven dense module types while both
closed-loop runners built rank-32 alpha-64 attention-only ones. Only the rank difference raised;
alpha and the three missing module families would have been applied silently. Every test here
fails against that behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spatialvlm.eval.adapter_contract import (
    infer_rank_and_targets,
    resolve_adapter_contract,
    verify_adapter_keys,
)

DENSE = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTENTION_ONLY = ("k_proj", "o_proj", "q_proj", "v_proj")


class _Tensor:
    """Only ``.shape`` is read, so these tests need no torch."""

    def __init__(self, *shape: int) -> None:
        self.shape = shape


def adapter_state(rank: int, modules: tuple[str, ...], layers: int = 2) -> dict[str, _Tensor]:
    state: dict[str, _Tensor] = {}
    for layer in range(layers):
        for module in modules:
            stem = f"base_model.model.model.language_model.layers.{layer}.block.{module}"
            state[f"{stem}.lora_A.weight"] = _Tensor(rank, 4096)
            state[f"{stem}.lora_B.weight"] = _Tensor(4096, rank)
    return state


def write_manifest(directory: Path, **fields: object) -> Path:
    (directory / "manifest.json").write_text(json.dumps(fields), encoding="utf-8")
    return directory / "ckpt_u03000.pt"


def test_rank_and_targets_come_from_the_tensors_not_a_default() -> None:
    rank, targets = infer_rank_and_targets(adapter_state(64, DENSE))
    assert rank == 64
    assert targets == tuple(sorted(DENSE))


def test_legacy_attention_only_checkpoint_still_resolves(tmp_path: Path) -> None:
    checkpoint = write_manifest(tmp_path, hyperparameters={"lora_alpha": 64})
    contract = resolve_adapter_contract(checkpoint, adapter_state(32, ATTENTION_ONLY))
    assert (contract.rank, contract.alpha, contract.scaling) == (32, 64, 2.0)
    assert contract.target_modules == ATTENTION_ONLY


def test_dense_rank64_contract_is_resolved_whole(tmp_path: Path) -> None:
    checkpoint = write_manifest(
        tmp_path, lora_rank=64, lora_targets=list(DENSE), hyperparameters={"lora_alpha": 128}
    )
    contract = resolve_adapter_contract(checkpoint, adapter_state(64, DENSE))
    assert contract.rank == 64
    assert contract.alpha == 128
    assert contract.scaling == 2.0
    assert contract.target_modules == tuple(sorted(DENSE))
    assert contract.as_dict()["lora_target_modules"] == sorted(DENSE)


def test_manifest_rank_disagreeing_with_the_tensors_is_fatal(tmp_path: Path) -> None:
    checkpoint = write_manifest(tmp_path, lora_rank=32, hyperparameters={"lora_alpha": 64})
    with pytest.raises(SystemExit, match="rank"):
        resolve_adapter_contract(checkpoint, adapter_state(64, DENSE))


def test_manifest_targets_disagreeing_with_the_tensors_are_fatal(tmp_path: Path) -> None:
    checkpoint = write_manifest(
        tmp_path, lora_targets=list(ATTENTION_ONLY), hyperparameters={"lora_alpha": 128}
    )
    with pytest.raises(SystemExit, match="Refusing to run a mismatched adapter"):
        resolve_adapter_contract(checkpoint, adapter_state(64, DENSE))


def test_unrecorded_alpha_fails_closed_rather_than_defaulting(tmp_path: Path) -> None:
    checkpoint = write_manifest(tmp_path, lora_rank=64)
    with pytest.raises(SystemExit, match="lora_alpha"):
        resolve_adapter_contract(checkpoint, adapter_state(64, DENSE))


def test_explicit_alpha_overrides_and_is_attributed(tmp_path: Path) -> None:
    checkpoint = write_manifest(tmp_path, hyperparameters={"lora_alpha": 64})
    contract = resolve_adapter_contract(checkpoint, adapter_state(64, DENSE), alpha_override=128)
    assert (contract.alpha, contract.alpha_source) == (128, "--lora-alpha")


def test_mixed_ranks_are_refused() -> None:
    state = adapter_state(64, DENSE)
    state["base_model.model.model.language_model.layers.9.block.q_proj.lora_A.weight"] = _Tensor(
        32, 4096
    )
    with pytest.raises(SystemExit, match="mixes LoRA ranks"):
        infer_rank_and_targets(state)


def test_a_checkpoint_without_lora_tensors_is_refused() -> None:
    with pytest.raises(SystemExit, match="no recognisable LoRA tensors"):
        infer_rank_and_targets({"model.embed_tokens.weight": _Tensor(1000, 4096)})


def test_silently_dropped_mlp_weights_are_caught() -> None:
    """The failure that would have scored 42 rollouts on a partly loaded adapter."""
    with pytest.raises(SystemExit, match="nowhere to go"):
        verify_adapter_keys(adapter_state(64, DENSE), adapter_state(64, ATTENTION_ONLY))


def test_untouched_model_adapters_are_caught() -> None:
    with pytest.raises(SystemExit, match="left untouched"):
        verify_adapter_keys(adapter_state(64, ATTENTION_ONLY), adapter_state(64, DENSE))


def test_matching_load_passes_through_peft_key_spelling() -> None:
    checkpoint = adapter_state(64, DENSE)
    reloaded = {
        key.replace(".lora_A.", ".lora_A.default.").replace(".lora_B.", ".lora_B.default.")
        for key in checkpoint
    }
    verify_adapter_keys(checkpoint, dict.fromkeys(reloaded, _Tensor(1)))
