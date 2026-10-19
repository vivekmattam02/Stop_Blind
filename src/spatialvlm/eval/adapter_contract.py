"""Resolve the LoRA adapter contract a checkpoint was actually trained under.

Deploy must match train -- question 5 of the project's five design checks. Nothing forces an evaluation
runner to rebuild the adapter its checkpoint was trained with, and only ONE of the three ways the
two can diverge fails loudly:

* ``lora_rank``    changes tensor shapes, so ``load_state_dict`` raises.                    LOUD
* ``lora_alpha``   changes only the ``alpha / r`` scale applied at forward time.            SILENT
* target modules   PEFT's ``set_peft_model_state_dict`` loads with ``strict=False``, so weights
  for module families the runner did not adapt are dropped without a word.                 SILENT

A runner that hardcodes any of the three evaluates a model that is not the one that trained, and
reports well-formed numbers for it. This module resolves the contract from the checkpoint's own
tensors, cross-checks it against the arm manifest the trainer wrote, and fails closed on anything
it cannot establish -- rather than falling back on a default that used to be right.

``rank`` and ``target_modules`` are read from the adapter tensors themselves and are therefore
exact. ``lora_alpha`` leaves no trace in a state dict at all: it is recovered from the sibling
``manifest.json``, or supplied explicitly, or the run refuses to start.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``...layers.0.self_attn.q_proj.lora_A.weight`` and the ``.lora_A.default.weight`` variant.
_LORA_KEY = re.compile(r"\.([A-Za-z0-9_]+)\.lora_(A|B)\b")


@dataclass(frozen=True)
class AdapterContract:
    """The adapter geometry a checkpoint requires, with the provenance of each field."""

    rank: int
    alpha: int
    target_modules: tuple[str, ...]
    alpha_source: str

    @property
    def scaling(self) -> float:
        """LoRA forward scale. The quantity a wrong ``alpha`` silently changes."""
        return self.alpha / self.rank

    def describe(self) -> str:
        return (
            f"rank {self.rank} alpha {self.alpha} (scale {self.scaling:g}, "
            f"alpha from {self.alpha_source}) on {len(self.target_modules)} module types: "
            f"{', '.join(self.target_modules)}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "lora_rank": self.rank,
            "lora_alpha": self.alpha,
            "lora_scaling": self.scaling,
            "lora_target_modules": list(self.target_modules),
            "alpha_source": self.alpha_source,
            "resolved_by": "spatialvlm.eval.adapter_contract",
        }


def infer_rank_and_targets(adapter_state: Mapping[str, Any]) -> tuple[int, tuple[str, ...]]:
    """Read rank and adapted module names off the adapter tensors. Exact, not inferred policy."""
    ranks: dict[str, int] = {}
    targets: set[str] = set()
    for key, tensor in adapter_state.items():
        match = _LORA_KEY.search(key)
        if match is None:
            continue
        module, side = match.group(1), match.group(2)
        targets.add(module)
        shape = tuple(getattr(tensor, "shape", ()))
        if len(shape) != 2:
            continue
        ranks[key] = int(shape[0]) if side == "A" else int(shape[1])
    if not targets:
        raise SystemExit(
            "Checkpoint carries no recognisable LoRA tensors; refusing to guess an adapter shape."
        )
    distinct = sorted(set(ranks.values()))
    if len(distinct) != 1:
        raise SystemExit(
            f"Checkpoint mixes LoRA ranks {distinct}; refusing to build one adapter for it."
        )
    return distinct[0], tuple(sorted(targets))


def read_arm_manifest(checkpoint: Path) -> dict[str, Any]:
    """The training manifest written beside the checkpoint. Empty when the arm predates it."""
    path = Path(checkpoint).parent / "manifest.json"
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def resolve_adapter_contract(
    checkpoint: Path,
    adapter_state: Mapping[str, Any],
    *,
    alpha_override: int | None = None,
) -> AdapterContract:
    """Resolve rank, alpha and target modules, or exit explaining what could not be established."""
    rank, targets = infer_rank_and_targets(adapter_state)
    manifest = read_arm_manifest(checkpoint)
    hyper = manifest.get("hyperparameters") or {}

    recorded_rank = manifest.get("lora_rank", hyper.get("lora_rank"))
    if recorded_rank is not None and int(recorded_rank) != rank:
        raise SystemExit(
            f"{checkpoint} holds rank-{rank} tensors but its manifest records rank "
            f"{recorded_rank}. One of the two is wrong; refusing to pick."
        )
    recorded_targets = manifest.get("lora_targets")
    if recorded_targets and tuple(sorted(recorded_targets)) != targets:
        raise SystemExit(
            f"{checkpoint} adapts {list(targets)} but its manifest records "
            f"{sorted(recorded_targets)}. Refusing to run a mismatched adapter."
        )

    if alpha_override is not None:
        alpha, source = int(alpha_override), "--lora-alpha"
    elif hyper.get("lora_alpha") is not None:
        alpha, source = int(hyper["lora_alpha"]), "arm manifest"
    elif manifest.get("lora_alpha") is not None:
        alpha, source = int(manifest["lora_alpha"]), "arm manifest"
    else:
        raise SystemExit(
            f"No lora_alpha recorded for {checkpoint} and none supplied. Alpha leaves no trace in "
            "a state dict, and a wrong one rescales every adapted layer silently. Pass "
            "--lora-alpha with the value the training run used."
        )
    return AdapterContract(rank=rank, alpha=alpha, target_modules=targets, alpha_source=source)


def _normalise(key: str) -> str:
    return key.replace(".default.", ".").removeprefix("base_model.model.")


def verify_adapter_keys(
    checkpoint_state: Mapping[str, Any], model_state: Mapping[str, Any]
) -> None:
    """Fail closed unless every checkpoint tensor landed in the model, and none was left empty.

    ``set_peft_model_state_dict`` loads with ``strict=False``. Without this check, weights for a
    module family the runner did not adapt vanish silently and the rollout scores a partly
    untrained adapter.
    """
    from_ckpt = {_normalise(k) for k in checkpoint_state}
    in_model = {_normalise(k) for k in model_state}
    dropped = sorted(from_ckpt - in_model)
    unfilled = sorted(in_model - from_ckpt)
    if dropped or unfilled:
        raise SystemExit(
            "Adapter load did not match the checkpoint. "
            f"{len(dropped)} checkpoint tensor(s) had nowhere to go (e.g. {dropped[:3]}); "
            f"{len(unfilled)} model adapter tensor(s) were left untouched (e.g. {unfilled[:3]}). "
            "PEFT drops both silently; refusing to score a partly loaded model."
        )
