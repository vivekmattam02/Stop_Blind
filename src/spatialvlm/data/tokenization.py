"""Fail-closed Qwen conversation rendering for external spatial tokens.

The spatial block is represented by exactly one checkpoint-native image-pad token in the user
message, rendered through the tokenizer's saved chat template, and then expanded to the required
number of external spatial tokens. Supervised targets are never truncated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_SPATIAL_PLACEHOLDER = "<|image_pad|>"
CONVERSATION_RENDERER_VERSION = 1

SYSTEM_PROMPT = (
    "You are a spatial navigation assistant. Given an indoor RGB-D observation and a navigation "
    "instruction, choose exactly one legal action."
)


@dataclass(frozen=True)
class SpatialTokenizerConfig:
    """Versioned token and supervision contract for the external-spatial pathway."""

    num_spatial_tokens: int = 1369
    spatial_placeholder: str = DEFAULT_SPATIAL_PLACEHOLDER
    system_prompt: str = SYSTEM_PROMPT
    max_length: int = 2048
    ignore_index: int = -100
    require_chat_template: bool = True

    def __post_init__(self) -> None:
        if self.num_spatial_tokens <= 0:
            raise ValueError("num_spatial_tokens must be positive.")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")


def _resolve_placeholder_id(tokenizer: Any, placeholder: str) -> int:
    """Resolve the exact checkpoint token; never substitute pad/zero silently."""
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or placeholder not in vocab:
        raise ValueError(
            f"Tokenizer vocabulary does not contain required spatial placeholder {placeholder!r}."
        )
    token_id = vocab[placeholder]
    if not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"Invalid token ID for spatial placeholder {placeholder!r}: {token_id!r}.")
    return token_id


def _as_token_id_list(value: Any, *, label: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, int) for item in value):
        raise TypeError(f"{label} must be a flat sequence of integer token IDs.")
    return list(value)


def _render_chat_prefix(
    tokenizer: Any,
    *,
    instruction: str,
    config: SpatialTokenizerConfig,
) -> list[int]:
    messages = [
        {"role": "system", "content": config.system_prompt},
        {
            "role": "user",
            "content": f"Instruction: {instruction}\nObservation: {config.spatial_placeholder}",
        },
    ]
    renderer = getattr(tokenizer, "apply_chat_template", None)
    if renderer is None:
        if config.require_chat_template:
            raise ValueError("Tokenizer has no apply_chat_template; scientific SFT requires it.")
        text = (
            f"{config.system_prompt}\n\nInstruction: {instruction}\n"
            f"Observation: {config.spatial_placeholder}\n\nAssistant:"
        )
        return _as_token_id_list(
            tokenizer.encode(text, add_special_tokens=True), label="legacy rendered prefix"
        )

    rendered = renderer(messages, tokenize=True, add_generation_prompt=True)
    return _as_token_id_list(rendered, label="chat-template rendered prefix")


def build_input_ids(
    tokenizer: Any,
    instruction: str,
    target: str | None = None,
    config: SpatialTokenizerConfig | None = None,
) -> dict[str, torch.Tensor | int | str]:
    """Render one conversation, expand its spatial sentinel, and construct labels.

    For SFT, only the complete assistant target and EOS are supervised. If the rendered sequence
    exceeds ``max_length``, this function raises instead of dropping target tokens.
    """
    config = config or SpatialTokenizerConfig()
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string.")
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise ValueError("target must be None or a non-empty string.")

    placeholder_id = _resolve_placeholder_id(tokenizer, config.spatial_placeholder)
    rendered_prefix = _render_chat_prefix(tokenizer, instruction=instruction, config=config)
    placeholder_positions = [
        index for index, token_id in enumerate(rendered_prefix) if token_id == placeholder_id
    ]
    if len(placeholder_positions) != 1:
        raise ValueError(
            "Rendered conversation must contain exactly one spatial placeholder token; "
            f"found {len(placeholder_positions)}."
        )

    placeholder_index = placeholder_positions[0]
    prefix_ids = rendered_prefix[:placeholder_index]
    suffix_after_spatial = rendered_prefix[placeholder_index + 1 :]
    spatial_ids = [placeholder_id] * config.num_spatial_tokens
    spatial_start_idx = len(prefix_ids)
    target_start_idx = len(prefix_ids) + len(spatial_ids) + len(suffix_after_spatial)

    target_ids: list[int] = []
    if target is not None:
        target_ids = _as_token_id_list(
            tokenizer.encode(target, add_special_tokens=False), label="assistant target"
        )
        if not target_ids:
            raise ValueError("Assistant target tokenized to an empty sequence.")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_id, int) or eos_id < 0:
        raise ValueError("Tokenizer must define a valid eos_token_id.")
    target_ids.append(eos_id)

    all_ids = prefix_ids + spatial_ids + suffix_after_spatial + target_ids
    if len(all_ids) > config.max_length:
        raise ValueError(
            "Rendered sequence exceeds max_length; refusing to truncate supervision: "
            f"length={len(all_ids)}, max_length={config.max_length}, "
            f"target_tokens={len(target_ids)}."
        )

    input_ids = torch.tensor(all_ids, dtype=torch.long)
    attention_mask = torch.ones(len(all_ids), dtype=torch.long)
    labels = torch.full((len(all_ids),), config.ignore_index, dtype=torch.long)
    if target is not None:
        labels[target_start_idx:] = input_ids[target_start_idx:]
    else:
        labels[:] = input_ids
        labels[spatial_start_idx : spatial_start_idx + config.num_spatial_tokens] = (
            config.ignore_index
        )

    supervised_count = int((labels != config.ignore_index).sum().item())
    if supervised_count <= 0:
        raise ValueError("Rendered sequence contains no supervised tokens.")
    if target is not None and supervised_count != len(target_ids):
        raise RuntimeError("Supervised suffix length does not match target plus EOS.")

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "spatial_start_idx": spatial_start_idx,
        "target_start_idx": target_start_idx,
        "supervised_token_count": supervised_count,
        "conversation_renderer_version": CONVERSATION_RENDERER_VERSION,
    }
