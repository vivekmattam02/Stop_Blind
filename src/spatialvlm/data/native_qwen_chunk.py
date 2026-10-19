"""Native Qwen3-VL batch rendering for the k-action chunk objective.

This module is strictly additive. ``spatialvlm/data/native_qwen.py`` is hash-locked by 17 completed
run artifacts, several of them load-bearing for the submitted manuscript, so it must not be edited.
The private helpers imported below are reused deliberately rather than duplicated: the visual-count,
processor-field and token-contract checks are subtle, and a copy would be free to drift away from
the behaviour the frozen runs were produced with.

The chunk objective differs from the frozen single-token objective in exactly one respect. The
assistant target carries ``chunk_size`` official action IDs instead of one, and ``chunk_size``
tokenizer positions are supervised instead of one. Everything else -- the history contract, the
image ordering, the processor pathway, the RGB-only policy input, the refusal to truncate
supervision -- is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.history import HistoryWindow

# Reused from the hash-locked frozen renderer; see the module docstring.
from spatialvlm.data.native_qwen import (  # noqa: PLC2701
    NativeQwenBatch,
    _frame_label,
    _native_rgb,
    _processor_dict,
    _target_token_ids,
    _tokenizer_for,
    _visual_token_counts,
)
from spatialvlm.eval.action_chunk_decision import (
    ACTION_CHUNK_METHOD,
    ACTION_CHUNK_PROTOCOL_VERSION,
    DEFAULT_ACTION_CHUNK_SIZE,
    ActionChunkCodebook,
    action_chunk_system_prompt,
    build_action_chunk_codebook,
    canonical_action_chunk_target,
    validate_chunk_target,
)

NATIVE_QWEN_CHUNK_RENDERER_VERSION = 1
NATIVE_QWEN_CHUNK_SUPERVISION_SCHEMA_VERSION = 1
NATIVE_QWEN_CHUNK_SUPERVISION_METHOD = ACTION_CHUNK_METHOD


@dataclass(frozen=True)
class NativeQwenChunkConfig:
    """Versioned k-action-token supervision settings for the native Qwen3-VL chunk objective."""

    chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE
    max_length: int = 4096
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int):
            raise TypeError("chunk_size must be an integer.")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least one.")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")

    @property
    def system_prompt(self) -> str:
        """Return the frozen chunk system prompt for this chunk size."""
        return action_chunk_system_prompt(self.chunk_size)


def build_native_qwen_chunk_messages(
    window: HistoryWindow,
    *,
    instruction: str,
    config: NativeQwenChunkConfig | None = None,
) -> list[dict[str, Any]]:
    """Build an official multimodal-chat message with exact chronological image order."""
    config = config or NativeQwenChunkConfig()
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string.")

    content: list[dict[str, Any]] = []
    for frame in window.frames:
        content.append({"type": "text", "text": _frame_label(frame)})
        content.append({"type": "image", "image": _native_rgb(frame.observation)})
    content.append(
        {
            "type": "text",
            "text": (
                f"Navigation instruction: {instruction.strip()}\n"
                f"Choose the next {config.chunk_size} official actions now."
            ),
        }
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": config.system_prompt}]},
        {"role": "user", "content": content},
    ]


def build_native_qwen_chunk_batch(
    processor: Any,
    window: HistoryWindow,
    *,
    instruction: str,
    target_actions: Sequence[R2RCEAction | int],
    config: NativeQwenChunkConfig | None = None,
    codebook: ActionChunkCodebook | None = None,
) -> NativeQwenBatch:
    """Render one chunk-supervised example through the official Qwen processor.

    The processor owns image resizing, vision placeholders, optional multimodal token types, and
    chat-template rendering. This function appends the canonical chunk target and EOS, then labels
    exactly the ``chunk_size`` tokenizer positions that carry official action IDs. It fails closed
    on token-contract, visual-count, sequence-field, or supervision-count drift.
    """
    config = config or NativeQwenChunkConfig()
    if len(target_actions) != config.chunk_size:
        raise ValueError(
            f"target_actions must contain exactly {config.chunk_size} actions, "
            f"got {len(target_actions)}."
        )
    official_actions = tuple(R2RCEAction(int(action)) for action in target_actions)
    # CR-021 precondition: never supervise an action after a terminal STOP.
    validate_chunk_target(official_actions)

    codebook = codebook or build_action_chunk_codebook(
        _tokenizer_for(processor), chunk_size=config.chunk_size
    )
    if codebook.chunk_size != config.chunk_size:
        raise ValueError("Supplied codebook chunk size disagrees with the configured chunk size.")

    renderer = getattr(processor, "apply_chat_template", None)
    if not callable(renderer):
        raise ValueError("Native chunk rendering requires a processor with apply_chat_template().")
    messages = build_native_qwen_chunk_messages(window, instruction=instruction, config=config)
    rendered = _processor_dict(
        renderer(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    )
    if "token_type_ids" in rendered:
        # Matches the frozen renderer: generic text-only token types are not accepted by every
        # pinned Qwen3-VL Transformers path.
        rendered.pop("token_type_ids")

    input_ids = rendered.get("input_ids")
    attention_mask = rendered.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Native Qwen input_ids must have shape [1,L].")
    if (
        not isinstance(attention_mask, torch.Tensor)
        or attention_mask.shape != input_ids.shape
        or not bool((attention_mask == 1).all().item())
    ):
        raise ValueError("Native Qwen attention_mask must be an all-one [1,L] tensor.")

    visual_counts = _visual_token_counts(processor, rendered)
    if len(visual_counts) != len(window.frames):
        raise ValueError(
            "Native Qwen processor image count does not match the history contract: "
            f"images={len(visual_counts)}, frames={len(window.frames)}."
        )

    target = canonical_action_chunk_target(official_actions)
    suffix_ids = torch.tensor(
        [_target_token_ids(processor, target)], dtype=input_ids.dtype, device=input_ids.device
    )
    expected_target = codebook.target_ids(official_actions)
    observed_target = tuple(int(v) for v in suffix_ids[0, : len(expected_target)].tolist())
    if observed_target != expected_target:
        raise RuntimeError(
            "Chunk-token contract changed while rendering the target: "
            f"expected {expected_target}, observed {observed_target}."
        )

    target_start = input_ids.shape[1]
    combined_ids = torch.cat((input_ids, suffix_ids), dim=1)
    if combined_ids.shape[1] > config.max_length:
        raise ValueError(
            "Native Qwen chunk sequence exceeds max_length; refusing to truncate supervision: "
            f"length={combined_ids.shape[1]}, max_length={config.max_length}."
        )

    model_inputs: dict[str, torch.Tensor] = {}
    for key, value in rendered.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Native Qwen processor field {key!r} must be a tensor.")
        model_inputs[key] = value
    model_inputs["input_ids"] = combined_ids
    model_inputs["attention_mask"] = torch.cat(
        (attention_mask, torch.ones_like(suffix_ids, dtype=attention_mask.dtype)), dim=1
    )

    mm_types = model_inputs.get("mm_token_type_ids")
    if mm_types is not None:
        if mm_types.shape != input_ids.shape:
            raise ValueError("mm_token_type_ids must align with the rendered prefix input_ids.")
        visual_type_count = int((mm_types != 0).sum().item())
        if visual_type_count != sum(visual_counts):
            raise ValueError(
                "mm_token_type_ids visual count does not match image_grid_thw: "
                f"types={visual_type_count}, grid={sum(visual_counts)}."
            )
        model_inputs["mm_token_type_ids"] = torch.cat(
            (mm_types, torch.zeros_like(suffix_ids, dtype=mm_types.dtype)), dim=1
        )

    supervised_positions = tuple(
        target_start + codebook.offset(index) for index in range(config.chunk_size)
    )
    labels = torch.full_like(combined_ids, config.ignore_index)
    for index, position in enumerate(supervised_positions):
        expected_token = codebook.action_token_id(official_actions[index])
        if int(combined_ids[0, position].item()) != expected_token:
            raise RuntimeError(
                f"Chunk position {index} does not carry its official action token after rendering."
            )
        labels[:, position] = combined_ids[:, position]
    model_inputs["labels"] = labels

    supervised_count = int((labels != config.ignore_index).sum().item())
    if supervised_count != config.chunk_size:
        raise RuntimeError(
            "Native Qwen chunk must supervise exactly "
            f"{config.chunk_size} official-action tokens, supervised {supervised_count}."
        )

    record = {
        "renderer_version": NATIVE_QWEN_CHUNK_RENDERER_VERSION,
        "supervision_schema_version": NATIVE_QWEN_CHUNK_SUPERVISION_SCHEMA_VERSION,
        "supervision_method": NATIVE_QWEN_CHUNK_SUPERVISION_METHOD,
        "action_chunk_protocol_version": ACTION_CHUNK_PROTOCOL_VERSION,
        "pathway": "native_qwen3_vl_processor_and_visual_encoder",
        "policy_visual_fields": ["rgb"],
        "explicitly_excluded_native_fields": ["depth", "simulator_pose"],
        "history": window.to_record(),
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(_tokenizer_for(processor)).__name__,
        "chunk_size": config.chunk_size,
        "rendered_prefix_length": target_start,
        "sequence_length": combined_ids.shape[1],
        "target_start_idx": target_start,
        "target_token_count": suffix_ids.shape[1],
        "action_token_offsets": list(codebook.action_token_offsets),
        "supervised_token_positions": list(supervised_positions),
        "supervised_token_ids": [codebook.action_token_id(action) for action in official_actions],
        "supervised_token_count": supervised_count,
        "target_action_ids": [int(action) for action in official_actions],
        "target_action_names": [action.name for action in official_actions],
        "target_chunk_is_constant": len({int(a) for a in official_actions}) == 1,
        "image_count": len(visual_counts),
        "visual_tokens_per_image": list(visual_counts),
        "total_visual_tokens": sum(visual_counts),
        "model_input_keys": sorted(model_inputs),
        "target_surface": target,
    }
    return NativeQwenBatch(model_inputs=model_inputs, record=record)
