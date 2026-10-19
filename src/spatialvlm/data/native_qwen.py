"""Native Qwen3-VL rendering for the C0 workshop baseline.

C0 must exercise the checkpoint's own multimodal processor and visual pathway.  It must not reuse
the external ``<|image_pad|>`` expansion used by the SpatialVLM geometry arms.  This module keeps
that boundary explicit and returns model inputs separately from provenance metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.history import HistoryFrame, HistoryWindow

NATIVE_QWEN_RENDERER_VERSION = 1
NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION = 1
NATIVE_QWEN_SUPERVISION_METHOD = "single_shared_position_official_action_id_token_v1"
NATIVE_QWEN_SYSTEM_PROMPT = (
    "You are a spatial navigation assistant. Given chronological indoor visual observations and "
    "a navigation instruction, choose exactly one official action ID: 0=STOP, "
    "1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT. Respond only with `Action ID: N`, "
    "replacing N with the chosen ID."
)


@dataclass(frozen=True)
class NativeQwenSFTConfig:
    """Versioned single-action-token supervision settings for native Qwen3-VL C0."""

    max_length: int = 4096
    ignore_index: int = -100
    system_prompt: str = NATIVE_QWEN_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must be non-empty.")


@dataclass(frozen=True)
class NativeQwenBatch:
    """A native processor batch plus a non-forwarded audit record."""

    model_inputs: dict[str, torch.Tensor]
    record: dict[str, Any]


def canonical_action_target(action: R2RCEAction | int) -> str:
    """Return the one allowed C0/C1-C4 assistant action surface."""
    if isinstance(action, bool) or not isinstance(action, int):
        raise ValueError("Action must be an official integer action ID.")
    try:
        official = R2RCEAction(int(action))
    except (TypeError, ValueError) as exc:
        raise ValueError("Action must be an official R2R-CE action.") from exc
    return f"Action ID: {int(official)}"


def _frame_label(frame: HistoryFrame) -> str:
    label = f"Frame t{frame.relative_timestep:+d}"
    if frame.is_padding:
        return f"{label} (left-padding copy of the initial observation; previous action: NONE)."
    if frame.previous_action_id is None:
        return f"{label} (previous action: NONE)."
    action = R2RCEAction(frame.previous_action_id)
    return f"{label} (previous action: {int(action)}={action.name})."


def _native_rgb(observation: Any) -> Any:
    """Select RGB only when a shared history observation contains multiple sensor fields."""
    if isinstance(observation, Mapping):
        if "rgb" not in observation:
            raise ValueError(
                "A mapping observation passed to native C0 must contain an 'rgb' field."
            )
        return observation["rgb"]
    return observation


def build_native_qwen_messages(
    window: HistoryWindow,
    *,
    instruction: str,
    config: NativeQwenSFTConfig | None = None,
) -> list[dict[str, Any]]:
    """Build an official multimodal-chat message with exact chronological image order."""
    config = config or NativeQwenSFTConfig()
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
                "Choose the next official action now."
            ),
        }
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": config.system_prompt}]},
        {"role": "user", "content": content},
    ]


def _processor_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    items = getattr(value, "items", None)
    if callable(items):
        return dict(items())
    raise TypeError("Qwen processor must return a mapping-like batch.")


def _tokenizer_for(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    return tokenizer if tokenizer is not None else processor


def _encoded_target_token_ids(processor: Any, target: str) -> list[int]:
    tokenizer = _tokenizer_for(processor)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise ValueError("Qwen processor must expose a tokenizer with encode().")
    values = encode(target, add_special_tokens=False)
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    if not isinstance(values, list | tuple) or not values:
        raise ValueError("Canonical action target tokenized to an empty or unsupported value.")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Canonical action target contains an invalid token ID.")
        result.append(value)
    return result


def _target_token_ids(processor: Any, target: str) -> list[int]:
    tokenizer = _tokenizer_for(processor)
    result = _encoded_target_token_ids(processor, target)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_id, int) or eos_id < 0:
        raise ValueError("Qwen tokenizer must define a valid eos_token_id.")
    return [*result, eos_id]


def _action_token_contract(processor: Any) -> tuple[int, dict[R2RCEAction, int]]:
    """Locate the one shared token position that distinguishes all official actions."""
    tokenized = {
        action: _encoded_target_token_ids(processor, canonical_action_target(action))
        for action in R2RCEAction
    }
    shortest = min(len(values) for values in tokenized.values())
    common_prefix_length = 0
    for index in range(shortest):
        if len({values[index] for values in tokenized.values()}) != 1:
            break
        common_prefix_length += 1
    if common_prefix_length == 0:
        raise ValueError("Canonical action targets do not share a tokenizer prefix.")

    suffixes = {
        action: values[common_prefix_length:] for action, values in tokenized.items()
    }
    if any(len(values) != 1 for values in suffixes.values()):
        lengths = {action.name: len(values) for action, values in suffixes.items()}
        raise ValueError(
            "Canonical action targets must differ by exactly one token after their shared "
            f"prefix; observed suffix lengths={lengths}."
        )
    token_ids = {action: values[0] for action, values in suffixes.items()}
    if len(set(token_ids.values())) != len(R2RCEAction):
        raise ValueError("Official actions do not map to four distinct action token IDs.")
    return common_prefix_length, token_ids


def _visual_token_counts(processor: Any, batch: dict[str, Any]) -> tuple[int, ...]:
    grid = batch.get("image_grid_thw")
    if not isinstance(grid, torch.Tensor) or grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("Native Qwen image processing must return image_grid_thw shaped [N,3].")
    image_processor = getattr(processor, "image_processor", None)
    merge_size = getattr(image_processor, "merge_size", 2)
    if isinstance(merge_size, bool) or not isinstance(merge_size, int) or merge_size <= 0:
        raise ValueError("Qwen image processor must expose a positive integer merge_size.")
    counts = grid.to(dtype=torch.long).prod(dim=1)
    divisor = merge_size**2
    if bool((counts % divisor != 0).any().item()):
        raise ValueError("Qwen image grids are incompatible with the processor merge size.")
    return tuple(int(value) for value in (counts // divisor).detach().cpu().tolist())


def build_native_qwen_sft_batch(
    processor: Any,
    window: HistoryWindow,
    *,
    instruction: str,
    target_action: R2RCEAction | int,
    config: NativeQwenSFTConfig | None = None,
) -> NativeQwenBatch:
    """Render one action-token-supervised C0 example through the official Qwen processor.

    The official processor owns image resizing, vision placeholders, optional multimodal token
    types, and chat-template rendering. This function appends the canonical assistant target and
    EOS, but labels only the one shared tokenizer position that distinguishes the four official
    action IDs. It fails closed on token-contract, visual-count, or sequence-field drift.
    """
    config = config or NativeQwenSFTConfig()
    renderer = getattr(processor, "apply_chat_template", None)
    if not callable(renderer):
        raise ValueError("Native C0 requires a processor with apply_chat_template().")
    messages = build_native_qwen_messages(window, instruction=instruction, config=config)
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
        # Generic text-only token types are not accepted by every pinned Qwen3-VL Transformers
        # path. Some processor revisions emit the distinct mm_token_type_ids field and some do not.
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

    target = canonical_action_target(target_action)
    official_action = R2RCEAction(int(target_action))
    action_token_offset, official_action_token_ids = _action_token_contract(processor)
    suffix_ids = torch.tensor(
        [_target_token_ids(processor, target)], dtype=input_ids.dtype, device=input_ids.device
    )
    expected_action_token_id = official_action_token_ids[official_action]
    observed_action_token_id = int(suffix_ids[0, action_token_offset].item())
    if observed_action_token_id != expected_action_token_id:
        raise RuntimeError("Canonical action-token contract changed while rendering the target.")
    target_start = input_ids.shape[1]
    combined_ids = torch.cat((input_ids, suffix_ids), dim=1)
    if combined_ids.shape[1] > config.max_length:
        raise ValueError(
            "Native Qwen sequence exceeds max_length; refusing to truncate supervision: "
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

    supervised_token_position = target_start + action_token_offset
    labels = torch.full_like(combined_ids, config.ignore_index)
    labels[:, supervised_token_position] = combined_ids[:, supervised_token_position]
    model_inputs["labels"] = labels
    supervised_count = int((labels != config.ignore_index).sum().item())
    if supervised_count != 1:
        raise RuntimeError("Native Qwen must supervise exactly one official-action token.")

    record = {
        "renderer_version": NATIVE_QWEN_RENDERER_VERSION,
        "supervision_schema_version": NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
        "supervision_method": NATIVE_QWEN_SUPERVISION_METHOD,
        "pathway": "native_qwen3_vl_processor_and_visual_encoder",
        "policy_visual_fields": ["rgb"],
        "explicitly_excluded_native_fields": ["depth", "simulator_pose"],
        "history": window.to_record(),
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(_tokenizer_for(processor)).__name__,
        "rendered_prefix_length": target_start,
        "sequence_length": combined_ids.shape[1],
        "target_start_idx": target_start,
        "target_token_count": suffix_ids.shape[1],
        "action_token_offset": action_token_offset,
        "supervised_token_position": supervised_token_position,
        "supervised_token_id": observed_action_token_id,
        "supervised_token_count": supervised_count,
        "image_count": len(visual_counts),
        "visual_tokens_per_image": list(visual_counts),
        "total_visual_tokens": sum(visual_counts),
        "model_input_keys": sorted(model_inputs),
        "target_surface": target,
    }
    return NativeQwenBatch(model_inputs=model_inputs, record=record)
