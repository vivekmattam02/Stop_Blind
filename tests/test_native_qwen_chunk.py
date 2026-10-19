"""No-download tests for the native Qwen chunk-objective processor boundary."""

from __future__ import annotations

import pytest
import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode, build_history_window
from spatialvlm.data.native_qwen_chunk import (
    NATIVE_QWEN_CHUNK_SUPERVISION_METHOD,
    NativeQwenChunkConfig,
    build_native_qwen_chunk_batch,
    build_native_qwen_chunk_messages,
)
from spatialvlm.eval.action_chunk_decision import build_action_chunk_codebook

# STOP may only be the final action of a chunk; see validate_chunk_target.
CHUNK = (
    R2RCEAction.MOVE_FORWARD,
    R2RCEAction.TURN_LEFT,
    R2RCEAction.TURN_RIGHT,
    R2RCEAction.STOP,
)


class _Tokenizer:
    """Mimics the real Qwen layout: prefix, then an alternating space/digit run."""

    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        assert text.startswith("Action IDs: ")
        digits = text[len("Action IDs: ") :].split(" ")
        out = [100, 101, 25]
        for digit in digits:
            out.extend([220, 10 + int(digit)])
        return out


class _ImageProcessor:
    merge_size = 2


class _OfficialProcessorFixture:
    tokenizer = _Tokenizer()
    image_processor = _ImageProcessor()

    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        images = [
            item
            for item in messages[1]["content"]
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        image_count = len(images)
        prefix_length = 8 + image_count
        mm_types = torch.zeros(1, prefix_length, dtype=torch.long)
        mm_types[0, 2 : 2 + image_count] = 1
        return {
            "input_ids": torch.arange(prefix_length).unsqueeze(0),
            "attention_mask": torch.ones(1, prefix_length, dtype=torch.long),
            "mm_token_type_ids": mm_types,
            "token_type_ids": torch.zeros(1, prefix_length, dtype=torch.long),
            "pixel_values": torch.zeros(image_count, 3, 4, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * image_count),
        }


def _window():
    return build_history_window(
        ["image-0", "image-1", "image-2", "image-3"],
        [R2RCEAction.MOVE_FORWARD, R2RCEAction.TURN_LEFT, R2RCEAction.TURN_RIGHT],
        contract=HISTORY_CONTRACTS[HistoryMode.H4_SHORT_WINDOW],
    )


def _build(**kwargs):
    processor = _OfficialProcessorFixture()
    batch = build_native_qwen_chunk_batch(
        processor,
        _window(),
        instruction="Walk to the kitchen.",
        target_actions=kwargs.pop("target_actions", CHUNK),
        **kwargs,
    )
    return processor, batch


# --------------------------------------------------------------------------- codebook wiring


def test_codebook_offsets_match_the_alternating_layout() -> None:
    codebook = build_action_chunk_codebook(_Tokenizer(), chunk_size=4)
    assert codebook.action_token_offsets == (4, 6, 8, 10)
    assert codebook.action_token_ids == (10, 11, 12, 13)


# --------------------------------------------------------------------------- supervision


def test_batch_supervises_exactly_the_chunk_positions() -> None:
    _, batch = _build()
    labels = batch.model_inputs["labels"]
    supervised = (labels != -100).nonzero(as_tuple=True)[1].tolist()
    assert supervised == batch.record["supervised_token_positions"]
    assert len(supervised) == 4
    assert batch.record["supervised_token_count"] == 4


def test_supervised_positions_carry_the_official_action_tokens() -> None:
    _, batch = _build()
    ids = batch.model_inputs["input_ids"]
    labels = batch.model_inputs["labels"]
    expected = [10 + int(action) for action in CHUNK]
    for position, token in zip(batch.record["supervised_token_positions"], expected):
        assert int(ids[0, position]) == token
        assert int(labels[0, position]) == token
    assert batch.record["supervised_token_ids"] == expected


def test_every_unsupervised_position_is_ignored() -> None:
    _, batch = _build()
    labels = batch.model_inputs["labels"]
    supervised = set(batch.record["supervised_token_positions"])
    for index in range(labels.shape[1]):
        if index not in supervised:
            assert int(labels[0, index]) == -100


def test_record_declares_the_chunk_method_and_targets() -> None:
    _, batch = _build()
    record = batch.record
    assert record["supervision_method"] == NATIVE_QWEN_CHUNK_SUPERVISION_METHOD
    assert record["chunk_size"] == 4
    assert record["target_action_ids"] == [int(a) for a in CHUNK]
    assert record["target_action_names"] == [a.name for a in CHUNK]
    assert record["target_surface"] == "Action IDs: 1 2 3 0"
    assert record["target_chunk_is_constant"] is False
    assert record["policy_visual_fields"] == ["rgb"]
    assert record["explicitly_excluded_native_fields"] == ["depth", "simulator_pose"]


def test_constant_chunk_is_flagged() -> None:
    _, batch = _build(target_actions=(R2RCEAction.MOVE_FORWARD,) * 4)
    assert batch.record["target_chunk_is_constant"] is True


# --------------------------------------------------------------------------- processor contract


def test_messages_carry_one_image_per_frame_and_the_chunk_prompt() -> None:
    messages = build_native_qwen_chunk_messages(
        _window(), instruction="Walk to the kitchen.", config=NativeQwenChunkConfig(chunk_size=4)
    )
    images = [item for item in messages[1]["content"] if item.get("type") == "image"]
    assert len(images) == 4
    assert "next 4 official action IDs" in messages[0]["content"][0]["text"]
    assert "Choose the next 4 official actions now." in messages[1]["content"][-1]["text"]


def test_generic_token_type_ids_are_dropped_and_mm_types_extended() -> None:
    _, batch = _build()
    assert "token_type_ids" not in batch.model_inputs
    mm = batch.model_inputs["mm_token_type_ids"]
    ids = batch.model_inputs["input_ids"]
    assert mm.shape == ids.shape
    assert int((mm != 0).sum()) == batch.record["total_visual_tokens"]


def test_attention_mask_covers_the_appended_target() -> None:
    _, batch = _build()
    mask = batch.model_inputs["attention_mask"]
    assert mask.shape == batch.model_inputs["input_ids"].shape
    assert bool((mask == 1).all())


def test_visual_token_count_matches_the_history_contract() -> None:
    _, batch = _build()
    assert batch.record["image_count"] == 4
    assert batch.record["visual_tokens_per_image"] == [1, 1, 1, 1]


# --------------------------------------------------------------------------- fail-closed


def test_rejects_a_chunk_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly 4 actions"):
        _build(target_actions=(R2RCEAction.STOP, R2RCEAction.STOP))


def test_refuses_to_truncate_supervision() -> None:
    with pytest.raises(ValueError, match="refusing to truncate supervision"):
        _build(config=NativeQwenChunkConfig(chunk_size=4, max_length=8))


def test_rejects_a_codebook_of_a_different_chunk_size() -> None:
    codebook = build_action_chunk_codebook(_Tokenizer(), chunk_size=2)
    with pytest.raises(ValueError, match="codebook chunk size disagrees"):
        _build(codebook=codebook)


def test_rejects_an_empty_instruction() -> None:
    processor = _OfficialProcessorFixture()
    with pytest.raises(ValueError, match="non-empty string"):
        build_native_qwen_chunk_batch(processor, _window(), instruction="   ", target_actions=CHUNK)


def test_config_rejects_a_nonpositive_chunk_size() -> None:
    with pytest.raises(ValueError, match="at least one"):
        NativeQwenChunkConfig(chunk_size=0)


def test_chunk_size_two_supervises_two_positions() -> None:
    _, batch = _build(
        target_actions=(R2RCEAction.MOVE_FORWARD, R2RCEAction.STOP),
        config=NativeQwenChunkConfig(chunk_size=2),
    )
    assert batch.record["supervised_token_count"] == 2
    assert batch.record["target_surface"] == "Action IDs: 1 0"


def test_renderer_rejects_a_post_stop_tail() -> None:
    """The CR-021 STOP-tail precondition is enforced at the training boundary, not assumed."""
    with pytest.raises(ValueError, match="only as its final action"):
        _build(
            target_actions=(
                R2RCEAction.MOVE_FORWARD,
                R2RCEAction.STOP,
                R2RCEAction.TURN_LEFT,
                R2RCEAction.TURN_RIGHT,
            )
        )
