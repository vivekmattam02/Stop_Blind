"""No-download tests for the native Qwen C0 processor boundary."""

from __future__ import annotations

import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode, build_history_window
from spatialvlm.data.native_qwen import (
    NATIVE_QWEN_SUPERVISION_METHOD,
    NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION,
    NativeQwenSFTConfig,
    build_native_qwen_messages,
    build_native_qwen_sft_batch,
)


class _Tokenizer:
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        assert text.startswith("Action ID: ")
        return [100, 101, 10 + int(text[-1])]


class _ImageProcessor:
    merge_size = 2


class _OfficialProcessorFixture:
    tokenizer = _Tokenizer()
    image_processor = _ImageProcessor()

    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
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


def _window(mode: HistoryMode):
    return build_history_window(
        ["image-0", "image-1", "image-2", "image-3"],
        [
            R2RCEAction.MOVE_FORWARD,
            R2RCEAction.TURN_LEFT,
            R2RCEAction.TURN_RIGHT,
        ],
        contract=HISTORY_CONTRACTS[mode],
    )


def test_native_messages_are_chronological_and_contain_no_depth_or_pose() -> None:
    messages = build_native_qwen_messages(_window(HistoryMode.H4_SHORT_WINDOW), instruction="go")
    content = messages[1]["content"]

    assert [item["image"] for item in content if item["type"] == "image"] == [
        "image-0",
        "image-1",
        "image-2",
        "image-3",
    ]
    rendered_text = " ".join(item["text"] for item in content if item["type"] == "text")
    assert "t-3" in rendered_text and "t+0" in rendered_text
    assert "simulator pose" not in rendered_text.lower()
    assert "depth" not in rendered_text.lower()


def test_native_messages_select_rgb_from_shared_rgbd_observations() -> None:
    window = build_history_window(
        [{"rgb": "rgb-0", "depth": "secret-depth", "simulator_pose": "secret-pose"}],
        [],
        contract=HISTORY_CONTRACTS[HistoryMode.H0_CURRENT],
    )
    messages = build_native_qwen_messages(window, instruction="go")
    content = messages[1]["content"]

    assert [item["image"] for item in content if item["type"] == "image"] == ["rgb-0"]
    assert "secret-depth" not in str(messages)
    assert "secret-pose" not in str(messages)


def test_native_batch_uses_processor_visual_fields_and_one_action_label() -> None:
    processor = _OfficialProcessorFixture()
    batch = build_native_qwen_sft_batch(
        processor,
        _window(HistoryMode.H4_SHORT_WINDOW),
        instruction="walk past the sofa",
        target_action=R2RCEAction.TURN_LEFT,
    )

    inputs = batch.model_inputs
    target_start = batch.record["target_start_idx"]
    assert "token_type_ids" not in inputs
    assert inputs["pixel_values"].shape[0] == 4
    assert inputs["image_grid_thw"].shape == (4, 3)
    assert inputs["labels"][0, :target_start].tolist() == [-100] * target_start
    assert inputs["input_ids"][0, target_start:].tolist() == [100, 101, 12, 2]
    assert inputs["labels"][0, target_start:].tolist() == [-100, -100, 12, -100]
    assert int((inputs["labels"] != -100).sum().item()) == 1
    assert inputs["mm_token_type_ids"].shape == inputs["input_ids"].shape
    assert inputs["mm_token_type_ids"][0, target_start:].tolist() == [0, 0, 0, 0]
    assert batch.record["image_count"] == 4
    assert batch.record["visual_tokens_per_image"] == [1, 1, 1, 1]
    assert batch.record["target_surface"] == "Action ID: 2"
    assert batch.record["target_token_count"] == 4
    assert batch.record["action_token_offset"] == 2
    assert batch.record["supervised_token_position"] == target_start + 2
    assert batch.record["supervised_token_id"] == 12
    assert batch.record["supervised_token_count"] == 1
    assert batch.record["supervision_schema_version"] == NATIVE_QWEN_SUPERVISION_SCHEMA_VERSION
    assert batch.record["supervision_method"] == NATIVE_QWEN_SUPERVISION_METHOD


def test_h0_and_h1_use_one_identical_current_image_but_different_metadata() -> None:
    h0 = build_native_qwen_messages(_window(HistoryMode.H0_CURRENT), instruction="go")
    h1 = build_native_qwen_messages(_window(HistoryMode.H1_PREVIOUS_ACTION), instruction="go")
    h0_content = h0[1]["content"]
    h1_content = h1[1]["content"]

    assert [item["image"] for item in h0_content if item["type"] == "image"] == ["image-3"]
    assert [item["image"] for item in h1_content if item["type"] == "image"] == ["image-3"]
    h0_text = " ".join(item["text"] for item in h0_content if item["type"] == "text")
    h1_text = " ".join(item["text"] for item in h1_content if item["type"] == "text")
    assert "TURN_RIGHT" not in h0_text
    assert "TURN_RIGHT" in h1_text


def test_native_batch_refuses_target_truncation() -> None:
    processor = _OfficialProcessorFixture()
    prefix = 8 + 1
    try:
        build_native_qwen_sft_batch(
            processor,
            _window(HistoryMode.H0_CURRENT),
            instruction="go",
            target_action=R2RCEAction.STOP,
            config=NativeQwenSFTConfig(max_length=prefix + 3),
        )
    except ValueError as exc:
        assert "refusing to truncate" in str(exc)
    else:
        raise AssertionError("Expected max-length failure.")


def test_native_batch_refuses_multi_token_action_branches() -> None:
    class _InvalidTokenizer(_Tokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            values = super().encode(text, add_special_tokens=add_special_tokens)
            if text.endswith("2"):
                values.append(999)
            return values

    processor = _OfficialProcessorFixture()
    processor.tokenizer = _InvalidTokenizer()

    try:
        build_native_qwen_sft_batch(
            processor,
            _window(HistoryMode.H0_CURRENT),
            instruction="go",
            target_action=R2RCEAction.STOP,
        )
    except ValueError as exc:
        assert "differ by exactly one token" in str(exc)
    else:
        raise AssertionError("Expected the multi-token action branch to fail closed.")


def test_native_batch_refuses_non_distinct_action_tokens() -> None:
    class _InvalidTokenizer(_Tokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            values = super().encode(text, add_special_tokens=add_special_tokens)
            # Preserve the shared prefix and one-token suffix while making only STOP and
            # MOVE_FORWARD collide. Making every target identical would instead exercise the
            # earlier suffix-length guard because the whole sequence becomes the common prefix.
            if text.endswith("1"):
                values[-1] = 10
            return values

    processor = _OfficialProcessorFixture()
    processor.tokenizer = _InvalidTokenizer()

    try:
        build_native_qwen_sft_batch(
            processor,
            _window(HistoryMode.H0_CURRENT),
            instruction="go",
            target_action=R2RCEAction.STOP,
        )
    except ValueError as exc:
        assert "four distinct" in str(exc)
    else:
        raise AssertionError("Expected duplicate action tokens to fail closed.")
