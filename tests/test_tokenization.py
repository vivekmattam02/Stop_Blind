"""Fail-closed tests for the scientific conversation and action-label contract."""

from __future__ import annotations

import pytest

from spatialvlm.data.tokenization import SpatialTokenizerConfig, build_input_ids


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, *, placeholder: bool = True, duplicate: bool = False) -> None:
        self.placeholder = placeholder
        self.duplicate = duplicate

    def get_vocab(self) -> dict[str, int]:
        vocab = {"word": 10}
        if self.placeholder:
            vocab["<|image_pad|>"] = 3
        return vocab

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize and add_generation_prompt
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        placeholders = [3, 3] if self.duplicate else [3]
        return [1, 11, *placeholders, 12, 13]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [20 + index for index, _ in enumerate(text.split())]


def test_sft_labels_exact_complete_target_and_eos() -> None:
    output = build_input_ids(
        _Tokenizer(),
        instruction="turn left at the sofa",
        target="Action: TURN_LEFT",
        config=SpatialTokenizerConfig(num_spatial_tokens=4, max_length=64),
    )

    start = output["spatial_start_idx"]
    target_start = output["target_start_idx"]
    assert isinstance(start, int) and isinstance(target_start, int)
    assert output["input_ids"][start : start + 4].tolist() == [3, 3, 3, 3]
    assert output["labels"][:target_start].tolist() == [-100] * target_start
    assert output["labels"][target_start:].tolist() == [20, 21, 2]
    assert output["supervised_token_count"] == 3


def test_overlength_sequence_raises_instead_of_truncating_target() -> None:
    with pytest.raises(ValueError, match="refusing to truncate supervision"):
        build_input_ids(
            _Tokenizer(),
            instruction="go",
            target="Action: MOVE_FORWARD",
            config=SpatialTokenizerConfig(num_spatial_tokens=20, max_length=8),
        )


def test_missing_checkpoint_placeholder_fails_closed() -> None:
    with pytest.raises(ValueError, match="required spatial placeholder"):
        build_input_ids(_Tokenizer(placeholder=False), "go", "Action: MOVE_FORWARD")


def test_rendered_placeholder_must_be_unique() -> None:
    with pytest.raises(ValueError, match="exactly one spatial placeholder"):
        build_input_ids(_Tokenizer(duplicate=True), "go", "Action: MOVE_FORWARD")


def test_missing_chat_template_fails_closed() -> None:
    class NoTemplate:
        eos_token_id = 2

        def get_vocab(self) -> dict[str, int]:
            return {"<|image_pad|>": 3}

    with pytest.raises(ValueError, match="scientific SFT requires"):
        build_input_ids(NoTemplate(), "go", "Action: MOVE_FORWARD")
