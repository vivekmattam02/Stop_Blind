"""Tests for batch-aware SpatialVLM collation."""

from __future__ import annotations

import numpy as np
import torch

from spatialvlm.data.collation import SpatialVLMCollator
from spatialvlm.data.tokenization import SpatialTokenizerConfig


class _LengthTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def get_vocab(self) -> dict[str, int]:
        return {"<|image_pad|>": 3}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        token_ids = [10 + i for i, _ in enumerate(text.split())]
        return ([1] if add_special_tokens else []) + token_ids

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize and add_generation_prompt
        instruction_length = len(messages[1]["content"].split())
        return [1] + list(range(10, 10 + instruction_length)) + [3, 90]


class _SiglipProcessor:
    def __call__(self, *, images, return_tensors, max_num_patches):
        assert images.dtype == np.uint8 and return_tensors == "pt"
        return {
            "pixel_values": torch.zeros(1, max_num_patches, 12),
            "pixel_attention_mask": torch.ones(1, max_num_patches, dtype=torch.long),
            "spatial_shapes": torch.tensor([[24, 24]], dtype=torch.long),
        }


class _DinoProcessor:
    def __call__(self, *, images, return_tensors, size, do_center_crop):
        assert images.dtype == np.uint8 and return_tensors == "pt"
        assert do_center_crop is False
        return {"pixel_values": torch.zeros(1, 3, size["height"], size["width"])}


def _frame(instruction: str) -> dict[str, object]:
    return {
        "rgb": torch.rand(3, 8, 8),
        "depth": torch.rand(8, 8) + 0.5,
        "instruction": instruction,
        "target": "Action: MOVE_FORWARD",
    }


def test_collator_preserves_per_sample_spatial_starts():
    collator = SpatialVLMCollator(
        _LengthTokenizer(),
        siglip_processor=_SiglipProcessor(),
        dinov2_processor=_DinoProcessor(),
        stage="sft",
        tok_config=SpatialTokenizerConfig(num_spatial_tokens=4, max_length=128),
    )

    batch = collator(
        [
            _frame("go left"),
            _frame("walk forward through the long hallway and then turn left"),
        ]
    )

    starts = batch["spatial_start_idx"]
    assert isinstance(starts, torch.Tensor)
    assert starts.shape == (2,)
    assert starts.dtype == torch.long
    assert starts[0] != starts[1]
