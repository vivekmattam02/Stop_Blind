"""Tests for the autoregressive chunk scoring contract (CR-021 precondition)."""

from __future__ import annotations

import pytest
import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode, build_history_window
from spatialvlm.eval.action_chunk_decision import build_action_chunk_codebook
from spatialvlm.eval.native_chunk_scorer import (
    CHUNK_SCORER_METHOD,
    AutoregressiveChunkScorer,
    ChunkScorerConfig,
    chunk_separator_token_ids,
)


class _Tokenizer:
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        digits = text[len("Action IDs: ") :].split(" ")
        out = [100, 101, 25]
        for d in digits:
            out.extend([220, 10 + int(d)])
        return out


class _ImageProcessor:
    merge_size = 2


class _Processor:
    tokenizer = _Tokenizer()
    image_processor = _ImageProcessor()

    def apply_chat_template(self, messages, **kwargs):
        images = [i for i in messages[1]["content"] if i.get("type") == "image"]
        n = 8 + len(images)
        return {
            "input_ids": torch.arange(n).unsqueeze(0),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(1, n, dtype=torch.long),
            "token_type_ids": torch.zeros(1, n, dtype=torch.long),
            "pixel_values": torch.zeros(len(images), 3, 4, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }


class _Out:
    def __init__(self, logits):
        self.logits = logits
        self.past_key_values = ("cache",)


class _Model(torch.nn.Module):
    """Emits a scripted action at each decode step and records how it was called."""

    def __init__(self, script, codebook):
        super().__init__()
        self.script = list(script)
        self.codebook = codebook
        self.calls: list[dict] = []
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **kwargs):
        self.calls.append(
            {
                "has_pixels": "pixel_values" in kwargs,
                "has_cache": kwargs.get("past_key_values") is not None,
                "len": int(kwargs["input_ids"].shape[1]),
            }
        )
        action = self.script[len(self.calls) - 1]
        logits = torch.full((1, int(kwargs["input_ids"].shape[1]), 64), -10.0)
        logits[0, -1, self.codebook.action_token_id(action)] = 10.0
        return _Out(logits)


def _codebook(k=4):
    return build_action_chunk_codebook(_Tokenizer(), chunk_size=k)


def _window():
    return build_history_window(
        ["a", "b", "c", "d"],
        [R2RCEAction.MOVE_FORWARD, R2RCEAction.TURN_LEFT, R2RCEAction.TURN_RIGHT],
        contract=HISTORY_CONTRACTS[HistoryMode.H4_SHORT_WINDOW],
    )


def _scorer(script, k=4):
    cb = _codebook(k)
    model = _Model(script, cb)
    return AutoregressiveChunkScorer(
        model, _Processor(), cb, config=ChunkScorerConfig(chunk_size=k), device=torch.device("cpu")
    ), model


def test_separators_are_derived_from_the_codebook() -> None:
    assert chunk_separator_token_ids(_codebook()) == ((220,), (220,), (220,))


def test_decodes_the_scripted_chunk_autoregressively() -> None:
    script = [R2RCEAction.MOVE_FORWARD, R2RCEAction.TURN_LEFT,
              R2RCEAction.TURN_RIGHT, R2RCEAction.MOVE_FORWARD]
    scorer, model = _scorer(script)
    actions, diag = scorer.score_chunk(window=_window(), instruction="go", chunk_length=4)
    assert list(actions) == script
    assert diag["proposed_chunk"] == [a.name for a in script]
    assert diag["scorer_method"] == CHUNK_SCORER_METHOD
    assert diag["model_calls"] == 4


def test_one_prefill_then_cached_decodes() -> None:
    """The saving comes from KV-cache reuse, not from a single forward pass."""
    scorer, model = _scorer([R2RCEAction.MOVE_FORWARD] * 4)
    scorer.score_chunk(window=_window(), instruction="go", chunk_length=4)
    assert model.calls[0]["has_pixels"] is True
    assert model.calls[0]["has_cache"] is False
    for call in model.calls[1:]:
        assert call["has_pixels"] is False
        assert call["has_cache"] is True
        assert call["len"] == 2  # action token + one separator


def test_stop_tail_is_discarded_not_executed() -> None:
    script = [R2RCEAction.MOVE_FORWARD, R2RCEAction.STOP,
              R2RCEAction.TURN_LEFT, R2RCEAction.TURN_RIGHT]
    scorer, _ = _scorer(script)
    actions, diag = scorer.score_chunk(window=_window(), instruction="go", chunk_length=4)
    assert list(actions) == [R2RCEAction.MOVE_FORWARD, R2RCEAction.STOP]
    assert diag["terminated_by_stop"] is True
    assert diag["discarded_after_stop"] == 2
    assert diag["proposed_chunk_ids"] == [1, 0, 2, 3]
    assert diag["executed_chunk_ids"] == [1, 0]


def test_stop_at_first_position_executes_only_stop() -> None:
    scorer, _ = _scorer([R2RCEAction.STOP] + [R2RCEAction.MOVE_FORWARD] * 3)
    actions, diag = scorer.score_chunk(window=_window(), instruction="go", chunk_length=4)
    assert list(actions) == [R2RCEAction.STOP]
    assert diag["discarded_after_stop"] == 3


def test_constant_chunk_is_flagged_and_timed() -> None:
    scorer, _ = _scorer([R2RCEAction.MOVE_FORWARD] * 4)
    _, diag = scorer.score_chunk(window=_window(), instruction="go", chunk_length=4)
    assert diag["proposed_chunk_is_constant"] is True
    assert diag["prefill_seconds"] > 0
    assert len(diag["decode_seconds"]) == 3
    assert diag["seconds_per_executed_action"] > 0


def test_restricted_distributions_are_four_way_and_normalised() -> None:
    scorer, _ = _scorer([R2RCEAction.TURN_LEFT] * 4)
    _, diag = scorer.score_chunk(window=_window(), instruction="go", chunk_length=4)
    assert len(diag["restricted_distributions"]) == 4
    for row in diag["restricted_distributions"]:
        assert len(row) == 4
        assert abs(sum(row) - 1.0) < 1e-5


def test_rejects_a_chunk_length_it_was_not_configured_for() -> None:
    scorer, _ = _scorer([R2RCEAction.STOP] * 4)
    with pytest.raises(ValueError, match="configured for chunk_size=4"):
        scorer.score_chunk(window=_window(), instruction="go", chunk_length=2)


def test_rejects_a_codebook_of_a_different_size() -> None:
    cb = _codebook(2)
    with pytest.raises(ValueError, match="disagrees with the supplied codebook"):
        AutoregressiveChunkScorer(
            _Model([], cb), _Processor(), cb,
            config=ChunkScorerConfig(chunk_size=4), device=torch.device("cpu"),
        )
