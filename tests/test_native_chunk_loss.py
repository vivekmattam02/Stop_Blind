"""Tests for the restricted four-way chunk loss."""

from __future__ import annotations

import math

import pytest
import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.action_chunk_decision import build_action_chunk_codebook
from spatialvlm.training.native_chunk_loss import restricted_chunk_loss

PREFIX = {"Action": 100, "IDs:": 101}


class _Tokenizer:
    name_or_path = "fixture/tokenizer"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        out: list[int] = []
        for piece in text.split(" "):
            out.append(PREFIX[piece] if piece in PREFIX else 10 + int(piece))
        return out


def _codebook(k: int = 4):
    return build_action_chunk_codebook(_Tokenizer(), chunk_size=k)


CHUNK = (R2RCEAction.MOVE_FORWARD, R2RCEAction.TURN_LEFT, R2RCEAction.STOP, R2RCEAction.TURN_RIGHT)


def test_uniform_logits_give_ln_four_at_every_position() -> None:
    cb = _codebook()
    logits = torch.zeros(1, 16, 64)
    loss, per = restricted_chunk_loss(
        logits, action_token_positions=(5, 6, 7, 8), codebook=cb, target_actions=CHUNK
    )
    assert math.isclose(float(loss), math.log(4), rel_tol=1e-6)
    assert all(math.isclose(v, math.log(4), rel_tol=1e-6) for v in per)


def test_confident_correct_chunk_drives_loss_to_zero() -> None:
    cb = _codebook()
    logits = torch.full((1, 16, 64), -20.0)
    for i, action in enumerate(CHUNK):
        logits[0, 5 + i - 1, cb.action_token_id(action)] = 20.0
    loss, per = restricted_chunk_loss(
        logits, action_token_positions=(5, 6, 7, 8), codebook=cb, target_actions=CHUNK
    )
    assert float(loss) < 1e-6
    assert max(per) < 1e-6


def test_loss_is_the_mean_not_the_sum() -> None:
    """A sum would rescale the effective learning rate against the frozen runs."""
    cb = _codebook()
    logits = torch.zeros(1, 16, 64)
    loss, per = restricted_chunk_loss(
        logits, action_token_positions=(5, 6, 7, 8), codebook=cb, target_actions=CHUNK
    )
    assert math.isclose(float(loss), sum(per) / len(per), rel_tol=1e-6)
    assert not math.isclose(float(loss), sum(per), rel_tol=1e-6)


def test_only_the_four_legal_logits_matter() -> None:
    """Mass outside the four action tokens cannot change the restricted loss."""
    cb = _codebook()
    a = torch.zeros(1, 16, 64)
    b = a.clone()
    b[0, :, 40:] = 50.0  # huge mass on illegal tokens
    la, _ = restricted_chunk_loss(
        a, action_token_positions=(5, 6, 7, 8), codebook=cb, target_actions=CHUNK
    )
    lb, _ = restricted_chunk_loss(
        b, action_token_positions=(5, 6, 7, 8), codebook=cb, target_actions=CHUNK
    )
    assert math.isclose(float(la), float(lb), rel_tol=1e-9)


def test_gradient_flows_to_every_supervised_position() -> None:
    cb = _codebook()
    logits = torch.zeros(1, 16, 64, requires_grad=True)
    loss, _ = restricted_chunk_loss(
        logits, action_token_positions=(5, 6, 7, 8), codebook=cb, target_actions=CHUNK
    )
    loss.backward()
    grad = logits.grad
    assert grad is not None
    for position in (5, 6, 7, 8):
        assert float(grad[0, position - 1].abs().sum()) > 0.0
    assert float(grad[0, 0].abs().sum()) == 0.0


def test_rejects_mismatched_lengths_and_ordering() -> None:
    cb = _codebook()
    logits = torch.zeros(1, 16, 64)
    with pytest.raises(ValueError, match="Expected 4 action positions"):
        restricted_chunk_loss(
            logits, action_token_positions=(5, 6), codebook=cb, target_actions=CHUNK
        )
    with pytest.raises(ValueError, match="Expected 4 target actions"):
        restricted_chunk_loss(
            logits,
            action_token_positions=(5, 6, 7, 8),
            codebook=cb,
            target_actions=(R2RCEAction.STOP,),
        )
    with pytest.raises(ValueError, match="must increase"):
        restricted_chunk_loss(
            logits, action_token_positions=(8, 7, 6, 5), codebook=cb, target_actions=CHUNK
        )
