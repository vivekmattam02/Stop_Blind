"""Adversarial tests for the length-fair four-action decision contract."""

from __future__ import annotations

import math

import pytest
import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.action_decision import (
    ACTION_DECISION_METHOD,
    ActionTokenCodebook,
    build_action_token_codebook,
    canonical_action_target,
    locate_action_token,
    locate_shared_action_position,
    score_action_token_logits,
    select_action_from_token_nll,
    validate_action_decision_record,
)


class _Tokenizer:
    name_or_path = "fixture/tokenizer"

    def __init__(self, suffixes: dict[str, list[int]] | None = None) -> None:
        self.suffixes = suffixes or {"0": [10], "1": [11], "2": [12], "3": [13]}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [100, 101, *self.suffixes[text[-1]]]


def _codebook() -> ActionTokenCodebook:
    return build_action_token_codebook(_Tokenizer())


def test_codebook_uses_official_ids_and_one_distinct_token() -> None:
    codebook = _codebook()
    record = codebook.to_record()

    assert codebook.shared_prefix_token_ids == (100, 101)
    assert codebook.action_token_ids == (10, 11, 12, 13)
    assert record["method"] == ACTION_DECISION_METHOD
    assert [record["actions"][action.name]["action_id"] for action in R2RCEAction] == [0, 1, 2, 3]
    assert canonical_action_target(R2RCEAction.TURN_RIGHT) == "Action ID: 3"


def test_codebook_rejects_variable_length_action_suffix() -> None:
    tokenizer = _Tokenizer({"0": [10], "1": [11, 21], "2": [12], "3": [13]})
    with pytest.raises(ValueError, match="exactly one action-specific token"):
        build_action_token_codebook(tokenizer)


def test_codebook_rejects_colliding_action_tokens() -> None:
    tokenizer = _Tokenizer({"0": [10], "1": [10], "2": [12], "3": [13]})
    with pytest.raises(ValueError, match="four distinct"):
        build_action_token_codebook(tokenizer)


def test_locate_action_token_requires_exact_rendered_target_and_supervision() -> None:
    codebook = _codebook()
    rendered = {
        "input_ids": torch.tensor([1, 2, 100, 101, 12, 99]),
        "labels": torch.tensor([-100, -100, 100, 101, 12, 99]),
        "target_start_idx": 2,
    }
    assert locate_action_token(rendered, action=R2RCEAction.TURN_LEFT, codebook=codebook) == 4

    rendered["labels"] = torch.tensor([-100, -100, 100, 101, -100, 99])
    with pytest.raises(ValueError, match="not supervised"):
        locate_action_token(rendered, action=R2RCEAction.TURN_LEFT, codebook=codebook)


def test_shared_position_rejects_candidate_context_drift() -> None:
    codebook = _codebook()
    rendered = {
        action: {
            "input_ids": torch.tensor([1, 2, 100, 101, 10 + int(action), 99]),
            "labels": torch.tensor([-100, -100, 100, 101, 10 + int(action), 99]),
            "target_start_idx": 2,
        }
        for action in R2RCEAction
    }
    assert locate_shared_action_position(rendered, codebook=codebook) == 4

    rendered[R2RCEAction.TURN_RIGHT]["input_ids"][1] = 7
    with pytest.raises(ValueError, match="identical pre-decision context"):
        locate_shared_action_position(rendered, codebook=codebook)


def test_single_position_scoring_is_not_biased_by_surface_length() -> None:
    codebook = _codebook()
    logits = torch.zeros(1, 8, 32)
    logits[0, 4, 10] = 2.0
    logits[0, 4, 11] = 1.0
    logits[0, 4, 12] = 0.0
    logits[0, 4, 13] = 3.0

    decision = score_action_token_logits(logits, action_token_position=5, codebook=codebook)

    assert decision["predicted_action"] == "TURN_RIGHT"
    assert decision["scores"]["TURN_RIGHT"]["scored_token_count"] == 1
    assert sum(
        score["restricted_probability"] for score in decision["scores"].values()
    ) == pytest.approx(1.0)
    assert validate_action_decision_record(decision, codebook_record=codebook.to_record()) == []


def test_selection_ties_break_in_official_action_order() -> None:
    scores = {
        action.name: {
            "action_id": int(action),
            "action_token_id": 10 + int(action),
            "scored_token_position": 7,
            "causal_logit_position": 6,
            "scored_token_count": 1,
            "token_nll": 1.0,
        }
        for action in R2RCEAction
    }
    assert select_action_from_token_nll(scores) == "STOP"


def test_selection_rejects_different_positions_and_nonfinite_values() -> None:
    scores = {
        action.name: {
            "action_id": int(action),
            "action_token_id": 10 + int(action),
            "scored_token_position": 7 + int(action == R2RCEAction.TURN_RIGHT),
            "causal_logit_position": 6 + int(action == R2RCEAction.TURN_RIGHT),
            "scored_token_count": 1,
            "token_nll": 1.0,
        }
        for action in R2RCEAction
    }
    with pytest.raises(ValueError, match="same causal position"):
        select_action_from_token_nll(scores)

    scores["TURN_RIGHT"]["scored_token_position"] = 7
    scores["TURN_RIGHT"]["causal_logit_position"] = 6
    scores["STOP"]["token_nll"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        select_action_from_token_nll(scores)


def test_decision_validator_detects_probability_and_codebook_tampering() -> None:
    codebook = _codebook()
    logits = torch.zeros(1, 8, 32)
    decision = score_action_token_logits(logits, action_token_position=5, codebook=codebook)
    decision["scores"]["STOP"]["restricted_probability"] = 0.9
    decision["scores"]["MOVE_FORWARD"]["action_token_id"] = 999

    errors = validate_action_decision_record(decision, codebook_record=codebook.to_record())
    assert "restricted action probabilities do not match token NLLs" in errors
    assert "MOVE_FORWARD: codebook token ID mismatch" in errors


def test_scoring_rejects_nonfinite_logits_and_out_of_vocab_codebook() -> None:
    codebook = _codebook()
    logits = torch.zeros(1, 8, 32)
    logits[0, 4, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        score_action_token_logits(logits, action_token_position=5, codebook=codebook)

    invalid = ActionTokenCodebook(
        shared_prefix_token_ids=(100,),
        action_token_ids=(10, 11, 12, 99),
        target_token_ids=((100, 10), (100, 11), (100, 12), (100, 99)),
        tokenizer_class="fixture",
        tokenizer_name_or_path=None,
    )
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        score_action_token_logits(torch.zeros(1, 8, 32), action_token_position=5, codebook=invalid)
