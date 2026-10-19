"""Adversarial tests for the length-fair action-chunk decision contract."""

from __future__ import annotations

import pytest
import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.action_chunk_decision import (
    ACTION_CHUNK_METHOD,
    ACTION_CHUNK_PROTOCOL_VERSION,
    ActionChunkCodebook,
    action_chunk_system_prompt,
    build_action_chunk_codebook,
    canonical_action_chunk_target,
    iter_action_chunks,
    locate_chunk_action_positions,
    score_action_chunk_logits,
    validate_action_chunk_record,
)

PREFIX = {"Action": 100, "IDs:": 101}
DIGITS = {"0": 10, "1": 11, "2": 12, "3": 13}


class _Tokenizer:
    """Whitespace tokenizer mimicking a checkpoint that gives each digit one token."""

    name_or_path = "fixture/tokenizer"

    def __init__(
        self,
        digits: dict[str, list[int]] | None = None,
        position_shift: int = 0,
    ) -> None:
        self.digits = digits or {k: [v] for k, v in DIGITS.items()}
        self.position_shift = position_shift

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        out: list[int] = []
        digit_index = 0
        for piece in text.split(" "):
            if piece in PREFIX:
                out.append(PREFIX[piece])
                continue
            ids = list(self.digits[piece])
            if self.position_shift:
                ids = [i + self.position_shift * digit_index for i in ids]
            out.extend(ids)
            digit_index += 1
        return out


def _codebook(chunk_size: int = 4, **kwargs: object) -> ActionChunkCodebook:
    return build_action_chunk_codebook(_Tokenizer(**kwargs), chunk_size=chunk_size)  # type: ignore[arg-type]


def _rendered(
    actions: list[R2RCEAction], codebook: ActionChunkCodebook, *, supervise: bool = True
) -> dict[str, torch.Tensor | int | str]:
    lead = [7, 8, 9]
    target = list(codebook.target_ids(actions))
    input_ids = torch.tensor(lead + target, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    if supervise:
        for position in range(codebook.chunk_size):
            index = len(lead) + codebook.offset(position)
            labels[index] = input_ids[index]
    return {"input_ids": input_ids, "labels": labels, "target_start_idx": len(lead)}


# --------------------------------------------------------------------------- codebook


def test_codebook_maps_each_position_to_one_distinct_token() -> None:
    codebook = _codebook()
    assert codebook.chunk_size == 4
    assert codebook.shared_prefix_token_ids == (100, 101)
    assert codebook.action_token_ids == (10, 11, 12, 13)
    assert codebook.action_token_offsets == (2, 3, 4, 5)
    record = codebook.to_record()
    assert record["method"] == ACTION_CHUNK_METHOD
    assert record["supervised_token_count"] == 4
    assert record["action_specific_token_count_per_position"] == 1


def test_canonical_target_and_prompt_are_frozen() -> None:
    chunk = [R2RCEAction.MOVE_FORWARD, R2RCEAction.TURN_LEFT, R2RCEAction.STOP, R2RCEAction.STOP]
    assert canonical_action_chunk_target(chunk) == "Action IDs: 1 2 0 0"
    assert "next 4 official action IDs" in action_chunk_system_prompt(4)
    assert "`Action IDs: N N N N`" in action_chunk_system_prompt(4)


def test_target_ids_reconstructs_every_position() -> None:
    codebook = _codebook()
    chunk = [R2RCEAction.TURN_RIGHT, R2RCEAction.STOP, R2RCEAction.MOVE_FORWARD, R2RCEAction.STOP]
    assert codebook.target_ids(chunk) == (100, 101, 13, 10, 11, 10)


def test_codebook_rejects_variable_length_action_token() -> None:
    with pytest.raises(ValueError, match="same length"):
        _codebook(digits={"0": [10], "1": [11, 21], "2": [12], "3": [13]})


def test_codebook_rejects_action_colliding_with_the_reference() -> None:
    """A digit indistinguishable from STOP changes no token index and must be caught there."""
    with pytest.raises(ValueError, match="changed 0 token indices"):
        _codebook(digits={"0": [10], "1": [10], "2": [12], "3": [13]})


def test_codebook_rejects_two_colliding_non_reference_actions() -> None:
    """A collision between two non-reference digits survives to the distinctness check."""
    with pytest.raises(ValueError, match="four distinct"):
        _codebook(digits={"0": [10], "1": [11], "2": [12], "3": [12]})


def test_codebook_rejects_position_dependent_action_tokens() -> None:
    """A checkpoint whose digit token depends on where the digit sits is inadmissible."""
    with pytest.raises(ValueError, match="same action token IDs"):
        _codebook(position_shift=100)


def test_codebook_rejects_nonpositive_chunk_size() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _codebook(chunk_size=0)


def test_offset_rejects_out_of_range_position() -> None:
    with pytest.raises(IndexError):
        _codebook().offset(4)


def test_target_ids_rejects_wrong_chunk_length() -> None:
    with pytest.raises(ValueError, match="exactly 4 actions"):
        _codebook().target_ids([R2RCEAction.STOP])


# --------------------------------------------------------------------------- chunking


def test_iter_chunks_anchors_final_chunk_so_stop_is_covered() -> None:
    actions = [R2RCEAction.MOVE_FORWARD] * 5 + [R2RCEAction.STOP]
    chunks = list(iter_action_chunks(actions, chunk_size=4))
    assert [start for start, _ in chunks] == [0, 2]
    assert chunks[-1][1][-1] is R2RCEAction.STOP


def test_iter_chunks_exact_multiple_does_not_duplicate() -> None:
    actions = [R2RCEAction.MOVE_FORWARD] * 3 + [R2RCEAction.STOP]
    assert [start for start, _ in iter_action_chunks(actions, chunk_size=4)] == [0]


def test_iter_chunks_skips_episodes_shorter_than_the_chunk() -> None:
    actions = [R2RCEAction.MOVE_FORWARD, R2RCEAction.STOP]
    assert list(iter_action_chunks(actions, chunk_size=4)) == []


def test_iter_chunks_honours_stride() -> None:
    actions = [R2RCEAction.MOVE_FORWARD] * 7 + [R2RCEAction.STOP]
    starts = [start for start, _ in iter_action_chunks(actions, chunk_size=4, stride=2)]
    assert starts == [0, 2, 4]


# --------------------------------------------------------------------------- location


def test_locate_returns_every_supervised_position() -> None:
    codebook = _codebook()
    chunk = [R2RCEAction.MOVE_FORWARD, R2RCEAction.TURN_LEFT, R2RCEAction.STOP, R2RCEAction.STOP]
    positions = locate_chunk_action_positions(
        _rendered(chunk, codebook), actions=chunk, codebook=codebook
    )
    assert positions == (5, 6, 7, 8)


def test_locate_rejects_unsupervised_chunk_token() -> None:
    codebook = _codebook()
    chunk = [R2RCEAction.STOP] * 4
    with pytest.raises(ValueError, match="not supervised"):
        locate_chunk_action_positions(
            _rendered(chunk, codebook, supervise=False), actions=chunk, codebook=codebook
        )


def test_locate_rejects_target_that_does_not_match_the_codebook() -> None:
    codebook = _codebook()
    chunk = [R2RCEAction.STOP] * 4
    rendered = _rendered(chunk, codebook)
    ids = rendered["input_ids"]
    assert isinstance(ids, torch.Tensor)
    ids[5] = 999
    with pytest.raises(ValueError, match="do not match the frozen codebook"):
        locate_chunk_action_positions(rendered, actions=chunk, codebook=codebook)


# --------------------------------------------------------------------------- scoring


def _logits(preferred: list[R2RCEAction], codebook: ActionChunkCodebook) -> torch.Tensor:
    logits = torch.full((1, 16, 64), -8.0)
    for index, action in enumerate(preferred):
        row = 5 + index - 1
        logits[0, row, codebook.action_token_id(action)] = 6.0
    return logits


def test_scoring_recovers_the_preferred_action_at_every_position() -> None:
    codebook = _codebook()
    preferred = [
        R2RCEAction.MOVE_FORWARD,
        R2RCEAction.TURN_LEFT,
        R2RCEAction.TURN_RIGHT,
        R2RCEAction.STOP,
    ]
    record = score_action_chunk_logits(
        _logits(preferred, codebook), action_token_positions=(5, 6, 7, 8), codebook=codebook
    )
    assert record["predicted_chunk"] == [a.name for a in preferred]
    assert record["predicted_chunk_ids"] == [int(a) for a in preferred]
    assert record["predicted_chunk_is_constant"] is False
    assert record["supervised_token_count"] == 4
    assert validate_action_chunk_record(record, chunk_size=4) == tuple(a.name for a in preferred)


def test_scoring_flags_a_constant_chunk() -> None:
    codebook = _codebook()
    preferred = [R2RCEAction.MOVE_FORWARD] * 4
    record = score_action_chunk_logits(
        _logits(preferred, codebook), action_token_positions=(5, 6, 7, 8), codebook=codebook
    )
    assert record["predicted_chunk_is_constant"] is True


def test_scoring_rejects_nonfinite_logits() -> None:
    codebook = _codebook()
    logits = _logits([R2RCEAction.STOP] * 4, codebook)
    logits[0, 4, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        score_action_chunk_logits(logits, action_token_positions=(5, 6, 7, 8), codebook=codebook)


def test_scoring_rejects_unordered_or_duplicate_positions() -> None:
    codebook = _codebook()
    logits = _logits([R2RCEAction.STOP] * 4, codebook)
    with pytest.raises(ValueError, match="must increase"):
        score_action_chunk_logits(logits, action_token_positions=(8, 7, 6, 5), codebook=codebook)
    with pytest.raises(ValueError, match="distinct"):
        score_action_chunk_logits(logits, action_token_positions=(5, 5, 6, 7), codebook=codebook)


def test_scoring_rejects_wrong_position_count() -> None:
    codebook = _codebook()
    logits = _logits([R2RCEAction.STOP] * 4, codebook)
    with pytest.raises(ValueError, match="Expected 4 action positions"):
        score_action_chunk_logits(logits, action_token_positions=(5, 6), codebook=codebook)


# --------------------------------------------------------------------------- validation


def _valid_record() -> dict[str, object]:
    codebook = _codebook()
    preferred = [R2RCEAction.MOVE_FORWARD, R2RCEAction.STOP, R2RCEAction.STOP, R2RCEAction.STOP]
    return score_action_chunk_logits(
        _logits(preferred, codebook), action_token_positions=(5, 6, 7, 8), codebook=codebook
    )


def test_validate_rejects_a_foreign_method() -> None:
    record = _valid_record()
    record["method"] = "single_shared_position_official_action_id_token_v1"
    with pytest.raises(ValueError, match="frozen chunk decision method"):
        validate_action_chunk_record(record, chunk_size=4)


def test_validate_rejects_a_mismatched_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk size other than 2"):
        validate_action_chunk_record(_valid_record(), chunk_size=2)


def test_validate_rejects_predicted_chunk_that_disagrees_with_scores() -> None:
    record = _valid_record()
    record["predicted_chunk"] = ["STOP", "STOP", "STOP", "STOP"]
    with pytest.raises(ValueError, match="disagrees with its own scores"):
        validate_action_chunk_record(record, chunk_size=4)


def test_validate_rejects_a_position_scored_on_three_actions() -> None:
    record = _valid_record()
    del record["positions"][1]["scores"]["STOP"]  # type: ignore[index]
    with pytest.raises(ValueError, match="exactly the four official actions"):
        validate_action_chunk_record(record, chunk_size=4)


def test_validate_rejects_a_broken_causal_offset() -> None:
    record = _valid_record()
    record["positions"][0]["scores"]["STOP"]["causal_logit_position"] = 99  # type: ignore[index]
    with pytest.raises(ValueError, match="invalid causal logit position"):
        validate_action_chunk_record(record, chunk_size=4)


def test_record_declares_the_frozen_protocol_version() -> None:
    assert _valid_record()["protocol_version"] == ACTION_CHUNK_PROTOCOL_VERSION


# --------------------------------------------------------------------------- STOP-tail mask


def test_validate_accepts_stop_only_as_the_final_action() -> None:
    from spatialvlm.eval.action_chunk_decision import validate_chunk_target

    validate_chunk_target([R2RCEAction.MOVE_FORWARD] * 3 + [R2RCEAction.STOP])
    validate_chunk_target([R2RCEAction.MOVE_FORWARD] * 4)


def test_validate_rejects_a_post_stop_tail() -> None:
    from spatialvlm.eval.action_chunk_decision import validate_chunk_target

    with pytest.raises(ValueError, match="only as its final action"):
        validate_chunk_target(
            [
                R2RCEAction.MOVE_FORWARD,
                R2RCEAction.STOP,
                R2RCEAction.MOVE_FORWARD,
                R2RCEAction.TURN_LEFT,
            ]
        )


def test_truncate_discards_everything_after_a_predicted_stop() -> None:
    from spatialvlm.eval.action_chunk_decision import truncate_chunk_at_stop

    actions, stopped = truncate_chunk_at_stop(
        [R2RCEAction.MOVE_FORWARD, R2RCEAction.STOP, R2RCEAction.TURN_LEFT, R2RCEAction.TURN_RIGHT]
    )
    assert actions == (R2RCEAction.MOVE_FORWARD, R2RCEAction.STOP)
    assert stopped is True


def test_truncate_passes_through_a_chunk_without_stop() -> None:
    from spatialvlm.eval.action_chunk_decision import truncate_chunk_at_stop

    chunk = [R2RCEAction.MOVE_FORWARD, R2RCEAction.TURN_LEFT] * 2
    actions, stopped = truncate_chunk_at_stop(chunk)
    assert actions == tuple(chunk)
    assert stopped is False


def test_truncate_at_position_zero_executes_only_stop() -> None:
    from spatialvlm.eval.action_chunk_decision import truncate_chunk_at_stop

    actions, stopped = truncate_chunk_at_stop([R2RCEAction.STOP] + [R2RCEAction.MOVE_FORWARD] * 3)
    assert actions == (R2RCEAction.STOP,)
    assert stopped is True


def test_every_iterated_training_chunk_satisfies_the_stop_invariant() -> None:
    from spatialvlm.eval.action_chunk_decision import validate_chunk_target

    episode = [R2RCEAction.MOVE_FORWARD] * 9 + [R2RCEAction.TURN_LEFT, R2RCEAction.STOP]
    for _, chunk in iter_action_chunks(episode, chunk_size=4):
        validate_chunk_target(chunk)


def test_validate_rejects_stop_followed_by_nothing_but_not_final() -> None:
    """Regression: an index-based [:-1] slice let a mid-chunk STOP pass when it was the only one."""
    from spatialvlm.eval.action_chunk_decision import validate_chunk_target

    with pytest.raises(ValueError, match="only as its final action"):
        validate_chunk_target([R2RCEAction.STOP, R2RCEAction.MOVE_FORWARD])
    with pytest.raises(ValueError, match="only as its final action"):
        validate_chunk_target([R2RCEAction.MOVE_FORWARD, R2RCEAction.STOP, R2RCEAction.TURN_LEFT])
