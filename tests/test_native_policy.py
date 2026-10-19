"""Tests for the learned-policy bridge (`spatialvlm.eval.native_policy`).

These run with no model and no simulator: the `ActionScorer` seam is filled with deterministic
fakes. What is under test is the decision logic — causal history alignment, chunk replay, and the
fail-loud contract from CR-018 — not Qwen itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode, HistoryWindow
from spatialvlm.eval.metrics import R2RCEAction
from spatialvlm.eval.native_policy import (
    LEGAL_ACTIONS,
    NativePolicyConfig,
    NativeQwenPolicy,
)


class _Episode:
    def __init__(self, instruction: str = "Walk to the kitchen and stop.") -> None:
        self.episode_id = "ep-1"
        self.instruction_text = instruction


# CR-026: LEGAL_ACTIONS' canonical first member is STOP, so a fake that defaulted its cycle to
# that tuple would emit STOP on its very first scored call — the opposite of what tests exercising
# several continuing steps intend, and later calls would trip build_history_window's fail-closed
# STOP check. Continuing tests get a non-STOP default; STOP is only ever requested explicitly.
_DEFAULT_CONTINUING_ACTIONS: tuple[R2RCEAction, ...] = (
    R2RCEAction.MOVE_FORWARD,
    R2RCEAction.TURN_LEFT,
    R2RCEAction.TURN_RIGHT,
)


class _FakeScorer:
    """Returns a fixed cycle of actions and records the windows it was shown."""

    def __init__(self, script: Sequence[R2RCEAction] | None = None) -> None:
        self.script = list(script) if script else list(_DEFAULT_CONTINUING_ACTIONS)
        self.seen_windows: list[HistoryWindow] = []
        self.calls = 0

    def score_chunk(
        self, *, window: HistoryWindow, instruction: str, chunk_length: int
    ) -> tuple[Sequence[R2RCEAction], Mapping[str, Any]]:
        self.seen_windows.append(window)
        self.calls += 1
        actions = [
            self.script[(self.calls - 1 + i) % len(self.script)] for i in range(chunk_length)
        ]
        return actions, {"call": self.calls, "instruction_len": len(instruction)}


def _contract(mode: HistoryMode = HistoryMode.H4_SHORT_WINDOW):
    """Always the frozen registry entry — an ad-hoc contract is now rejected by design."""
    return HISTORY_CONTRACTS[mode]


def _policy(
    chunk_length: int = 1,
    mode: HistoryMode = HistoryMode.H4_SHORT_WINDOW,
    scorer: Any = None,
) -> NativeQwenPolicy:
    return NativeQwenPolicy(
        scorer=scorer or _FakeScorer(),
        config=NativePolicyConfig.for_mode(mode, chunk_length=chunk_length, step_cap=50),
    )


def test_act_before_reset_raises() -> None:
    with pytest.raises(RuntimeError, match="before reset"):
        _policy().act({"rgb": 0})


def test_history_stays_causally_aligned() -> None:
    """N executed actions must always be paired with N+1 observations."""
    scorer = _FakeScorer()
    policy = _policy(chunk_length=1, scorer=scorer)
    policy.reset(_Episode())
    for step in range(6):
        policy.act({"rgb": step})
        window = scorer.seen_windows[-1]
        assert window.decision_step_index == step
        # the final frame is always the observation just handed in, never a future one
        assert window.frames[-1].source_step_index == step
        assert not window.frames[-1].is_padding


def test_chunking_reduces_decode_calls() -> None:
    """CR-021: one forward pass should yield `chunk_length` actions."""
    scorer = _FakeScorer()
    policy = _policy(chunk_length=4, scorer=scorer)
    policy.reset(_Episode())
    for step in range(8):
        policy.act({"rgb": step})
    assert scorer.calls == 2, "8 steps at chunk_length=4 must cost exactly 2 decode calls"
    record = policy.episode_record()
    assert record["steps_taken"] == 8
    assert record["decode_calls"] == 2
    assert record["steps_per_decode_call"] == pytest.approx(4.0)


def test_chunk_length_one_scores_every_step() -> None:
    scorer = _FakeScorer()
    policy = _policy(chunk_length=1, scorer=scorer)
    policy.reset(_Episode())
    for step in range(5):
        policy.act({"rgb": step})
    assert scorer.calls == 5


def test_every_observation_is_recorded_even_when_not_scored() -> None:
    """Replayed chunk steps must still extend history, or alignment silently drifts."""
    scorer = _FakeScorer()
    policy = _policy(chunk_length=4, scorer=scorer)
    policy.reset(_Episode())
    for step in range(5):
        policy.act({"rgb": step})
    # second decode call happens at step 4, having seen observations 0..4
    assert scorer.seen_windows[-1].decision_step_index == 4


class _WrongLengthScorer:
    def score_chunk(self, *, window, instruction, chunk_length):  # noqa: ANN001, ARG002
        return [R2RCEAction.MOVE_FORWARD], {}


class _IllegalActionScorer:
    def score_chunk(self, *, window, instruction, chunk_length):  # noqa: ANN001, ARG002
        return ["FLY"] * chunk_length, {}


def test_wrong_chunk_length_raises_rather_than_defaulting() -> None:
    """CR-018: a malformed scorer output must crash, never fall back to a default action."""
    policy = _policy(chunk_length=4, scorer=_WrongLengthScorer())
    policy.reset(_Episode())
    with pytest.raises(RuntimeError, match="refusing to substitute"):
        policy.act({"rgb": 0})


def test_illegal_action_raises() -> None:
    policy = _policy(chunk_length=2, scorer=_IllegalActionScorer())
    policy.reset(_Episode())
    with pytest.raises(RuntimeError, match="non-legal actions"):
        policy.act({"rgb": 0})


def test_step_cap_is_enforced() -> None:
    policy = NativeQwenPolicy(
        scorer=_FakeScorer(),
        config=NativePolicyConfig.for_mode(HistoryMode.H4_SHORT_WINDOW, step_cap=3),
    )
    policy.reset(_Episode())
    for step in range(3):
        policy.act({"rgb": step})
    with pytest.raises(RuntimeError, match="Step cap 3 exceeded"):
        policy.act({"rgb": 3})


def test_reset_clears_all_episode_state() -> None:
    scorer = _FakeScorer()
    policy = _policy(chunk_length=4, scorer=scorer)
    policy.reset(_Episode())
    for step in range(6):
        policy.act({"rgb": step})
    policy.reset(_Episode())
    assert policy.decisions == ()
    assert policy.episode_record()["steps_taken"] == 0
    policy.act({"rgb": 0})
    # a stale pending chunk would have been replayed instead of rescoring
    assert scorer.seen_windows[-1].decision_step_index == 0


def test_missing_instruction_fails_closed() -> None:
    class _NoInstruction:
        episode_id = "ep-2"

    policy = _policy()
    with pytest.raises(ValueError, match="instruction text"):
        policy.reset(_NoInstruction())


def test_ground_truth_is_accepted_and_ignored() -> None:
    """A learned policy that read ground truth would silently become an oracle."""
    policy = _policy()
    policy.reset(_Episode(), ground_truth={"actions": [R2RCEAction.STOP]})
    action = policy.act({"rgb": 0})
    assert action in LEGAL_ACTIONS


def test_config_rejects_degenerate_values() -> None:
    with pytest.raises(ValueError, match="chunk_length"):
        NativePolicyConfig(history_contract=_contract(), chunk_length=0)
    with pytest.raises(ValueError, match="step_cap"):
        NativePolicyConfig(history_contract=_contract(), step_cap=0)


def test_multi_episode_telemetry_survives_reset() -> None:
    """CR-023 defect 4: the evaluator reuses one policy, so records must accumulate."""
    scorer = _FakeScorer()
    policy = _policy(chunk_length=1, scorer=scorer)
    policy.reset(_Episode())
    for step in range(3):
        policy.act({"rgb": step})
    policy.reset(_Episode())
    for step in range(2):
        policy.act({"rgb": step})
    records = policy.collected_episode_records()
    assert len(records) == 2, "both episodes must remain reachable after a multi-episode run"
    assert [r["steps_taken"] for r in records] == [3, 2]


def test_one_step_stop_is_returned_once() -> None:
    """CR-026: a scorer emitting STOP must be honored exactly once and not silently skipped.

    The evaluator-facing contract is that the caller (not this policy) terminates the episode as
    soon as STOP is executed, so this test only ever issues one `act()` call.
    """
    scorer = _FakeScorer(script=[R2RCEAction.STOP])
    policy = _policy(chunk_length=1, scorer=scorer)
    policy.reset(_Episode())
    action = policy.act({"rgb": 0})
    assert action is R2RCEAction.STOP
    record = policy.episode_record()
    assert record["steps_taken"] == 1
    assert record["actions"] == ["STOP"]


class _MidChunkStopScorer:
    def score_chunk(self, *, window, instruction, chunk_length):  # noqa: ANN001, ARG002
        actions = [R2RCEAction.MOVE_FORWARD] * chunk_length
        actions[0] = R2RCEAction.STOP
        return actions, {}


def test_mid_chunk_stop_is_rejected_not_truncated() -> None:
    """CR-026: STOP before a chunk's final position must raise, never be silently truncated."""
    policy = _policy(chunk_length=4, scorer=_MidChunkStopScorer())
    policy.reset(_Episode())
    with pytest.raises(RuntimeError, match="STOP must only be the chunk's final action"):
        policy.act({"rgb": 0})


def test_reset_fails_closed_on_malformed_episode() -> None:
    """CR-023 defect 3: a bad episode must not leave the previous one logically active."""

    class _NoInstruction:
        episode_id = "bad"

    scorer = _FakeScorer()
    policy = _policy(chunk_length=1, scorer=scorer)
    policy.reset(_Episode())
    policy.act({"rgb": 0})
    with pytest.raises(ValueError, match="instruction text"):
        policy.reset(_NoInstruction())
    # the prior episode is untouched, not half-cleared
    assert policy.episode_record()["steps_taken"] == 1
