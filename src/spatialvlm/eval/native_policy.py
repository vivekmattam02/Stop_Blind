"""Learned-policy bridge: native Qwen3-VL checkpoint -> Habitat closed-loop R2R-CE rollout.

This is the component identified as the critical-path blocker by both 2026-08-24 audits
(`docs/CROSS_REVIEW.md` CR-010). It implements the `R2RCEPolicy` protocol consumed by the
canonical evaluator in `spatialvlm.eval.r2r_ce`, so a trained checkpoint can be scored by the
same verified harness that scored the 854-run control suite.

Design constraints, each traceable to a cross-review entry:

- **CR-012 / CR-021**: the policy emits exactly one constrained action token per decision (never a
  chain-of-thought trace), and supports *action chunking* — one forward pass yields `chunk_length`
  future actions, cutting decode calls per episode by that factor. Chunk length is recorded in
  telemetry so it is never conflated with the step cap.
- **CR-018**: the model seam must fail loudly rather than silently degrade. A scorer that returns a
  malformed or non-finite distribution raises; it never falls back to a default action.
- **CR-019**: this class is split-agnostic. It records what it did and nothing about which split it
  ran on; split discipline is enforced by the caller's manifest.
- **CR-015**: competence is judged against preregistered controls on a locked development manifest,
  not against published cross-protocol success rates. Nothing here encodes a numeric threshold.

The `ActionScorer` seam exists so the decision logic is unit-testable on a machine with no model
and no simulator. The real implementation wraps `Qwen3VLBackbone` plus
`spatialvlm.eval.action_decision.score_action_token_logits`; tests inject a deterministic fake.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from spatialvlm.data.history import (
    HISTORY_CONTRACTS,
    HistoryContract,
    HistoryMode,
    HistoryWindow,
    build_history_window,
)
from spatialvlm.eval.metrics import R2RCEAction

__all__ = [
    "ActionScorer",
    "NativeQwenPolicy",
    "NativePolicyConfig",
    "PolicyDecisionRecord",
]

# The four legal actions, in the canonical order used by the action-token codebook.
LEGAL_ACTIONS: tuple[R2RCEAction, ...] = (
    R2RCEAction.STOP,
    R2RCEAction.MOVE_FORWARD,
    R2RCEAction.TURN_LEFT,
    R2RCEAction.TURN_RIGHT,
)


class ActionScorer(Protocol):
    """Model seam: one history window and instruction in, one action chunk out.

    Implementations must return exactly `chunk_length` actions drawn from `LEGAL_ACTIONS`.
    Returning a different length, a non-legal action, or raising is all preferable to returning a
    silently-wrong default: the failure mode this bridge exists to prevent is a policy that keeps
    producing plausible actions after its visual pathway has been disconnected (CR-018).
    """

    def score_chunk(
        self,
        *,
        window: HistoryWindow,
        instruction: str,
        chunk_length: int,
    ) -> tuple[Sequence[R2RCEAction], Mapping[str, Any]]:
        """Return (actions, diagnostics). `diagnostics` is recorded verbatim in telemetry."""
        ...


@dataclass(frozen=True)
class NativePolicyConfig:
    """Frozen rollout contract for one closed-loop evaluation.

    `chunk_length` defaults to 1. The CR-021 verdict rejected four-action chunking as the primary
    interface, and the model-backed scorer accepts only single-step scoring; chunking above 1
    remains available as an efficiency experiment but must be declared, never inherited.

    `history_contract` should come from the frozen `HISTORY_CONTRACTS` registry via `for_mode`.
    No redundant guard is applied here: `HistoryContract.__post_init__` already validates every
    field against its mode, so any validly-constructed contract is equal to its registry entry.
    """

    history_contract: HistoryContract
    chunk_length: int = 1
    step_cap: int = 500

    def __post_init__(self) -> None:
        if self.chunk_length < 1:
            raise ValueError("chunk_length must be at least 1.")
        if self.step_cap < 1:
            raise ValueError("step_cap must be at least 1.")

    @classmethod
    def for_mode(
        cls, mode: HistoryMode, *, chunk_length: int = 1, step_cap: int = 500
    ) -> NativePolicyConfig:
        """Build a config from the frozen registry — the only recommended constructor."""
        return cls(
            history_contract=HISTORY_CONTRACTS[mode],
            chunk_length=chunk_length,
            step_cap=step_cap,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "history_contract": self.history_contract.to_record(),
            "chunk_length": self.chunk_length,
            "step_cap": self.step_cap,
        }


@dataclass
class PolicyDecisionRecord:
    """One scored decision. Actions replayed from a chunk are not decisions."""

    step_index: int
    decision_index: int
    chunk_actions: tuple[str, ...]
    window: dict[str, Any]
    diagnostics: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "decision_index": self.decision_index,
            "chunk_actions": list(self.chunk_actions),
            "window": self.window,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class NativeQwenPolicy:
    """Closed-loop R2R-CE policy backed by a native Qwen3-VL action scorer.

    Satisfies the `R2RCEPolicy` protocol: `reset(episode, ground_truth)` then repeated
    `act(observation)`. The evaluator owns simulator stepping, metrics, and termination; this class
    owns only history bookkeeping and action selection.
    """

    scorer: ActionScorer
    config: NativePolicyConfig

    _observations: list[Any] = field(default_factory=list, init=False)
    _executed: list[R2RCEAction] = field(default_factory=list, init=False)
    _pending: list[R2RCEAction] = field(default_factory=list, init=False)
    _decisions: list[PolicyDecisionRecord] = field(default_factory=list, init=False)
    _instruction: str = field(default="", init=False)
    _episode_id: str = field(default="", init=False)
    _reset_called: bool = field(default=False, init=False)
    _completed: list[dict[str, Any]] = field(default_factory=list, init=False)

    # ── protocol surface ────────────────────────────────────────────────────

    def reset(self, episode: Any, ground_truth: Any = None) -> None:  # noqa: ARG002
        """Begin a new episode. Ground truth is accepted for protocol conformance and ignored.

        Accepting `ground_truth` and never reading it is deliberate: the evaluator hands it to
        every policy, and a learned policy that consulted it would silently become an oracle.

        Fails closed. The instruction is extracted *before* any state is mutated, so a malformed
        episode cannot leave the previous episode logically active while appearing reset. On the
        way in, the finished episode's telemetry is archived, because the canonical evaluator
        reuses one policy across episodes and would otherwise discard every record but the last.
        """
        instruction = _episode_instruction(episode)  # may raise; nothing mutated yet
        episode_id = str(getattr(episode, "episode_id", ""))

        if self._reset_called and self._executed:
            self._completed.append(self.episode_record())

        self._observations = []
        self._executed = []
        self._pending = []
        self._decisions = []
        self._instruction = instruction
        self._episode_id = episode_id
        self._reset_called = True

    def act(self, observation: Mapping[str, Any]) -> R2RCEAction:
        """Return the next action for `observation`.

        Every observation is recorded, because they are all real simulator states and the history
        window must stay causally aligned. Only every `chunk_length`-th observation is *scored*.
        """
        if not self._reset_called:
            raise RuntimeError("NativeQwenPolicy.act called before reset; episode state is unset.")

        self._observations.append(observation)

        step_index = len(self._executed)
        if step_index >= self.config.step_cap:
            raise RuntimeError(
                f"Step cap {self.config.step_cap} exceeded at step {step_index}; "
                "the evaluator must terminate the episode at its own cap."
            )

        if not self._pending:
            self._score_new_chunk(step_index)

        action = self._pending.pop(0)
        self._executed.append(action)
        return action

    # ── internals ───────────────────────────────────────────────────────────

    def _score_new_chunk(self, step_index: int) -> None:
        window = build_history_window(
            self._observations,
            self._executed,
            contract=self.config.history_contract,
        )
        actions, diagnostics = self.scorer.score_chunk(
            window=window,
            instruction=self._instruction,
            chunk_length=self.config.chunk_length,
        )
        actions = tuple(actions)

        # Fail loudly (CR-018). A scorer whose visual pathway has been severed can still emit
        # fluent, well-formed actions; a scorer that emits a malformed chunk must never be
        # papered over with a default.
        if len(actions) != self.config.chunk_length:
            raise RuntimeError(
                f"Scorer returned {len(actions)} actions, expected exactly "
                f"{self.config.chunk_length}; refusing to substitute a default action."
            )
        illegal = [a for a in actions if a not in LEGAL_ACTIONS]
        if illegal:
            raise RuntimeError(f"Scorer returned non-legal actions: {illegal}.")

        # CR-026: a chunk with STOP anywhere but its final position has no defined meaning —
        # every action after STOP would be executed post-episode-termination. Reject it outright
        # rather than inventing a truncation/masking policy that has not been specified or tested.
        if self.config.chunk_length > 1:
            stop_positions = [i for i, a in enumerate(actions) if a is R2RCEAction.STOP]
            if stop_positions and stop_positions[0] != len(actions) - 1:
                raise RuntimeError(
                    f"Scorer returned STOP at position {stop_positions[0]} of a "
                    f"{len(actions)}-action chunk; STOP must only be the chunk's final action. "
                    "Truncation or masking past STOP is not implemented."
                )

        self._decisions.append(
            PolicyDecisionRecord(
                step_index=step_index,
                decision_index=len(self._decisions),
                chunk_actions=tuple(a.name for a in actions),
                window=window.to_record(),
                diagnostics=dict(diagnostics),
            )
        )
        self._pending = list(actions)

    # ── telemetry ───────────────────────────────────────────────────────────

    @property
    def decisions(self) -> tuple[PolicyDecisionRecord, ...]:
        return tuple(self._decisions)

    def collected_episode_records(self) -> tuple[dict[str, Any], ...]:
        """Every episode's telemetry, including the one still in progress.

        This is the seam CR-023 asks for: the canonical metric evaluator is not modified, and a
        runner reads per-episode model telemetry from here after the run completes.
        """
        records = list(self._completed)
        if self._reset_called and self._executed:
            records.append(self.episode_record())
        return tuple(records)

    def episode_record(self) -> dict[str, Any]:
        """Compact per-episode telemetry for the run artifact.

        `decode_calls` versus `steps_taken` is the measured chunking saving (CR-021), and
        `steps_per_decode_call` is the number CR-012's evaluation budget must be re-derived from
        once it is observed rather than estimated.
        """
        steps = len(self._executed)
        calls = len(self._decisions)
        return {
            "episode_id": self._episode_id,
            "config": self.config.to_record(),
            "steps_taken": steps,
            "decode_calls": calls,
            "steps_per_decode_call": (steps / calls) if calls else 0.0,
            "actions": [action.name for action in self._executed],
            "decisions": [record.to_record() for record in self._decisions],
        }


def _episode_instruction(episode: Any) -> str:
    """Extract the canonical instruction text, failing closed rather than guessing."""
    for attribute in ("instruction_text", "instruction"):
        value = getattr(episode, attribute, None)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, Mapping):
            text = value.get("instruction_text")
            if isinstance(text, str) and text.strip():
                return text
    raise ValueError("Episode does not expose a non-empty instruction text.")
