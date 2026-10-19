"""Length-fair action-chunk decisions for the four-action R2R-CE policy.

This is a *new* protocol that sits beside the frozen single-token protocol in
``spatialvlm.eval.action_decision``. Nothing here modifies or reinterprets the frozen
``single_shared_position_official_action_id_token_v1`` objective or any artifact produced under it.

Motivation, measured on the released R2R-CE ``train`` ground truth (10,819 episodes,
631,244 decisions). Under the single-action objective the best constant-action policy --
always ``MOVE_FORWARD`` -- attains an exact-match rate of ``404,912 / 631,244 = 64.1%``. Under a
``k=4`` chunk objective the best constant chunk ``(1, 1, 1, 1)`` attains only
``38,683 / 153,693 = 25.2%``. The chunked objective therefore removes most of the payoff that the
single-token objective grants to a policy that never adapts.

Two facts bound what this protocol can claim. Only 49 of the 256 possible ``k=4`` chunks occur in
the training ground truth, so the target space is structured rather than uniform. And the share of
chunks containing ``STOP`` (1.70%) is essentially the share of single decisions that are ``STOP``
(1.71%), so chunking on its own does **not** improve ``STOP`` coverage. Any ``STOP`` remedy must be
separate from this protocol.

The decision contract mirrors the single-token one. A chunk target renders as one shared prefix
followed by exactly ``chunk_size`` action-specific tokens, one per decision position. Each position
is scored as a restricted four-way choice over the same four checkpoint tokens, at a position that
is identical across all candidate chunks. Variable-length surface forms are never compared.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction

ACTION_CHUNK_PROTOCOL_VERSION = 1
ACTION_CHUNK_METHOD = "shared_position_official_action_id_chunk_v1"
ACTION_CHUNK_TARGET_PREFIX = "Action IDs:"
DEFAULT_ACTION_CHUNK_SIZE = 4

ACTION_CHUNK_SYSTEM_PROMPT_TEMPLATE = (
    "You are a spatial navigation assistant. Given an indoor RGB-D observation and a navigation "
    "instruction, choose the next {chunk_size} official action IDs: 0=STOP, 1=MOVE_FORWARD, "
    "2=TURN_LEFT, 3=TURN_RIGHT. Respond only with `Action IDs: {pattern}`, replacing each N with "
    "the chosen ID."
)


def action_chunk_system_prompt(chunk_size: int) -> str:
    """Return the frozen system prompt for a given chunk size."""
    _validate_chunk_size(chunk_size)
    return ACTION_CHUNK_SYSTEM_PROMPT_TEMPLATE.format(
        chunk_size=chunk_size, pattern=" ".join(["N"] * chunk_size)
    )


def _validate_chunk_size(chunk_size: int) -> None:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one.")


def canonical_action_chunk_target(actions: Sequence[R2RCEAction | int]) -> str:
    """Return the frozen assistant target for one ordered chunk of official actions."""
    if len(actions) < 1:
        raise ValueError("An action chunk must contain at least one action.")
    ids = [int(R2RCEAction(int(action))) for action in actions]
    return f"{ACTION_CHUNK_TARGET_PREFIX} " + " ".join(str(value) for value in ids)


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().reshape(-1).tolist()
    if not isinstance(encoded, (list, tuple)) or not encoded:
        raise ValueError(f"Action chunk target {text!r} tokenized to no IDs.")
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in encoded):
        raise TypeError(f"Action chunk target {text!r} did not tokenize to non-negative IDs.")
    return tuple(encoded)


@dataclass(frozen=True)
class ActionChunkCodebook:
    """Runtime-verified chunk codebook with one distinct token per decision position."""

    chunk_size: int
    shared_prefix_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, int, int, int]
    action_token_offsets: tuple[int, ...]
    reference_target_token_ids: tuple[int, ...]
    tokenizer_class: str
    tokenizer_name_or_path: str | None

    def action_token_id(self, action: R2RCEAction | int) -> int:
        """Return the unique checkpoint token assigned to `action`."""
        return self.action_token_ids[int(R2RCEAction(int(action)))]

    def offset(self, position: int) -> int:
        """Return the token offset of decision `position` within the rendered target."""
        if not 0 <= position < self.chunk_size:
            raise IndexError(f"Chunk position {position} is outside 0..{self.chunk_size - 1}.")
        return self.action_token_offsets[position]

    def target_ids(self, actions: Sequence[R2RCEAction | int]) -> tuple[int, ...]:
        """Return the expected complete tokenization of one chunk target."""
        if len(actions) != self.chunk_size:
            raise ValueError(
                f"Chunk must contain exactly {self.chunk_size} actions, got {len(actions)}."
            )
        tokens = list(self.reference_target_token_ids)
        for position, action in enumerate(actions):
            tokens[self.action_token_offsets[position]] = self.action_token_id(action)
        return tuple(tokens)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-safe provenance record."""
        return {
            "protocol_version": ACTION_CHUNK_PROTOCOL_VERSION,
            "method": ACTION_CHUNK_METHOD,
            "chunk_size": self.chunk_size,
            "system_prompt": action_chunk_system_prompt(self.chunk_size),
            "target_prefix": ACTION_CHUNK_TARGET_PREFIX,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "shared_prefix_token_ids": list(self.shared_prefix_token_ids),
            "shared_prefix_token_count": len(self.shared_prefix_token_ids),
            "action_token_offsets": list(self.action_token_offsets),
            "reference_target_token_ids": list(self.reference_target_token_ids),
            "action_specific_token_count_per_position": 1,
            "supervised_token_count": self.chunk_size,
            "actions": {
                action.name: {
                    "action_id": int(action),
                    "action_token_id": self.action_token_id(action),
                }
                for action in R2RCEAction
            },
        }


def build_action_chunk_codebook(
    tokenizer: Any, *, chunk_size: int = DEFAULT_ACTION_CHUNK_SIZE
) -> ActionChunkCodebook:
    """Build the chunk codebook and fail unless every position is one distinct branch token.

    Tokenization is inspected at runtime because BPE boundaries are checkpoint-specific. The
    verification holds every other position at ``STOP`` and varies one position at a time. For the
    codebook to be admitted, each such variation must change the rendered target in exactly one
    token index, that index must depend only on the position, and the substituted token must be the
    same for a given action at every position.
    """
    _validate_chunk_size(chunk_size)
    reference_actions = [R2RCEAction.STOP] * chunk_size
    reference = _token_ids(tokenizer, canonical_action_chunk_target(reference_actions))

    offsets: list[int] = []
    per_position_tokens: list[dict[int, int]] = []
    for position in range(chunk_size):
        changed_indices: set[int] = set()
        tokens_here: dict[int, int] = {int(R2RCEAction.STOP): reference[0]}
        for action in R2RCEAction:
            variant_actions = list(reference_actions)
            variant_actions[position] = action
            variant = _token_ids(tokenizer, canonical_action_chunk_target(variant_actions))
            if len(variant) != len(reference):
                raise ValueError(
                    "Length-fair chunk codebook requires every chunk target to tokenize to the "
                    f"same length; position {position} action {action.name} produced "
                    f"{len(variant)} tokens against a reference of {len(reference)}."
                )
            differing = [i for i, (a, b) in enumerate(zip(reference, variant)) if a != b]
            if action is R2RCEAction.STOP:
                if differing:
                    raise ValueError(
                        f"Chunk target for the reference action differs from itself at {differing}."
                    )
                continue
            if len(differing) != 1:
                raise ValueError(
                    "Length-fair chunk codebook requires exactly one action-specific token per "
                    f"position; position {position} action {action.name} changed "
                    f"{len(differing)} token indices."
                )
            index = differing[0]
            changed_indices.add(index)
            tokens_here[int(action)] = variant[index]
        if len(changed_indices) != 1:
            raise ValueError(
                f"Chunk position {position} does not map to a single token index; "
                f"observed {sorted(changed_indices)}."
            )
        index = next(iter(changed_indices))
        tokens_here[int(R2RCEAction.STOP)] = reference[index]
        offsets.append(index)
        per_position_tokens.append(tokens_here)

    if len(set(offsets)) != chunk_size:
        raise ValueError(f"Chunk positions must map to distinct token indices; got {offsets}.")
    if offsets != sorted(offsets):
        raise ValueError(f"Chunk position offsets must increase with position; got {offsets}.")

    first = per_position_tokens[0]
    for position, mapping in enumerate(per_position_tokens[1:], start=1):
        if mapping != first:
            raise ValueError(
                "Every chunk position must use the same action token IDs; position "
                f"{position} disagrees with position 0."
            )
    action_token_ids = tuple(first[int(action)] for action in R2RCEAction)
    if len(set(action_token_ids)) != len(R2RCEAction):
        raise ValueError("Length-fair chunk codebook requires four distinct action token IDs.")

    shared_prefix = reference[: offsets[0]]
    if not shared_prefix:
        raise ValueError("Chunk targets have no shared tokenizer prefix before the first action.")

    name_or_path = getattr(tokenizer, "name_or_path", None)
    if name_or_path is not None and not isinstance(name_or_path, str):
        name_or_path = str(name_or_path)
    return ActionChunkCodebook(
        chunk_size=chunk_size,
        shared_prefix_token_ids=shared_prefix,
        action_token_ids=action_token_ids,  # type: ignore[arg-type]
        action_token_offsets=tuple(offsets),
        reference_target_token_ids=reference,
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_name_or_path=name_or_path,
    )


def iter_action_chunks(
    actions: Sequence[R2RCEAction | int], *, chunk_size: int, stride: int | None = None
) -> Iterator[tuple[int, tuple[R2RCEAction, ...]]]:
    """Yield ``(start_index, chunk)`` pairs covering an episode's ordered action sequence.

    The final chunk is anchored at ``len(actions) - chunk_size`` rather than dropped, so an
    episode's terminal ``STOP`` is always represented whenever the episode is at least
    ``chunk_size`` actions long. Anchoring can overlap the previous chunk; callers that require
    disjoint coverage should deduplicate on the returned start index.
    """
    _validate_chunk_size(chunk_size)
    step = chunk_size if stride is None else stride
    if step < 1:
        raise ValueError("stride must be at least one.")
    total = len(actions)
    if total < chunk_size:
        return
    starts = list(range(0, total - chunk_size + 1, step))
    last_start = total - chunk_size
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        yield start, tuple(R2RCEAction(int(a)) for a in actions[start : start + chunk_size])


def validate_chunk_target(actions: Sequence[R2RCEAction | int]) -> None:
    """Fail unless ``STOP`` appears only as the final action of a supervised chunk.

    ``STOP`` ends an episode, so any action after it is undefined and must never be supervised.
    Measured on the released R2R-CE ``train`` ground truth, all 10,819 episodes end in ``STOP`` and
    none contains ``STOP`` anywhere else, so chunks drawn by ``iter_action_chunks`` satisfy this by
    construction. The check is kept as a fail-closed assertion rather than an assumption: a
    different split, a different iterator, or a future dataset revision could break the invariant
    silently, and the resulting objective would train on post-terminal garbage.
    """
    ids = [int(R2RCEAction(int(action))) for action in actions]
    final = len(ids) - 1
    offending = [i for i, value in enumerate(ids) if value == int(R2RCEAction.STOP) and i != final]
    if offending:
        raise ValueError(
            "A supervised chunk may contain STOP only as its final action; "
            f"STOP occurs at non-final position(s) {offending} of {ids}."
        )


def truncate_chunk_at_stop(
    actions: Sequence[R2RCEAction | int],
) -> tuple[tuple[R2RCEAction, ...], bool]:
    """Return the executable prefix of a *predicted* chunk and whether it terminates.

    This is the inference-side STOP-tail mask. During a rollout the policy may emit ``STOP`` at any
    chunk position; every action the model proposed after that point refers to an episode that has
    already ended and must be discarded rather than executed. Returns the prefix up to and
    including the first ``STOP``, and ``True`` when a ``STOP`` was found.
    """
    executable: list[R2RCEAction] = []
    for action in actions:
        official = R2RCEAction(int(action))
        executable.append(official)
        if official is R2RCEAction.STOP:
            return tuple(executable), True
    return tuple(executable), False


def locate_chunk_action_positions(
    rendered: dict[str, torch.Tensor | int | str],
    *,
    actions: Sequence[R2RCEAction | int],
    codebook: ActionChunkCodebook,
) -> tuple[int, ...]:
    """Validate a rendered chunk candidate and return its absolute action-token positions."""
    target_start = rendered.get("target_start_idx")
    input_ids = rendered.get("input_ids")
    labels = rendered.get("labels")
    if not isinstance(target_start, int):
        raise TypeError("Rendered target_start_idx must be an integer.")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
        raise TypeError("Rendered input_ids must be a one-dimensional tensor.")
    if not isinstance(labels, torch.Tensor) or labels.shape != input_ids.shape:
        raise TypeError("Rendered labels must match the one-dimensional input_ids tensor.")

    expected = codebook.target_ids(actions)
    actual = tuple(int(v) for v in input_ids[target_start : target_start + len(expected)])
    if actual != expected:
        raise ValueError(
            "Rendered chunk target tokens do not match the frozen codebook: "
            f"expected {expected}, observed {actual}."
        )

    positions: list[int] = []
    for index, action in enumerate(actions):
        position = target_start + codebook.offset(index)
        if position >= input_ids.numel():
            raise ValueError("A rendered chunk action position lies outside input_ids.")
        expected_token = codebook.action_token_id(action)
        if int(input_ids[position].item()) != expected_token:
            raise ValueError(
                f"Rendered chunk token at position {index} does not match its codebook ID."
            )
        if int(labels[position].item()) != expected_token:
            raise ValueError(f"Rendered chunk token at position {index} is not supervised.")
        positions.append(position)
    return tuple(positions)


def score_action_chunk_logits(
    logits: torch.Tensor,
    *,
    action_token_positions: Sequence[int],
    codebook: ActionChunkCodebook,
) -> dict[str, Any]:
    """Score every chunk position as an independent restricted four-way decision.

    Each position is scored from the causal logit row that precedes it, exactly as the single-token
    protocol does. Restricted probabilities renormalize only across the four legal action tokens;
    ``token_nll`` remains a full-vocabulary conditional NLL.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"logits must have shape [1,T,V], got {tuple(logits.shape)}.")
    if len(action_token_positions) != codebook.chunk_size:
        raise ValueError(
            f"Expected {codebook.chunk_size} action positions, got {len(action_token_positions)}."
        )
    if len(set(action_token_positions)) != len(action_token_positions):
        raise ValueError("Chunk action positions must be distinct.")
    if list(action_token_positions) != sorted(action_token_positions):
        raise ValueError("Chunk action positions must increase with chunk position.")
    if max(codebook.action_token_ids) >= logits.shape[2]:
        raise ValueError("An action token ID lies outside the model vocabulary.")

    token_ids = list(codebook.action_token_ids)
    per_position: list[dict[str, Any]] = []
    predicted: list[str] = []
    for index, position in enumerate(action_token_positions):
        if position < 1 or position >= logits.shape[1]:
            raise ValueError("Every action position must identify a non-initial token inside T.")
        row = logits[0, position - 1].detach().float()
        if not bool(torch.isfinite(row).all().item()):
            raise ValueError(f"Chunk position {index} logit row contains non-finite values.")
        full_log_probabilities = torch.log_softmax(row, dim=-1)
        restricted = torch.softmax(row[token_ids], dim=0)
        scores = {
            action.name: {
                "action_id": int(action),
                "action_token_id": codebook.action_token_id(action),
                "scored_token_position": int(position),
                "causal_logit_position": int(position) - 1,
                "scored_token_count": 1,
                "token_nll": float(-full_log_probabilities[codebook.action_token_id(action)]),
                "restricted_probability": float(restricted[int(action)].item()),
            }
            for action in R2RCEAction
        }
        choice = min(R2RCEAction, key=lambda a: (scores[a.name]["token_nll"], int(a)))
        probabilities = [scores[a.name]["restricted_probability"] for a in R2RCEAction]
        per_position.append(
            {
                "chunk_position": index,
                "scores": scores,
                "predicted_action": choice.name,
                "predicted_action_id": int(choice),
                "restricted_entropy": -sum(p * math.log(p) for p in probabilities if p > 0.0),
            }
        )
        predicted.append(choice.name)

    return {
        "protocol_version": ACTION_CHUNK_PROTOCOL_VERSION,
        "method": ACTION_CHUNK_METHOD,
        "chunk_size": codebook.chunk_size,
        "legal_actions": [action.name for action in R2RCEAction],
        "supervised_token_count": codebook.chunk_size,
        "action_token_positions": [int(p) for p in action_token_positions],
        "positions": per_position,
        "predicted_chunk": predicted,
        "predicted_chunk_ids": [int(R2RCEAction[name]) for name in predicted],
        "predicted_chunk_is_constant": len(set(predicted)) == 1,
    }


def validate_action_chunk_record(record: dict[str, Any], *, chunk_size: int) -> tuple[str, ...]:
    """Validate a scored chunk record and return its predicted action names."""
    _validate_chunk_size(chunk_size)
    if record.get("method") != ACTION_CHUNK_METHOD:
        raise ValueError("Chunk record does not use the frozen chunk decision method.")
    if record.get("protocol_version") != ACTION_CHUNK_PROTOCOL_VERSION:
        raise ValueError("Chunk record does not use the frozen chunk protocol version.")
    if record.get("chunk_size") != chunk_size:
        raise ValueError(f"Chunk record declares a chunk size other than {chunk_size}.")
    positions = record.get("positions")
    if not isinstance(positions, list) or len(positions) != chunk_size:
        raise ValueError(f"Chunk record must contain exactly {chunk_size} scored positions.")

    predicted: list[str] = []
    seen_positions: set[int] = set()
    for index, entry in enumerate(positions):
        if not isinstance(entry, dict) or entry.get("chunk_position") != index:
            raise ValueError(f"Chunk record position {index} is missing or out of order.")
        scores = entry.get("scores")
        if not isinstance(scores, dict) or set(scores) != {a.name for a in R2RCEAction}:
            raise ValueError(
                f"Chunk position {index} must score exactly the four official actions."
            )
        token_ids: set[int] = set()
        causal: set[int] = set()
        values: dict[str, float] = {}
        for action in R2RCEAction:
            score = scores[action.name]
            if not isinstance(score, dict):
                raise ValueError(f"Chunk position {index} action {action.name} must be a mapping.")
            value = float(score.get("token_nll", float("nan")))
            if not math.isfinite(value):
                raise ValueError(f"Chunk position {index} action {action.name} has non-finite NLL.")
            if score.get("scored_token_count") != 1:
                raise ValueError(
                    f"Chunk position {index} action {action.name} scored more than one token."
                )
            if score.get("action_id") != int(action):
                raise ValueError(
                    f"Chunk position {index} action {action.name} has the wrong action ID."
                )
            scored_position = score.get("scored_token_position")
            if not isinstance(scored_position, int) or scored_position < 1:
                raise ValueError(f"Chunk position {index} has an invalid scored token position.")
            if score.get("causal_logit_position") != scored_position - 1:
                raise ValueError(f"Chunk position {index} has an invalid causal logit position.")
            token_id = score.get("action_token_id")
            if not isinstance(token_id, int) or token_id < 0:
                raise ValueError(f"Chunk position {index} has an invalid tokenizer ID.")
            token_ids.add(token_id)
            causal.add(scored_position)
            values[action.name] = value
        if len(token_ids) != len(R2RCEAction):
            raise ValueError(f"Chunk position {index} must use four distinct tokenizer IDs.")
        if len(causal) != 1:
            raise ValueError(f"Chunk position {index} must score all actions at one position.")
        seen_positions |= causal
        predicted.append(min(R2RCEAction, key=lambda a: (values[a.name], int(a))).name)

    if len(seen_positions) != chunk_size:
        raise ValueError("Chunk positions must be scored at distinct causal positions.")
    declared = record.get("predicted_chunk")
    if declared is not None and list(declared) != predicted:
        raise ValueError("Chunk record predicted_chunk disagrees with its own scores.")
    return tuple(predicted)
