"""Length-fair constrained action decisions for the four-action R2R-CE policy.

The protocol maps the official action IDs 0--3 to four distinct tokenizer tokens after one shared
target prefix. A policy decision compares those tokens at the same causal-LM position. It never
compares variable-length surface-form likelihoods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction

ACTION_DECISION_PROTOCOL_VERSION = 1
ACTION_DECISION_METHOD = "single_shared_position_official_action_id_token_v1"
ACTION_TARGET_PREFIX = "Action ID:"
ACTION_DECISION_SYSTEM_PROMPT = (
    "You are a spatial navigation assistant. Given an indoor RGB-D observation and a navigation "
    "instruction, choose exactly one official action ID: 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, "
    "3=TURN_RIGHT. Respond only with `Action ID: N`, replacing N with the chosen ID."
)


def canonical_action_target(action: R2RCEAction) -> str:
    """Return the frozen assistant target for one official action."""
    return f"{ACTION_TARGET_PREFIX} {int(action)}"


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().reshape(-1).tolist()
    if not isinstance(encoded, (list, tuple)) or not encoded:
        raise ValueError(f"Action target {text!r} tokenized to no IDs.")
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in encoded):
        raise TypeError(f"Action target {text!r} did not tokenize to non-negative integer IDs.")
    return tuple(encoded)


def _longest_common_prefix(sequences: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    limit = min(len(sequence) for sequence in sequences)
    prefix_length = 0
    while prefix_length < limit and len({sequence[prefix_length] for sequence in sequences}) == 1:
        prefix_length += 1
    return sequences[0][:prefix_length]


@dataclass(frozen=True)
class ActionTokenCodebook:
    """Runtime-verified one-token codebook in official action-ID order."""

    shared_prefix_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, int, int, int]
    target_token_ids: tuple[tuple[int, ...], ...]
    tokenizer_class: str
    tokenizer_name_or_path: str | None

    @property
    def action_token_offset(self) -> int:
        """Offset of the decision token within the assistant target."""
        return len(self.shared_prefix_token_ids)

    def action_token_id(self, action: R2RCEAction) -> int:
        """Return the unique checkpoint token assigned to `action`."""
        return self.action_token_ids[int(action)]

    def target_ids(self, action: R2RCEAction) -> tuple[int, ...]:
        """Return the complete target tokenization, excluding EOS."""
        return self.target_token_ids[int(action)]

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-safe provenance record."""
        return {
            "protocol_version": ACTION_DECISION_PROTOCOL_VERSION,
            "method": ACTION_DECISION_METHOD,
            "system_prompt": ACTION_DECISION_SYSTEM_PROMPT,
            "target_prefix": ACTION_TARGET_PREFIX,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "shared_prefix_token_ids": list(self.shared_prefix_token_ids),
            "shared_prefix_token_count": len(self.shared_prefix_token_ids),
            "action_token_offset": self.action_token_offset,
            "actions": {
                action.name: {
                    "action_id": int(action),
                    "target_text": canonical_action_target(action),
                    "target_token_ids": list(self.target_ids(action)),
                    "action_token_id": self.action_token_id(action),
                    "action_specific_token_count": 1,
                }
                for action in R2RCEAction
            },
        }


def build_action_token_codebook(tokenizer: Any) -> ActionTokenCodebook:
    """Build the frozen codebook and fail unless every action is one distinct branch token.

    Tokenization is inspected at runtime because BPE boundaries are checkpoint-specific. The
    complete targets may contain any number of shared prefix tokens, but exactly one token must
    remain after their longest common prefix for every legal action.
    """
    targets = tuple(
        _token_ids(tokenizer, canonical_action_target(action)) for action in R2RCEAction
    )
    shared = _longest_common_prefix(targets)
    if not shared:
        raise ValueError("Canonical action targets have no shared tokenizer prefix.")
    suffixes = tuple(target[len(shared) :] for target in targets)
    bad_lengths = {
        action.name: len(suffixes[int(action)])
        for action in R2RCEAction
        if len(suffixes[int(action)]) != 1
    }
    if bad_lengths:
        raise ValueError(
            "Length-fair action codebook requires exactly one action-specific token per target; "
            f"observed suffix lengths {bad_lengths}."
        )
    action_token_ids = tuple(suffix[0] for suffix in suffixes)
    if len(set(action_token_ids)) != len(R2RCEAction):
        raise ValueError("Length-fair action codebook requires four distinct action token IDs.")
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if name_or_path is not None and not isinstance(name_or_path, str):
        name_or_path = str(name_or_path)
    return ActionTokenCodebook(
        shared_prefix_token_ids=shared,
        action_token_ids=action_token_ids,  # type: ignore[arg-type]
        target_token_ids=targets,
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_name_or_path=name_or_path,
    )


def locate_action_token(
    rendered: dict[str, torch.Tensor | int | str],
    *,
    action: R2RCEAction,
    codebook: ActionTokenCodebook,
) -> int:
    """Validate a rendered candidate and return its absolute action-token position."""
    target_start = rendered.get("target_start_idx")
    input_ids = rendered.get("input_ids")
    labels = rendered.get("labels")
    if not isinstance(target_start, int):
        raise TypeError("Rendered target_start_idx must be an integer.")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
        raise TypeError("Rendered input_ids must be a one-dimensional tensor.")
    if not isinstance(labels, torch.Tensor) or labels.shape != input_ids.shape:
        raise TypeError("Rendered labels must match the one-dimensional input_ids tensor.")

    expected = codebook.target_ids(action)
    actual = tuple(int(value) for value in input_ids[target_start : target_start + len(expected)])
    if actual != expected:
        raise ValueError(
            f"Rendered target tokens for {action.name} do not match the frozen codebook: "
            f"expected {expected}, observed {actual}."
        )
    position = target_start + codebook.action_token_offset
    if position >= input_ids.numel():
        raise ValueError("Rendered action-token position lies outside input_ids.")
    expected_token = codebook.action_token_id(action)
    if int(input_ids[position].item()) != expected_token:
        raise ValueError(f"Rendered action token for {action.name} does not match its codebook ID.")
    if int(labels[position].item()) != expected_token:
        raise ValueError(f"Rendered action token for {action.name} is not supervised.")
    return position


def locate_shared_action_position(
    rendered_candidates: dict[R2RCEAction, dict[str, torch.Tensor | int | str]],
    *,
    codebook: ActionTokenCodebook,
) -> int:
    """Require all four candidates to share the exact context before their action token."""
    if set(rendered_candidates) != set(R2RCEAction):
        raise ValueError("Rendered candidates must contain exactly the four official actions.")
    positions = {
        action: locate_action_token(
            rendered_candidates[action],
            action=action,
            codebook=codebook,
        )
        for action in R2RCEAction
    }
    if len(set(positions.values())) != 1:
        raise ValueError("Rendered candidates place action tokens at different positions.")
    position = next(iter(positions.values()))
    reference = rendered_candidates[R2RCEAction.STOP]["input_ids"]
    if not isinstance(reference, torch.Tensor):
        raise TypeError("Rendered STOP input_ids must be a tensor.")
    reference_prefix = reference[:position]
    for action in R2RCEAction:
        input_ids = rendered_candidates[action]["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"Rendered {action.name} input_ids must be a tensor.")
        if not torch.equal(input_ids[:position], reference_prefix):
            raise ValueError("Rendered candidates do not share an identical pre-decision context.")
    return position


def select_action_from_token_nll(scores: dict[str, dict[str, Any]]) -> str:
    """Validate four one-token scores and return a deterministic official-order argmin."""
    legal = {action.name for action in R2RCEAction}
    if set(scores) != legal:
        raise ValueError("Action scores must contain exactly the four official actions.")
    values: dict[str, float] = {}
    positions: set[int] = set()
    token_ids: set[int] = set()
    for action in R2RCEAction:
        score = scores[action.name]
        if not isinstance(score, dict):
            raise ValueError(f"Action {action.name} score must be a mapping.")
        value = float(score.get("token_nll", float("nan")))
        if not math.isfinite(value):
            raise ValueError(f"Action {action.name} has a non-finite token NLL.")
        if score.get("scored_token_count") != 1:
            raise ValueError(f"Action {action.name} was not scored with exactly one token.")
        if score.get("action_id") != int(action):
            raise ValueError(f"Action {action.name} has the wrong official action ID.")
        position = score.get("scored_token_position")
        causal_position = score.get("causal_logit_position")
        token_id = score.get("action_token_id")
        if not isinstance(position, int) or position < 1:
            raise ValueError(f"Action {action.name} has an invalid scored token position.")
        if causal_position != position - 1:
            raise ValueError(f"Action {action.name} has an invalid causal logit position.")
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"Action {action.name} has an invalid tokenizer ID.")
        positions.add(position)
        token_ids.add(token_id)
        values[action.name] = value
    if len(positions) != 1:
        raise ValueError("All actions must be scored at the same causal position.")
    if len(token_ids) != len(R2RCEAction):
        raise ValueError("All actions must use distinct tokenizer IDs.")
    return min(R2RCEAction, key=lambda action: (values[action.name], int(action))).name


def score_action_token_logits(
    logits: torch.Tensor,
    *,
    action_token_position: int,
    codebook: ActionTokenCodebook,
) -> dict[str, Any]:
    """Score the four action tokens at one shared causal position.

    `action_token_position` is the input/label position of the action token. Causal logits at
    the preceding position predict it. The returned restricted probabilities renormalize only
    across the four legal action tokens; `token_nll` remains the full-vocabulary conditional NLL.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"logits must have shape [1,T,V], got {tuple(logits.shape)}.")
    if action_token_position < 1 or action_token_position >= logits.shape[1]:
        raise ValueError("action_token_position must identify a non-initial token inside T.")
    row = logits[0, action_token_position - 1].detach().float()
    if not bool(torch.isfinite(row).all().item()):
        raise ValueError("Action-decision logit row contains non-finite values.")
    if max(codebook.action_token_ids) >= row.numel():
        raise ValueError("An action token ID lies outside the model vocabulary.")

    full_log_probabilities = torch.log_softmax(row, dim=-1)
    selected_logits = row[list(codebook.action_token_ids)]
    restricted_probabilities = torch.softmax(selected_logits, dim=0)
    scores: dict[str, dict[str, Any]] = {}
    for action in R2RCEAction:
        token_id = codebook.action_token_id(action)
        scores[action.name] = {
            "action_id": int(action),
            "action_token_id": token_id,
            "scored_token_position": action_token_position,
            "causal_logit_position": action_token_position - 1,
            "scored_token_count": 1,
            "token_nll": float(-full_log_probabilities[token_id].item()),
            "restricted_probability": float(restricted_probabilities[int(action)].item()),
        }
    predicted = select_action_from_token_nll(scores)
    ordered_nll = sorted(float(score["token_nll"]) for score in scores.values())
    probabilities = [float(scores[action.name]["restricted_probability"]) for action in R2RCEAction]
    entropy = -sum(
        probability * math.log(probability) for probability in probabilities if probability
    )
    return {
        "protocol_version": ACTION_DECISION_PROTOCOL_VERSION,
        "method": ACTION_DECISION_METHOD,
        "legal_actions": [action.name for action in R2RCEAction],
        "scores": scores,
        "predicted_action": predicted,
        "top_two_margin_nats": ordered_nll[1] - ordered_nll[0],
        "restricted_entropy_nats": entropy,
    }


def validate_action_decision_record(
    decision: dict[str, Any],
    *,
    codebook_record: dict[str, Any] | None = None,
    tolerance: float = 1e-6,
) -> list[str]:
    """Return independent structural and numerical errors for a decision record."""
    errors: list[str] = []
    if decision.get("protocol_version") != ACTION_DECISION_PROTOCOL_VERSION:
        errors.append("action-decision protocol version mismatch")
    if decision.get("method") != ACTION_DECISION_METHOD:
        errors.append("action-decision method mismatch")
    scores = decision.get("scores")
    if not isinstance(scores, dict):
        return [*errors, "action scores are missing"]
    try:
        predicted = select_action_from_token_nll(scores)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        return errors
    if decision.get("predicted_action") != predicted:
        errors.append("recorded action is not the deterministic minimum-token-NLL action")

    nll_values = [float(scores[action.name]["token_nll"]) for action in R2RCEAction]
    minimum = min(nll_values)
    weights = [math.exp(-(value - minimum)) for value in nll_values]
    normalizer = sum(weights)
    expected_probabilities = [weight / normalizer for weight in weights]
    try:
        observed_probabilities = [
            float(scores[action.name].get("restricted_probability", float("nan")))
            for action in R2RCEAction
        ]
    except (TypeError, ValueError):
        observed_probabilities = [float("nan")] * len(R2RCEAction)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in observed_probabilities):
        errors.append("restricted action probabilities are invalid")
    elif any(
        abs(observed - expected) > tolerance
        for observed, expected in zip(observed_probabilities, expected_probabilities, strict=True)
    ):
        errors.append("restricted action probabilities do not match token NLLs")

    ordered = sorted(nll_values)
    expected_margin = ordered[1] - ordered[0]
    try:
        observed_margin = float(decision.get("top_two_margin_nats", float("nan")))
    except (TypeError, ValueError):
        observed_margin = float("nan")
    if not math.isfinite(observed_margin) or abs(observed_margin - expected_margin) > tolerance:
        errors.append("top-two action margin is invalid")
    expected_entropy = -sum(
        probability * math.log(probability) for probability in expected_probabilities if probability
    )
    try:
        observed_entropy = float(decision.get("restricted_entropy_nats", float("nan")))
    except (TypeError, ValueError):
        observed_entropy = float("nan")
    if not math.isfinite(observed_entropy) or abs(observed_entropy - expected_entropy) > tolerance:
        errors.append("restricted action entropy is invalid")

    if codebook_record is not None:
        if codebook_record.get("protocol_version") != ACTION_DECISION_PROTOCOL_VERSION:
            errors.append("codebook protocol version mismatch")
        if codebook_record.get("method") != ACTION_DECISION_METHOD:
            errors.append("codebook method mismatch")
        actions = codebook_record.get("actions", {})
        if not isinstance(actions, dict):
            errors.append("codebook actions are not a mapping")
            return errors
        for action in R2RCEAction:
            expected = actions.get(action.name, {})
            observed = scores[action.name]
            if expected.get("action_id") != observed.get("action_id"):
                errors.append(f"{action.name}: codebook action ID mismatch")
            if expected.get("action_token_id") != observed.get("action_token_id"):
                errors.append(f"{action.name}: codebook token ID mismatch")
    return errors
