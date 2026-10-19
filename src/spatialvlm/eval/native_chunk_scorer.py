"""Autoregressive chunk scoring for closed-loop rollout — the CR-021 precondition.

CR-021 rejected four-action chunking as the primary frozen interface but allowed it as a named
efficiency pilot once four things existed: a versioned renderer/objective, a STOP-tail mask, an
**autoregressive scoring contract**, and matched quality/latency evidence. This module is the third.

Why a separate contract is needed at all. During training the chunk target is rendered in full, so
position *i* is scored with the ground-truth actions at positions ``< i`` already present in the
context. A rollout has no target to render: it must emit action 0, feed that back, then emit
action 1 conditioned on its own choice. Training therefore teacher-forces *inside* the chunk while
inference does not — the same distribution shift the manuscript documents between reference states
and executed states, reproduced at chunk scale. The gap is bounded by ``chunk_size`` rather than by
episode length, but it is real and must be declared rather than assumed away.

Where the speed-up actually comes from. Internal review CR-021 correctly noted that a fourfold
reduction in model calls does not follow from chunking alone: Efficient-VLN's reported latency uses
KV-cache reuse. This implementation matches that structure — one expensive prefill over the visual
context, then ``chunk_size`` cheap single-token decode steps that reuse the cache. The saving is
measured per call in the returned diagnostics rather than asserted.

The STOP-tail mask is applied here too: a predicted ``STOP`` ends the episode, so any action the
model proposed after it is discarded rather than executed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.history import HistoryWindow
from spatialvlm.data.native_qwen_chunk import (
    NativeQwenChunkConfig,
    build_native_qwen_chunk_messages,
)
from spatialvlm.eval.action_chunk_decision import (
    ACTION_CHUNK_METHOD,
    ACTION_CHUNK_PROTOCOL_VERSION,
    ActionChunkCodebook,
    truncate_chunk_at_stop,
)

CHUNK_SCORER_METHOD = "autoregressive_kv_cached_restricted_chunk_decode_v1"


def chunk_separator_token_ids(codebook: ActionChunkCodebook) -> tuple[tuple[int, ...], ...]:
    """Return the tokens that sit between consecutive action positions.

    Derived from the codebook's own reference tokenization rather than hardcoded, so a checkpoint
    whose chat surface separates digits differently still produces a correct decode sequence.
    """
    reference = codebook.reference_target_token_ids
    offsets = codebook.action_token_offsets
    return tuple(
        tuple(reference[offsets[i] + 1 : offsets[i + 1]]) for i in range(len(offsets) - 1)
    )


@dataclass(frozen=True)
class ChunkScorerConfig:
    """Frozen inference contract for autoregressive chunk decoding."""

    chunk_size: int = 4
    max_length: int = 4096

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least one.")


class AutoregressiveChunkScorer:
    """Emit one action chunk per visual context, decoding positions autoregressively.

    Satisfies the ``ActionScorer`` protocol in ``spatialvlm.eval.native_policy``. Each call performs
    one multimodal prefill and ``chunk_length`` restricted single-token decodes, reusing the KV
    cache between them.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        codebook: ActionChunkCodebook,
        *,
        config: ChunkScorerConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.codebook = codebook
        self.config = config or ChunkScorerConfig(chunk_size=codebook.chunk_size)
        if self.config.chunk_size != codebook.chunk_size:
            raise ValueError("Scorer chunk size disagrees with the supplied codebook.")
        self.device = device or next(model.parameters()).device
        self.separators = chunk_separator_token_ids(codebook)
        self._action_ids = torch.tensor(
            [codebook.action_token_id(a) for a in R2RCEAction], device=self.device
        )

    def _prompt_inputs(self, window: HistoryWindow, instruction: str) -> dict[str, torch.Tensor]:
        messages = build_native_qwen_chunk_messages(
            window,
            instruction=instruction,
            config=NativeQwenChunkConfig(chunk_size=self.config.chunk_size),
        )
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        rendered = dict(rendered)
        rendered.pop("token_type_ids", None)

        prefix = torch.tensor(
            [list(self.codebook.shared_prefix_token_ids)],
            dtype=rendered["input_ids"].dtype,
        )
        rendered["input_ids"] = torch.cat((rendered["input_ids"], prefix), dim=1)
        rendered["attention_mask"] = torch.cat(
            (rendered["attention_mask"], torch.ones_like(prefix)), dim=1
        )
        mm = rendered.get("mm_token_type_ids")
        if mm is not None:
            rendered["mm_token_type_ids"] = torch.cat((mm, torch.zeros_like(prefix)), dim=1)
        if rendered["input_ids"].shape[1] > self.config.max_length:
            raise ValueError(
                "Chunk rollout prompt exceeds max_length: "
                f"{rendered['input_ids'].shape[1]} > {self.config.max_length}."
            )
        inputs = {k: v.to(self.device) for k, v in rendered.items() if isinstance(v, torch.Tensor)}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs

    def _restricted_choice(self, logits: torch.Tensor) -> tuple[R2RCEAction, list[float]]:
        row = logits[0, -1].detach().float()
        if not bool(torch.isfinite(row).all().item()):
            raise ValueError("Chunk decode produced a non-finite logit row.")
        restricted = row[self._action_ids]
        probabilities = torch.softmax(restricted, dim=0)
        action = R2RCEAction(int(torch.argmax(restricted).item()))
        return action, [float(p) for p in probabilities.detach().cpu().tolist()]

    @torch.no_grad()
    def score_chunk(
        self,
        *,
        window: HistoryWindow,
        instruction: str,
        chunk_length: int,
    ) -> tuple[Sequence[R2RCEAction], Mapping[str, Any]]:
        """Return the executable action prefix and verbatim rollout diagnostics."""
        if chunk_length != self.config.chunk_size:
            raise ValueError(
                f"Scorer is configured for chunk_size={self.config.chunk_size}, "
                f"asked for {chunk_length}."
            )
        was_training = self.model.training
        self.model.eval()
        try:
            inputs = self._prompt_inputs(window, instruction)
            prompt_length = int(inputs["input_ids"].shape[1])

            started = time.perf_counter()
            outputs = self.model(**inputs, use_cache=True)
            prefill_seconds = time.perf_counter() - started
            past = outputs.past_key_values

            proposed: list[R2RCEAction] = []
            distributions: list[list[float]] = []
            decode_seconds: list[float] = []
            attention = inputs["attention_mask"]

            for index in range(chunk_length):
                action, probabilities = self._restricted_choice(outputs.logits)
                proposed.append(action)
                distributions.append(probabilities)
                if index == chunk_length - 1:
                    break
                nxt = [self.codebook.action_token_id(action), *self.separators[index]]
                step_ids = torch.tensor(
                    [nxt], dtype=inputs["input_ids"].dtype, device=self.device
                )
                attention = torch.cat((attention, torch.ones_like(step_ids)), dim=1)
                started = time.perf_counter()
                outputs = self.model(
                    input_ids=step_ids,
                    attention_mask=attention,
                    past_key_values=past,
                    use_cache=True,
                )
                decode_seconds.append(time.perf_counter() - started)
                past = outputs.past_key_values

            executable, terminated = truncate_chunk_at_stop(proposed)
        finally:
            if was_training:
                self.model.train()

        diagnostics = {
            "protocol_version": ACTION_CHUNK_PROTOCOL_VERSION,
            "objective_method": ACTION_CHUNK_METHOD,
            "scorer_method": CHUNK_SCORER_METHOD,
            "chunk_size": chunk_length,
            "proposed_chunk": [a.name for a in proposed],
            "proposed_chunk_ids": [int(a) for a in proposed],
            "executed_chunk": [a.name for a in executable],
            "executed_chunk_ids": [int(a) for a in executable],
            "discarded_after_stop": len(proposed) - len(executable),
            "terminated_by_stop": terminated,
            "proposed_chunk_is_constant": len({int(a) for a in proposed}) == 1,
            "restricted_distributions": distributions,
            "legal_actions": [a.name for a in R2RCEAction],
            "prompt_token_count": prompt_length,
            "model_calls": 1 + len(decode_seconds),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": prefill_seconds + sum(decode_seconds),
            "seconds_per_executed_action": (
                (prefill_seconds + sum(decode_seconds)) / len(executable) if executable else None
            ),
            "teacher_forcing_note": (
                "Training renders the full chunk target, so position i is supervised with "
                "ground-truth actions at positions < i. This rollout conditions position i on its "
                "own earlier predictions. The mismatch is bounded by chunk_size."
            ),
        }
        return executable, diagnostics
