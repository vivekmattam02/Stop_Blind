"""Closed-loop chunk scoring with the geometry branch attached. Additive module.

``AutoregressiveChunkScorer`` renders a prompt with no geometry tokens. A checkpoint trained by
``run_chunk_geometry_pilot.py`` cannot be evaluated with it: that adapter was trained on sequences
carrying 256 spliced geometry tokens, and running it on sequences without them is a train/deploy
mismatch that produces fluent, plausible, meaningless actions. The generic runner loads only
``state["adapter"]`` and would silently discard the 409 geometry tensors, so the failure is
invisible at the call site --- which is precisely why this module exists rather than a flag.

The layout is reproduced exactly. Training renders the chat template, then splices the geometry
block at ``target_start`` (the end of the generation prompt), then appends the action target:

    [chat template][GEO x N]["Action IDs: a b c d"]

Rollout has no target to render, so it emits the same prefix and decodes:

    [chat template][GEO x N]["Action IDs:"] -> decode a, b, c, d

The geometry block sits at the identical boundary in both, which is the fifth pre-flight question
(distribution match) satisfied by construction rather than by assertion.

Depth comes from the live simulator, taken from the same four history frames the model sees, so no
cached observation is consulted during rollout.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from spatialvlm.backbone.rope_patch_v2 import stash_geometry
from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.history import HistoryWindow
from spatialvlm.data.native_qwen_chunk_geometry import filler_token_id
from spatialvlm.eval.action_chunk_decision import truncate_chunk_at_stop
from spatialvlm.eval.native_chunk_scorer import (
    CHUNK_SCORER_METHOD,
    AutoregressiveChunkScorer,
)

GEOMETRY_SCORER_METHOD = "geometry_autoregressive_kv_cached_restricted_chunk_decode_v1"


class GeometryChunkScorer(AutoregressiveChunkScorer):
    """Autoregressive chunk decoding with metric-depth geometry tokens spliced into the prompt."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        codebook: Any,
        *,
        branch: Any,
        rotary: Any,
        embed: Any,
        config: Any | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(model, processor, codebook, config=config, device=device)
        self.branch = branch
        self.rotary = rotary
        self.embed = embed
        self.filler_id = filler_token_id(
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self._last_geometry: dict[str, Any] = {}

    def _depth_stack(self, window: HistoryWindow) -> torch.Tensor:
        """Metric depth for the window's frames, in chronological order, from the live simulator."""
        frames = []
        for frame in window.frames:
            obs = frame.observation
            depth = obs.get("depth_meters") if isinstance(obs, Mapping) else None
            if depth is None:
                raise RuntimeError(
                    "A history frame carries no 'depth_meters'. The geometry arm cannot be "
                    "evaluated against an RGB-only observation stream."
                )
            arr = np.asarray(depth, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            frames.append(torch.from_numpy(arr))
        stack = torch.stack(frames, dim=0)
        if stack.shape[0] != self.branch.n_frames:
            raise RuntimeError(
                f"Geometry branch expects {self.branch.n_frames} frames, window gave "
                f"{stack.shape[0]}."
            )
        return stack

    def _prompt_inputs(  # type: ignore[override]
        self, window: HistoryWindow, instruction: str
    ) -> dict[str, torch.Tensor]:
        """Render the prompt with the geometry block spliced at the training boundary."""
        from spatialvlm.data.native_qwen_chunk import (
            NativeQwenChunkConfig,
            build_native_qwen_chunk_messages,
        )

        messages = build_native_qwen_chunk_messages(
            window,
            instruction=instruction,
            config=NativeQwenChunkConfig(chunk_size=self.config.chunk_size),
        )
        rendered = dict(
            self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
        )
        rendered.pop("token_type_ids", None)
        ids = rendered["input_ids"]
        splice_at = int(ids.shape[1])  # end of the generation prompt == training's target_start
        n = self.branch.n_tokens

        geo = torch.full((1, n), self.filler_id, dtype=ids.dtype)
        prefix = torch.tensor(
            [list(self.codebook.shared_prefix_token_ids)], dtype=ids.dtype
        )
        rendered["input_ids"] = torch.cat((ids[:, :splice_at], geo, prefix), dim=1)

        for key in ("attention_mask", "mm_token_type_ids"):
            value = rendered.get(key)
            if value is None:
                continue
            pad = 1 if key == "attention_mask" else 0
            rendered[key] = torch.cat(
                (
                    value[:, :splice_at],
                    torch.full((1, n), pad, dtype=value.dtype),
                    torch.full((1, prefix.shape[1]), pad, dtype=value.dtype),
                ),
                dim=1,
            )

        total = int(rendered["input_ids"].shape[1])
        if total > self.config.max_length:
            raise ValueError(f"Geometry rollout prompt exceeds max_length: {total}.")

        mask = torch.zeros((1, total), dtype=torch.bool)
        mask[:, splice_at : splice_at + n] = True
        if int(mask.sum()) != n:
            raise RuntimeError("Geometry mask does not mark exactly the spliced block.")

        inputs = {k: v.to(self.device) for k, v in rendered.items() if isinstance(v, torch.Tensor)}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        coords, validity = self.branch.points_from_depth(self._depth_stack(window))
        coords = coords.to(self.device)
        embeds = self.branch(coords, validity.to(self.device))
        self._last_geometry = {
            "geometry_tokens": n,
            "splice_at": splice_at,
            "prompt_token_count": total,
            "valid_patch_fraction": float(validity.mean()),
            "mean_abs_coord_m": float(coords.abs().mean()),
        }
        stash_geometry(
            self.rotary, self.embed,
            coords_m=coords, spatial_mask=mask.to(self.device), geometry_embeds=embeds,
        )
        return inputs

    @torch.no_grad()
    def score_chunk(  # type: ignore[override]
        self, *, window: HistoryWindow, instruction: str, chunk_length: int
    ) -> tuple[Sequence[R2RCEAction], Mapping[str, Any]]:
        """Prefill once with geometry armed, then decode the chunk from the KV cache.

        The stash is consumed by the prefill and cleared; decode steps run on the cache, where the
        geometry rows already live. That mirrors training, where the rotary patch and embedding
        hook fire once per forward.
        """
        if chunk_length != self.config.chunk_size:
            raise ValueError(f"Configured for chunk_size={self.config.chunk_size}.")
        was_training = self.model.training
        self.model.eval()
        self.branch.eval()
        try:
            inputs = self._prompt_inputs(window, instruction)  # arms the stash
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
                    input_ids=step_ids, attention_mask=attention,
                    past_key_values=past, use_cache=True,
                )
                decode_seconds.append(time.perf_counter() - started)
                past = outputs.past_key_values

            executable, terminated = truncate_chunk_at_stop(proposed)
        finally:
            if was_training:
                self.model.train()

        diagnostics = {
            "scorer_method": GEOMETRY_SCORER_METHOD,
            "base_scorer_method": CHUNK_SCORER_METHOD,
            "chunk_size": chunk_length,
            "proposed_chunk": [a.name for a in proposed],
            "executed_chunk": [a.name for a in executable],
            "discarded_after_stop": len(proposed) - len(executable),
            "terminated_by_stop": terminated,
            "proposed_chunk_is_constant": len({int(a) for a in proposed}) == 1,
            "restricted_distributions": distributions,
            "model_calls": 1 + len(decode_seconds),
            "prefill_seconds": prefill_seconds,
            "total_seconds": prefill_seconds + sum(decode_seconds),
            **self._last_geometry,
        }
        return executable, diagnostics
