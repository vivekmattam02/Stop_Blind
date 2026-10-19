"""Post-render geometry splice for the chunk objective. Additive module.

The frozen chunk renderer (`native_qwen_chunk.py`, hash-locked lineage) is called **unchanged**;
this module then inserts ``n_tokens`` filler token ids at ``target_start`` — after the assistant
generation header, before the action target — and mechanically shifts everything downstream.
The filler id is irrelevant to the forward pass: the embedding hook replaces those rows and the
rotary patch overwrites their phases. What matters, and what is verified fail-closed, is that the
positions are exact: every supervised position still carries its official action token after the
shift, and the mask marks exactly the spliced block.

Layout, identical at train and deploy::

    [system][user: 4 labeled frames + instruction][<assistant>][GEO x N]["Action IDs: a b c d"]

At closed-loop decode the prefill ends after the GEO block and decoding proceeds over action ids,
so the sequence the policy was trained on is the sequence it is executed with.
"""

from __future__ import annotations

from typing import Any

import torch

from spatialvlm.data.native_qwen import NativeQwenBatch  # noqa: PLC2701 - shared dataclass

GEOMETRY_SPLICE_VERSION = 1
FILLER_SURFACE = "!"  # any single-token text surface; embeddings at these rows are replaced


def filler_token_id(tokenizer: Any) -> int:
    """Resolve the filler surface to exactly one vocabulary id, failing closed otherwise."""
    ids = tokenizer.encode(FILLER_SURFACE, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Filler surface {FILLER_SURFACE!r} must be one token, got {ids}.")
    return int(ids[0])


def splice_geometry_block(
    batch: NativeQwenBatch,
    *,
    n_tokens: int,
    filler_id: int,
    ignore_index: int = -100,
) -> tuple[dict[str, torch.Tensor], tuple[int, ...], torch.Tensor, dict[str, Any]]:
    """Insert the geometry block and return (model_inputs, supervised_positions, mask, record).

    Every tensor field of the rendered batch is either length-preserving metadata (image grids,
    pixel values) or a [1, L] sequence field; sequence fields are split at ``target_start`` and
    re-joined around the filler block. Labels are rebuilt from the shifted positions and verified
    token-by-token against the frozen record's ``supervised_token_ids``.
    """
    if n_tokens < 1:
        raise ValueError("n_tokens must be at least one.")
    record = batch.record
    ts = int(record["target_start_idx"])
    old_positions = tuple(int(p) for p in record["supervised_token_positions"])
    expected_ids = tuple(int(v) for v in record["supervised_token_ids"])

    inputs = dict(batch.model_inputs)
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Expected [1, L] input_ids.")
    seq_len = ids.shape[1]
    if not all(0 < p < seq_len for p in old_positions) or min(old_positions) < ts:
        raise ValueError("Frozen supervised positions do not sit inside the target region.")

    filler = torch.full((1, n_tokens), filler_id, dtype=ids.dtype, device=ids.device)
    new_ids = torch.cat([ids[:, :ts], filler, ids[:, ts:]], dim=1)

    out: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if key in ("input_ids", "labels"):
            continue
        if value.ndim == 2 and value.shape[:2] == (1, seq_len):
            pad_value = 1 if key == "attention_mask" else 0
            pad = torch.full((1, n_tokens), pad_value, dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value[:, :ts], pad, value[:, ts:]], dim=1)
        else:
            out[key] = value  # pixel_values, image_grid_thw, ... untouched
    out["input_ids"] = new_ids

    new_positions = tuple(p + n_tokens for p in old_positions)
    labels = torch.full_like(new_ids, ignore_index)
    for position, expected in zip(new_positions, expected_ids):
        observed = int(new_ids[0, position].item())
        if observed != expected:
            raise RuntimeError(
                f"Post-splice position {position} carries token {observed}, expected {expected}; "
                "the splice arithmetic is wrong and supervision would be misaligned."
            )
        labels[0, position] = expected
    out["labels"] = labels

    mask = torch.zeros_like(new_ids, dtype=torch.bool)
    mask[:, ts : ts + n_tokens] = True
    if int(mask.sum()) != n_tokens:
        raise RuntimeError("Spatial mask does not mark exactly the spliced block.")

    splice_record = {
        "geometry_splice_version": GEOMETRY_SPLICE_VERSION,
        "n_geometry_tokens": n_tokens,
        "filler_token_id": filler_id,
        "splice_at": ts,
        "sequence_length": int(new_ids.shape[1]),
        "supervised_token_positions": list(new_positions),
    }
    return out, new_positions, mask, splice_record
