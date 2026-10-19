"""Restricted four-way cross-entropy for the k-action chunk objective.

The frozen single-token objective computes cross-entropy over exactly the four official action
token logits at one causal position (``scripts/run_spark_native_development_measurements.py::
_restricted_loss``). Full-vocabulary cross-entropy is never used for the training signal, so a
model cannot lower its loss by manipulating probability mass outside the four legal actions.

The chunk objective keeps that property per position and averages. With ``chunk_size`` positions the
mean is directly comparable to the frozen scalar: an uninformative policy scores ``ln 4 = 1.3863``
under both, so chunk losses and single-token losses sit on the same scale even though they supervise
different numbers of tokens.

Averaging rather than summing is deliberate. A sum would make the chunk loss ``chunk_size`` times
larger and silently rescale the effective learning rate relative to the frozen runs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.eval.action_chunk_decision import ActionChunkCodebook

CHUNK_LOSS_METHOD = "mean_restricted_four_way_cross_entropy_over_chunk_positions_v1"


def restricted_chunk_loss(
    logits: torch.Tensor,
    *,
    action_token_positions: Sequence[int],
    codebook: ActionChunkCodebook,
    target_actions: Sequence[R2RCEAction | int],
) -> tuple[torch.Tensor, list[float]]:
    """Return the mean restricted four-way CE over chunk positions and its per-position values.

    ``logits`` must be ``[1, T, V]``. Each supervised position is scored from the causal row that
    precedes it, matching the frozen single-token convention exactly.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"logits must have shape [1,T,V], got {tuple(logits.shape)}.")
    if len(action_token_positions) != codebook.chunk_size:
        raise ValueError(
            f"Expected {codebook.chunk_size} action positions, got {len(action_token_positions)}."
        )
    if len(target_actions) != codebook.chunk_size:
        raise ValueError(
            f"Expected {codebook.chunk_size} target actions, got {len(target_actions)}."
        )
    if len(set(action_token_positions)) != len(action_token_positions):
        raise ValueError("Chunk action positions must be distinct.")
    if list(action_token_positions) != sorted(action_token_positions):
        raise ValueError("Chunk action positions must increase with chunk position.")

    action_ids = torch.tensor(
        [codebook.action_token_id(action) for action in R2RCEAction], device=logits.device
    )
    if int(action_ids.max().item()) >= logits.shape[2]:
        raise ValueError("An action token ID lies outside the model vocabulary.")

    ordered = list(R2RCEAction)
    per_position: list[torch.Tensor] = []
    for index, position in enumerate(action_token_positions):
        if position < 1 or position >= logits.shape[1]:
            raise ValueError("Every action position must identify a non-initial token inside T.")
        row = logits[0, position - 1].float()
        restricted = row[action_ids]
        target_index = ordered.index(R2RCEAction(int(target_actions[index])))
        per_position.append(
            functional.cross_entropy(
                restricted.unsqueeze(0),
                torch.tensor([target_index], device=row.device),
            )
        )
    stacked = torch.stack(per_position)
    return stacked.mean(), [float(value.detach().cpu().item()) for value in per_position]
