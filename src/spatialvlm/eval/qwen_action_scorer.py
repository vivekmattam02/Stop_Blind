"""Real native-Qwen scorer for the single-step closed-loop policy bridge.

The bridge in :mod:`spatialvlm.eval.native_policy` deliberately depends on a small ``ActionScorer``
protocol.  This module is the model-backed implementation for C0: it renders the current causal
history with the pinned Qwen processor, runs ``Qwen3VLBackbone`` once, and compares the four legal
action-token logits at one shared causal position.

CR-021 is only partially accepted.  Version 1 therefore supports exactly one closed-loop action
per model call.  Four-action chunks require a different, versioned renderer, STOP-tail mask, and
autoregressive scoring contract; returning repeated copies of one one-step decision would be a
silent scientific error.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from spatialvlm.backbone.qwen3_vl import Qwen3VLBackbone
from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.history import HistoryWindow
from spatialvlm.data.native_qwen import NativeQwenSFTConfig, build_native_qwen_sft_batch
from spatialvlm.eval.action_decision import (
    ActionTokenCodebook,
    build_action_token_codebook,
    locate_action_token,
    score_action_token_logits,
)
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    file_sha256,
    load_native_c0_selected_checkpoint,
)

NATIVE_QWEN_ACTION_SCORER_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _candidate_record(batch: Any) -> dict[str, torch.Tensor | int | str]:
    """Return the narrow surface required by ``locate_action_token``."""
    return {
        "input_ids": batch.model_inputs["input_ids"][0],
        "labels": batch.model_inputs["labels"][0],
        "target_start_idx": int(batch.record["target_start_idx"]),
    }


def _synchronize(device: torch.device) -> None:
    """Synchronize CUDA only when timing an actual CUDA forward."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@dataclass
class NativeQwenActionScorer:
    """Score one legal R2R-CE action with an already-loaded native Qwen backbone.

    ``checkpoint_sha256`` is mandatory provenance.  The caller may construct and load the
    backbone itself, or use :meth:`from_native_c0_checkpoint` for the admitted checkpoint format.
    """

    backbone: Qwen3VLBackbone
    processor: Any
    device: torch.device
    checkpoint_sha256: str
    model_id: str = NATIVE_C0_MODEL_ID
    model_revision: str = NATIVE_C0_MODEL_REVISION
    render_config: NativeQwenSFTConfig = field(default_factory=NativeQwenSFTConfig)
    codebook: ActionTokenCodebook = field(init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if not _SHA256_RE.fullmatch(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be one lowercase SHA-256 digest.")
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.codebook = build_action_token_codebook(tokenizer)

    @classmethod
    def from_native_c0_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
        expected_checkpoint_sha256: str,
        expected_checkpoint_gate: str,
        expected_manifest_sha256: str,
        expected_trainable_state_sha256: str,
        expected_update_step: int,
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = NATIVE_C0_MODEL_ID,
        model_revision: str = NATIVE_C0_MODEL_REVISION,
        local_files_only: bool = True,
    ) -> NativeQwenActionScorer:
        """Load the exact native-C0 model, processor, and trainable checkpoint on Spark.

        This factory accepts only one exact, independently verified selected-checkpoint identity.
        The caller must supply its producing gate, manifest, update, file hash, and trainable-state
        hash; the loader independently reconstructs tensor-name and shape parity.
        """
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        checkpoint_sha256 = file_sha256(checkpoint)
        if checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError("Selected native-C0 scorer checkpoint hash mismatch.")
        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=model_revision,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        backbone = Qwen3VLBackbone(
            model_id=model_id,
            lora_rank=32,
            lora_alpha=64,
            lora_dropout=0.0,
            enable_lora=True,
            freeze_base_model=True,
            apply_peft_2880_workaround=False,
            apply_spatial_rope=False,
            device=torch.device(device),
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            revision=model_revision,
        )
        load_native_c0_selected_checkpoint(
            checkpoint,
            model=backbone,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_gate=expected_checkpoint_gate,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_trainable_state_sha256=expected_trainable_state_sha256,
            expected_update_step=expected_update_step,
        )
        backbone.eval()
        return cls(
            backbone=backbone,
            processor=processor,
            device=torch.device(device),
            checkpoint_sha256=checkpoint_sha256,
            model_id=model_id,
            model_revision=model_revision,
        )

    def score_chunk(
        self,
        *,
        window: HistoryWindow,
        instruction: str,
        chunk_length: int,
    ) -> tuple[tuple[R2RCEAction, ...], dict[str, Any]]:
        """Return one constrained action and JSON-safe decision telemetry."""
        if chunk_length != 1:
            raise ValueError(
                "NativeQwenActionScorer v1 supports chunk_length=1 only; four-action scoring "
                "requires the unresolved CR-021 renderer/objective and must not be synthesized "
                "from one-step logits."
            )

        processor_started = time.perf_counter()
        # The suffix action is only a causal-position probe.  Its token cannot affect logits at
        # the preceding position, and labels are removed before the inference forward.
        probe_action = R2RCEAction.STOP
        batch = build_native_qwen_sft_batch(
            self.processor,
            window,
            instruction=instruction,
            target_action=probe_action,
            config=self.render_config,
        )
        action_position = locate_action_token(
            _candidate_record(batch),
            action=probe_action,
            codebook=self.codebook,
        )
        processor_seconds = time.perf_counter() - processor_started

        model_inputs = {
            key: tensor.to(device=self.device, non_blocking=True)
            for key, tensor in batch.model_inputs.items()
            if key != "labels"
        }
        self.backbone.eval()
        _synchronize(self.device)
        model_started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.backbone(**model_inputs)
        _synchronize(self.device)
        model_seconds = time.perf_counter() - model_started

        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Native Qwen backbone output must expose a tensor-valued logits field.")
        decision = score_action_token_logits(
            logits,
            action_token_position=action_position,
            codebook=self.codebook,
        )
        predicted = R2RCEAction[decision["predicted_action"]]
        diagnostics = {
            "schema_version": NATIVE_QWEN_ACTION_SCORER_VERSION,
            "method": "native_qwen_single_shared_action_token_v1",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "chunk_length": 1,
            "history_contract_id": window.contract.contract_id,
            "action_token_position": action_position,
            "probe_suffix_action": probe_action.name,
            "processor_seconds": processor_seconds,
            "model_forward_seconds": model_seconds,
            "sequence_length": int(batch.record["sequence_length"]),
            "image_count": int(batch.record["image_count"]),
            "total_visual_tokens": int(batch.record["total_visual_tokens"]),
            "model_input_shapes": {
                key: list(tensor.shape) for key, tensor in sorted(model_inputs.items())
            },
            "action_decision": decision,
        }
        return (predicted,), diagnostics
