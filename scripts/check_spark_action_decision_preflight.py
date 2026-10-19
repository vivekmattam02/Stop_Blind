#!/usr/bin/env python3
"""Verify the length-fair action codebook with the real Qwen checkpoint tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from spatialvlm.config.model import SpatialVLMConfig
from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.tokenization import SpatialTokenizerConfig, build_input_ids
from spatialvlm.eval.action_decision import (
    ACTION_DECISION_SYSTEM_PROMPT,
    build_action_token_codebook,
    canonical_action_target,
    locate_shared_action_position,
)
from spatialvlm.eval.action_decision_artifacts import (
    ACTION_DECISION_GATE_SCHEMA_VERSION,
    ACTION_DECISION_PREFLIGHT_GATE_NAME,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prefix_sha256(input_ids: torch.Tensor, position: int) -> str:
    prefix = input_ids[:position].detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(prefix.dtype).encode("ascii"))
    digest.update(repr(tuple(prefix.shape)).encode("ascii"))
    digest.update(prefix.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": ACTION_DECISION_GATE_SCHEMA_VERSION,
        "gate": ACTION_DECISION_PREFLIGHT_GATE_NAME,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    try:
        model_config = SpatialVLMConfig()
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.backbone.model_id,
            trust_remote_code=True,
        )
        codebook = build_action_token_codebook(tokenizer)
        token_config = SpatialTokenizerConfig(
            num_spatial_tokens=model_config.fusion.sva_num_queries,
            max_length=2048,
            system_prompt=ACTION_DECISION_SYSTEM_PROMPT,
        )
        rendered = {
            action: build_input_ids(
                tokenizer,
                instruction="Walk forward and stop near the doorway.",
                target=canonical_action_target(action),
                config=token_config,
            )
            for action in R2RCEAction
        }
        position = locate_shared_action_position(rendered, codebook=codebook)
        reference_ids = rendered[R2RCEAction.STOP]["input_ids"]
        if not isinstance(reference_ids, torch.Tensor):
            raise TypeError("Rendered reference input_ids are not a tensor.")
        payload.update(
            {
                "status": "passed",
                "backbone_model_id": model_config.backbone.model_id,
                "codebook": codebook.to_record(),
                "shared_action_token_position": position,
                "shared_context_sha256": _prefix_sha256(reference_ids, position),
                "candidates": {
                    action.name: {
                        "sequence_length": int(rendered[action]["input_ids"].numel()),
                        "target_start_idx": rendered[action]["target_start_idx"],
                        "supervised_token_count": rendered[action]["supervised_token_count"],
                        "action_token_position": position,
                    }
                    for action in R2RCEAction
                },
            }
        )
        _atomic_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _atomic_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
