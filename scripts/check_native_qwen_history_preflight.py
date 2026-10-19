#!/usr/bin/env python3
"""Validate H0/H1/H4 rendering through the pinned native Qwen3-VL processor.

This is a processor-only gate: it does not load model weights and it does not establish training
or navigation competence. Its purpose is to catch chat-template, visual-token, chronological
history, and exact one-action-token supervision drift before a C0 training run is allowed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor

from spatialvlm.data.datasets import R2RCEAction
from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode, build_history_window
from spatialvlm.data.native_qwen import build_native_qwen_sft_batch
from spatialvlm.data.native_qwen_artifacts import (
    NATIVE_QWEN_HISTORY_MODEL_ID,
    NATIVE_QWEN_HISTORY_MODEL_REVISION,
    NATIVE_QWEN_HISTORY_PREFLIGHT_GATE_NAME,
    NATIVE_QWEN_HISTORY_PREFLIGHT_SCHEMA_VERSION,
    processor_metadata_locks,
    source_locks,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _synthetic_observations() -> list[Image.Image]:
    return [
        Image.new("RGB", (224, 224), color=(32 + index * 40, 96, 160 - index * 20))
        for index in range(4)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", default=NATIVE_QWEN_HISTORY_MODEL_ID)
    parser.add_argument("--model-revision", default=NATIVE_QWEN_HISTORY_MODEL_REVISION)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--docker-image-ref", required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    payload: dict[str, Any] = {
        "schema_version": NATIVE_QWEN_HISTORY_PREFLIGHT_SCHEMA_VERSION,
        "gate": NATIVE_QWEN_HISTORY_PREFLIGHT_GATE_NAME,
        "run_id": args.run_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "docker_image": {
            "reference": args.docker_image_ref,
            "id": args.docker_image_id,
        },
        "local_files_only": args.local_files_only,
        "evidence_boundary": {
            "established_if_passed": (
                "The pinned Qwen processor renders the versioned H0/H1/H4 RGB/action histories, "
                "native image tensors, processor-selected optional multimodal inputs, and "
                "exactly one supervised official-action token without truncation for one "
                "deterministic synthetic example."
            ),
            "not_established": (
                "Model forward/backward correctness, visual-token budget selection, training "
                "quality, closed-loop navigation competence, or any C0-C4 performance claim."
            ),
        },
    }
    try:
        payload["processor_metadata_locks"] = processor_metadata_locks(args.model_snapshot)
        payload["source_locks"] = source_locks(repo)
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            revision=args.model_revision,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        observations = _synthetic_observations()
        actions = (
            R2RCEAction.MOVE_FORWARD,
            R2RCEAction.TURN_LEFT,
            R2RCEAction.TURN_RIGHT,
        )
        modes: dict[str, Any] = {}
        for mode in HistoryMode:
            window = build_history_window(
                observations,
                actions,
                contract=HISTORY_CONTRACTS[mode],
            )
            batch = build_native_qwen_sft_batch(
                processor,
                window,
                instruction="Walk past the sofa, turn left, and stop near the doorway.",
                target_action=R2RCEAction.TURN_LEFT,
            )
            labels = batch.model_inputs["labels"]
            supervised_positions = (labels[0] != -100).nonzero(as_tuple=False).reshape(-1)
            if supervised_positions.numel() != 1:
                raise RuntimeError("Processor preflight must supervise exactly one action token.")
            observed_position = int(supervised_positions[0].item())
            if observed_position != batch.record["supervised_token_position"]:
                raise RuntimeError("Processor preflight label and audit position disagree.")
            if int(labels[0, observed_position].item()) != batch.record["supervised_token_id"]:
                raise RuntimeError("Processor preflight label and audit token ID disagree.")
            model_shapes = {key: list(value.shape) for key, value in batch.model_inputs.items()}
            modes[mode.value] = {
                **batch.record,
                "model_input_shapes": model_shapes,
                "optional_model_inputs": {
                    "mm_token_type_ids_present": "mm_token_type_ids" in batch.model_inputs,
                },
            }

        if modes["H0"]["image_count"] != 1 or modes["H1"]["image_count"] != 1:
            raise ValueError("H0 and H1 must each render exactly one native Qwen image.")
        if modes["H4"]["image_count"] != 4:
            raise ValueError("H4 must render exactly four native Qwen images.")
        if modes["H0"]["visual_tokens_per_image"] != modes["H1"]["visual_tokens_per_image"]:
            raise ValueError("H0 and H1 must have identical current-image visual tokenization.")

        payload.update(
            {
                "status": "passed",
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "processor_class": type(processor).__name__,
                "tokenizer_class": type(processor.tokenizer).__name__,
                "history_modes": modes,
            }
        )
        _write_json(output, payload)
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
        _write_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
