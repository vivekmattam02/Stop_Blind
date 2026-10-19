#!/usr/bin/env python3
"""Validate real checkpoint processors and chat rendering before loading model weights."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoImageProcessor, AutoTokenizer

from spatialvlm.config.model import SpatialVLMConfig
from spatialvlm.data.preprocessing import preprocess_encoder_rgb
from spatialvlm.data.tokenization import SpatialTokenizerConfig, build_input_ids


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": 1,
        "gate": "spark_real_batch_checkpoint_processor_preflight_v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    try:
        config = SpatialVLMConfig()
        siglip = AutoImageProcessor.from_pretrained(config.encoder.siglip_model_id)
        dino = AutoImageProcessor.from_pretrained(config.encoder.dinov2_model_id)
        tokenizer = AutoTokenizer.from_pretrained(config.backbone.model_id, trust_remote_code=True)
        image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
        processed = preprocess_encoder_rgb(
            image,
            siglip_processor=siglip,
            dinov2_processor=dino,
        )
        tokens = build_input_ids(
            tokenizer,
            instruction="Walk forward and stop near the doorway.",
            target="Action: MOVE_FORWARD",
            config=SpatialTokenizerConfig(
                num_spatial_tokens=config.fusion.sva_num_queries,
                max_length=2048,
            ),
        )
        payload.update(
            {
                "status": "passed",
                "siglip_processor_class": type(siglip).__name__,
                "dinov2_processor_class": type(dino).__name__,
                "siglip_pixel_values_shape": list(processed["siglip_pixel_values"].shape),
                "siglip_attention_mask_shape": list(processed["siglip_pixel_attention_mask"].shape),
                "siglip_spatial_shapes": processed["siglip_spatial_shapes"].tolist(),
                "dinov2_pixel_values_shape": list(processed["dinov2_pixel_values"].shape),
                "token_sequence_length": int(tokens["input_ids"].numel()),
                "spatial_start_idx": tokens["spatial_start_idx"],
                "target_start_idx": tokens["target_start_idx"],
                "supervised_token_count": tokens["supervised_token_count"],
                "conversation_renderer_version": tokens["conversation_renderer_version"],
            }
        )
        _write(output, payload)
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
        _write(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
