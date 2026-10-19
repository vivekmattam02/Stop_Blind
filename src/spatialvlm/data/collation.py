"""Batch collation: convert cached frames into model-ready tensors.

The collator takes a list of frame dicts (from ``CachedFrameDataset``) and
produces a single dict consumable by ``SpatialVLM.forward()``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from spatialvlm.data.preprocessing import preprocess_encoder_rgb, preprocess_metric_depth
from spatialvlm.data.tokenization import SpatialTokenizerConfig, build_input_ids
from spatialvlm.utils.camera import CameraIntrinsics


class SpatialVLMCollator:
    """Collates cached frames into a model-ready batch.

    Parameters
    ----------
    tokenizer
        HuggingFace tokenizer (e.g. ``AutoTokenizer.from_pretrained(...)``).
    stage : str
        ``"prealign"`` or ``"sft"``.  Controls whether ``target`` is used.
    tok_config : SpatialTokenizerConfig | None
        Override tokenization defaults.
    siglip_size : tuple[int, int]
        Resize for SigLIP input.
    dinov2_size : tuple[int, int]
        Resize for DINOv2 input.
    """

    def __init__(
        self,
        tokenizer: Any,
        siglip_processor: Any,
        dinov2_processor: Any,
        stage: str = "sft",
        tok_config: SpatialTokenizerConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.siglip_processor = siglip_processor
        self.dinov2_processor = dinov2_processor
        self.stage = stage
        self.tok_config = tok_config or SpatialTokenizerConfig()

    def __call__(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        siglip_pixels_list: list[torch.Tensor] = []
        siglip_mask_list: list[torch.Tensor] = []
        siglip_shapes_list: list[torch.Tensor] = []
        dinov2_pixels_list: list[torch.Tensor] = []
        depth_list: list[torch.Tensor] = []
        input_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        attn_mask_list: list[torch.Tensor] = []
        spatial_start_indices: list[int] = []
        batch_intrinsics: CameraIntrinsics | None = None

        for frame in frames:
            rgb = frame["rgb"]  # [3, H, W] float
            depth = frame["depth"]  # [H, W] float, metric metres

            processed_rgb = preprocess_encoder_rgb(
                rgb,
                siglip_processor=self.siglip_processor,
                dinov2_processor=self.dinov2_processor,
            )
            siglip_pixels_list.append(processed_rgb["siglip_pixel_values"][0])
            siglip_mask_list.append(processed_rgb["siglip_pixel_attention_mask"][0])
            siglip_shapes_list.append(processed_rgb["siglip_spatial_shapes"][0])
            dinov2_pixels_list.append(processed_rgb["dinov2_pixel_values"][0])

            intr = frame.get("intrinsics")
            if intr is not None and isinstance(intr, dict):
                source_intrinsics = CameraIntrinsics(**intr)
            else:
                source_intrinsics = CameraIntrinsics.from_hfov(
                    width=int(depth.shape[1]),
                    height=int(depth.shape[0]),
                    hfov_degrees=90.0,
                )
            processed_depth, resized_intrinsics, _ = preprocess_metric_depth(
                depth, intrinsics=source_intrinsics, target_height_width=(518, 518)
            )
            if batch_intrinsics is None:
                batch_intrinsics = resized_intrinsics
            elif resized_intrinsics != batch_intrinsics:
                raise ValueError("All frames in a batch must share resized camera intrinsics.")
            depth_list.append(processed_depth)

            # Tokenize
            instruction = frame.get("instruction", "Navigate the environment.")
            target = frame.get("target") if self.stage == "sft" else None
            tok_out = build_input_ids(
                self.tokenizer,
                instruction=instruction,
                target=target,
                config=self.tok_config,
            )
            input_ids_list.append(tok_out["input_ids"])
            labels_list.append(tok_out["labels"])
            attn_mask_list.append(tok_out["attention_mask"])

            spatial_start = tok_out["spatial_start_idx"]
            if not isinstance(spatial_start, int):
                raise TypeError("Tokenization `spatial_start_idx` must be an integer.")
            spatial_start_indices.append(spatial_start)

        # Pad sequences to uniform length
        max_len = max(ids.shape[0] for ids in input_ids_list)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        for i in range(len(input_ids_list)):
            pad_len = max_len - input_ids_list[i].shape[0]
            if pad_len > 0:
                input_ids_list[i] = functional.pad(input_ids_list[i], (0, pad_len), value=pad_id)
                labels_list[i] = functional.pad(
                    labels_list[i], (0, pad_len), value=self.tok_config.ignore_index
                )
                attn_mask_list[i] = functional.pad(attn_mask_list[i], (0, pad_len), value=0)

        if batch_intrinsics is None:
            raise ValueError("Cannot collate an empty frame batch.")

        return {
            "siglip_pixels": torch.stack(siglip_pixels_list),  # [B, 576, patch_dim]
            "siglip_attention_mask": torch.stack(siglip_mask_list),  # [B, 576]
            "siglip_spatial_shapes": torch.stack(siglip_shapes_list),  # [B, 2]
            "dinov2_pixels": torch.stack(dinov2_pixels_list),  # [B, 3, 518, 518]
            "depth": torch.stack(depth_list),  # [B, 518, 518]
            "intrinsics": batch_intrinsics,
            "input_ids": torch.stack(input_ids_list),  # [B, seq_len]
            "labels": torch.stack(labels_list),  # [B, seq_len]
            "attention_mask": torch.stack(attn_mask_list),  # [B, seq_len]
            # Instructions have different token lengths, so every sample needs
            # its own spatial-block start rather than reusing sample zero's index.
            "spatial_start_idx": torch.tensor(spatial_start_indices, dtype=torch.long),
        }
