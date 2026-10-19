"""Image/depth preprocessing utilities for SpatialVLM data pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from spatialvlm.utils.camera import CameraIntrinsics


@dataclass(frozen=True)
class EncoderPreprocessConfig:
    """Deliberate external-encoder preprocessing choices for the C1-C4 path."""

    siglip_max_num_patches: int = 576
    dinov2_height_width: tuple[int, int] = (518, 518)

    def __post_init__(self) -> None:
        if self.siglip_max_num_patches <= 0:
            raise ValueError("siglip_max_num_patches must be positive.")
        if any(value <= 0 for value in self.dinov2_height_width):
            raise ValueError("DINOv2 output dimensions must be positive.")


def _to_rgb_hwc_uint8(rgb: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu()
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = rgb.permute(1, 2, 0)
        rgb = rgb.numpy()
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"RGB must be HWC or CHW with three channels, got {array.shape}.")
    if array.dtype != np.uint8:
        if not np.isfinite(array).all():
            raise ValueError("RGB contains non-finite values.")
        maximum = float(array.max()) if array.size else 0.0
        minimum = float(array.min()) if array.size else 0.0
        if 0.0 <= minimum and maximum <= 1.0:
            array = np.rint(array * 255.0)
        if float(array.min()) < 0.0 or float(array.max()) > 255.0:
            raise ValueError("RGB values must lie in [0,1] or [0,255].")
        array = array.astype(np.uint8)
    return np.ascontiguousarray(array)


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def preprocess_encoder_rgb(
    rgb: np.ndarray | torch.Tensor,
    *,
    siglip_processor: Any,
    dinov2_processor: Any,
    config: EncoderPreprocessConfig | None = None,
) -> dict[str, Any]:
    """Run checkpoint-native SigLIP2 and DINOv2 processors and validate their contracts."""
    config = config or EncoderPreprocessConfig()
    rgb_uint8 = _to_rgb_hwc_uint8(rgb)

    siglip = siglip_processor(
        images=rgb_uint8,
        return_tensors="pt",
        max_num_patches=config.siglip_max_num_patches,
    )
    required_siglip = {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
    missing = required_siglip - set(siglip)
    if missing:
        raise ValueError(f"SigLIP2 processor omitted required fields: {sorted(missing)}.")
    siglip_pixels = siglip["pixel_values"]
    siglip_mask = siglip["pixel_attention_mask"]
    siglip_shapes = siglip["spatial_shapes"]
    if siglip_pixels.ndim != 3 or siglip_pixels.shape[:2] != (
        1,
        config.siglip_max_num_patches,
    ):
        raise ValueError(
            "SigLIP2 pixel_values must be [1,max_num_patches,patch_dim], got "
            f"{tuple(siglip_pixels.shape)}."
        )
    if siglip_mask.shape != siglip_pixels.shape[:2] or siglip_shapes.shape != (1, 2):
        raise ValueError("SigLIP2 attention-mask or spatial-shape contract is inconsistent.")
    valid_patches = int(siglip_mask.to(dtype=torch.int64).sum().item())
    shape_product = int(siglip_shapes[0, 0].item() * siglip_shapes[0, 1].item())
    if valid_patches != shape_product:
        raise ValueError(
            f"SigLIP2 valid-patch count {valid_patches} != spatial-shape product {shape_product}."
        )
    if valid_patches != config.siglip_max_num_patches:
        raise ValueError(
            "The current SVA contract requires an unpadded square SigLIP2 grid: "
            f"valid={valid_patches}, configured={config.siglip_max_num_patches}."
        )

    height, width = config.dinov2_height_width
    dinov2 = dinov2_processor(
        images=rgb_uint8,
        return_tensors="pt",
        size={"height": height, "width": width},
        do_center_crop=False,
    )
    if "pixel_values" not in dinov2:
        raise ValueError("DINOv2 processor omitted pixel_values.")
    dinov2_pixels = dinov2["pixel_values"]
    if dinov2_pixels.shape != (1, 3, height, width):
        raise ValueError(
            f"DINOv2 pixel_values must be {(1, 3, height, width)}, "
            f"got {tuple(dinov2_pixels.shape)}."
        )

    return {
        "siglip_pixel_values": siglip_pixels,
        "siglip_pixel_attention_mask": siglip_mask,
        "siglip_spatial_shapes": siglip_shapes,
        "dinov2_pixel_values": dinov2_pixels,
        "manifest": {
            "contract_version": 1,
            "config": asdict(config),
            "raw_rgb": {
                "shape": list(rgb_uint8.shape),
                "dtype": str(rgb_uint8.dtype),
                "minimum": int(rgb_uint8.min()),
                "maximum": int(rgb_uint8.max()),
            },
            "siglip_valid_patches": valid_patches,
            "siglip_spatial_shapes": siglip_shapes.detach().cpu().tolist(),
            "siglip_pixel_values": _tensor_stats(siglip_pixels),
            "dinov2_pixel_values": _tensor_stats(dinov2_pixels),
        },
    }


def preprocess_metric_depth(
    depth_meters: np.ndarray | torch.Tensor,
    *,
    intrinsics: CameraIntrinsics,
    target_height_width: tuple[int, int] = (518, 518),
) -> tuple[torch.Tensor, CameraIntrinsics, dict[str, Any]]:
    """Resize metric depth and keep its pinhole calibration consistent."""
    depth = torch.as_tensor(depth_meters, dtype=torch.float32)
    if depth.ndim != 2:
        raise ValueError(f"Metric depth must be [H,W], got {tuple(depth.shape)}.")
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("Metric-depth shape does not match its source intrinsics.")
    if not torch.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("Metric depth must contain finite, non-negative values.")
    target_height, target_width = target_height_width
    resized = functional.interpolate(
        depth[None, None], size=(target_height, target_width), mode="nearest"
    )[0, 0]
    resized_intrinsics = intrinsics.rescaled(width=target_width, height=target_height)
    manifest = {
        "contract_version": 1,
        "resize_mode": "nearest",
        "coordinate_transform": "align_corners_false_half_pixel",
        "source_intrinsics": asdict(intrinsics),
        "resized_intrinsics": asdict(resized_intrinsics),
        "depth_meters": _tensor_stats(resized),
    }
    return resized, resized_intrinsics, manifest


def to_float_rgb01(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor to float in [0,1].

    Supports:
    - uint8 [0,255]
    - float [0,255] or [0,1]
    """
    if rgb.dtype == torch.uint8:
        return rgb.float() / 255.0
    out = rgb.float()
    if out.numel() > 0 and out.max() > 1.0:
        out = out / 255.0
    return out


def resize_rgb_bchw(
    rgb: torch.Tensor,
    size: tuple[int, int] = (518, 518),
) -> torch.Tensor:
    """Resize RGB tensor [B,3,H,W] with bilinear interpolation."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"RGB must be [B,3,H,W], got {tuple(rgb.shape)}.")
    return functional.interpolate(
        rgb,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


def resize_depth_bhw(
    depth: torch.Tensor,
    size: tuple[int, int] = (518, 518),
) -> torch.Tensor:
    """Resize depth tensor [B,H,W] using nearest-neighbor interpolation."""
    if depth.ndim != 3:
        raise ValueError(f"Depth must be [B,H,W], got {tuple(depth.shape)}.")
    depth_4d = depth.unsqueeze(1)  # [B,1,H,W]
    out = functional.interpolate(depth_4d, size=size, mode="nearest")
    return out[:, 0]  # [B,H,W]


def normalize_depth_bhw(
    depth: torch.Tensor,
    max_depth: float | None = None,
    percentile: float = 99.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize depth [B,H,W] into [0,1], robust to zeros/NaNs."""
    if depth.ndim != 3:
        raise ValueError(f"Depth must be [B,H,W], got {tuple(depth.shape)}.")

    d = depth.float().clone()
    d[~torch.isfinite(d)] = 0.0
    d = torch.clamp(d, min=0.0)

    if max_depth is not None:
        denom = max(max_depth, eps)
        return torch.clamp(d / denom, 0.0, 1.0)

    bsz = d.shape[0]
    out = torch.zeros_like(d)
    for i in range(bsz):
        flat = d[i].reshape(-1)
        valid = flat[flat > 0]
        if valid.numel() == 0:
            continue
        scale = torch.quantile(valid, q=percentile / 100.0).item()
        scale = max(scale, eps)
        out[i] = torch.clamp(d[i] / scale, 0.0, 1.0)
    return out


def preprocess_rgb_depth(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    size: tuple[int, int] = (518, 518),
    depth_percentile: float = 99.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess RGB + depth into model-ready tensors.

    Parameters
    ----------
    rgb : Tensor[B, 3, H, W]
    depth : Tensor[B, H, W]
    """
    rgb_float = to_float_rgb01(rgb)
    rgb_resized = resize_rgb_bchw(rgb_float, size=size)
    depth_resized = resize_depth_bhw(depth, size=size)
    depth_norm = normalize_depth_bhw(depth_resized, percentile=depth_percentile)
    return rgb_resized, depth_norm
