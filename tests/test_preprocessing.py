"""Tests for RGB/depth preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spatialvlm.data.preprocessing import (
    EncoderPreprocessConfig,
    normalize_depth_bhw,
    preprocess_encoder_rgb,
    preprocess_metric_depth,
    preprocess_rgb_depth,
    resize_depth_bhw,
    resize_rgb_bchw,
    to_float_rgb01,
)
from spatialvlm.utils.camera import CameraIntrinsics


def test_to_float_rgb01_uint8():
    rgb = torch.full((1, 3, 4, 4), fill_value=255, dtype=torch.uint8)
    out = to_float_rgb01(rgb)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.ones_like(out))


def test_to_float_rgb01_float255():
    rgb = torch.full((1, 3, 4, 4), fill_value=128.0, dtype=torch.float32)
    out = to_float_rgb01(rgb)
    expected = torch.full_like(out, 128.0 / 255.0)
    assert torch.allclose(out, expected)


def test_resize_rgb_bchw_shape():
    rgb = torch.randn(2, 3, 32, 40)
    out = resize_rgb_bchw(rgb, size=(518, 518))
    assert out.shape == (2, 3, 518, 518)


def test_resize_rgb_bchw_rejects_invalid_shape():
    rgb = torch.randn(2, 32, 40)
    with pytest.raises(ValueError, match="RGB must be"):
        resize_rgb_bchw(rgb)


def test_resize_depth_bhw_shape():
    depth = torch.randn(2, 30, 50)
    out = resize_depth_bhw(depth, size=(518, 518))
    assert out.shape == (2, 518, 518)


def test_normalize_depth_handles_nan_and_zero():
    depth = torch.tensor([[[float("nan"), 0.0], [2.0, 4.0]]], dtype=torch.float32)
    out = normalize_depth_bhw(depth, percentile=100.0)
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out[0, 0, 0] == 0.0  # NaN replaced
    assert out[0, 0, 1] == 0.0  # zero depth stays zero


def test_normalize_depth_with_max_depth():
    depth = torch.tensor([[[0.0, 2.0], [5.0, 20.0]]], dtype=torch.float32)
    out = normalize_depth_bhw(depth, max_depth=10.0)
    assert torch.allclose(out[0, 0, 1], torch.tensor(0.2))
    assert torch.allclose(out[0, 1, 0], torch.tensor(0.5))
    assert out[0, 1, 1] == 1.0  # clipped


def test_preprocess_rgb_depth_end_to_end():
    rgb = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
    depth = torch.rand(1, 32, 32) * 5.0
    rgb_out, depth_out = preprocess_rgb_depth(rgb, depth, size=(518, 518), depth_percentile=99.0)
    assert rgb_out.shape == (1, 3, 518, 518)
    assert depth_out.shape == (1, 518, 518)
    assert rgb_out.dtype == torch.float32
    assert depth_out.dtype == torch.float32
    assert rgb_out.min() >= 0.0
    assert rgb_out.max() <= 1.0
    assert depth_out.min() >= 0.0
    assert depth_out.max() <= 1.0


class _SiglipProcessor:
    def __call__(self, *, images, return_tensors, max_num_patches):
        assert images.dtype == np.uint8
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.linspace(
                -1, 1, max_num_patches * 12, dtype=torch.float32
            ).reshape(1, max_num_patches, 12),
            "pixel_attention_mask": torch.ones(1, max_num_patches, dtype=torch.long),
            "spatial_shapes": torch.tensor([[2, 2]], dtype=torch.long),
        }


class _DinoProcessor:
    def __call__(self, *, images, return_tensors, size, do_center_crop):
        assert images.dtype == np.uint8
        assert return_tensors == "pt"
        assert do_center_crop is False
        return {"pixel_values": torch.zeros(1, 3, size["height"], size["width"])}


def test_checkpoint_native_encoder_processor_contract() -> None:
    rgb = torch.rand(3, 8, 8)
    output = preprocess_encoder_rgb(
        rgb,
        siglip_processor=_SiglipProcessor(),
        dinov2_processor=_DinoProcessor(),
        config=EncoderPreprocessConfig(
            siglip_max_num_patches=4,
            dinov2_height_width=(14, 14),
        ),
    )

    assert output["siglip_pixel_values"].shape == (1, 4, 12)
    assert output["siglip_pixel_attention_mask"].shape == (1, 4)
    assert output["siglip_spatial_shapes"].tolist() == [[2, 2]]
    assert output["dinov2_pixel_values"].shape == (1, 3, 14, 14)
    assert output["manifest"]["siglip_valid_patches"] == 4


def test_metric_depth_resize_updates_intrinsics() -> None:
    source = CameraIntrinsics.from_hfov(width=4, height=4, hfov_degrees=90.0)
    depth, intrinsics, manifest = preprocess_metric_depth(
        torch.ones(4, 4), intrinsics=source, target_height_width=(8, 8)
    )

    assert depth.shape == (8, 8)
    assert torch.all(depth == 1.0)
    assert intrinsics.fx == pytest.approx(source.fx * 2.0)
    assert intrinsics.cx == pytest.approx(3.5)
    assert manifest["coordinate_transform"] == "align_corners_false_half_pixel"


def test_siglip_padding_is_rejected_for_fixed_sva_contract() -> None:
    class PaddedSiglip(_SiglipProcessor):
        def __call__(self, **kwargs):
            output = super().__call__(**kwargs)
            output["pixel_attention_mask"][0, -1] = 0
            return output

    with pytest.raises(ValueError, match="valid-patch count"):
        preprocess_encoder_rgb(
            torch.rand(3, 8, 8),
            siglip_processor=PaddedSiglip(),
            dinov2_processor=_DinoProcessor(),
            config=EncoderPreprocessConfig(
                siglip_max_num_patches=4,
                dinov2_height_width=(14, 14),
            ),
        )
