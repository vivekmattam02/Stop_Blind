"""Tests for the v2 geometry stack: masked aggregation, metric/direction readout, splice, hook.

The GATr-dependent tests skip cleanly where the dependency is absent (host); the launch gate runs
this file inside the training container, where they must pass.
"""

from __future__ import annotations

import pytest
import torch

from spatialvlm.geometry.geometry_branch import (
    UNIT_SCALE_M,
    aggregate_patches_percentile_masked,
)

try:  # container-only dependency
    from spatialvlm.geometry.geometry_branch import GeometryBranch

    _branch = GeometryBranch(hidden_size=64, gatr_blocks=2, depth_hw=(64, 64), patch_size_px=32)
    HAS_GATR = True
except ImportError:
    HAS_GATR = False


def _scene(depth_value: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    depth = torch.full((1, 64, 64), depth_value)
    points = torch.zeros(1, 64, 64, 3)
    points[..., 2] = depth
    return points, depth


class TestMaskedAggregation:
    def test_valid_patch_selects_valid_pixel_at_percentile(self) -> None:
        points, depth = _scene()
        pts, val = aggregate_patches_percentile_masked(points, depth, patch_size=32)
        assert pts.shape == (1, 4, 3) and val.shape == (1, 4)
        assert torch.allclose(val, torch.ones_like(val))
        assert torch.allclose(pts[..., 2], torch.full_like(pts[..., 2], 2.0))

    def test_dead_patch_gets_origin_and_zero_validity(self) -> None:
        points, depth = _scene()
        depth[0, :32, :32] = 0.0  # kill patch 0 entirely
        points[0, :32, :32] = 5.0  # poison its point map: must NOT leak through
        pts, val = aggregate_patches_percentile_masked(points, depth, patch_size=32)
        assert float(val[0, 0]) == 0.0
        assert torch.allclose(pts[0, 0], torch.zeros(3))
        assert float(val[0, 3]) == 1.0  # untouched patch unaffected

    def test_mostly_invalid_patch_still_picks_a_valid_pixel(self) -> None:
        """The v1 defect: a fixed 15%-of-all-pixels rank lands on an invalid pixel here."""
        points, depth = _scene()
        depth[0, :32, :32] = 0.0
        depth[0, 0, 0:3] = 1.5  # exactly three valid pixels in patch 0
        points[0, 0, 0:3, 2] = 1.5
        pts, val = aggregate_patches_percentile_masked(points, depth, patch_size=32)
        assert float(pts[0, 0, 2]) == pytest.approx(1.5)  # a valid pixel, not the origin
        assert 0.0 < float(val[0, 0]) < 0.01


@pytest.mark.skipif(not HAS_GATR, reason="gatr not installed on this host")
class TestGeometryBranchReadout:
    def test_metric_scale_is_not_divided_away(self) -> None:
        """A room and the same room at 2x scale must produce different features (v1: identical)."""
        torch.manual_seed(0)
        coords = torch.rand(1, _branch.n_tokens, 3) * 4.0 + 0.5
        validity = torch.ones(1, _branch.n_tokens)
        with torch.no_grad():
            a = _branch(coords, validity)
            b = _branch(coords * 2.0, validity)
        assert (a - b).abs().max() > 1e-4

    def test_readout_is_direction_sensitive(self) -> None:
        """Mirroring the scene left-right must change the features (v1 invariants: blind)."""
        torch.manual_seed(0)
        coords = torch.rand(1, _branch.n_tokens, 3) * 4.0 + 0.5
        mirrored = coords.clone()
        mirrored[..., 0] = -mirrored[..., 0]
        validity = torch.ones(1, _branch.n_tokens)
        with torch.no_grad():
            a = _branch(coords, validity)
            b = _branch(mirrored, validity)
        assert (a - b).abs().max() > 1e-4

    def test_dead_tokens_receive_only_the_frame_embedding(self) -> None:
        coords = torch.rand(1, _branch.n_tokens, 3) + 0.5
        validity = torch.zeros(1, _branch.n_tokens)
        with torch.no_grad():
            out = _branch(coords, validity)
            zero_feats = _branch.projector(torch.zeros(1, _branch.n_tokens, _branch.readout_dim))
        frame_idx = torch.arange(_branch.n_frames).repeat_interleave(_branch.tokens_per_frame)
        expected = zero_feats + _branch.frame_embedding(frame_idx).unsqueeze(0)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_points_from_depth_units_and_shapes(self) -> None:
        depth = torch.full((4, 64, 64), 3.0)
        coords, validity = _branch.points_from_depth(depth)
        assert coords.shape == (1, _branch.n_tokens, 3)
        assert float(validity.min()) == 1.0
        # Centre-ish patches look straight ahead: Z equals the metric depth.
        assert float(coords[..., 2].max()) == pytest.approx(3.0, abs=1e-5)
        assert UNIT_SCALE_M == 10.0


class TestRopePatchV2Math:
    def test_v2_phase_layout_pads_to_head_dim(self) -> None:
        from spatialvlm.geometry.gridcell_rope3d_v2 import IcosahedralRoPE3Dv2

        enc = IcosahedralRoPE3Dv2()
        angles = enc(torch.zeros(1, 5, 3))  # origin
        assert angles.shape == (1, 5, 120)
        pairs = angles.view(1, 5, 60, 2)
        assert torch.allclose(pairs[..., 0], torch.zeros_like(pairs[..., 0]))  # sin(0)
        assert torch.allclose(pairs[..., 1], torch.ones_like(pairs[..., 1]))  # cos(0)

    def test_norm_matching_hook_matches_text_rms(self) -> None:
        from spatialvlm.backbone.rope_patch_v2 import _norm_matched_embedding_hook

        class Carrier(torch.nn.Module):
            pass

        module = Carrier()
        output = torch.randn(1, 10, 8) * 5.0
        mask = torch.zeros(1, 10, dtype=torch.bool)
        mask[0, 3:6] = True
        module._v2_embeds = torch.randn(1, 3, 8) * 100.0
        module._v2_mask = mask
        result = _norm_matched_embedding_hook(module, (), output)
        text_rms = result[~mask].pow(2).mean(-1).sqrt().mean()
        geo_rms = result[mask].pow(2).mean(-1).sqrt().mean()
        assert float((geo_rms - text_rms).abs() / text_rms) < 0.05
        assert module._v2_embeds is None  # one-shot

    def test_zero_embeds_stay_zero(self) -> None:
        from spatialvlm.backbone.rope_patch_v2 import _norm_matched_embedding_hook

        class Carrier(torch.nn.Module):
            pass

        module = Carrier()
        output = torch.randn(1, 6, 4)
        mask = torch.zeros(1, 6, dtype=torch.bool)
        mask[0, 2:4] = True
        module._v2_embeds = torch.zeros(1, 2, 4)
        module._v2_mask = mask
        result = _norm_matched_embedding_hook(module, (), output)
        assert torch.allclose(result[mask], torch.zeros_like(result[mask]))


class TestGeometrySplice:
    def _fake_batch(self):
        from spatialvlm.data.native_qwen import NativeQwenBatch

        ids = torch.arange(100, 120).unsqueeze(0)  # [1, 20]
        target_start = 14
        positions = (15, 16, 17, 18)
        inputs = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.randn(4, 8),
        }
        record = {
            "target_start_idx": target_start,
            "supervised_token_positions": list(positions),
            "supervised_token_ids": [int(ids[0, p]) for p in positions],
        }
        return NativeQwenBatch(model_inputs=inputs, record=record)

    def test_splice_shifts_and_verifies(self) -> None:
        from spatialvlm.data.native_qwen_chunk_geometry import splice_geometry_block

        out, positions, mask, record = splice_geometry_block(
            self._fake_batch(), n_tokens=6, filler_id=999
        )
        assert out["input_ids"].shape == (1, 26)
        assert positions == (21, 22, 23, 24)
        assert int(mask.sum()) == 6 and bool(mask[0, 14:20].all())
        assert (out["input_ids"][0, 14:20] == 999).all()
        assert int((out["labels"] != -100).sum()) == 4
        assert bool((out["attention_mask"] == 1).all())
        assert out["pixel_values"].shape == (4, 8)  # non-sequence fields untouched

    def test_splice_refuses_misaligned_supervision(self) -> None:
        from spatialvlm.data.native_qwen_chunk_geometry import splice_geometry_block

        batch = self._fake_batch()
        batch.record["supervised_token_ids"][2] = -1  # sabotage
        with pytest.raises(RuntimeError, match="misaligned"):
            splice_geometry_block(batch, n_tokens=6, filler_id=999)
