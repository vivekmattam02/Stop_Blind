"""Tests for the scale-corrected 3D rotary encoding."""

from __future__ import annotations

import math

import pytest
import torch

from spatialvlm.geometry.gridcell_rope3d import IcosahedralRoPE3D
from spatialvlm.geometry.gridcell_rope3d_v2 import IcosahedralRoPE3Dv2


def test_wavelengths_span_the_scene_not_a_hand_width() -> None:
    r = IcosahedralRoPE3Dv2()
    short, long = r.wavelengths_m
    assert math.isclose(long, 12.0, rel_tol=1e-6)
    assert 0.5 < short < 0.7
    assert long >= 10.0, "longest wavelength must cover a Matterport scene"


def test_v1_pathology_is_present_and_v2_fixes_it() -> None:
    """v1 scores a 5 m separation as more similar than one 0.25 m forward step."""
    old = IcosahedralRoPE3D()
    f = old.freqs
    k = lambda d: float(torch.cos(torch.tensor([d]) * f).mean())  # noqa: E731
    assert k(5.0) > k(0.25), "v1 pathology should still be reproducible"

    new = IcosahedralRoPE3Dv2()
    assert float(new.kernel(torch.tensor([0.25]))[0]) > float(new.kernel(torch.tensor([5.0]))[0])


def test_near_range_inversions_improve_over_v1() -> None:
    d = IcosahedralRoPE3Dv2().kernel_diagnostics()
    assert d["prefers_near_over_far"] is True
    assert d["near_range_rank_inversions"] < 400, d


def test_kernel_decreases_from_zero() -> None:
    r = IcosahedralRoPE3Dv2()
    v = r.kernel(torch.tensor([0.0, 0.1, 0.5, 1.0]))
    assert float(v[0]) > float(v[1]) > float(v[2]) > float(v[3])


def test_output_width_fits_the_rotary_budget() -> None:
    r = IcosahedralRoPE3Dv2()
    assert r.output_dim == 120 <= r.MAX_ROTARY_DIMS
    out = r(torch.zeros(2, 7, 3))
    assert out.shape == (2, 7, 120)


def test_rejects_a_design_that_would_overflow_the_rotary_width() -> None:
    with pytest.raises(ValueError, match="exceeds the 128-dimension rotary budget"):
        IcosahedralRoPE3Dv2(n_freqs=11)


def test_rejects_nonpositive_scene_extent() -> None:
    with pytest.raises(ValueError, match="scene_extent_m must be positive"):
        IcosahedralRoPE3Dv2(scene_extent_m=0.0)


def test_encoding_is_deterministic_and_parameter_free() -> None:
    r = IcosahedralRoPE3Dv2()
    assert list(r.parameters()) == []
    p = torch.randn(1, 5, 3)
    assert torch.equal(r(p), r(p))


def test_translation_changes_the_encoding() -> None:
    """A positional encoding that ignores position is the failure this replaces."""
    r = IcosahedralRoPE3Dv2()
    a = r(torch.zeros(1, 1, 3))
    b = r(torch.full((1, 1, 3), 0.25))
    assert not torch.allclose(a, b, atol=1e-4)


def test_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"positions must be \[B, N, 3\]"):
        IcosahedralRoPE3Dv2()(torch.zeros(4, 3))
