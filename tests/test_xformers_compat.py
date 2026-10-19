"""Tests for the optional xFormers boundary used by GATr."""

from __future__ import annotations

import pytest

from spatialvlm.geometry.xformers_compat import ensure_xformers_import_surface


def test_xformers_import_surface_is_always_available():
    has_real_xformers = ensure_xformers_import_surface()

    from xformers.ops import AttentionBias, memory_efficient_attention

    assert isinstance(AttentionBias, type)
    assert callable(memory_efficient_attention)

    if not has_real_xformers:
        with pytest.raises(RuntimeError, match="xFormers attention was requested"):
            memory_efficient_attention(None, None, None)
