"""Optional xFormers import surface for GATr's native PyTorch attention path.

The vendored GATr package imports two names from ``xformers.ops`` even when its
default execution path uses ``torch.nn.functional.scaled_dot_product_attention``.
DGX Spark is ARM64, where a compatible xFormers wheel may not be available.

This module supplies only the import surface GATr needs when xFormers is absent.
It does not emulate xFormers kernels: asking for an xFormers-specific mask or
forcing the xFormers backend still fails with a clear error.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


class _UnavailableAttentionBias:
    """Placeholder type used only when the real xFormers package is absent."""


def _unavailable_memory_efficient_attention(*args, **kwargs):
    _ = args, kwargs
    raise RuntimeError(
        "xFormers attention was requested, but xFormers is not installed. "
        "Use GATr's default PyTorch SDPA path or install a compatible xFormers build."
    )


def ensure_xformers_import_surface() -> bool:
    """Make GATr importable without xFormers and report whether xFormers is real.

    Returns
    -------
    bool
        ``True`` when the real ``xformers.ops`` imported successfully. ``False``
        when a minimal import-only fallback was registered.
    """
    existing_ops = sys.modules.get("xformers.ops")
    if getattr(existing_ops, "_spatialvlm_import_fallback", False):
        return False

    try:
        importlib.import_module("xformers.ops")
        return True
    except ModuleNotFoundError as exc:
        if exc.name is not None and not exc.name.startswith("xformers"):
            raise

    xformers_module = sys.modules.get("xformers")
    if xformers_module is None:
        xformers_module = ModuleType("xformers")

    ops_module = ModuleType("xformers.ops")
    setattr(ops_module, "AttentionBias", _UnavailableAttentionBias)
    setattr(
        ops_module,
        "memory_efficient_attention",
        _unavailable_memory_efficient_attention,
    )
    setattr(ops_module, "_spatialvlm_import_fallback", True)
    setattr(xformers_module, "ops", ops_module)

    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module
    return False
