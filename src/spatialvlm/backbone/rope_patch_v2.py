"""RoPE patch v2 — scale-corrected geometry injection into Qwen3-VL. Additive module.

Differences from ``rope_patch.py`` (which is left untouched — frozen gates hash it):

- The positional code is ``IcosahedralRoPE3Dv2``: wavelengths 0.597–12.0 m instead of v1's
  0.061–0.628 m, 120 rotary dims → 60 real pairs + 4 identity pairs.
- Norm matching is *wired and stateless*: at the injection site the geometry rows are rescaled so
  their RMS matches the RMS of the same sample's non-geometry embedding rows. No EMA, no state,
  identical behaviour at train and deploy. This replaces the never-connected ``RMSNormMatching``.
- The runner stashes tensors directly on the resolved modules via :func:`stash_geometry` — no
  kwargs are threaded through PEFT/HF forwards, so a silent ``AttributeError`` no-op (the v1
  failure mode) cannot occur; resolution failures raise at setup instead.
- Stash attributes are namespaced ``_v2_*`` so v1 and v2 cannot interact.

During autoregressive decode the geometry rows live in the KV cache from prefill; the stash is
cleared after its first use, so decode steps are exact no-ops, as in v1.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from spatialvlm.geometry.gridcell_rope3d_v2 import IcosahedralRoPE3Dv2

ROPE_PATCH_V2_VERSION = 1


def resolve_language_rotary(model: Any) -> Any:
    """Locate the text-stack rotary-embedding module through any PEFT/HF wrapping."""
    hits = [
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ == "Qwen3VLTextRotaryEmbedding"
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one Qwen3VLTextRotaryEmbedding, found {[n for n, _ in hits]}."
        )
    return hits[0][1]


def resolve_embed_tokens(model: Any) -> Any:
    """Locate the word-embedding module (works through PEFT wrapping)."""
    embed = getattr(model, "get_input_embeddings", None)
    if callable(embed):
        module = embed()
        if module is not None:
            return module
    raise RuntimeError("Cannot resolve the input embedding module.")


def stash_geometry(
    rotary: Any,
    embed: Any,
    *,
    coords_m: torch.Tensor,
    spatial_mask: torch.Tensor,
    geometry_embeds: torch.Tensor,
) -> None:
    """Arm one forward pass with geometry tensors.

    Parameters
    ----------
    coords_m : Tensor[B, N, 3] — metric camera-space coordinates for the phase code.
    spatial_mask : Tensor[B, L] bool — True at the geometry token positions.
    geometry_embeds : Tensor[B, N, hidden] — fp32 branch output; cast at the boundary.
    """
    if spatial_mask.sum().item() != coords_m.shape[0] * coords_m.shape[1]:
        raise ValueError(
            f"spatial_mask marks {int(spatial_mask.sum())} positions but coords carry "
            f"{coords_m.shape[0] * coords_m.shape[1]} tokens."
        )
    rotary._v2_coords = coords_m
    rotary._v2_mask = spatial_mask
    embed._v2_embeds = geometry_embeds
    embed._v2_mask = spatial_mask


def _patched_rotary_forward(
    original_forward: Any, rotary_self: Any, x: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = original_forward(x, position_ids)  # [B, L, D] or [S, B, L, D] (mrope sections)
    coords = getattr(rotary_self, "_v2_coords", None)
    mask = getattr(rotary_self, "_v2_mask", None)
    if coords is None or mask is None or not bool(mask.any()):
        return cos, sin
    seq_dim = 1 if cos.ndim == 3 else 2
    if mask.shape[-1] != cos.shape[seq_dim]:
        # Decode step on a shorter sequence than the stashed mask: geometry already in KV cache.
        return cos, sin

    encoder: IcosahedralRoPE3Dv2 = rotary_self._v2_encoder.to(coords.device)
    angles = encoder(coords)  # [B, N, 120] interleaved (sin, cos) pairs
    b, n, dims = angles.shape
    pairs = dims // 2  # 60
    interleaved = angles.view(b, n, pairs, 2)
    sin_vals, cos_vals = interleaved[..., 0], interleaved[..., 1]  # [B, N, 60]

    head_dim = cos.shape[-1]
    pad = head_dim // 2 - pairs  # 4 identity pairs for head_dim 128
    if pad < 0:
        raise RuntimeError(f"RoPE3D-v2 needs {pairs} pairs but head_dim {head_dim} offers fewer.")
    if pad:
        sin_vals = torch.cat([sin_vals, torch.zeros(b, n, pad, device=coords.device)], dim=-1)
        cos_vals = torch.cat([cos_vals, torch.ones(b, n, pad, device=coords.device)], dim=-1)
    ico_cos = torch.cat([cos_vals, cos_vals], dim=-1).to(cos.dtype)  # [B, N, head_dim]
    ico_sin = torch.cat([sin_vals, sin_vals], dim=-1).to(sin.dtype)

    cos, sin = cos.clone(), sin.clone()
    if cos.ndim == 3:
        expanded = mask.unsqueeze(-1).expand_as(cos)  # [B, L, D]
        cos[expanded] = ico_cos.reshape(-1)
        sin[expanded] = ico_sin.reshape(-1)
    else:  # [S, B, L, D]: identical phases written into every mrope section
        sections = cos.shape[0]
        expanded = mask.unsqueeze(0).unsqueeze(-1).expand_as(cos)
        cos[expanded] = ico_cos.reshape(-1).repeat(sections)
        sin[expanded] = ico_sin.reshape(-1).repeat(sections)

    rotary_self._v2_coords = None
    rotary_self._v2_mask = None
    return cos, sin


def _norm_matched_embedding_hook(module: Any, args: tuple, output: torch.Tensor) -> torch.Tensor:
    """Replace geometry rows, rescaled to the sample's own text-row RMS. One-shot per forward."""
    embeds = getattr(module, "_v2_embeds", None)
    mask = getattr(module, "_v2_mask", None)
    if embeds is None or mask is None:
        return output
    if mask.shape != output.shape[:2]:
        return output  # decode step; geometry already injected at prefill

    flat = embeds.reshape(-1, output.shape[-1]).float()  # [K, D]
    # The scale is a detached statistic: geometry rows are rescaled to the text-row RMS, but the
    # matching itself is not a differentiable pathway (standard for norm matching; keeps the
    # branch gradient about content, not magnitude bookkeeping).
    text_rms = output[~mask].detach().float().pow(2).mean(dim=-1).sqrt().mean()
    geo_rms = flat.detach().pow(2).mean(dim=-1).sqrt().mean()
    scale = (text_rms / geo_rms).item() if geo_rms.item() > 1e-8 else 0.0
    output = output.clone()
    output[mask] = (flat * scale).to(output.dtype)

    module._v2_embeds = None
    module._v2_mask = None
    return output


def apply_rope_patch_v2(
    model: Any,
    scene_extent_m: float = IcosahedralRoPE3Dv2.DEFAULT_SCENE_EXTENT_M,
    n_freqs: int = IcosahedralRoPE3Dv2.DEFAULT_N_FREQS,
) -> tuple[Any, Any]:
    """Patch the rotary math and register the norm-matched injection hook.

    Returns the resolved ``(rotary, embed)`` modules; the runner stashes onto these directly.
    """
    rotary = resolve_language_rotary(model)
    embed = resolve_embed_tokens(model)

    rotary._v2_encoder = IcosahedralRoPE3Dv2(scene_extent_m=scene_extent_m, n_freqs=n_freqs)
    rotary._v2_coords = None
    rotary._v2_mask = None
    original = rotary.forward
    rotary.forward = functools.partial(_patched_rotary_forward, original, rotary)

    embed._v2_embeds = None
    embed._v2_mask = None
    embed.register_forward_hook(_norm_matched_embedding_hook)
    return rotary, embed
