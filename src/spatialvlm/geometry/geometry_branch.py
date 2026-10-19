"""Geometry branch v2 — metric-preserving, direction-aware, validity-masked. Additive module.

This is the first version of the geometry stack intended to run inside a trained model. It repairs,
in new code, the three input-side defects confirmed against the original modules (which stay
untouched for gate-hash reconstructability):

1. **Metric scale.** `gatr_wrapper.py` divides every sample's coordinates by that sample's max
   radius, so a 12 m room and a 1.2 m closet become identical inputs. Here coordinates are scaled
   by a *fixed* unit constant — 10 m, the Habitat depth clip — identical for every sample, which is
   a change of units rather than a destruction of information, and the true metric radius is passed
   as a scalar channel.

2. **Direction.** The v1 readout was scalars ⊕ multivector norms, both rotation-invariant by
   construction, so "free space to the left" and "free space to the right" produced identical
   features. The input frame is egocentric (camera space), where absolute direction is meaningful,
   so this readout adds the `extract_point` coordinates of every output multivector channel.

3. **Validity.** Zero depth backprojects to the camera origin, and the original percentile
   aggregation could select an invalid pixel when a patch was mostly invalid. Here ranking is over
   valid pixels only with a per-patch percentile index, a patch with no valid pixel gets coords
   (0,0,0) and validity 0, and invalid tokens have their features zeroed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spatialvlm.encoders.projector import MLPProjector
from spatialvlm.geometry.xformers_compat import ensure_xformers_import_surface
from spatialvlm.utils.camera import CameraIntrinsics, backproject_pixel, make_pixel_grid

GEOMETRY_BRANCH_VERSION = 1
UNIT_SCALE_M = 10.0  # Habitat depth clip; a documented unit constant, not a per-sample statistic.


def aggregate_patches_percentile_masked(
    point_map: torch.Tensor,
    depth: torch.Tensor,
    patch_size: int,
    percentile: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Percentile-depth patch aggregation that ranks only valid pixels.

    Parameters
    ----------
    point_map : Tensor[B, H, W, 3]
        Backprojected camera-space points in metres.
    depth : Tensor[B, H, W]
        Metric depth; zero or NaN marks an invalid pixel.
    patch_size : int
        Square patch edge in pixels; H and W must divide evenly.
    percentile : float
        Rank fraction among the *valid* pixels of each patch.

    Returns
    -------
    patch_points : Tensor[B, n_patches, 3]
        Selected point per patch; (0, 0, 0) where the patch has no valid pixel.
    validity : Tensor[B, n_patches]
        Valid-pixel fraction per patch; exactly 0.0 marks a dead patch.
    """
    b, h, w, _ = point_map.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"{h}x{w} does not tile with patch_size={patch_size}.")
    ph, pw = h // patch_size, w // patch_size
    n, px = ph * pw, patch_size * patch_size

    pm = point_map.view(b, ph, patch_size, pw, patch_size, 3)
    pm = pm.permute(0, 1, 3, 2, 4, 5).reshape(b, n, px, 3)  # [B, n, px, 3]
    dp = depth.view(b, ph, patch_size, pw, patch_size)
    dp = dp.permute(0, 1, 3, 2, 4).reshape(b, n, px)  # [B, n, px]

    valid = (dp > 0) & torch.isfinite(dp)  # [B, n, px]
    valid_count = valid.sum(dim=-1)  # [B, n]
    ranked = torch.where(valid, dp, torch.full_like(dp, float("inf")))
    order = torch.argsort(ranked, dim=-1)  # invalid pixels sort last

    # Per-patch percentile index into the valid prefix; rank < valid_count guarantees a valid pick.
    k = (percentile * valid_count.to(point_map.dtype)).long()
    k = torch.minimum(k, (valid_count - 1).clamp_min(0))  # [B, n]
    sel = order.gather(-1, k.unsqueeze(-1))  # [B, n, 1]
    points = pm.gather(2, sel.unsqueeze(-1).expand(-1, -1, 1, 3)).squeeze(2)  # [B, n, 3]

    dead = valid_count == 0
    points = torch.where(dead.unsqueeze(-1), torch.zeros_like(points), points)
    return points, valid.float().mean(dim=-1)  # [B, n, 3], [B, n]


class GeometryBranch(nn.Module):
    """Depth → metric points → GATr → LLM-width geometry tokens.

    forward() consumes one H4 window's depth stack and returns, per token: an embedding in the
    backbone's hidden size (fp32; the injection site casts), the metric camera-space coordinate
    that RoPE3D-v2 will phase-encode, and the validity weight.
    """

    def __init__(
        self,
        hidden_size: int,
        n_frames: int = 4,
        patch_size_px: int = 32,
        depth_hw: tuple[int, int] = (256, 256),
        hfov_degrees: float = 90.0,
        gatr_blocks: int = 8,
        gatr_mv_channels: int = 16,
        gatr_s_channels: int = 32,
        percentile: float = 0.15,
        output_rms_norm: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        ensure_xformers_import_surface()
        from gatr import GATr, MLPConfig, SelfAttentionConfig
        from gatr.interface import embed_point, extract_point
        from gatr.utils.einsum import enable_cached_einsum

        enable_cached_einsum(False)  # cached opt_einsum planner fails on this numpy stack

        self._embed_point = embed_point
        self._extract_point = extract_point
        self.hidden_size = int(hidden_size)
        self.n_frames = int(n_frames)
        self.patch_size_px = int(patch_size_px)
        self.percentile = float(percentile)
        # Scale-free normalisation of the branch output. OFF by default so
        # ``native_chunk_geometry_134k_v1`` reproduces bit-identically; the rescue Stage A turns it
        # on. Without it the branch magnitude is a FREE DIRECTION: the injection site rescales
        # geometry rows to the text RMS with a *detached* factor, so the loss is blind to how large
        # this output grows. It grew to |x|~3e18, the rescale factor fell to ~1e-18, and the branch
        # gradient underflowed to exactly zero -- the crashes at u1088, u1594 and u1607.
        self.output_rms_norm = bool(output_rms_norm)
        self.intrinsics = CameraIntrinsics.from_hfov(
            width=depth_hw[1], height=depth_hw[0], hfov_degrees=hfov_degrees
        )
        grid_h = depth_hw[0] // patch_size_px
        grid_w = depth_hw[1] // patch_size_px
        self.tokens_per_frame = grid_h * grid_w
        self.n_tokens = self.tokens_per_frame * n_frames

        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=gatr_mv_channels,
            hidden_mv_channels=gatr_mv_channels,
            in_s_channels=2,  # (metric radius / UNIT_SCALE_M, validity)
            out_s_channels=gatr_s_channels,
            hidden_s_channels=gatr_s_channels,
            num_blocks=gatr_blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )
        self.readout_dim = gatr_s_channels + gatr_mv_channels + gatr_mv_channels * 3
        self.projector = MLPProjector(in_dim=self.readout_dim, out_dim=self.hidden_size)
        self.frame_embedding = nn.Embedding(n_frames, self.hidden_size)
        nn.init.normal_(self.frame_embedding.weight, std=0.02)
        if device is not None:
            self.to(device)

    def points_from_depth(self, depth_frames_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Backproject and aggregate one window's depth stack.

        Parameters
        ----------
        depth_frames_m : Tensor[F, H, W]
            Metric depth per chronological frame.

        Returns
        -------
        coords_m : Tensor[1, F*P, 3]; validity : Tensor[1, F*P]
        """
        f, h, w = depth_frames_m.shape
        if f != self.n_frames:
            raise ValueError(f"Expected {self.n_frames} frames, got {f}.")
        depth = depth_frames_m.to(torch.float32)
        grid = make_pixel_grid(w, h, device=depth.device, dtype=depth.dtype)  # [H, W, 2]
        u = grid[..., 0].unsqueeze(0).expand(f, -1, -1)
        v = grid[..., 1].unsqueeze(0).expand(f, -1, -1)
        point_map = backproject_pixel(u, v, depth, self.intrinsics)  # [F, H, W, 3]
        pts, val = aggregate_patches_percentile_masked(
            point_map, depth, self.patch_size_px, self.percentile
        )  # [F, P, 3], [F, P]
        return pts.reshape(1, -1, 3), val.reshape(1, -1)  # [1, F*P, 3], [1, F*P]

    def forward(
        self, coords_m: torch.Tensor, validity: torch.Tensor
    ) -> torch.Tensor:
        """Encode metric points into LLM-width geometry tokens.

        Parameters
        ----------
        coords_m : Tensor[1, N, 3] — camera-space metres, per-frame egocentric.
        validity : Tensor[1, N] — 0 marks a dead patch; its features are zeroed.

        Returns
        -------
        embeds : Tensor[1, N, hidden_size], fp32.
        """
        if coords_m.shape[1] != self.n_tokens:
            raise ValueError(f"Expected {self.n_tokens} tokens, got {coords_m.shape[1]}.")
        scaled = coords_m / UNIT_SCALE_M  # unit change, identical for every sample
        radius = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)  # [1, N, 1]
        scalars = torch.cat([radius, validity.unsqueeze(-1)], dim=-1)  # [1, N, 2]

        mv_out, s_out = self.gatr(
            self._embed_point(scaled).unsqueeze(-2), scalars=scalars
        )  # [1, N, C_mv, 16], [1, N, C_s]
        mv_norms = torch.linalg.vector_norm(mv_out, dim=-1)  # [1, N, C_mv]
        points = self._extract_point(mv_out) * UNIT_SCALE_M  # [1, N, C_mv, 3], back to metres
        feats = torch.cat(
            [s_out, mv_norms, points.flatten(-2)], dim=-1
        )  # [1, N, readout_dim]
        feats = feats * validity.unsqueeze(-1)  # dead patches carry nothing

        embeds = self.projector(feats)  # [1, N, hidden]
        frame_idx = torch.arange(
            self.n_frames, device=embeds.device
        ).repeat_interleave(self.tokens_per_frame)
        embeds = embeds + self.frame_embedding(frame_idx).unsqueeze(0)  # [1, N, hidden]
        if self.output_rms_norm:
            # Differentiable and scale-free: divides by its own RMS, so the gradient carries
            # direction, never magnitude, and downstream norm matching sees an O(1) input.
            rms = embeds.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            embeds = embeds / rms
        return embeds  # [1, N, hidden]
