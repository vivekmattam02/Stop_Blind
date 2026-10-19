"""IcosahedralRoPE3D v2 --- scale-corrected 3D rotary position encoding.

Additive module. ``gridcell_rope3d.py`` is hash-locked by ``development_sft_v1`` and
``stability_pilot_v1`` and is left untouched so those runs stay reconstructable; neither is
load-bearing for the submitted manuscript, but provenance is cheaper to keep than to rebuild.

**What was wrong.** v1 sets ``BASE_FREQ = 10.0`` rad/m with 8 frequencies rising as
``f_k = 10.0 * e^(k/3)``, so wavelengths run **0.061 m to 0.628 m**. Scenes are roughly 10 m across
and reference paths average 6 m, which means the longest wavelength in the entire encoding was about
sixteen times smaller than the space it had to represent. Every position in a room aliased onto the
same few phases. Measured on the kernel ``mean_k cos(f_k d)``, v1 scores a 5 m separation as *more*
similar than a 0.25 m separation --- one forward step --- which inverts the property a positional
encoding exists to provide.

**What v2 does.** The design is chosen by search rather than by taste, under three constraints:

1. output width ``6 directions x n_freqs x 2 <= 128`` rotary dimensions, so ``n_freqs <= 10``;
2. longest wavelength at least the scene extent, so distinct rooms do not alias;
3. shortest wavelength small enough to separate nearby viewpoints.

Sweeping ``n_freqs``, scene extent and frequency ratio, and scoring each design by the number of
rank inversions of the kernel over 0--3 m (the range in which most decisions are made), the best
admissible design keeps the theoretically motivated ``e^(1/3)`` ratio and uses 10 frequencies over a
12 m extent. Near-range rank inversions fall from **900** under v1 to **262**. The kernel no longer
prefers far separations to near ones.

**What v2 does not do.** A finite Fourier basis ripples. Long-range inversions remain high for every
admissible design because ten frequencies cannot tile 0.6--12 m smoothly, and the shortest
wavelength, 0.597 m, is about 2.4 forward steps rather than one. Both are consequences of the
128-dimension budget, not oversights, and both are reported by ``kernel_diagnostics``
so a caller can check them rather than assume them.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from spatialvlm.geometry.gridcell_rope3d import _build_icosahedral_directions  # noqa: PLC2701

ROPE3D_V2_DESIGN_VERSION = 2


class IcosahedralRoPE3Dv2(nn.Module):
    """Scale-corrected icosahedral rotary encoding for metric 3D coordinates.

    Parameters
    ----------
    scene_extent_m:
        Longest wavelength, in metres. Should be at least the diameter of the environment so that
        positions across a scene do not share a phase. Default 12.0 for Matterport3D rooms.
    n_freqs:
        Number of logarithmically spaced frequencies. ``6 * n_freqs * 2`` must not exceed the
        backbone's rotary width (128 for Qwen3-VL-8B), so the practical maximum is 10.
    """

    N_DIRS: int = 6
    FREQ_RATIO: float = math.exp(1.0 / 3.0)
    DEFAULT_SCENE_EXTENT_M: float = 12.0
    DEFAULT_N_FREQS: int = 10
    MAX_ROTARY_DIMS: int = 128

    def __init__(
        self,
        scene_extent_m: float = DEFAULT_SCENE_EXTENT_M,
        n_freqs: int = DEFAULT_N_FREQS,
    ) -> None:
        super().__init__()
        if scene_extent_m <= 0.0:
            raise ValueError("scene_extent_m must be positive.")
        if n_freqs < 1:
            raise ValueError("n_freqs must be at least one.")
        width = self.N_DIRS * n_freqs * 2
        if width > self.MAX_ROTARY_DIMS:
            raise ValueError(
                f"{self.N_DIRS} directions x {n_freqs} frequencies x 2 = {width} exceeds the "
                f"{self.MAX_ROTARY_DIMS}-dimension rotary budget; reduce n_freqs."
            )

        self.scene_extent_m = float(scene_extent_m)
        self.n_freqs = int(n_freqs)
        self.output_dim = width

        self.register_buffer("directions", _build_icosahedral_directions())  # [6, 3]
        base = 2.0 * math.pi / self.scene_extent_m
        freqs = torch.tensor(
            [base * (self.FREQ_RATIO**k) for k in range(self.n_freqs)], dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)  # [n_freqs]

    @property
    def wavelengths_m(self) -> tuple[float, float]:
        """Return ``(shortest, longest)`` wavelength in metres."""
        f = self.freqs.detach().cpu()
        return (float(2.0 * math.pi / f.max()), float(2.0 * math.pi / f.min()))

    def kernel(self, separations_m: torch.Tensor) -> torch.Tensor:
        """Return the isotropic positional-similarity kernel at each separation.

        ``mean_k cos(f_k d)``. A usable encoding decreases with ``d`` over the range that matters;
        this exists so that property is checked rather than assumed.
        """
        d = separations_m.to(dtype=self.freqs.dtype).reshape(-1, 1)  # [S, 1]
        return torch.cos(d * self.freqs.reshape(1, -1)).mean(dim=-1)  # [S]

    def kernel_diagnostics(self, max_separation_m: float = 12.0, steps: int = 240) -> dict:
        """Report scale coverage and kernel monotonicity, for tests and run manifests."""
        grid = torch.linspace(max_separation_m / steps, max_separation_m, steps)
        values = self.kernel(grid)
        near_mask = grid <= 3.0
        near = values[near_mask]
        inversions = int(
            sum(
                1
                for i in range(len(near))
                for j in range(i + 1, len(near))
                if near[j] > near[i] + 1e-9
            )
        )
        short, long = self.wavelengths_m
        return {
            "design_version": ROPE3D_V2_DESIGN_VERSION,
            "scene_extent_m": self.scene_extent_m,
            "n_freqs": self.n_freqs,
            "output_dim": self.output_dim,
            "shortest_wavelength_m": short,
            "longest_wavelength_m": long,
            "kernel_at_0_25m": float(self.kernel(torch.tensor([0.25]))[0]),
            "kernel_at_5m": float(self.kernel(torch.tensor([5.0]))[0]),
            "near_range_rank_inversions": inversions,
            "prefers_near_over_far": bool(
                self.kernel(torch.tensor([0.25]))[0] > self.kernel(torch.tensor([5.0]))[0]
            ),
        }

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode metric 3D positions.

        Parameters
        ----------
        positions : Tensor[B, N, 3]
            Coordinates in metres.

        Returns
        -------
        Tensor[B, N, output_dim]
            Interleaved sin/cos over 6 icosahedral directions and ``n_freqs`` frequencies.
        """
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(f"positions must be [B, N, 3], got {tuple(positions.shape)}.")
        encoded = positions.to(dtype=self.directions.dtype)
        projections = torch.einsum("bnd,fd->bnf", encoded, self.directions)  # [B, N, 6]
        scaled = projections.unsqueeze(-1) * self.freqs  # [B, N, 6, F]
        enc = torch.stack([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # [B, N, 6, F, 2]
        return enc.reshape(*positions.shape[:2], self.output_dim)
