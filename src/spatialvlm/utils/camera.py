"""Camera intrinsics and projection utilities.

Implements the standard pinhole camera model:
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

where (u, v) are pixel coordinates (col, row) and (cx, cy) is the principal point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters.

    Follows OpenCV convention: u = col index (x), v = row index (y).

    Parameters
    ----------
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point in pixels.
    width, height : int
        Image dimensions in pixels.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(
        cls,
        *,
        width: int,
        height: int,
        hfov_degrees: float,
    ) -> CameraIntrinsics:
        """Construct centered pixel intrinsics from a horizontal field of view.

        Pixel centers use zero-based coordinates, so the centered principal point is
        ``((width - 1) / 2, (height - 1) / 2)``. Square pixels are assumed.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Camera width and height must be positive.")
        if not (0.0 < hfov_degrees < 180.0):
            raise ValueError("hfov_degrees must be strictly between 0 and 180.")
        focal = width / (2.0 * math.tan(math.radians(hfov_degrees) / 2.0))
        return cls(
            fx=focal,
            fy=focal,
            cx=(width - 1.0) / 2.0,
            cy=(height - 1.0) / 2.0,
            width=width,
            height=height,
        )

    def rescaled(self, *, width: int, height: int) -> CameraIntrinsics:
        """Rescale intrinsics for an ``align_corners=False`` image resize.

        The half-pixel transform matches PyTorch and common image-processor resize conventions:
        ``x' = (x + 0.5) * scale - 0.5``.
        """
        if width <= 0 or height <= 0:
            raise ValueError("Target width and height must be positive.")
        scale_x = width / self.width
        scale_y = height / self.height
        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=(self.cx + 0.5) * scale_x - 0.5,
            cy=(self.cy + 0.5) * scale_y - 0.5,
            width=width,
            height=height,
        )


def make_pixel_grid(
    width: int,
    height: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a dense grid of pixel (u, v) coordinates.

    Parameters
    ----------
    width, height : int
        Image dimensions.
    device : torch.device | None
    dtype : torch.dtype

    Returns
    -------
    grid : Tensor[H, W, 2]
        Last dimension is (u, v) = (col, row), both 0-indexed.
    """
    u = torch.arange(width, device=device, dtype=dtype)  # [W]
    v = torch.arange(height, device=device, dtype=dtype)  # [H]
    # indexing='xy': first arg varies along W (u), second arg along H (v)
    uu, vv = torch.meshgrid(u, v, indexing="xy")  # both [H, W]
    return torch.stack([uu, vv], dim=-1)  # [H, W, 2]


def backproject_pixel(
    u: torch.Tensor,
    v: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: CameraIntrinsics,
) -> torch.Tensor:
    """Backproject pixels with given depths into 3D camera-space coordinates.

    Parameters
    ----------
    u : Tensor[...]
        Pixel column coordinates (float).
    v : Tensor[...]
        Pixel row coordinates (float).
    depth : Tensor[...]
        Metric depth values (same shape as u, v).
        Zero or NaN values produce (0, 0, 0) or (NaN, NaN, NaN) respectively.
    intrinsics : CameraIntrinsics

    Returns
    -------
    points : Tensor[..., 3]
        3D coordinates [X, Y, Z] in camera space. Shape = (*u.shape, 3).
    """
    # X = (u - cx) * d / fx
    # Y = (v - cy) * d / fy
    # Z = d
    x = (u - intrinsics.cx) * depth / intrinsics.fx
    y = (v - intrinsics.cy) * depth / intrinsics.fy
    z = depth
    return torch.stack([x, y, z], dim=-1)  # [..., 3]
