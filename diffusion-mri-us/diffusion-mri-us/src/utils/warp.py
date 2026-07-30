"""Warping / deformation-field helpers.

Includes a scaling-and-squaring integrator (for an optional diffeomorphic
variant of the registration output) and a Jacobian-determinant utility for
reporting the fraction of folding voxels, a standard deformation-regularity
diagnostic.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.registration import SpatialTransformer


class VecInt(torch.nn.Module):
    """Integrate a stationary velocity field via scaling and squaring."""

    def __init__(self, size, steps: int = 7):
        super().__init__()
        self.steps = steps
        self.scale = 1.0 / (2 ** steps)
        self.stn = SpatialTransformer(size)

    def forward(self, vel: torch.Tensor) -> torch.Tensor:
        flow = vel * self.scale
        for _ in range(self.steps):
            flow = flow + self.stn(flow, flow)
        return flow


def jacobian_determinant(flow: torch.Tensor) -> torch.Tensor:
    """Jacobian determinant of the deformation ``phi = id + flow`` (2-D).

    ``flow`` is ``[B, 2, H, W]`` with channel order ``(dx, dy)``. Returns
    ``[B, H-1, W-1]``; negative entries indicate folding.
    """
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    dudx = dx[:, 0, :-1, :]
    dvdx = dx[:, 1, :-1, :]
    dudy = dy[:, 0, :, :-1]
    dvdy = dy[:, 1, :, :-1]
    return (1 + dudx) * (1 + dvdy) - dudy * dvdx


def folding_fraction(flow: torch.Tensor) -> float:
    jac = jacobian_determinant(flow)
    return float((jac <= 0).float().mean().item())
