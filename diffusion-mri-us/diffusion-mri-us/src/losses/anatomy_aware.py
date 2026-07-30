"""Anatomy-aware registration losses.

The similarity term reweights each voxel so that low-intensity organ interiors
contribute more than bright, noisy boundaries. When a segmentation mask is
available it is used directly; otherwise an intensity-driven sigmoid surrogate
approximates the interior weighting. A diffusion (gradient) penalty keeps the
displacement field smooth.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def local_ncc(a: torch.Tensor, b: torch.Tensor, win: int = 9, eps: float = 1e-5) -> torch.Tensor:
    """Per-voxel local normalised cross-correlation (higher is better)."""
    pad = win // 2
    k = torch.ones(1, 1, win, win, device=a.device) / (win * win)
    mu_a = F.conv2d(a, k, padding=pad)
    mu_b = F.conv2d(b, k, padding=pad)
    a2 = F.conv2d(a * a, k, padding=pad) - mu_a ** 2
    b2 = F.conv2d(b * b, k, padding=pad) - mu_b ** 2
    ab = F.conv2d(a * b, k, padding=pad) - mu_a * mu_b
    cc = (ab * ab) / (a2 * b2 + eps)
    return cc.clamp(0, 1)


def interior_weight(fixed: torch.Tensor, mask: torch.Tensor | None, sharpness: float = 10.0) -> torch.Tensor:
    if mask is not None:
        w = mask.float()
        return 0.25 + 0.75 * w  # never fully ignore the exterior
    # intensity surrogate: emphasise mid/low-intensity interior, damp bright edges
    grad_x = torch.abs(fixed[..., :, 1:] - fixed[..., :, :-1])
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_y = torch.abs(fixed[..., 1:, :] - fixed[..., :-1, :])
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    edge = (grad_x + grad_y)
    w = torch.sigmoid(sharpness * (0.5 - edge))
    return 0.25 + 0.75 * w


class AnatomyAwareLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 1.0, lambda_bridge: float = 0.5, win: int = 9):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_bridge = lambda_bridge
        self.win = win

    def smoothness(self, flow: torch.Tensor) -> torch.Tensor:
        dx = flow[..., :, 1:] - flow[..., :, :-1]
        dy = flow[..., 1:, :] - flow[..., :-1, :]
        return (dx.pow(2).mean() + dy.pow(2).mean())

    def forward(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        flow: torch.Tensor,
        bridge: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        w = interior_weight(fixed, mask)
        sim_map = local_ncc(warped, fixed, self.win)
        sim = 1.0 - (w * sim_map).sum() / (w.sum() + 1e-6)
        total = sim + self.lambda_smooth * self.smoothness(flow)
        out = {"sim": sim, "smooth": self.smoothness(flow)}
        if bridge is not None and self.lambda_bridge > 0:
            b_map = local_ncc(warped, bridge, self.win)
            bridge_term = 1.0 - (w * b_map).sum() / (w.sum() + 1e-6)
            total = total + self.lambda_bridge * bridge_term
            out["bridge"] = bridge_term
        out["total"] = total
        return total, out
