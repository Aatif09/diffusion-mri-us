"""Deformable registration network (VoxelMorph-style) and a differentiable warp.

The network predicts a dense displacement field from a moving/fixed pair; the
:class:`SpatialTransformer` resamples the moving image with that field. This is
the single learned deformable registration used in the framework.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Enc(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.net(x)


class _Dec(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.net(x)


class RegistrationNet(nn.Module):
    """U-Net regressing a 2-channel (2D) displacement field."""

    def __init__(self, in_ch: int = 2, enc=(16, 32, 32, 32), dec=(32, 32, 32, 16)):
        super().__init__()
        self.downs = nn.ModuleList()
        prev = in_ch
        for c in enc:
            self.downs.append(_Enc(prev, c)); prev = c
        self.ups = nn.ModuleList()
        skips = list(enc[::-1])
        for k, c in enumerate(dec):
            extra = skips[k] if k < len(skips) else 0
            self.ups.append(_Dec(prev + extra, c)); prev = c
        self.flow = nn.Conv2d(prev, 2, 3, padding=1)
        nn.init.normal_(self.flow.weight, 0, 1e-5)
        nn.init.zeros_(self.flow.bias)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([moving, fixed], dim=1)
        feats = []
        for d in self.downs:
            x = d(x); feats.append(x)
        feats = feats[::-1]
        for k, u in enumerate(self.ups):
            if k < len(feats):
                x = F.interpolate(x, size=feats[k].shape[-2:], mode="nearest")
                x = torch.cat([x, feats[k]], dim=1)
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = u(x)
        flow = self.flow(x)
        if flow.shape[-2:] != moving.shape[-2:]:
            flow = F.interpolate(flow, size=moving.shape[-2:], mode="bilinear", align_corners=False)
        return flow


class SpatialTransformer(nn.Module):
    """Resample an image with a displacement field (pixel units)."""

    def __init__(self, size):
        super().__init__()
        vectors = [torch.arange(0, s) for s in size]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij")).float()  # [2,H,W] (y,x)
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, src: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        new = self.grid + flow.flip(1)  # flow is (dx,dy) -> reorder to (dy,dx)
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new[:, i] = 2 * (new[:, i] / (shape[i] - 1) - 0.5)
        new = new.permute(0, 2, 3, 1)[..., [1, 0]]  # -> (x,y) for grid_sample
        return F.grid_sample(src, new, mode=mode, align_corners=True, padding_mode="border")
