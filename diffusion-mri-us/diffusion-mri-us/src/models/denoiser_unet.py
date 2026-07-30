"""Conditional U-Net denoiser operating in the VAE latent space.

Timestep information is injected through FiLM-style embeddings, and MRI context
(the ViT patch tokens) is injected through cross-attention at the coarser
resolutions, so global anatomical context guides denoising at every scale.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8):
        super().__init__()
        g_in = math.gcd(groups, in_ch) or 1
        g_out = math.gcd(groups, out_ch) or 1
        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Spatial queries attend to MRI context tokens (key/value)."""

    def __init__(self, ch: int, ctx_dim: int, heads: int = 4, groups: int = 8):
        super().__init__()
        g = math.gcd(groups, ch) or 1
        self.norm = nn.GroupNorm(g, ch)
        self.heads = heads
        self.to_q = nn.Conv2d(ch, ch, 1)
        self.to_k = nn.Linear(ctx_dim, ch)
        self.to_v = nn.Linear(ctx_dim, ch)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.to_q(self.norm(x)).reshape(b, self.heads, c // self.heads, h * w)
        k = self.to_k(ctx).reshape(b, ctx.shape[1], self.heads, c // self.heads).permute(0, 2, 3, 1)
        v = self.to_v(ctx).reshape(b, ctx.shape[1], self.heads, c // self.heads).permute(0, 2, 1, 3)
        attn = torch.softmax((q.transpose(-1, -2) @ k) / math.sqrt(c // self.heads), dim=-1)
        out = (attn @ v).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.op(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        latent_ch: int = 4,
        base: int = 64,
        ch_mult=(1, 2, 4),
        ctx_dim: int = 768,
        t_dim: int = 256,
        attn_levels=(1, 2),
    ):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.t_dim = t_dim
        self.in_conv = nn.Conv2d(latent_ch, base, 3, padding=1)

        chs = [base * m for m in ch_mult]
        self.downs = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        prev = base
        for lvl, ch in enumerate(chs):
            self.downs.append(ResBlock(prev, ch, t_dim))
            self.down_attn.append(CrossAttention(ch, ctx_dim) if lvl in attn_levels else nn.Identity())
            self.downsamplers.append(Down(ch) if lvl < len(chs) - 1 else nn.Identity())
            prev = ch

        self.mid1 = ResBlock(prev, prev, t_dim)
        self.mid_attn = CrossAttention(prev, ctx_dim)
        self.mid2 = ResBlock(prev, prev, t_dim)

        self.ups = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for lvl, ch in reversed(list(enumerate(chs))):
            self.upsamplers.append(Up(prev) if lvl < len(chs) - 1 else nn.Identity())
            self.ups.append(ResBlock(prev + ch, ch, t_dim))
            self.up_attn.append(CrossAttention(ch, ctx_dim) if lvl in attn_levels else nn.Identity())
            prev = ch

        g = math.gcd(8, prev) or 1
        self.out = nn.Sequential(nn.GroupNorm(g, prev), nn.SiLU(), nn.Conv2d(prev, latent_ch, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))
        h = self.in_conv(x)
        skips = []
        for res, attn, ds in zip(self.downs, self.down_attn, self.downsamplers):
            h = res(h, temb)
            h = attn(h, ctx) if isinstance(attn, CrossAttention) else h
            skips.append(h)
            h = ds(h)
        h = self.mid1(h, temb); h = self.mid_attn(h, ctx); h = self.mid2(h, temb)
        for us, res, attn in zip(self.upsamplers, self.ups, self.up_attn):
            h = us(h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = res(torch.cat([h, skip], dim=1), temb)
            h = attn(h, ctx) if isinstance(attn, CrossAttention) else h
        return self.out(h)
