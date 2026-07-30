"""Shared Vision Transformer encoder.

Produces a sequence of patch tokens that condition the diffusion denoiser
through cross-attention. Structurally follows ViT-B/16 (patch 16, depth 12,
12 heads, width 768). If ``timm`` is installed and ``pretrained=True`` the
ImageNet-pretrained ViT-B/16 backbone is loaded and adapted to single-channel
input; otherwise a self-contained implementation is used so the repository
runs offline.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 1, patch: int = 16, dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                    # [B, dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)    # [B, N, dim]
        return x


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Returns L2-normalised patch tokens [B, N, dim] used as diffusion context."""

    def __init__(
        self,
        img_size: int = 224,
        patch: int = 16,
        in_ch: int = 1,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        pretrained: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.img_size = img_size
        n_patches = (img_size // patch) ** 2

        self._timm = None
        if pretrained:
            self._timm = self._try_load_timm(in_ch)

        if self._timm is None:
            self.patch_embed = PatchEmbed(in_ch, patch, dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
            self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _try_load_timm(in_ch: int):
        try:
            import timm  # type: ignore
            m = timm.create_model("vit_base_patch16_224", pretrained=True, in_chans=in_ch)
            m.head = nn.Identity()
            return m
        except Exception:
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )
        if self._timm is not None:
            tokens = self._timm.forward_features(x)      # [B, N(+1), dim]
        else:
            b = x.shape[0]
            tokens = self.patch_embed(x)
            cls = self.cls_token.expand(b, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
            for blk in self.blocks:
                tokens = blk(tokens)
            tokens = self.norm(tokens)
        # drop the class token; keep spatial patch tokens as context
        tokens = tokens[:, 1:, :]
        return torch.nn.functional.normalize(tokens, dim=-1)
