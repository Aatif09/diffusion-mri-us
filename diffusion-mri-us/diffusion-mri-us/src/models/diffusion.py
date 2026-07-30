"""Latent stage (a compact VAE) and the Gaussian diffusion process.

The VAE follows the two-stage latent-diffusion recipe: images are compressed to
a low-resolution latent, diffusion is trained in that latent space, and samples
are decoded back to image space. A DDIM sampler is provided for fast inference.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(math.gcd(8, o) or 1, o), nn.SiLU(),
            nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(math.gcd(8, o) or 1, o), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class TinyVAE(nn.Module):
    """8x spatial compression, ``latent_ch`` channels. KL-regularised."""

    def __init__(self, in_ch: int = 1, latent_ch: int = 4, base: int = 32):
        super().__init__()
        self.enc = nn.ModuleList([
            _ConvBlock(in_ch, base), _ConvBlock(base, base * 2), _ConvBlock(base * 2, base * 4),
        ])
        self.down = nn.AvgPool2d(2)
        self.to_moments = nn.Conv2d(base * 4, latent_ch * 2, 1)
        self.from_latent = nn.Conv2d(latent_ch, base * 4, 1)
        self.dec = nn.ModuleList([
            _ConvBlock(base * 4, base * 2), _ConvBlock(base * 2, base), _ConvBlock(base, base),
        ])
        self.out = nn.Conv2d(base, in_ch, 3, padding=1)
        self.latent_ch = latent_ch

    def encode(self, x):
        h = x
        for b in self.enc:
            h = self.down(b(h))
        mu, logvar = self.to_moments(h).chunk(2, dim=1)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        h = self.from_latent(z)
        for b in self.dec:
            h = b(F.interpolate(h, scale_factor=2, mode="nearest"))
        return torch.tanh(self.out(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, z, kl


class GaussianDiffusion:
    """Standard DDPM forward process with an epsilon-prediction objective."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, device="cpu"):
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_acp[t][:, None, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None, None]
        return a * x0 + b * noise, noise

    def loss(self, model, x0, ctx):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        pred = model(x_t, t, ctx)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def ddim_sample(self, model, shape, ctx, steps: int = 50, eta: float = 0.0, device="cpu"):
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        for i, t in enumerate(ts):
            t_b = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
            eps = model(x, t_b, ctx)
            acp_t = self.alphas_cumprod[t]
            x0 = (x - torch.sqrt(1 - acp_t) * eps) / torch.sqrt(acp_t)
            x0 = x0.clamp(-3, 3)
            if i < len(ts) - 1:
                acp_next = self.alphas_cumprod[ts[i + 1]]
                sigma = eta * torch.sqrt((1 - acp_next) / (1 - acp_t)) * torch.sqrt(1 - acp_t / acp_next)
                dir_xt = torch.sqrt(1 - acp_next - sigma ** 2) * eps
                x = torch.sqrt(acp_next) * x0 + dir_xt + sigma * torch.randn_like(x)
            else:
                x = x0
        return x
