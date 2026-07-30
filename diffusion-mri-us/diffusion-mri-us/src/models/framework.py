"""DiffUSion framework: ViT-conditioned latent diffusion + deformable registration.

Stage 1 (translation): the MRI is encoded by the shared Vision Transformer and
conditions a latent diffusion model that synthesises an MRI-calibrated
pseudo-ultrasound (US*).

Stage 2 (registration): a single deformable registration aligns the *real*
ultrasound to the MRI, using US* as an intermediate bridge that narrows the
modality appearance gap. The synthetic image is never registered back to the
MRI as a separate step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .vit_encoder import ViTEncoder
from .denoiser_unet import ConditionalUNet
from .diffusion import TinyVAE, GaussianDiffusion
from .registration import RegistrationNet, SpatialTransformer


@dataclass
class FrameworkConfig:
    image_size: int = 256
    latent_ch: int = 4
    vit_dim: int = 768
    vit_depth: int = 12
    vit_heads: int = 12
    vit_pretrained: bool = False
    base_ch: int = 64
    ch_mult: tuple = (1, 2, 4)
    timesteps: int = 1000
    sample_steps: int = 50


class DiffUSion(nn.Module):
    def __init__(self, cfg: FrameworkConfig, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.vit = ViTEncoder(
            img_size=224, dim=cfg.vit_dim, depth=cfg.vit_depth,
            heads=cfg.vit_heads, pretrained=cfg.vit_pretrained,
        )
        self.vae = TinyVAE(in_ch=1, latent_ch=cfg.latent_ch)
        self.denoiser = ConditionalUNet(
            latent_ch=cfg.latent_ch, base=cfg.base_ch, ch_mult=cfg.ch_mult, ctx_dim=cfg.vit_dim,
        )
        self.diffusion = GaussianDiffusion(cfg.timesteps, device=device)
        self.reg = RegistrationNet()
        self.stn = SpatialTransformer((cfg.image_size, cfg.image_size))

    # ------- Stage 1 helpers -------
    def encode_latent(self, img: torch.Tensor) -> torch.Tensor:
        mu, _ = self.vae.encode(img)
        return mu

    def synthesise(self, mri: torch.Tensor) -> torch.Tensor:
        """Sample an MRI-calibrated pseudo-ultrasound from an MRI (inference)."""
        ctx = self.vit(mri)
        lat = self.encode_latent(mri)
        z = self.diffusion.ddim_sample(
            self.denoiser, lat.shape, ctx, steps=self.cfg.sample_steps, device=mri.device
        )
        return self.vae.decode(z)

    # ------- Stage 2 helpers -------
    def register(self, moving: torch.Tensor, fixed: torch.Tensor):
        flow = self.reg(moving, fixed)
        warped = self.stn(moving, flow)
        return warped, flow

    # ------- training forward -------
    def training_step(self, mri, us, us_mask=None, train_vae=False):
        """Returns per-term losses for one optimisation step (Algorithm 2)."""
        ctx = self.vit(mri)
        out = {}

        # VAE stage-1 (warmup): reconstruct the ultrasound
        recon, _, kl = self.vae(us)
        vae_recon = nn.functional.l1_loss(recon, us)
        out["vae_recon"] = vae_recon
        out["vae_kl"] = kl
        if train_vae:
            out["total_vae"] = vae_recon + 1e-4 * kl

        # diffusion loss on the ultrasound latent, conditioned on the MRI
        with torch.no_grad():
            us_lat = self.encode_latent(us)
        out["diff"] = self.diffusion.loss(self.denoiser, us_lat, ctx)
        return out

    @torch.no_grad()
    def full_inference(self, mri, us):
        """Algorithm 3: synthesise US*, then register real US to MRI."""
        us_star = self.synthesise(mri)
        warped, flow = self.register(us, mri)
        return {"synthetic_us": us_star, "registered_us": warped, "flow": flow}
