"""Training entry point (Algorithm 2).

Phase 1 warms up the VAE (stage-1 reconstruction). Phase 2 freezes the VAE and
jointly trains the ViT-conditioned diffusion denoiser and the anatomy-aware
deformable registration network.

Example
-------
    python -m src.train --config configs/liver.yaml
    python -m src.train --config configs/prostate.yaml --steps 200000
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from .models.framework import DiffUSion, FrameworkConfig
from .losses.anatomy_aware import AnatomyAwareLoss
from .data.datasets import PairedMRIUS


def load_cfg(path: str | None) -> dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def build(cfg: dict, device: str):
    fc = FrameworkConfig(
        image_size=cfg.get("image_size", 256),
        latent_ch=cfg.get("latent_ch", 4),
        vit_pretrained=cfg.get("vit_pretrained", False),
        base_ch=cfg.get("base_ch", 64),
        ch_mult=tuple(cfg.get("ch_mult", (1, 2, 4))),
        timesteps=cfg.get("timesteps", 1000),
        sample_steps=cfg.get("sample_steps", 50),
    )
    return DiffUSion(fc, device=device).to(device), fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_root", default=None, help="omit to use the synthetic smoke dataset")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--vae_warmup", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lambda_reg", type=float, default=None)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    steps = args.steps or cfg.get("steps", 100000)
    warmup = args.vae_warmup if args.vae_warmup is not None else cfg.get("vae_warmup", 5000)
    bs = args.batch_size or cfg.get("batch_size", 8)
    lr = args.lr or cfg.get("lr", 1e-4)
    lambda_reg = args.lambda_reg if args.lambda_reg is not None else cfg.get("lambda_reg", 1.0)

    model, fc = build(cfg, args.device)
    reg_loss = AnatomyAwareLoss(
        lambda_smooth=cfg.get("lambda_smooth", 1.0),
        lambda_bridge=cfg.get("lambda_bridge", 0.5),
    )

    ds = PairedMRIUS(root=args.data_root, image_size=fc.image_size)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=cfg.get("workers", 0))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    os.makedirs(args.out, exist_ok=True)

    step = 0
    model.train()
    while step < steps:
        for batch in dl:
            mri = batch["mri"].to(args.device)
            us = batch["us"].to(args.device)
            mask = batch["mask"].to(args.device)

            opt.zero_grad()
            if step < warmup:
                out = model.training_step(mri, us, train_vae=True)
                loss = out["total_vae"]
            else:
                out = model.training_step(mri, us)
                us_star = model.synthesise(mri)
                warped, flow = model.register(us, mri)
                rloss, _ = reg_loss(warped, mri, flow, bridge=us_star, mask=mask)
                loss = out["diff"] + lambda_reg * rloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % cfg.get("log_every", 50) == 0:
                print(f"step {step:>7d} | loss {loss.item():.4f}")
            if step and step % cfg.get("ckpt_every", 5000) == 0:
                torch.save(model.state_dict(), os.path.join(args.out, f"diffusion_{step}.pt"))
            step += 1
            if step >= steps:
                break

    torch.save(model.state_dict(), os.path.join(args.out, "diffusion_final.pt"))
    print("done ->", os.path.join(args.out, "diffusion_final.pt"))


if __name__ == "__main__":
    main()
