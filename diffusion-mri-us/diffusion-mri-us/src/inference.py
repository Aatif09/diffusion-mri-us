"""Inference entry point (Algorithm 3).

Loads a trained checkpoint, synthesises an MRI-calibrated ultrasound from a test
MRI, then registers the real ultrasound to the MRI using that synthetic image as
the intermediate bridge. Saves the synthetic US, the registered US, and the
displacement field.

Example
-------
    python -m src.inference --config configs/liver.yaml \
        --ckpt checkpoints/diffusion_final.pt --index 0 --out outputs
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .train import load_cfg, build
from .data.datasets import PairedMRIUS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    model, fc = build(cfg, args.device)
    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model.eval()

    ds = PairedMRIUS(root=args.data_root, image_size=fc.image_size)
    sample = ds[args.index]
    mri = sample["mri"][None].to(args.device)
    us = sample["us"][None].to(args.device)

    with torch.no_grad():
        res = model.full_inference(mri, us)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "synthetic_us.npy"), res["synthetic_us"].cpu().numpy()[0, 0])
    np.save(os.path.join(args.out, "registered_us.npy"), res["registered_us"].cpu().numpy()[0, 0])
    np.save(os.path.join(args.out, "flow.npy"), res["flow"].cpu().numpy()[0])
    print("saved synthetic_us / registered_us / flow to", args.out)


if __name__ == "__main__":
    main()
