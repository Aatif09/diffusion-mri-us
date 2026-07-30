# DiffUSion — Vision Transformer-Conditioned Diffusion for MRI-Calibrated Ultrasound Synthesis and Registration

Reference implementation for the paper *"A Vision Transformer-Conditioned
Diffusion Framework for MRI-Calibrated Ultrasound Synthesis and Registration in
Liver and Prostate Imaging."*

The framework couples cross-modal synthesis and deformable registration:

- **Stage 1 — translation.** A shared Vision Transformer encodes the MRI and,
  through cross-attention, conditions a latent diffusion model that synthesises
  an MRI-calibrated pseudo-ultrasound (US\*).
- **Stage 2 — registration.** A single VoxelMorph-style deformable registration
  aligns the *real* ultrasound to the MRI, using US\* as an intermediate bridge
  that narrows the modality appearance gap. The synthetic image is never
  registered back to the MRI as a separate step.

An anatomy-aware loss reweights the similarity term toward low-intensity organ
interiors and penalises non-smooth deformation fields.

## Repository layout

```
configs/            liver.yaml, prostate.yaml
src/models/
  vit_encoder.py    shared ViT-B/16-style encoder (MRI context tokens)
  denoiser_unet.py  conditional U-Net denoiser with cross-attention
  diffusion.py      TinyVAE (latent stage) + Gaussian diffusion + DDIM sampler
  registration.py   VoxelMorph-style RegNet + spatial-transformer warp
  framework.py      DiffUSion: ties Stage 1 and Stage 2 together
src/losses/
  anatomy_aware.py  interior-weighted similarity + smoothness (+ bridge) losses
src/data/
  datasets.py       paired MRI/US loader (+ synthetic fallback)
src/utils/
  metrics.py        PSNR, SSIM, LPIPS, Dice, ASD, HD95, TRE
  warp.py           differentiable warping helpers
src/train.py        Algorithm 2 (VAE warmup -> diffusion + registration)
src/inference.py    Algorithm 3 (synthesise US*, then register)
scripts/prepare_data.py  convert public datasets to the .npy layout
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`timm` and `lpips` are optional. Without `timm` the ViT encoder runs from a
self-contained implementation (no pretrained weights); without `lpips` all
other metrics still work.

## Quickstart (no data required)

The loader generates a small synthetic dataset when `--data_root` is omitted, so
the full pipeline can be exercised end to end:

```bash
# train a few steps on CPU/GPU
python -m src.train --config configs/liver.yaml --steps 40 --vae_warmup 20 --batch_size 2

# run inference and write synthetic_us.npy / registered_us.npy / flow.npy
python -m src.inference --config configs/liver.yaml --index 0 --out outputs
```

## Training on real data

1. Convert each source into `root/{mri,us,mask}/<id>.npy` (single-channel,
   values in `[-1, 1]`) using `scripts/prepare_data.py`.
2. Point training at it:

```bash
python -m src.train --config configs/prostate.yaml --data_root data/prostate
```

## Datasets

| Source | Modality | Role |
| --- | --- | --- |
| CHAOS | liver MRI | Stage-1 conditioning / registration target |
| Annotated liver US | liver ultrasound | real ultrasound input |
| PROSTATEx | prostate MRI | Stage-1 conditioning / registration target |
| OpenPros | simulated prostate US | controlled validation (known geometry) |

OpenPros pairs speed-of-sound phantoms with simulated full-waveform ultrasound,
giving known ground-truth geometry for the Dice / ASD / HD95 / TRE reported in
the paper. Please follow each dataset's own licence and citation terms.

## Notes

This is a clean-room reference implementation intended for reproducibility and
extension. It is written in 2-D for clarity; the encoder, denoiser, registration
network, and losses extend to 3-D by swapping the 2-D operators for their 3-D
counterparts. Trained checkpoints will be linked here on release.

## Citation

```bibtex
@article{diffusion_mri_us,
  title   = {A Vision Transformer-Conditioned Diffusion Framework for
             MRI-Calibrated Ultrasound Synthesis and Registration in Liver and
             Prostate Imaging},
  author  = {Gupta, Priti and Agarwal, Neha and Adlakha, Richa and Dubey, Anil
             and Dubey, Gaurav and Jamshed, Aatif and Das, Nripendra Narayan},
  year    = {2025}
}
```

## License

MIT — see [LICENSE](LICENSE).
