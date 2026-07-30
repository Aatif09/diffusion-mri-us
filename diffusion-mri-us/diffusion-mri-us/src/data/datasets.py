"""Paired MRI / ultrasound dataset.

Expected on-disk layout (see ``scripts/prepare_data.py``)::

    root/
      mri/   0001.npy 0002.npy ...   # single-channel, values in [-1, 1]
      us/    0001.npy 0002.npy ...
      mask/  0001.npy 0002.npy ...   # optional organ mask (0/1)

Supported public sources: CHAOS (liver MRI), an annotated liver ultrasound set,
PROSTATEx (prostate MRI), and OpenPros (simulated prostate ultrasound with
known geometry, used for controlled validation). Preprocessing scripts convert
each into the ``.npy`` layout above.

If ``root`` is ``None`` a small synthetic dataset is generated so the pipeline
can be exercised end to end without any downloads.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedMRIUS(Dataset):
    def __init__(self, root: str | None = None, image_size: int = 256, n_synthetic: int = 32):
        self.image_size = image_size
        self.root = root
        if root is None:
            self.ids = [f"{i:04d}" for i in range(n_synthetic)]
            self._synthetic = True
        else:
            mri_dir = os.path.join(root, "mri")
            self.ids = sorted(p[:-4] for p in os.listdir(mri_dir) if p.endswith(".npy"))
            self._synthetic = False

    def __len__(self):
        return len(self.ids)

    def _load(self, sub, name):
        arr = np.load(os.path.join(self.root, sub, f"{name}.npy")).astype(np.float32)
        return torch.from_numpy(arr)[None]  # [1,H,W]

    def _make_synthetic(self, seed: int):
        rng = np.random.default_rng(seed)
        s = self.image_size
        yy, xx = np.mgrid[0:s, 0:s] / s
        cx, cy, r = rng.uniform(0.35, 0.65, 2).tolist() + [rng.uniform(0.15, 0.3)]
        organ = ((xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2).astype(np.float32)
        mri = 0.6 * organ + 0.2 * np.sin(6 * np.pi * xx) * np.cos(6 * np.pi * yy)
        us = 0.5 * organ + 0.4 * rng.standard_normal((s, s)).astype(np.float32) * (0.5 + organ)
        norm = lambda a: np.clip(a / (np.abs(a).max() + 1e-6), -1, 1).astype(np.float32)
        to_t = lambda a: torch.from_numpy(norm(a))[None]
        return to_t(mri), to_t(us), torch.from_numpy(organ)[None]

    def __getitem__(self, idx):
        name = self.ids[idx]
        if self._synthetic:
            mri, us, mask = self._make_synthetic(idx)
        else:
            mri, us = self._load("mri", name), self._load("us", name)
            mask_path = os.path.join(self.root, "mask", f"{name}.npy")
            mask = self._load("mask", name) if os.path.exists(mask_path) else torch.ones_like(us)
        return {"mri": mri, "us": us, "mask": mask, "id": name}
