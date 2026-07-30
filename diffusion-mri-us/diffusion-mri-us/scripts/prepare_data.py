"""Convert public datasets into the paired ``.npy`` layout used for training.

This is a template: fill in the reader for each source you use. Every case is
resampled to a common in-plane size, intensity-normalised to ``[-1, 1]`` and
written as ``root/{mri,us,mask}/<id>.npy``.

Supported sources
-----------------
* CHAOS ........... liver MRI (T1/T2) with liver masks.
* Liver US ........ annotated liver ultrasound frames.
* PROSTATEx ....... prostate MRI (T2) with gland masks.
* OpenPros ........ simulated prostate ultrasound with known geometry
                    (used for controlled validation).

Usage
-----
    python scripts/prepare_data.py --source chaos --in /path/raw --out data/liver
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def normalise(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-6)  # -> [0,1]
    return (arr * 2 - 1).astype(np.float32)  # -> [-1,1]


def resize_to(arr: np.ndarray, size: int) -> np.ndarray:
    try:
        from skimage.transform import resize
        return resize(arr, (size, size), order=1, preserve_range=True, anti_aliasing=True)
    except Exception:
        # nearest-neighbour fallback without skimage
        ys = (np.linspace(0, arr.shape[0] - 1, size)).astype(int)
        xs = (np.linspace(0, arr.shape[1] - 1, size)).astype(int)
        return arr[np.ix_(ys, xs)]


def save_case(out_root: str, case_id: str, mri: np.ndarray, us: np.ndarray, mask=None):
    for sub in ("mri", "us", "mask"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    np.save(os.path.join(out_root, "mri", f"{case_id}.npy"), mri)
    np.save(os.path.join(out_root, "us", f"{case_id}.npy"), us)
    if mask is not None:
        np.save(os.path.join(out_root, "mask", f"{case_id}.npy"), mask.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["chaos", "liver_us", "prostatex", "openpros"])
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    # Implement the per-source reader here. Each reader should yield tuples of
    # (case_id, mri_slice, us_slice, optional_mask) as raw 2-D numpy arrays.
    raise NotImplementedError(
        "Add the reader for '%s'. For each paired case call save_case(args.out, id, "
        "normalise(resize_to(mri, args.size)), normalise(resize_to(us, args.size)), mask)."
        % args.source
    )


if __name__ == "__main__":
    main()
