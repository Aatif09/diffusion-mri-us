"""Evaluation metrics for translation and registration.

Translation fidelity: PSNR, SSIM, and (optionally) LPIPS.
Registration accuracy: Dice, average surface distance (ASD), 95th-percentile
Hausdorff distance (HD95), and target registration error (TRE) on landmarks.
"""
from __future__ import annotations

import numpy as np

try:  # optional, only needed if you call ssim/psnr on numpy arrays
    from skimage.metrics import structural_similarity as _ssim
    from skimage.metrics import peak_signal_noise_ratio as _psnr
except Exception:  # pragma: no cover
    _ssim = _psnr = None

try:
    from scipy.ndimage import distance_transform_edt, binary_erosion
except Exception:  # pragma: no cover
    distance_transform_edt = binary_erosion = None


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> float:
    if _psnr is not None:
        return float(_psnr(target, pred, data_range=data_range))
    mse = np.mean((pred - target) ** 2)
    return float(20 * np.log10(data_range) - 10 * np.log10(mse + 1e-12))


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> float:
    if _ssim is not None:
        return float(_ssim(target, pred, data_range=data_range))
    # minimal global SSIM fallback
    mu_x, mu_y = pred.mean(), target.mean()
    vx, vy = pred.var(), target.var()
    cov = ((pred - mu_x) * (target - mu_y)).mean()
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return float(((2 * mu_x * mu_y + c1) * (2 * cov + c2)) /
                 ((mu_x ** 2 + mu_y ** 2 + c1) * (vx + vy + c2)))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denom)


def _surface_distances(a: np.ndarray, b: np.ndarray):
    assert distance_transform_edt is not None, "scipy is required for ASD/HD95"
    a = a.astype(bool); b = b.astype(bool)
    a_surf = a ^ binary_erosion(a)
    b_surf = b ^ binary_erosion(b)
    dt_b = distance_transform_edt(~b_surf)
    dt_a = distance_transform_edt(~a_surf)
    d_ab = dt_b[a_surf]
    d_ba = dt_a[b_surf]
    return np.concatenate([d_ab, d_ba]) if d_ab.size and d_ba.size else np.array([0.0])


def asd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(_surface_distances(a, b)))


def hd95(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.percentile(_surface_distances(a, b), 95))


def tre(pred_pts: np.ndarray, gt_pts: np.ndarray, spacing=(1.0, 1.0)) -> float:
    """Mean target registration error over matched landmark pairs (physical units)."""
    spacing = np.asarray(spacing)
    d = np.linalg.norm((pred_pts - gt_pts) * spacing, axis=1)
    return float(d.mean())
