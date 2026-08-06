"""Similarity metrics for intensity-based registration."""

import numpy as np


def ncc(a, b, mask=None):
    """Normalised cross-correlation between two equally shaped images.

    Returns a value in [-1, 1]; 1 means perfectly correlated. A boolean
    mask restricts the comparison to valid (e.g. tissue-covered) pixels.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def mutual_information(a, b, bins=32, mask=None):
    """Histogram-based mutual information (in nats) between two images.

    Preferred over NCC for cross-modality pairs (e.g. H&E vs IF) where
    intensities are related non-linearly.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel()
        a, b = a[m], b[m]
    h2d, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = h2d / h2d.sum()
    px, py = pxy.sum(axis=1), pxy.sum(axis=0)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz])))
