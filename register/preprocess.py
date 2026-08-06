"""Intensity normalisation and denoising for registration preprocessing."""

import numpy as np


def normalize_intensity(img, method="minmax", clip_percentile=None):
    """Normalise image intensities.

    method="minmax" scales to [0, 1]; method="zscore" standardises to
    zero mean / unit variance. clip_percentile=(lo, hi) winsorises the
    intensities first, which is useful for fluorescence images with
    bright hot pixels.
    """
    img = np.asarray(img, dtype=np.float32)
    if clip_percentile is not None:
        lo, hi = np.percentile(img, clip_percentile)
        img = np.clip(img, lo, hi)
    if method == "minmax":
        span = img.max() - img.min()
        return (img - img.min()) / span if span > 0 else np.zeros_like(img)
    if method == "zscore":
        std = img.std()
        return (img - img.mean()) / std if std > 0 else np.zeros_like(img)
    raise ValueError(f"unknown normalisation method: {method!r}")


def gaussian_smooth(img, sigma=1.0, radius=None):
    """Separable Gaussian smoothing implemented with numpy only.

    Keeps the dependency footprint small for environments where scipy is
    unavailable (e.g. embedded analysis pipelines).
    """
    img = np.asarray(img, dtype=np.float32)
    radius = radius if radius is not None else max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    out = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 0, img)
    out = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, out)
    return out
