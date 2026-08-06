"""Image loading and grayscale conversion utilities."""

from pathlib import Path

import numpy as np


def read_image(path, grayscale=True):
    """Read an image file into a float32 array in [0, 1].

    Uses imageio when available; falls back to PIL. Channel-first and
    channel-last inputs are both normalised to (H, W) or (H, W, C).
    """
    path = Path(path)
    try:
        import imageio.v3 as iio
        img = iio.imread(path)
    except ImportError:
        from PIL import Image
        img = np.asarray(Image.open(path))
    img = np.asarray(img, dtype=np.float32)
    if img.max() > 1.0:
        img /= np.iinfo(np.uint8).max if img.max() <= 255 else img.max()
    return to_grayscale(img) if grayscale else img


def to_grayscale(img, weights=(0.299, 0.587, 0.114)):
    """Convert an (H, W, C) RGB image to (H, W) luminance.

    Single-channel images are returned unchanged. The ITU-R BT.601
    weights are used by default because they match most microscopy
    viewers' greyscale rendering.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2 or img.shape[-1] == 1:
        return img.squeeze()
    w = np.asarray(weights, dtype=np.float32)
    return np.tensordot(img[..., :3], w, axes=([-1], [0]))
