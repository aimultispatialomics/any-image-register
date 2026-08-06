"""Gaussian image pyramid for coarse-to-fine registration."""

import numpy as np


def downsample2x(img):
    """2x block-mean downsampling; odd edges are truncated."""
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    img = img[: h - h % 2, : w - w % 2]
    out = img.reshape(h // 2, 2, w // 2, 2, *img.shape[2:]).mean(axis=(1, 3))
    return out


def build_pyramid(img, levels=4, min_size=32):
    """Return [img, img/2, ..., img/2**(levels-1)] coarse-to-fine inputs.

    Construction stops early if the next level would fall below
    min_size on any axis, so callers can request a generous level count
    without guarding the image size themselves.
    """
    pyramid = [np.asarray(img, dtype=np.float32)]
    for _ in range(levels - 1):
        nxt = downsample2x(pyramid[-1])
        if min(nxt.shape[:2]) < min_size:
            break
        pyramid.append(nxt)
    return pyramid
