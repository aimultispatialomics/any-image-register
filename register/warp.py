"""Image warping: resample a moving image into the fixed image space."""

import numpy as np


def warp_image(moving, matrix, output_shape=None, fill=0.0):
    """Warp `moving` with a (2, 3) affine matrix using bilinear sampling.

    The matrix maps output (fixed) pixel coordinates to input (moving)
    coordinates, i.e. the inverse of the forward registration transform.
    Pixels sampled outside the moving image are set to `fill`.
    """
    moving = np.asarray(moving, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float64)
    h, w = output_shape or moving.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ones = np.ones_like(xx)
    coords = np.stack([xx, yy, ones], axis=-1) @ matrix.T
    sx, sy = coords[..., 0], coords[..., 1]

    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    H, W = moving.shape[:2]
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)

    def sample(xs, ys):
        xs = np.clip(xs, 0, W - 1)
        ys = np.clip(ys, 0, H - 1)
        return moving[ys, xs]

    wx = sx - x0
    wy = sy - y0
    out = (
        sample(y0, x0) * (1 - wx) * (1 - wy)
        + sample(y0, x1) * wx * (1 - wy)
        + sample(y1, x0) * (1 - wx) * wy
        + sample(y1, x1) * wx * wy
    )
    out[~valid] = fill
    return out
