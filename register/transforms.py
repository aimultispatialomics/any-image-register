"""Rigid and affine transform estimation from point correspondences."""

import numpy as np


def estimate_rigid(src, dst, allow_reflection=False):
    """Kabsch/Umeyama estimate of rotation + translation mapping src -> dst.

    Returns (R, t) with dst ~= src @ R.T + t. Scale is preserved; use
    estimate_similarity-like fitting via estimate_affine if scaling is
    expected.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    u, _, vt = np.linalg.svd(src_c.T @ dst_c)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    if not allow_reflection:
        vt[-1, :] *= d
    R = vt.T @ u.T
    t = dst.mean(axis=0) - src.mean(axis=0) @ R.T
    return R, t


def estimate_affine(src, dst):
    """Least-squares 2x3 affine matrix mapping src -> dst.

    Solves dst = [src, 1] @ M.T for M with shape (2, 3).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    ones = np.ones((len(src), 1))
    M, *_ = np.linalg.lstsq(np.hstack([src, ones]), dst, rcond=None)
    return M.T


def apply_transform(points, matrix):
    """Apply a (2, 3) affine or (3, 3) homogeneous matrix to (N, 2) points."""
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    hom = np.hstack([points, np.ones((len(points), 1))])
    if matrix.shape == (2, 3):
        return hom @ matrix.T
    out = hom @ matrix.T
    return out[:, :2] / out[:, 2:3]
