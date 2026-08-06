"""RANSAC outlier rejection for keypoint correspondences."""

import numpy as np

from .transforms import apply_transform, estimate_affine


def ransac_affine(src, dst, iterations=500, threshold=3.0, min_samples=3, seed=0):
    """Fit an affine transform with RANSAC.

    Randomly samples `min_samples` correspondences, fits an affine
    candidate, and keeps the model with the largest inlier set
    (residual < threshold pixels). The final model is refit by least
    squares on all inliers.

    Returns (matrix, inlier_mask); inlier_mask flags the correspondences
    consistent with the returned matrix.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < min_samples:
        raise ValueError(f"need at least {min_samples} correspondences")
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(src), dtype=bool)

    for _ in range(iterations):
        idx = rng.choice(len(src), size=min_samples, replace=False)
        try:
            M = estimate_affine(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(apply_transform(src, M) - dst, axis=1)
        inliers = residual < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if not best_inliers.any():
        raise ValueError("RANSAC found no inlier set; check threshold/inputs")
    matrix = estimate_affine(src[best_inliers], dst[best_inliers])
    return matrix, best_inliers
