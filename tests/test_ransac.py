"""Unit tests for RANSAC affine fitting."""

import numpy as np
import pytest

from register.ransac import ransac_affine
from register.transforms import apply_transform


def _make_correspondences(n_inliers=40, n_outliers=10, seed=0):
    M_true = np.array([[1.1, 0.05, 3.0],
                       [-0.02, 0.95, -4.0]])
    rng = np.random.default_rng(seed)
    src = rng.random((n_inliers + n_outliers, 2)) * 100
    dst = apply_transform(src, M_true)
    dst[n_inliers:] += rng.normal(0, 50, size=(n_outliers, 2))
    return src, dst, M_true, n_inliers


def test_recovers_transform_despite_outliers():
    src, dst, M_true, n_inliers = _make_correspondences()
    M, inliers = ransac_affine(src, dst, iterations=1000, threshold=1.0)
    assert np.allclose(M, M_true, atol=1e-6)
    assert inliers.sum() == n_inliers
    assert not inliers[n_inliers:].any()


def test_all_inliers_flagged_on_clean_data():
    rng = np.random.default_rng(2)
    src = rng.random((15, 2)) * 50
    M_true = np.array([[1.0, 0.2, 1.0],
                       [0.0, 1.0, 2.0]])
    dst = apply_transform(src, M_true)
    M, inliers = ransac_affine(src, dst, threshold=0.5)
    assert inliers.all()
    assert np.allclose(M, M_true, atol=1e-8)


def test_raises_when_too_few_correspondences():
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="at least"):
        ransac_affine(pts, pts, min_samples=3)


def test_deterministic_given_seed():
    src, dst, _, _ = _make_correspondences(seed=7)
    M1, mask1 = ransac_affine(src, dst, seed=42)
    M2, mask2 = ransac_affine(src, dst, seed=42)
    assert np.array_equal(M1, M2)
    assert np.array_equal(mask1, mask2)
