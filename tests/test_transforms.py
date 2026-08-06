"""Unit tests for transform estimation."""

import numpy as np

from register.transforms import apply_transform, estimate_affine, estimate_rigid


def test_rigid_recovers_known_transform():
    theta = np.deg2rad(30)
    R_true = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    t_true = np.array([5.0, -3.0])
    rng = np.random.default_rng(0)
    src = rng.random((50, 2)) * 100
    dst = src @ R_true.T + t_true
    R, t = estimate_rigid(src, dst)
    assert np.allclose(R, R_true, atol=1e-8)
    assert np.allclose(t, t_true, atol=1e-8)


def test_affine_recovers_known_matrix():
    M_true = np.array([[1.2, 0.1, 4.0],
                       [-0.05, 0.9, -2.0]])
    rng = np.random.default_rng(1)
    src = rng.random((20, 2)) * 50
    dst = apply_transform(src, M_true)
    M = estimate_affine(src, dst)
    assert np.allclose(M, M_true, atol=1e-8)


def test_apply_transform_homogeneous():
    pts = np.array([[0.0, 0.0], [2.0, 4.0]])
    M = np.array([[2.0, 0.0, 1.0],
                  [0.0, 2.0, 1.0],
                  [0.0, 0.0, 1.0]])
    out = apply_transform(pts, M)
    assert np.allclose(out, [[1.0, 1.0], [5.0, 9.0]])
