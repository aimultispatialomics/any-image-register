"""Unit tests for similarity metrics."""

import numpy as np

from register.metrics import mutual_information, ncc


def test_ncc_identical_images():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64))
    assert ncc(img, img) == np.float64(1.0)


def test_ncc_anticorrelated():
    img = np.linspace(0, 1, 100).reshape(10, 10)
    assert ncc(img, 1 - img) < -0.99


def test_ncc_shape_mismatch_raises():
    a = np.zeros((10, 10))
    b = np.zeros((10, 12))
    try:
        ncc(a, b)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_mi_independent_images_near_zero():
    rng = np.random.default_rng(1)
    a = rng.random(10_000)
    b = rng.random(10_000)
    assert mutual_information(a, b, bins=16) < 0.05


def test_mi_dependent_images_positive():
    rng = np.random.default_rng(2)
    a = rng.random(10_000)
    assert mutual_information(a, a + 0.01 * rng.random(10_000)) > 0.5
