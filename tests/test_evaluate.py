"""Unit tests for registration evaluation metrics."""

import numpy as np
import pytest

from register.evaluate import dice_coefficient, target_registration_error


def test_tre_zero_for_perfect_alignment():
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, -2.0]])
    result = target_registration_error(pts, pts)
    assert result["mean_tre"] == 0.0
    assert result["max_tre"] == 0.0


def test_tre_with_known_errors():
    moving = np.array([[1.0, 0.0], [0.0, 2.0]])
    fixed = np.zeros((2, 2))
    result = target_registration_error(moving, fixed)
    assert result["mean_tre"] == pytest.approx(1.5)
    assert result["max_tre"] == pytest.approx(2.0)


def test_tre_applies_matrix_first():
    moving = np.array([[0.0, 0.0], [1.0, 0.0]])
    shift = np.array([[1.0, 0.0, 5.0],
                      [0.0, 1.0, -3.0]])
    fixed = np.array([[5.0, -3.0], [6.0, -3.0]])
    result = target_registration_error(moving, fixed, matrix=shift)
    assert result["mean_tre"] == pytest.approx(0.0)


def test_tre_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        target_registration_error(np.zeros((3, 2)), np.zeros((4, 2)))


def test_dice_identical_masks():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:3, 1:3] = True
    assert dice_coefficient(mask, mask) == 1.0


def test_dice_disjoint_masks():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[0, 0] = True
    b[3, 3] = True
    assert dice_coefficient(a, b) == 0.0


def test_dice_half_overlap():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[:2, :] = True
    b[1:3, :] = True
    # |a|=8, |b|=8, intersection=4 → 2*4/16 = 0.5
    assert dice_coefficient(a, b) == pytest.approx(0.5)


def test_dice_two_empty_masks_is_perfect_match():
    assert dice_coefficient(np.zeros((3, 3)), np.zeros((3, 3))) == 1.0


def test_dice_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        dice_coefficient(np.zeros((2, 2)), np.zeros((3, 3)))
