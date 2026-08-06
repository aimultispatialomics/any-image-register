"""Registration accuracy evaluation: landmark TRE and region Dice."""

import numpy as np


def target_registration_error(moving_landmarks, fixed_landmarks, matrix=None):
    """Mean and max target registration error (TRE) in pixels.

    If `matrix` is given, moving landmarks are transformed first; pass
    matrix=None to evaluate already-transformed landmarks.
    """
    moving = np.asarray(moving_landmarks, dtype=np.float64)
    fixed = np.asarray(fixed_landmarks, dtype=np.float64)
    if moving.shape != fixed.shape:
        raise ValueError(f"shape mismatch: {moving.shape} vs {fixed.shape}")
    if matrix is not None:
        hom = np.hstack([moving, np.ones((len(moving), 1))])
        moving = hom @ np.asarray(matrix, dtype=np.float64).T
    errors = np.linalg.norm(moving - fixed, axis=1)
    return {"mean_tre": float(errors.mean()), "max_tre": float(errors.max())}


def dice_coefficient(mask_a, mask_b):
    """Dice overlap between two boolean region masks.

    Returns 1.0 for identical non-empty masks and 0.0 for disjoint ones;
    two empty masks are defined as a perfect match.
    """
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * (a & b).sum() / denom)
