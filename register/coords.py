"""Map spatial barcode / spot coordinates through an image transform."""

import numpy as np

from .transforms import apply_transform


def map_spot_coordinates(spots, matrix):
    """Apply the image-space transform to (N, 2) spot pixel coordinates.

    `spots` may be an array of [x, y] columns or a DataFrame-like with
    'x'/'y' columns; the transformed coordinates are returned as an
    (N, 2) float array aligned with the input order.
    """
    if hasattr(spots, "columns"):
        pts = spots[["x", "y"]].to_numpy(dtype=np.float64)
    else:
        pts = np.asarray(spots, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("spots must be (N, 2) pixel coordinates")
    return apply_transform(pts, matrix)


def pixels_to_physical(pixels, pixel_size_um):
    """Convert pixel coordinates to physical microns for a uniform pitch."""
    return np.asarray(pixels, dtype=np.float64) * float(pixel_size_um)


def physical_to_pixels(coords_um, pixel_size_um):
    """Inverse of pixels_to_physical."""
    return np.asarray(coords_um, dtype=np.float64) / float(pixel_size_um)
