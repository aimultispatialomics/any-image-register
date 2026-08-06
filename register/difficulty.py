"""Registration difficulty index (1-5 stars), as documented in the README."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultyFactors:
    """The four factors that drive registration difficulty."""

    modality_gap: int = 0      # 0 = same modality, 1 = related, 2 = cross-modality
    deformation: int = 0       # 0 = rigid, 1 = affine, 2 = non-rigid
    resolution_gap: int = 0    # 0 = same, 1 = <=4x, 2 = >4x or tiled
    tissue_artifacts: int = 0  # 0 = clean, 1 = mild, 2 = folds/tears/bubbles

    def __post_init__(self):
        for name, value in vars(self).items():
            if not 0 <= value <= 2:
                raise ValueError(f"{name} must be 0, 1 or 2, got {value}")


def difficulty_index(factors: DifficultyFactors) -> int:
    """Combine the four factors into a 1-5 star difficulty index.

    Every factor beyond a plain rigid, same-modality alignment adds
    roughly one star, matching the rule of thumb in the README.
    """
    total = sum(vars(factors).values())
    return min(5, 1 + (total + 1) // 2)


def suggested_transform(factors: DifficultyFactors) -> str:
    """Return the transform family typically needed for these factors."""
    if factors.deformation == 2:
        return "non-rigid (TPS / deformation field)"
    if factors.resolution_gap == 2:
        return "affine + pyramid"
    if factors.deformation == 1 or factors.resolution_gap == 1:
        return "affine"
    return "rigid"


def describe(factors: DifficultyFactors) -> str:
    stars = "★" * difficulty_index(factors)
    return f"{stars}  suggested transform: {suggested_transform(factors)}"
