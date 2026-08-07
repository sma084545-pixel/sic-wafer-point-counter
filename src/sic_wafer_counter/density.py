"""Physical area and point-density calculations.

Only counting (Poisson) uncertainty is represented here.  Segmentation,
classification and physical-identification errors are systematic effects and
must be assessed separately with expert-labelled data.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
from scipy.stats import chi2


COUNTING_UNCERTAINTY_NOTE = (
    "该统计不确定度只反映有限计数造成的随机误差，不包含图像分割、漏检、"
    "误检以及物理判定错误造成的系统误差。"
)


@dataclasses.dataclass(frozen=True, slots=True)
class DensityResult:
    """Result of a point-count divided by a valid physical area."""

    count: int
    area_cm2: float
    density_cm2: float
    standard_uncertainty_cm2: float
    confidence_level: float
    count_ci_lower: float
    count_ci_upper: float
    density_ci_lower_cm2: float
    density_ci_upper_cm2: float
    unit: str = "cm^-2"
    uncertainty_note: str = COUNTING_UNCERTAINTY_NOTE

    @property
    def n(self) -> int:
        """Alias for the accepted point count."""

        return self.count

    @property
    def rho(self) -> float:
        """Alias for density in ``cm^-2``."""

        return self.density_cm2

    @property
    def sigma_rho(self) -> float:
        """Alias for the counting standard uncertainty in ``cm^-2``."""

        return self.standard_uncertainty_cm2

    @property
    def poisson_95_ci_cm2(self) -> tuple[float, float]:
        """Return the density interval (named for the default 95% use case)."""

        return self.density_ci_lower_cm2, self.density_ci_upper_cm2

    def to_dict(self) -> dict[str, Any]:
        """Return stable report field names and compatibility aliases."""

        return {
            "count": self.count,
            "n": self.count,
            "valid_area_cm2": self.area_cm2,
            "density_cm2": self.density_cm2,
            "rho_cm2": self.density_cm2,
            "counting_uncertainty_cm2": self.standard_uncertainty_cm2,
            "sigma_rho_cm2": self.standard_uncertainty_cm2,
            "confidence_level": self.confidence_level,
            "poisson_count_ci_lower": self.count_ci_lower,
            "poisson_count_ci_upper": self.count_ci_upper,
            "poisson_density_ci_lower_cm2": self.density_ci_lower_cm2,
            "poisson_density_ci_upper_cm2": self.density_ci_upper_cm2,
            "unit": self.unit,
            "uncertainty_note": self.uncertainty_note,
        }


def _validate_count(count: int) -> int:
    if isinstance(count, (bool, np.bool_)):
        raise TypeError("count must be a non-negative integer, not bool")
    try:
        integer = int(count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"count must be a non-negative integer, got {count!r}") from exc
    if integer != count or integer < 0:
        raise ValueError(f"count must be a non-negative integer, got {count!r}")
    return integer


def _validate_area(area_cm2: float) -> float:
    try:
        area = float(area_cm2)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"area_cm2 must be a positive finite number, got {area_cm2!r}") from exc
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError(f"area_cm2 must be positive and finite, got {area_cm2!r}")
    return area


def poisson_count_interval(
    count: int, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Return a two-sided exact Garwood interval for a Poisson mean.

    For ``count == 0`` the lower bound is exactly zero and the finite upper
    bound is retained (approximately 3.689 counts at 95% confidence).
    """

    n = _validate_count(count)
    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not (0.0 < confidence < 1.0):
        raise ValueError(
            f"confidence_level must be strictly between 0 and 1, got {confidence_level!r}"
        )
    alpha = 1.0 - confidence
    lower = 0.0 if n == 0 else 0.5 * float(chi2.ppf(alpha / 2.0, 2 * n))
    upper = 0.5 * float(chi2.ppf(1.0 - alpha / 2.0, 2 * (n + 1)))
    return lower, upper


def calculate_density(
    count: int,
    area_cm2: float,
    *,
    confidence_level: float = 0.95,
) -> DensityResult:
    """Calculate ``rho = n / S`` and counting uncertainty.

    Parameters
    ----------
    count:
        Accepted point-like target count ``n``.
    area_cm2:
        Pixel-mask-derived valid analysis area ``S`` in square centimetres.
    confidence_level:
        Coverage used for the exact Garwood count interval.
    """

    n = _validate_count(count)
    area = _validate_area(area_cm2)
    count_lower, count_upper = poisson_count_interval(n, confidence_level)
    density = n / area
    uncertainty = math.sqrt(n) / area  # Correctly returns zero for n == 0.
    return DensityResult(
        count=n,
        area_cm2=area,
        density_cm2=density,
        standard_uncertainty_cm2=uncertainty,
        confidence_level=float(confidence_level),
        count_ci_lower=count_lower,
        count_ci_upper=count_upper,
        density_ci_lower_cm2=count_lower / area,
        density_ci_upper_cm2=count_upper / area,
    )


def calculate_mask_area_cm2(mask: np.ndarray, cm_per_pixel: float) -> float:
    """Return physical area from the number of true pixels in a 2-D mask."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {array.shape}")
    scale = float(cm_per_pixel)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"cm_per_pixel must be positive and finite, got {cm_per_pixel!r}"
        )
    return float(np.count_nonzero(array)) * scale * scale


# Backwards-friendly aliases used by scripts and notebooks.
compute_density = calculate_density
calculate_point_density = calculate_density


__all__ = [
    "COUNTING_UNCERTAINTY_NOTE",
    "DensityResult",
    "calculate_density",
    "calculate_mask_area_cm2",
    "calculate_point_density",
    "compute_density",
    "poisson_count_interval",
]
