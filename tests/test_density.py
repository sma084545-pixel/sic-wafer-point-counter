"""Unit tests for units, area and Poisson counting calculations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sic_wafer_counter.density import calculate_density, calculate_mask_area_cm2
from sic_wafer_counter.wafer_detection import pixel_scale_from_diameter, theoretical_wafer_area_cm2


def test_100_mm_is_10_cm_and_complete_area_is_correct() -> None:
    assert 100.0 / 10.0 == 10.0
    assert theoretical_wafer_area_cm2(100.0) == pytest.approx(math.pi * 25.0, rel=1e-12)
    assert theoretical_wafer_area_cm2(100.0) == pytest.approx(78.5398163397, rel=1e-10)


def test_pixel_calibration_and_mask_area() -> None:
    mm_per_pixel, cm_per_pixel, pixel_area = pixel_scale_from_diameter(100.0, 1000.0)
    assert mm_per_pixel == pytest.approx(0.1)
    assert cm_per_pixel == pytest.approx(0.01)
    assert pixel_area == pytest.approx(1.0e-4)
    mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    assert calculate_mask_area_cm2(mask, cm_per_pixel) == pytest.approx(4.0e-4)


def test_density_and_zero_count_poisson_interval() -> None:
    result = calculate_density(25, 12.5)
    assert result.density_cm2 == pytest.approx(2.0)
    assert result.standard_uncertainty_cm2 == pytest.approx(0.4)
    assert result.density_ci_lower_cm2 < result.density_cm2 < result.density_ci_upper_cm2

    empty = calculate_density(0, 10.0)
    assert empty.density_cm2 == 0.0
    assert empty.standard_uncertainty_cm2 == 0.0
    assert empty.count_ci_lower == 0.0
    assert empty.count_ci_upper == pytest.approx(3.688879, rel=1e-5)
    assert empty.density_ci_upper_cm2 > 0.0

