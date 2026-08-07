"""Area-normalized radial, angular, and regional density tests."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from sic_wafer_counter.density import calculate_density
from sic_wafer_counter.visualization import (
    calculate_angular_density,
    calculate_radial_density,
    calculate_regional_density,
    save_area_normalized_distributions,
)


def _mask_and_points(seed: int = 17) -> tuple[np.ndarray, pd.DataFrame, tuple[float, float], float, float]:
    center = (120.0, 120.0)
    mm_per_pixel, radius_px = 0.25, 100
    valid = np.zeros((241, 241), dtype=np.uint8)
    cv2.circle(valid, (120, 120), radius_px, 1, thickness=-1)
    # Simulate both an invalid notch/flat feature and an excluded outer edge.
    valid[95:146, 195:221] = 0
    inner = np.zeros_like(valid)
    cv2.circle(inner, (120, 120), 92, 1, thickness=-1)
    valid &= inner
    rows, cols = np.nonzero(valid)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(rows), size=5000, replace=False)
    y, x = rows[selected], cols[selected]
    x_mm = (x - center[0]) * mm_per_pixel
    y_mm = -(y - center[1]) * mm_per_pixel
    frame = pd.DataFrame(
        {
            "accepted": True,
            "x_mm": x_mm,
            "y_mm": y_mm,
            "radial_distance_mm": np.hypot(x_mm, y_mm),
            "polar_angle_deg": np.degrees(np.arctan2(y_mm, x_mm)) % 360.0,
        }
    )
    return valid.astype(bool), frame, center, mm_per_pixel, radius_px * mm_per_pixel


def test_radial_and_angular_area_conservation_with_notch_invalid_and_edge() -> None:
    valid, frame, center, mm_per_pixel, radius_mm = _mask_and_points()
    pixel_area = (mm_per_pixel / 10.0) ** 2
    radial = calculate_radial_density(
        frame, valid_mask=valid, center_px=center, mm_per_pixel=mm_per_pixel,
        wafer_radius_mm=radius_mm, bins=6, mode="equal_area",
    )
    angular = calculate_angular_density(
        frame, valid_mask=valid, center_px=center, mm_per_pixel=mm_per_pixel, sectors=12,
    )
    expected_area = valid.sum() * pixel_area
    assert radial["valid_area_cm2"].sum() == pytest.approx(expected_area)
    assert angular["valid_area_cm2"].sum() == pytest.approx(expected_area)
    assert set(("r_inner_mm", "r_outer_mm", "valid_area_cm2", "count", "density_cm2", "poisson_lower_cm2", "poisson_upper_cm2")) <= set(radial.columns)
    assert angular["angle_reference"].eq("image_positive_x").all()


def test_uniform_random_points_have_stable_area_normalized_radial_density() -> None:
    valid, frame, center, mm_per_pixel, radius_mm = _mask_and_points(seed=31)
    radial = calculate_radial_density(
        frame, valid_mask=valid, center_px=center, mm_per_pixel=mm_per_pixel,
        wafer_radius_mm=radius_mm, bins=5, mode="equal_area",
    )
    finite = radial["density_cm2"].dropna()
    assert finite.max() / finite.min() < 1.3
    # Raw counts can differ modestly even when density is statistically stable.
    assert radial["count"].max() - radial["count"].min() > 0


def test_zero_valid_area_sector_is_na_and_total_density_is_not_mutated() -> None:
    valid = np.zeros((121, 121), dtype=np.uint8)
    cv2.circle(valid, (60, 60), 50, 1, thickness=-1)
    valid[:, :60] = 0  # Left-side angular sectors have no valid analysis area.
    frame = pd.DataFrame(columns=["accepted", "radial_distance_mm", "polar_angle_deg"])
    before = calculate_density(0, valid.sum() * 0.01**2)
    angular = calculate_angular_density(
        frame, valid_mask=valid.astype(bool), center_px=(60.0, 60.0),
        mm_per_pixel=0.1, sectors=8,
    )
    after = calculate_density(0, valid.sum() * 0.01**2)
    assert angular["valid_area_cm2"].eq(0.0).any()
    assert angular.loc[angular["valid_area_cm2"] == 0.0, "density_cm2"].isna().all()
    assert before == after


def test_regional_density_and_output_artifacts(tmp_path) -> None:
    valid, frame, center, mm_per_pixel, radius_mm = _mask_and_points()
    regional = calculate_regional_density(
        frame, valid_mask=valid, center_px=center, mm_per_pixel=mm_per_pixel,
        wafer_radius_mm=radius_mm,
    )
    assert regional["region"].tolist() == ["center", "middle", "edge"]
    assert regional["valid_area_cm2"].sum() == pytest.approx(valid.sum() * (mm_per_pixel / 10.0) ** 2)
    paths, _ = save_area_normalized_distributions(
        frame, tmp_path, valid_mask=valid, center_px=center, mm_per_pixel=mm_per_pixel,
        wafer_radius_mm=radius_mm,
    )
    for path in paths.values():
        assert path.is_file()
