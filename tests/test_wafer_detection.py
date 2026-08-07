"""Wafer geometry, masks, calibration and failure-mode coverage."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sic_wafer_counter.feature_extraction import pixel_to_wafer_coordinates
from sic_wafer_counter.wafer_detection import (
    WaferDetectionError,
    WaferGeometry,
    build_analysis_masks,
    calculate_area_statistics,
    detect_wafer,
)


def test_valid_mask_area_and_invalid_region_reconcile() -> None:
    geometry = WaferGeometry(
        center_x=500.0,
        center_y=500.0,
        radius_px=450.0,
        image_width=1001,
        image_height=1001,
        diameter_mm=100.0,
    )
    masks = build_analysis_masks(
        geometry,
        exclude_edge_mm=1.0,
        invalid_regions=[{"type": "rectangle", "x": 475, "y": 475, "width": 50, "height": 50}],
    )
    area = calculate_area_statistics(geometry, masks=masks)
    assert area.full_wafer_pixel_count == (
        area.edge_excluded_pixel_count + area.invalid_pixel_count + area.valid_pixel_count
    )
    assert area.edge_excluded_area_cm2 > 0.0
    assert area.invalid_area_cm2 > 0.0
    assert area.valid_area_cm2 < area.full_wafer_area_cm2
    assert area.theoretical_area_cm2 == pytest.approx(78.5398163397, rel=1e-10)


def test_image_coordinate_to_mm_coordinate_conversion() -> None:
    x_mm, y_mm, radial, angle = pixel_to_wafer_coordinates(
        600.0, 400.0, 500.0, 500.0, 0.1
    )
    assert x_mm == pytest.approx(10.0)
    assert y_mm == pytest.approx(10.0)
    assert radial == pytest.approx(np.sqrt(200.0))
    assert angle == pytest.approx(45.0)


def test_auto_detects_synthetic_wafer(generate_synthetic) -> None:
    generated = generate_synthetic("clean")
    preview = cv2.imread(str(generated["image_path"]), cv2.IMREAD_GRAYSCALE)
    geometry = detect_wafer(preview, full_shape=preview.shape, diameter_mm=100.0)
    assert geometry.confidence > 0.9
    assert geometry.center_x == pytest.approx(515.0, abs=2.0)
    assert geometry.center_y == pytest.approx(507.0, abs=2.0)
    assert geometry.radius_px == pytest.approx(448.0, abs=2.0)


def test_partial_image_refuses_automatic_density_geometry() -> None:
    partial = np.full((256, 256), 180, dtype=np.uint8)
    with pytest.raises(WaferDetectionError):
        detect_wafer(partial, full_shape=partial.shape, diameter_mm=100.0)

