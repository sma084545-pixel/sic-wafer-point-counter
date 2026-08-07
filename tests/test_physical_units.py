"""Physical-unit calibration and final-mask boundary-distance coverage."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sic_wafer_counter.density import calculate_density
from sic_wafer_counter.feature_extraction import (
    extract_candidate_features,
    valid_boundary_distance_transform,
)
from sic_wafer_counter.physical_parameters import resolve_physical_parameters
from sic_wafer_counter.image_io import ImageTile
from sic_wafer_counter.pipeline import _exact_tile_valid_boundary_distance
from sic_wafer_counter.wafer_detection import (
    WaferGeometry,
    build_analysis_masks,
    build_valid_boundary_index,
)


def _physical_config() -> dict[str, object]:
    """Rules intentionally specified in physical units for resolution tests."""

    return {
        "preprocessing": {
            "background_kernel_px": 101,
            "background_kernel_um": 1000.0,
            "gaussian_sigma": 1.0,
            "gaussian_sigma_um": 50.0,
        },
        "detection": {
            "blackhat_kernel_sizes_px": [9, 17, 31],
            "blackhat_kernel_sizes_um": [1000.0, 1500.0],
            "min_peak_distance_px": 5,
            "min_peak_distance_um": 500.0,
            "dog_min_sigma_px": 1.0,
            "dog_min_sigma_um": 100.0,
            "dog_max_sigma_px": 6.0,
            "dog_max_sigma_um": 500.0,
        },
        "filters": {
            "min_area_px": 5,
            "max_area_px": 1500,
            "min_area_um2": 100_000.0,
            "max_area_um2": 3_000_000.0,
            "min_equivalent_diameter_px": 2.0,
            "max_equivalent_diameter_px": 45.0,
            "min_equivalent_diameter_um": 750.0,
            "max_equivalent_diameter_um": 1700.0,
            "local_background_ring_px": 5,
            "local_background_ring_um": 400.0,
            "min_edge_distance_mm": 0.0,
            "min_edge_distance_um": 0.0,
            "min_circularity": 0.0,
            "max_eccentricity": 1.0,
            "max_aspect_ratio": 100.0,
            "min_solidity": 0.0,
            "min_contrast": 0.0,
            "min_valid_fraction": 1.0,
        },
    }


def _render_same_physical_scene(mm_per_pixel: float):
    """Render three 1.2 mm dots on one 100 mm physical wafer."""

    radius_px = int(round(50.0 / mm_per_pixel))
    side = radius_px * 2 + 5
    center = radius_px + 2
    geometry = WaferGeometry(
        center_x=float(center),
        center_y=float(center),
        radius_px=float(radius_px),
        image_width=side,
        image_height=side,
        diameter_mm=100.0,
    )
    valid = np.zeros((side, side), dtype=np.uint8)
    cv2.circle(valid, (center, center), radius_px, 1, thickness=-1)
    labels = np.zeros_like(valid, dtype=np.int32)
    dot_radius_px = max(1, int(round(0.6 / mm_per_pixel)))
    for identifier, (x_mm, y_mm) in enumerate(((0.0, 0.0), (10.0, 0.0), (-10.0, 8.0)), 1):
        x = int(round(center + x_mm / mm_per_pixel))
        y = int(round(center - y_mm / mm_per_pixel))
        cv2.circle(labels, (x, y), dot_radius_px, identifier, thickness=-1)
    raw = np.full(labels.shape, 0.4, dtype=np.float32)
    background = np.full(labels.shape, 0.6, dtype=np.float32)
    return geometry, labels, raw, background, valid.astype(bool)


def test_physical_rules_are_resolution_consistent() -> None:
    """The same physical scene keeps n, rho, and physical diameters stable."""

    counts: list[int] = []
    densities: list[float] = []
    diameters: list[float] = []
    kernel_values: list[list[int]] = []
    for mm_per_pixel in (0.2, 0.1, 0.05):
        geometry, labels, raw, background, valid = _render_same_physical_scene(mm_per_pixel)
        resolution = resolve_physical_parameters(
            _physical_config(), mm_per_pixel=geometry.mm_per_pixel
        )
        assert resolution.warnings  # Physical values intentionally override px defaults.
        assert resolution.report["um_per_pixel"] == pytest.approx(mm_per_pixel * 1000.0)
        kernel_values.append(
            list(resolution.config["detection"]["blackhat_kernel_sizes_px"])
        )
        features = extract_candidate_features(
            labels,
            raw,
            background=background,
            valid_mask=valid,
            config=resolution.config,
            geometry=geometry,
        )
        accepted = [feature for feature in features if feature.accepted]
        counts.append(len(accepted))
        densities.append(
            calculate_density(
                len(accepted), float(valid.sum()) * geometry.pixel_area_cm2
            ).density_cm2
        )
        diameters.extend(feature.equivalent_diameter_mm for feature in accepted)
        for feature in accepted:
            assert feature.equivalent_diameter_um == pytest.approx(
                feature.equivalent_diameter_px * geometry.um_per_pixel
            )
            assert feature.major_axis_length_um == pytest.approx(
                feature.major_axis_length_px * geometry.um_per_pixel
            )
            assert feature.minor_axis_length_um == pytest.approx(
                feature.minor_axis_length_px * geometry.um_per_pixel
            )

    assert counts == [3, 3, 3]
    assert max(densities) - min(densities) < 0.002
    assert max(diameters) - min(diameters) < 0.04
    assert kernel_values[0][0] < kernel_values[1][0] < kernel_values[2][0]


def test_legacy_pixel_configuration_remains_supported() -> None:
    config = _physical_config()
    for section in ("preprocessing", "detection", "filters"):
        for key in list(config[section]):
            if key.endswith("_um") or key.endswith("_um2"):
                del config[section][key]
    resolution = resolve_physical_parameters(config, mm_per_pixel=0.1)
    assert not resolution.warnings
    assert resolution.config == config
    assert all(
        item["source"].startswith("legacy")
        for item in resolution.report["parameters"].values()
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0])
def test_invalid_physical_size_is_rejected(bad_value: float) -> None:
    config = _physical_config()
    config["filters"]["min_area_um2"] = bad_value
    with pytest.raises(ValueError, match="finite"):
        resolve_physical_parameters(config, mm_per_pixel=0.1)


def test_final_valid_boundary_distance_beats_fitted_circle_for_notch() -> None:
    """A notch/invalid boundary is closer than the ideal fitted-circle edge."""

    geometry = WaferGeometry(
        center_x=100.0,
        center_y=100.0,
        radius_px=80.0,
        image_width=201,
        image_height=201,
        diameter_mm=100.0,
    )
    valid = np.zeros((201, 201), dtype=np.uint8)
    cv2.circle(valid, (100, 100), 80, 1, thickness=-1)
    valid[80:121, 145:181] = 0  # Notch or manually invalidated interior region.
    labels = np.zeros_like(valid, dtype=np.int32)
    cv2.circle(labels, (140, 100), 3, 1, thickness=-1)
    config = {
        "filters": {
            "min_area_px": 1,
            "max_area_px": 1000,
            "min_equivalent_diameter_px": 0.0,
            "max_equivalent_diameter_px": 100.0,
            "min_circularity": 0.0,
            "max_eccentricity": 1.0,
            "max_aspect_ratio": 100.0,
            "min_solidity": 0.0,
            "min_contrast": 0.0,
            "min_valid_fraction": 1.0,
            "local_background_ring_px": 2,
            "min_edge_distance_mm": 5.0,
        }
    }
    features = extract_candidate_features(
        labels,
        np.full(valid.shape, 0.4, dtype=np.float32),
        background=np.full(valid.shape, 0.6, dtype=np.float32),
        valid_mask=valid.astype(bool),
        config=config,
        geometry=geometry,
    )
    feature = features[0]
    assert feature.distance_to_valid_boundary_mm < feature.distance_to_fitted_circle_mm
    assert "near_wafer_edge" in feature.rejection_reason
    assert feature.distance_to_wafer_edge_mm == feature.distance_to_fitted_circle_mm


def test_tiled_boundary_distance_expands_beyond_tile_edges() -> None:
    """A central tile reports the wafer boundary, not its artificial tile edge."""

    geometry = WaferGeometry(
        center_x=250.0, center_y=250.0, radius_px=200.0,
        image_width=501, image_height=501, diameter_mm=100.0,
    )
    tile = ImageTile(
        image=np.zeros((80, 80), dtype=np.float32),
        x=210, y=210, width=80, height=80,
        core_x=210, core_y=210, core_width=80, core_height=80,
    )
    labels = np.zeros((80, 80), dtype=np.int32)
    labels[40, 40] = 1
    distance = _exact_tile_valid_boundary_distance(
        geometry, tile, labels, exclude_edge_mm=0.0, invalid_regions=[],
        invalid_mask=None, initial_margin_px=16,
    )
    full = build_analysis_masks(geometry).valid_analysis_mask
    expected = cv2.distanceTransform(full.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    assert distance[40, 40] == pytest.approx(expected[250, 250])
    assert distance[40, 40] > 100.0


def test_boundary_index_matches_notch_invalid_edge_and_center_distances() -> None:
    """The sparse index exactly reproduces the final raster-mask EDT."""

    contour_with_notch = [
        (10, 10),
        (290, 10),
        (290, 120),
        (240, 120),
        (240, 180),
        (290, 180),
        (290, 290),
        (10, 290),
    ]
    geometry = WaferGeometry(
        center_x=150.0,
        center_y=150.0,
        radius_px=140.0,
        image_width=301,
        image_height=301,
        diameter_mm=100.0,
        contour_polygon=contour_with_notch,
    )
    invalid_regions = [
        {
            "type": "polygon",
            "points": [(95, 190), (135, 185), (140, 215), (100, 220)],
        }
    ]
    masks = build_analysis_masks(
        geometry,
        exclude_edge_mm=2.0,
        invalid_regions=invalid_regions,
    )
    padded = np.pad(masks.valid_analysis_mask.astype(np.uint8), 1)
    expected = cv2.distanceTransform(
        padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1]
    index = build_valid_boundary_index(
        geometry,
        tile_size=48,
        exclude_edge_mm=2.0,
        invalid_regions=invalid_regions,
    )
    valid_coordinates = np.argwhere(masks.valid_analysis_mask)
    sampled = valid_coordinates[:: max(1, len(valid_coordinates) // 250)]
    center_coordinate = valid_coordinates[
        np.argmin(
            (valid_coordinates[:, 0] - geometry.center_y) ** 2
            + (valid_coordinates[:, 1] - geometry.center_x) ** 2
        )
    ][None, :]
    coordinates = np.concatenate((sampled, center_coordinate), axis=0)
    actual = index.query_yx(coordinates)
    assert actual == pytest.approx(
        expected[coordinates[:, 0], coordinates[:, 1]], abs=1e-6
    )


def test_full_and_tiled_boundary_distance_match_at_cropped_image_edge() -> None:
    """The image exterior is a valid-analysis boundary in both code paths."""

    geometry = WaferGeometry(
        center_x=20.0,
        center_y=50.0,
        radius_px=50.0,
        image_width=101,
        image_height=101,
        diameter_mm=100.0,
    )
    masks = build_analysis_masks(geometry)
    padded = np.pad(masks.valid_analysis_mask.astype(np.uint8), 1)
    expected = cv2.distanceTransform(
        padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1]

    full_distance = valid_boundary_distance_transform(masks.valid_analysis_mask)
    assert full_distance == pytest.approx(expected, abs=1e-6)
    assert full_distance[50, 0] == pytest.approx(1.0)

    index = build_valid_boundary_index(geometry, tile_size=32)
    coordinates = np.argwhere(masks.valid_analysis_mask)
    assert index.query_yx(coordinates) == pytest.approx(
        expected[coordinates[:, 0], coordinates[:, 1]], abs=1e-6
    )


def test_boundary_index_uses_only_combined_boundary_for_overlapping_regions() -> None:
    """Overlapping exclusions cannot leave false internal index boundaries."""

    geometry = WaferGeometry(
        center_x=100.0,
        center_y=100.0,
        radius_px=92.0,
        image_width=201,
        image_height=201,
        diameter_mm=100.0,
    )
    invalid_regions = [
        # The rectangle overlaps both the polygon and the excluded wafer edge.
        {"type": "rectangle", "x": 118, "y": 55, "width": 80, "height": 90},
        {
            "type": "polygon",
            "points": [(90, 75), (170, 65), (180, 135), (95, 145)],
        },
        # A second polygon hides part of both earlier analytic boundaries.
        {
            "type": "polygon",
            "points": [(105, 45), (155, 50), (150, 165), (100, 155)],
        },
    ]
    masks = build_analysis_masks(
        geometry,
        exclude_edge_mm=2.0,
        invalid_regions=invalid_regions,
    )
    padded = np.pad(masks.valid_analysis_mask.astype(np.uint8), 1)
    expected = cv2.distanceTransform(
        padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1]
    index = build_valid_boundary_index(
        geometry,
        tile_size=29,
        exclude_edge_mm=2.0,
        invalid_regions=invalid_regions,
    )
    coordinates = np.argwhere(masks.valid_analysis_mask)
    actual = index.query_yx(coordinates)
    assert actual == pytest.approx(
        expected[coordinates[:, 0], coordinates[:, 1]], abs=1e-6
    )


def test_boundary_index_matches_slanted_concave_contour_raster() -> None:
    """Tile clipping cannot alter the full OpenCV contour boundary."""

    contour = [
        (12, 30),
        (82, 8),
        (146, 20),
        (190, 74),
        (164, 112),
        (190, 184),
        (102, 192),
        (22, 169),
        (46, 111),
        (9, 79),
    ]
    geometry = WaferGeometry(
        center_x=100.0,
        center_y=100.0,
        radius_px=94.0,
        image_width=201,
        image_height=201,
        diameter_mm=100.0,
        contour_polygon=contour,
    )
    masks = build_analysis_masks(geometry, exclude_edge_mm=1.5)
    padded = np.pad(masks.valid_analysis_mask.astype(np.uint8), 1)
    expected = cv2.distanceTransform(
        padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1]
    index = build_valid_boundary_index(
        geometry,
        tile_size=23,
        exclude_edge_mm=1.5,
    )
    coordinates = np.argwhere(masks.valid_analysis_mask)
    assert index.query_yx(coordinates) == pytest.approx(
        expected[coordinates[:, 0], coordinates[:, 1]], abs=1e-6
    )


def test_boundary_index_matches_supplied_raster_invalid_mask() -> None:
    """Arbitrary raster holes retain exact final-mask distance semantics."""

    geometry = WaferGeometry(
        center_x=64.0,
        center_y=64.0,
        radius_px=55.0,
        image_width=129,
        image_height=129,
        diameter_mm=100.0,
    )
    invalid_mask = np.zeros((129, 129), dtype=bool)
    invalid_mask[55:75, 60:63] = True
    invalid_mask[31:40, 80:95] = True
    masks = build_analysis_masks(geometry, invalid_mask=invalid_mask)
    padded = np.pad(masks.valid_analysis_mask.astype(np.uint8), 1)
    expected = cv2.distanceTransform(
        padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1]
    index = build_valid_boundary_index(
        geometry,
        tile_size=31,
        invalid_mask=invalid_mask,
    )
    coordinates = np.argwhere(masks.valid_analysis_mask)
    assert index.query_yx(coordinates) == pytest.approx(
        expected[coordinates[:, 0], coordinates[:, 1]], abs=1e-6
    )


def test_boundary_index_mask_working_set_is_tile_bounded() -> None:
    """A central candidate no longer grows a distance raster to full-wafer size."""

    geometry = WaferGeometry(
        center_x=512.0,
        center_y=512.0,
        radius_px=500.0,
        image_width=1025,
        image_height=1025,
        diameter_mm=100.0,
    )
    scan_tile_size = 64
    exclude_edge_mm = 1.0
    index = build_valid_boundary_index(
        geometry,
        tile_size=scan_tile_size,
        exclude_edge_mm=exclude_edge_mm,
    )
    edge_width_px = exclude_edge_mm / geometry.mm_per_pixel
    edge_halo = int(np.ceil(edge_width_px)) + 2
    upper_bound = (scan_tile_size + 2 + 2 * edge_halo) ** 2
    assert index.max_mask_tile_pixels <= upper_bound
    assert index.max_mask_tile_pixels < geometry.image_width * geometry.image_height / 50

    masks = build_analysis_masks(geometry, exclude_edge_mm=exclude_edge_mm)
    padded = np.pad(masks.valid_analysis_mask.astype(np.uint8), 1)
    expected = cv2.distanceTransform(
        padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1]
    center = np.asarray([[512, 512]])
    assert index.query_yx(center)[0] == pytest.approx(expected[512, 512])


def test_pathological_raster_boundary_is_explicitly_rejected() -> None:
    """A high-frequency raster mask cannot silently exceed the tree budget."""

    geometry = WaferGeometry(
        center_x=64.0,
        center_y=64.0,
        radius_px=60.0,
        image_width=129,
        image_height=129,
        diameter_mm=100.0,
    )
    rows, cols = np.indices((129, 129))
    checkerboard = (rows + cols) % 2 == 0
    with pytest.raises(RuntimeError, match="boundary is too complex"):
        build_valid_boundary_index(
            geometry,
            tile_size=32,
            invalid_mask=checkerboard,
            max_boundary_points=1_000,
        )



def test_rigaku_reference_profile_is_diagnostic_and_warns_when_under_sampled() -> None:
    from sic_wafer_counter.pipeline import _reference_profile_report

    report, warnings = _reference_profile_report(
        {
            "reference_profile": {
                "enabled": True,
                "imaging_conditions_confirmed": True,
                "expected_major_axis_um": 50.0,
                "expected_minor_axis_um": 30.0,
                "minimum_axis_pixels": 3.0,
            }
        },
        um_per_pixel=12.0,
    )
    assert report["expected_major_axis_px"] == pytest.approx(50.0 / 12.0)
    assert report["expected_minor_axis_px"] == pytest.approx(2.5)
    assert report["status"] == "below_stable_minor_axis_sampling"
    assert report["classification_gating"] is False
    assert warnings and "not reliable" in warnings[0]

    unconfirmed, unconfirmed_warnings = _reference_profile_report(
        {
            "reference_profile": {
                "enabled": True,
                "imaging_conditions_confirmed": False,
            }
        },
        um_per_pixel=5.0,
    )
    assert unconfirmed["status"] == "diagnostic_only_imaging_conditions_unconfirmed"
    assert unconfirmed_warnings
