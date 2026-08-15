"""Candidate detection tests for lines, edge exclusion and external pixels."""

from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest

from sic_wafer_counter.feature_extraction import extract_candidate_features
from sic_wafer_counter.pipeline import analyze_image
from sic_wafer_counter.point_detection import (
    DetectionConfig,
    _conservative_watershed,
    detect_candidates,
)
from sic_wafer_counter.preprocessing import preprocess_image
from skimage.measure import label


def test_fast_watershed_splits_touching_points_but_skips_ordinary_small_blob() -> None:
    touching = np.zeros((41, 41), dtype=np.uint8)
    cv2.circle(touching, (16, 20), 7, 1, thickness=-1)
    cv2.circle(touching, (25, 20), 7, 1, thickness=-1)
    config = DetectionConfig(
        min_area_px=5,
        max_area_px=500,
        use_watershed=True,
        min_peak_distance_px=5,
        watershed_min_component_area_px=20,
    ).validated()
    split = _conservative_watershed(
        label(touching.astype(bool), connectivity=2).astype(np.int32), config
    )
    assert int(split.max()) == 2

    ordinary = np.zeros((15, 15), dtype=np.uint8)
    cv2.circle(ordinary, (7, 7), 2, 1, thickness=-1)
    unchanged = _conservative_watershed(
        label(ordinary.astype(bool), connectivity=2).astype(np.int32), config
    )
    assert int(unchanged.max()) == 1


def test_line_is_not_split_into_many_accepted_points(default_config) -> None:
    image = np.full((512, 512), 35, dtype=np.uint8)
    cv2.circle(image, (256, 256), 220, 215, thickness=-1)
    cv2.line(image, (120, 260), (390, 275), 12, thickness=4, lineType=cv2.LINE_AA)
    valid = np.zeros(image.shape, dtype=np.uint8)
    cv2.circle(valid, (256, 256), 220, 1, thickness=-1)
    preprocessing = preprocess_image(image, valid.astype(bool), default_config)
    detection = detect_candidates(
        preprocessing.filtered,
        valid.astype(bool),
        default_config,
        dark_response=preprocessing.dark_response,
    )
    features = extract_candidate_features(
        detection,
        preprocessing.image,
        preprocessing.dark_response,
        preprocessing.background,
        valid.astype(bool),
        default_config,
        center_x_px=256.0,
        center_y_px=256.0,
        radius_px=220.0,
        mm_per_pixel=100.0 / 440.0,
    )
    # A single line can have a small number of disconnected response fragments,
    # but it must not be transformed into a long list of accepted round points.
    assert detection.post_watershed_count <= 4
    assert not any(feature.accepted for feature in features)
    assert any("too_elongated" in feature.rejection_reason or "low_circularity" in feature.rejection_reason for feature in features)


def test_outside_and_excluded_edge_points_are_not_counted(tmp_path, default_config) -> None:
    image = np.full((601, 601), 30, dtype=np.uint8)
    cv2.circle(image, (300, 300), 280, 205, thickness=-1)
    cv2.circle(image, (300, 300), 6, 10, thickness=-1)  # valid centre point
    cv2.circle(image, (560, 300), 6, 10, thickness=-1)  # inside wafer, excluded edge band
    cv2.circle(image, (20, 20), 6, 0, thickness=-1)  # external point
    input_path = tmp_path / "edge_and_outside.png"
    assert cv2.imwrite(str(input_path), image)

    config = copy.deepcopy(default_config)
    config["wafer"]["exclude_edge_mm"] = 5.0
    result = analyze_image(
        input_path,
        tmp_path / "result",
        config,
        center_x=300.0,
        center_y=300.0,
        radius_px=280.0,
    )
    accepted = result.defects[result.defects["accepted"]]
    assert len(accepted) == 1
    assert accepted.iloc[0]["centroid_x_px"] == pytest.approx(300.0, abs=2.0)
    assert accepted.iloc[0]["centroid_y_px"] == pytest.approx(300.0, abs=2.0)
