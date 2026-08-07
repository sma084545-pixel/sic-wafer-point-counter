"""Morphological feature extraction and transparent candidate filtering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import logging
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.measure import regionprops

from .preprocessing import PreprocessingResult, as_float01

LOGGER = logging.getLogger(__name__)


DEFECT_COLUMNS: tuple[str, ...] = (
    "defect_id",
    "centroid_x_px",
    "centroid_y_px",
    "x_mm",
    "y_mm",
    "radial_distance_mm",
    "polar_angle_deg",
    "area_px",
    "area_mm2",
    "perimeter_px",
    "equivalent_diameter_px",
    "equivalent_diameter_mm",
    "major_axis_length_px",
    "minor_axis_length_px",
    "aspect_ratio",
    "eccentricity",
    "circularity",
    "solidity",
    "bounding_box",
    "mean_gray_raw",
    "mean_dark_response",
    "local_background_gray",
    "contrast",
    "distance_to_fitted_circle_mm",
    "distance_to_valid_boundary_mm",
    "distance_to_wafer_edge_mm",
    "accepted",
    "rejection_reason",
)


def _measurement_float01(image: NDArray[np.generic]) -> NDArray[np.float32]:
    """Reuse normalized float32 pipeline arrays; normalize other inputs safely."""

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got {array.shape}")
    if array.dtype == np.float32:
        minimum = float(array.min(initial=0.0))
        maximum = float(array.max(initial=0.0))
        if (
            np.isfinite(minimum)
            and np.isfinite(maximum)
            and minimum >= 0.0
            and maximum <= 1.0
        ):
            return array
    return as_float01(array)


def _section(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = mapping.get(name)
    return value if isinstance(value, Mapping) else mapping


@dataclass(frozen=True, slots=True)
class FeatureFilterConfig:
    """Configurable, auditable rules for point-versus-artifact classification."""

    min_area_px: float = 5.0
    max_area_px: float = 5000.0
    min_equivalent_diameter_px: float = 2.0
    max_equivalent_diameter_px: float = 80.0
    min_circularity: float = 0.25
    max_eccentricity: float = 0.92
    max_aspect_ratio: float = 4.0
    min_solidity: float = 0.60
    min_contrast: float = 0.02
    min_edge_distance_mm: float = 0.0
    min_valid_fraction: float = 0.95
    local_background_ring_px: int = 5

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "FeatureFilterConfig":
        """Build from a full YAML mapping or its ``filters`` section."""

        if mapping is None:
            return cls()
        values = _section(mapping, "filters")
        kwargs = {
            key: values[key]
            for key in cls.__dataclass_fields__.keys()
            if key in values
        }
        # Older configs may only define detection min/max area.  Use those as a
        # fallback, never as a hidden overwrite of explicit filter values.
        detection = mapping.get("detection") if isinstance(mapping, Mapping) else None
        if isinstance(detection, Mapping):
            defaults = cls()
            kwargs.setdefault(
                "min_area_px", detection.get("min_area_px", defaults.min_area_px)
            )
            kwargs.setdefault(
                "max_area_px", detection.get("max_area_px", defaults.max_area_px)
            )
        return cls(**kwargs)

    def validated(self) -> "FeatureFilterConfig":
        """Validate physical/morphological threshold consistency."""

        if (
            not np.isfinite(self.min_area_px)
            or not np.isfinite(self.max_area_px)
            or self.min_area_px < 0
            or self.max_area_px < self.min_area_px
        ):
            raise ValueError("Require 0 <= min_area_px <= max_area_px")
        if (
            not np.isfinite(self.min_equivalent_diameter_px)
            or not np.isfinite(self.max_equivalent_diameter_px)
            or self.min_equivalent_diameter_px < 0
            or self.max_equivalent_diameter_px < self.min_equivalent_diameter_px
        ):
            raise ValueError("Invalid equivalent diameter range")
        if not 0 <= self.min_circularity <= 1:
            raise ValueError("min_circularity must lie in [0, 1]")
        if not 0 <= self.max_eccentricity <= 1:
            raise ValueError("max_eccentricity must lie in [0, 1]")
        if self.max_aspect_ratio < 1:
            raise ValueError("max_aspect_ratio must be >= 1")
        if not 0 <= self.min_solidity <= 1:
            raise ValueError("min_solidity must lie in [0, 1]")
        if not 0 <= self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must lie in [0, 1]")
        if not np.isfinite(self.min_edge_distance_mm) or self.min_edge_distance_mm < 0:
            raise ValueError("min_edge_distance_mm must be finite and non-negative")
        if self.local_background_ring_px < 1:
            raise ValueError("local_background_ring_px must be >= 1")
        return replace(
            self,
            min_area_px=float(self.min_area_px),
            max_area_px=float(self.max_area_px),
            local_background_ring_px=int(self.local_background_ring_px),
        )


@dataclass(frozen=True, slots=True)
class DefectFeature:
    """One numbered candidate and all measurements needed for audit/review.

    ``bounding_box`` is ``(min_x, min_y, max_x_exclusive, max_y_exclusive)``.
    Intensities and contrast are measured on normalized ``[0, 1]`` images.
    """

    defect_id: int
    centroid_x_px: float
    centroid_y_px: float
    x_mm: float
    y_mm: float
    radial_distance_mm: float
    polar_angle_deg: float
    area_px: int
    area_mm2: float
    perimeter_px: float
    equivalent_diameter_px: float
    equivalent_diameter_mm: float
    major_axis_length_px: float
    minor_axis_length_px: float
    aspect_ratio: float
    eccentricity: float
    circularity: float
    solidity: float
    bounding_box: tuple[int, int, int, int]
    mean_gray_raw: float
    mean_dark_response: float
    local_background_gray: float
    contrast: float
    distance_to_fitted_circle_mm: float
    distance_to_valid_boundary_mm: float
    distance_to_wafer_edge_mm: float
    accepted: bool
    rejection_reason: str

    def to_record(self) -> dict[str, Any]:
        """Return a CSV/JSON-friendly record in a stable column order."""

        data = asdict(self)
        data["bounding_box"] = json.dumps(self.bounding_box, separators=(",", ":"))
        return {column: data[column] for column in DEFECT_COLUMNS}


def pixel_to_wafer_coordinates(
    x_px: float,
    y_px: float,
    center_x_px: float,
    center_y_px: float,
    mm_per_pixel: float,
) -> tuple[float, float, float, float]:
    """Convert image coordinates to center-relative mm and polar coordinates.

    Image rows increase downward, so the returned physical ``y_mm`` is negated.
    The polar angle is in ``[0, 360)`` degrees, counter-clockwise from +x.
    """

    if mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be positive")
    x_mm = (float(x_px) - float(center_x_px)) * float(mm_per_pixel)
    y_mm = -(float(y_px) - float(center_y_px)) * float(mm_per_pixel)
    radial = math.hypot(x_mm, y_mm)
    angle = math.degrees(math.atan2(y_mm, x_mm)) % 360.0
    return x_mm, y_mm, radial, angle


def _resolve_calibration(
    *,
    geometry: Any | None,
    center_x_px: float | None,
    center_y_px: float | None,
    radius_px: float | None,
    mm_per_pixel: float | None,
    wafer_diameter_mm: float,
) -> tuple[float, float, float, float]:
    """Resolve explicit calibration or common WaferGeometry attribute names."""

    if geometry is not None:
        center_x_px = center_x_px if center_x_px is not None else getattr(
            geometry, "center_x_px", getattr(geometry, "center_x", None)
        )
        center_y_px = center_y_px if center_y_px is not None else getattr(
            geometry, "center_y_px", getattr(geometry, "center_y", None)
        )
        radius_px = radius_px if radius_px is not None else getattr(
            geometry, "radius_px", None
        )
        mm_per_pixel = mm_per_pixel if mm_per_pixel is not None else getattr(
            geometry, "mm_per_pixel", None
        )
    if center_x_px is None or center_y_px is None or radius_px is None:
        raise ValueError("Wafer center_x_px, center_y_px, and radius_px are required")
    if radius_px <= 0:
        raise ValueError("radius_px must be positive")
    if mm_per_pixel is None:
        if wafer_diameter_mm <= 0:
            raise ValueError("wafer_diameter_mm must be positive")
        mm_per_pixel = float(wafer_diameter_mm) / (2.0 * float(radius_px))
    if mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be positive")
    return (
        float(center_x_px),
        float(center_y_px),
        float(radius_px),
        float(mm_per_pixel),
    )


def _local_background_from_ring(
    raw: NDArray[np.float32],
    all_labels: NDArray[np.int32],
    region: Any,
    valid_mask: NDArray[np.bool_],
    ring_px: int,
) -> float:
    """Estimate local background from a valid, candidate-free dilation ring."""

    min_row, min_col, max_row, max_col = region.bbox
    row0, col0 = max(0, min_row - ring_px), max(0, min_col - ring_px)
    row1 = min(raw.shape[0], max_row + ring_px)
    col1 = min(raw.shape[1], max_col + ring_px)
    local_component = all_labels[row0:row1, col0:col1] == region.label
    kernel_size = ring_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(local_component.astype(np.uint8), kernel).astype(bool)
    ring = (
        dilated
        & ~local_component
        & (all_labels[row0:row1, col0:col1] == 0)
        & valid_mask[row0:row1, col0:col1]
    )
    values = raw[row0:row1, col0:col1][ring]
    if values.size:
        return float(np.median(values))
    return float("nan")


def _rejection_reasons(
    *,
    area_px: int,
    equivalent_diameter_px: float,
    aspect_ratio: float,
    eccentricity: float,
    circularity: float,
    solidity: float,
    contrast: float,
    distance_to_valid_boundary_mm: float,
    valid_fraction: float,
    config: FeatureFilterConfig,
) -> list[str]:
    """Apply all configured filters and return stable machine-readable reasons."""

    reasons: list[str] = []
    if valid_fraction < config.min_valid_fraction or distance_to_valid_boundary_mm <= 0:
        reasons.append("outside_valid_mask")
    if area_px < config.min_area_px or equivalent_diameter_px < config.min_equivalent_diameter_px:
        reasons.append("too_small")
    if area_px > config.max_area_px or equivalent_diameter_px > config.max_equivalent_diameter_px:
        reasons.append("too_large")
    # Aspect and eccentricity jointly expose line-like objects.  Either one is
    # sufficient to reject; the single reason keeps review tables concise.
    if aspect_ratio > config.max_aspect_ratio or eccentricity > config.max_eccentricity:
        reasons.append("too_elongated")
    if circularity < config.min_circularity:
        reasons.append("low_circularity")
    if solidity < config.min_solidity:
        reasons.append("low_solidity")
    if not np.isfinite(contrast) or contrast < config.min_contrast:
        reasons.append("low_contrast")
    if (
        distance_to_valid_boundary_mm > 0
        and distance_to_valid_boundary_mm < config.min_edge_distance_mm
    ):
        reasons.append("near_wafer_edge")
    return reasons


def valid_boundary_distance_transform(
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Return exact in-mask Euclidean distance to the final valid-mask boundary.

    Zero denotes pixels outside the final analysis mask.  Candidates use the
    minimum distance across their pixels, conservatively respecting flats,
    notches, excluded edge bands, and user-defined invalid regions.
    """

    mask = np.asarray(valid_mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("valid_mask must be a two-dimensional array")
    if not np.any(mask):
        raise ValueError("valid_mask contains no valid pixels")
    return np.asarray(
        cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE),
        dtype=np.float32,
    )


def extract_candidate_features(
    labels: NDArray[np.integer] | Any,
    raw_image: NDArray[np.generic],
    dark_response: NDArray[np.generic] | None = None,
    background: NDArray[np.generic] | None = None,
    valid_mask: NDArray[np.bool_] | None = None,
    config: FeatureFilterConfig | Mapping[str, Any] | None = None,
    *,
    geometry: Any | None = None,
    center_x_px: float | None = None,
    center_y_px: float | None = None,
    radius_px: float | None = None,
    mm_per_pixel: float | None = None,
    wafer_diameter_mm: float = 100.0,
    coordinate_offset_xy: tuple[int, int] = (0, 0),
    valid_boundary_distance_px: NDArray[np.floating] | None = None,
) -> list[DefectFeature]:
    """Measure every labeled candidate and classify it with configured rules.

    ``labels`` may be a label array or an object exposing a ``labels`` attribute
    (for example :class:`point_detection.DetectionResult`).  Supplying the
    preprocessing background is preferred; otherwise an annular local median is
    used.  No candidate is discarded from the returned list.

    For a lazily read tile, pass its global ``(x, y)`` origin as
    ``coordinate_offset_xy``.  Centroids, physical coordinates, bboxes, and edge
    distances are then reported in the full source-image coordinate system.
    ``valid_boundary_distance_px`` is an optional precomputed distance map from
    the final valid analysis mask; when absent it is computed locally.
    """

    detection_obj = labels if hasattr(labels, "labels") else None
    label_image = np.asarray(getattr(labels, "labels", labels), dtype=np.int32)
    if label_image.ndim != 2:
        raise ValueError("labels must be a two-dimensional label image")
    raw = _measurement_float01(raw_image)
    if raw.shape != label_image.shape:
        raise ValueError("raw_image shape must match labels")
    if dark_response is None and detection_obj is not None:
        dark_response = getattr(detection_obj, "response", None)
    response = (
        np.zeros(raw.shape, dtype=np.float32)
        if dark_response is None
        else _measurement_float01(np.asarray(dark_response))
    )
    if response.shape != raw.shape:
        raise ValueError("dark_response shape must match labels")
    background_image = (
        None if background is None else _measurement_float01(np.asarray(background))
    )
    if background_image is not None and background_image.shape != raw.shape:
        raise ValueError("background shape must match labels")
    mask = (
        np.ones(raw.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if mask.shape != raw.shape:
        raise ValueError("valid_mask shape must match labels")
    boundary_distance = (
        valid_boundary_distance_transform(mask)
        if valid_boundary_distance_px is None
        else np.asarray(valid_boundary_distance_px, dtype=np.float32)
    )
    if boundary_distance.shape != raw.shape:
        raise ValueError("valid_boundary_distance_px shape must match labels")
    if not np.isfinite(boundary_distance).all() or np.any(boundary_distance < 0):
        raise ValueError("valid_boundary_distance_px must be finite and non-negative")
    cfg = (
        config.validated()
        if isinstance(config, FeatureFilterConfig)
        else FeatureFilterConfig.from_mapping(config).validated()
    )
    center_x, center_y, wafer_radius, scale = _resolve_calibration(
        geometry=geometry,
        center_x_px=center_x_px,
        center_y_px=center_y_px,
        radius_px=radius_px,
        mm_per_pixel=mm_per_pixel,
        wafer_diameter_mm=wafer_diameter_mm,
    )

    features: list[DefectFeature] = []
    offset_x, offset_y = int(coordinate_offset_xy[0]), int(coordinate_offset_xy[1])
    for defect_id, region in enumerate(regionprops(label_image), start=1):
        coords = region.coords
        rows, cols = coords[:, 0], coords[:, 1]
        local_centroid_y, local_centroid_x = region.centroid
        centroid_x = float(local_centroid_x + offset_x)
        centroid_y = float(local_centroid_y + offset_y)
        x_mm, y_mm, radial_mm, angle_deg = pixel_to_wafer_coordinates(
            centroid_x, centroid_y, center_x, center_y, scale
        )
        area_px = int(region.area)
        perimeter = float(region.perimeter)
        circularity = (
            float(4.0 * math.pi * area_px / (perimeter * perimeter))
            if perimeter > 0
            else 0.0
        )
        # axis_* names are current in scikit-image 0.26; the older aliases emit
        # a FutureWarning and would pollute run.log on every candidate.
        major = float(region.axis_major_length)
        minor = float(region.axis_minor_length)
        aspect = major / minor if minor > np.finfo(float).eps else float("inf")
        equivalent_diameter = float(region.equivalent_diameter_area)
        mean_raw = float(np.mean(raw[rows, cols]))
        mean_response = float(np.mean(response[rows, cols]))
        if background_image is not None:
            local_background = float(np.mean(background_image[rows, cols]))
        else:
            local_background = _local_background_from_ring(
                raw, label_image, region, mask, cfg.local_background_ring_px
            )
            if not np.isfinite(local_background):
                # If a component fills its crop, dark response still provides an
                # explicit background-minus-image estimate.
                local_background = mean_raw + mean_response
        contrast = float(local_background - mean_raw)
        radial_px = math.hypot(centroid_x - center_x, centroid_y - center_y)
        distance_to_fitted_circle_mm = float((wafer_radius - radial_px) * scale)
        distance_to_valid_boundary_mm = float(
            np.min(boundary_distance[rows, cols]) * scale
        )
        valid_fraction = float(np.mean(mask[rows, cols])) if area_px else 0.0
        reasons = _rejection_reasons(
            area_px=area_px,
            equivalent_diameter_px=equivalent_diameter,
            aspect_ratio=aspect,
            eccentricity=float(region.eccentricity),
            circularity=circularity,
            solidity=float(region.solidity),
            contrast=contrast,
            distance_to_valid_boundary_mm=distance_to_valid_boundary_mm,
            valid_fraction=valid_fraction,
            config=cfg,
        )
        min_row, min_col, max_row, max_col = region.bbox
        features.append(
            DefectFeature(
                defect_id=defect_id,
                centroid_x_px=float(centroid_x),
                centroid_y_px=float(centroid_y),
                x_mm=x_mm,
                y_mm=y_mm,
                radial_distance_mm=radial_mm,
                polar_angle_deg=angle_deg,
                area_px=area_px,
                area_mm2=float(area_px * scale * scale),
                perimeter_px=perimeter,
                equivalent_diameter_px=equivalent_diameter,
                equivalent_diameter_mm=float(equivalent_diameter * scale),
                major_axis_length_px=major,
                minor_axis_length_px=minor,
                aspect_ratio=float(aspect),
                eccentricity=float(region.eccentricity),
                circularity=circularity,
                solidity=float(region.solidity),
                bounding_box=(
                    int(min_col + offset_x),
                    int(min_row + offset_y),
                    int(max_col + offset_x),
                    int(max_row + offset_y),
                ),
                mean_gray_raw=mean_raw,
                mean_dark_response=mean_response,
                local_background_gray=local_background,
                contrast=contrast,
                distance_to_fitted_circle_mm=distance_to_fitted_circle_mm,
                distance_to_valid_boundary_mm=distance_to_valid_boundary_mm,
                # Compatibility field: it has always been a fitted-circle
                # distance.  New analyses should use the explicit name above.
                distance_to_wafer_edge_mm=distance_to_fitted_circle_mm,
                accepted=not reasons,
                rejection_reason=";".join(reasons),
            )
        )
    accepted = sum(feature.accepted for feature in features)
    LOGGER.info(
        "Extracted %d candidate feature records: %d accepted, %d rejected",
        len(features),
        accepted,
        len(features) - accepted,
    )
    return features


def features_to_records(features: Sequence[DefectFeature]) -> list[dict[str, Any]]:
    """Convert features to stable CSV/JSON-friendly dictionaries."""

    return [feature.to_record() for feature in features]


def features_to_dataframe(features: Sequence[DefectFeature]) -> Any:
    """Return a pandas DataFrame with stable columns (pandas imported lazily)."""

    import pandas as pd

    return pd.DataFrame.from_records(features_to_records(features), columns=DEFECT_COLUMNS)


def extract_features(
    detection: Any,
    preprocessing: PreprocessingResult,
    geometry: Any,
    valid_mask: NDArray[np.bool_] | None = None,
    config: FeatureFilterConfig | Mapping[str, Any] | None = None,
    *,
    wafer_diameter_mm: float = 100.0,
) -> list[DefectFeature]:
    """Convenience adapter for pipeline dataclasses used by the CLI."""

    return extract_candidate_features(
        detection,
        preprocessing.image,
        dark_response=preprocessing.dark_response,
        background=preprocessing.background,
        valid_mask=preprocessing.valid_mask if valid_mask is None else valid_mask,
        config=config,
        geometry=geometry,
        wafer_diameter_mm=wafer_diameter_mm,
    )


__all__ = [
    "DEFECT_COLUMNS",
    "DefectFeature",
    "FeatureFilterConfig",
    "extract_candidate_features",
    "extract_features",
    "features_to_dataframe",
    "features_to_records",
    "pixel_to_wafer_coordinates",
    "valid_boundary_distance_transform",
]
