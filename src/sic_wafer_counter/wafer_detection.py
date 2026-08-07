"""Wafer boundary detection, physical calibration and analysis masks.

Automatic detection runs on a normalized low-resolution preview.  The fitted
circle supplies the 100 mm physical scale, while the segmented contour is kept
separately and is preferred for area masks so flats, notches, cropping and
other silhouette departures affect the actual analysed area.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.measure import points_in_poly


LOGGER = logging.getLogger(__name__)


class WaferDetectionError(RuntimeError):
    """Raised when no physically credible wafer geometry can be established."""

    def __init__(self, message: str, geometry: "WaferGeometry | None" = None) -> None:
        super().__init__(message)
        self.geometry = geometry


@dataclasses.dataclass(slots=True)
class WaferGeometry:
    """Full-resolution wafer geometry and fit diagnostics."""

    center_x: float
    center_y: float
    radius_px: float
    image_width: int
    image_height: int
    diameter_mm: float = 100.0
    confidence: float = 1.0
    circularity: float = 1.0
    fit_residual: float = 0.0
    diameter_reasonable: bool = True
    is_cropped: bool = False
    contour_area_px: float | None = None
    border_contact_fraction: float = 0.0
    angular_coverage: float = 1.0
    detection_method: str = "manual"
    contour_polygon: list[tuple[float, float]] | None = None
    warnings: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        values = (self.center_x, self.center_y, self.radius_px, self.diameter_mm)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Wafer center, radius and diameter must be finite")
        if self.radius_px <= 0.0:
            raise ValueError(f"radius_px must be positive, got {self.radius_px}")
        if self.diameter_mm <= 0.0:
            raise ValueError(f"diameter_mm must be positive, got {self.diameter_mm}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")

    @property
    def diameter_px(self) -> float:
        return 2.0 * self.radius_px

    @property
    def mm_per_pixel(self) -> float:
        return self.diameter_mm / self.diameter_px

    @property
    def cm_per_pixel(self) -> float:
        return self.mm_per_pixel / 10.0

    @property
    def um_per_pixel(self) -> float:
        """Physical calibration in micrometres per source pixel."""

        return self.mm_per_pixel * 1000.0

    @property
    def pixel_area_cm2(self) -> float:
        return self.cm_per_pixel**2

    @property
    def theoretical_area_cm2(self) -> float:
        return theoretical_wafer_area_cm2(self.diameter_mm)

    @property
    def circle_fit_area_cm2(self) -> float:
        return math.pi * (self.radius_px * self.cm_per_pixel) ** 2

    @property
    def center(self) -> tuple[float, float]:
        return self.center_x, self.center_y

    def to_dict(self) -> dict[str, Any]:
        """Return geometry and derived physical quantities for reports."""

        return {
            "center_x_px": self.center_x,
            "center_y_px": self.center_y,
            "radius_px": self.radius_px,
            "diameter_px": self.diameter_px,
            "diameter_mm": self.diameter_mm,
            "mm_per_pixel": self.mm_per_pixel,
            "cm_per_pixel": self.cm_per_pixel,
            "um_per_pixel": self.um_per_pixel,
            "pixel_area_cm2": self.pixel_area_cm2,
            "theoretical_area_cm2": self.theoretical_area_cm2,
            "circle_fit_area_cm2": self.circle_fit_area_cm2,
            "confidence": self.confidence,
            "circularity": self.circularity,
            "fit_residual_fraction_radius": self.fit_residual,
            "diameter_reasonable": self.diameter_reasonable,
            "is_cropped": self.is_cropped,
            "contour_area_px": self.contour_area_px,
            "border_contact_fraction": self.border_contact_fraction,
            "angular_coverage": self.angular_coverage,
            "detection_method": self.detection_method,
            "contour_polygon": self.contour_polygon,
            "warnings": list(self.warnings),
        }


@dataclasses.dataclass(slots=True)
class AnalysisMasks:
    """Full-size boolean masks with non-overlapping exclusion semantics."""

    full_wafer_mask: np.ndarray
    edge_exclusion_mask: np.ndarray
    invalid_mask: np.ndarray
    valid_analysis_mask: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            self.full_wafer_mask.shape,
            self.edge_exclusion_mask.shape,
            self.invalid_mask.shape,
            self.valid_analysis_mask.shape,
        }
        if len(shapes) != 1:
            raise ValueError(f"Analysis mask shapes differ: {shapes}")


@dataclasses.dataclass(slots=True)
class AnalysisMaskTile:
    """A non-overlapping global tile of all four analysis masks."""

    x: int
    y: int
    masks: AnalysisMasks

    @property
    def width(self) -> int:
        return int(self.masks.valid_analysis_mask.shape[1])

    @property
    def height(self) -> int:
        return int(self.masks.valid_analysis_mask.shape[0])


class ValidBoundaryIndex:
    """Bounded-memory nearest-neighbour index of the final invalid boundary.

    Points are centres of invalid pixels adjacent to the final valid mask,
    including the one-pixel exterior ring around the source image.  For any
    valid candidate pixel, its nearest zero in the exact raster mask is
    necessarily one of these boundary points.  A KD-tree therefore reproduces
    the Euclidean distance-transform value without allocating a full-frame
    float32 distance raster.
    """

    def __init__(
        self,
        points_xy: np.ndarray,
        *,
        max_mask_tile_pixels: int,
        scan_tile_size: int,
    ) -> None:
        points = np.asarray(points_xy)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            raise ValueError("Valid-boundary index requires at least one (x, y) point")
        self._tree = cKDTree(np.asarray(points, dtype=np.float64))
        self.boundary_point_count = int(len(points))
        self.max_mask_tile_pixels = int(max_mask_tile_pixels)
        self.scan_tile_size = int(scan_tile_size)

    @property
    def tree_data_bytes(self) -> int:
        """Bytes in the KD-tree's coordinate table (excluding tree overhead)."""

        return int(self._tree.data.nbytes)

    def query_yx(self, coordinates_yx: np.ndarray) -> np.ndarray:
        """Return exact nearest final-invalid-pixel distances for global pixels."""

        coordinates = np.asarray(coordinates_yx)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates_yx must have shape (n, 2)")
        if not len(coordinates):
            return np.empty(0, dtype=np.float32)
        query_xy = np.column_stack((coordinates[:, 1], coordinates[:, 0]))
        distances, _ = self._tree.query(query_xy, k=1, eps=0.0)
        return np.asarray(distances, dtype=np.float32)


@dataclasses.dataclass(frozen=True, slots=True)
class AreaStatistics:
    """Pixel-exact, mutually reconciled wafer analysis areas."""

    full_wafer_pixel_count: int
    edge_excluded_pixel_count: int
    invalid_pixel_count: int
    valid_pixel_count: int
    pixel_area_cm2: float
    theoretical_area_cm2: float
    circle_fit_area_cm2: float

    @property
    def full_wafer_area_cm2(self) -> float:
        return self.full_wafer_pixel_count * self.pixel_area_cm2

    @property
    def edge_excluded_area_cm2(self) -> float:
        return self.edge_excluded_pixel_count * self.pixel_area_cm2

    @property
    def invalid_area_cm2(self) -> float:
        return self.invalid_pixel_count * self.pixel_area_cm2

    @property
    def valid_area_cm2(self) -> float:
        return self.valid_pixel_count * self.pixel_area_cm2

    @property
    def excluded_total_area_cm2(self) -> float:
        return self.edge_excluded_area_cm2 + self.invalid_area_cm2

    def to_dict(self) -> dict[str, float | int]:
        return {
            "theoretical_complete_wafer_area_cm2": self.theoretical_area_cm2,
            "circle_fit_area_cm2": self.circle_fit_area_cm2,
            "pixel_mask_full_wafer_area_cm2": self.full_wafer_area_cm2,
            "edge_excluded_area_cm2": self.edge_excluded_area_cm2,
            "other_invalid_area_cm2": self.invalid_area_cm2,
            "valid_analysis_area_cm2": self.valid_area_cm2,
            "full_wafer_pixel_count": self.full_wafer_pixel_count,
            "edge_excluded_pixel_count": self.edge_excluded_pixel_count,
            "invalid_pixel_count": self.invalid_pixel_count,
            "valid_pixel_count": self.valid_pixel_count,
            "pixel_area_cm2": self.pixel_area_cm2,
        }


@dataclasses.dataclass(slots=True)
class _FitCandidate:
    contour: np.ndarray
    center_x: float
    center_y: float
    radius: float
    area: float
    circularity: float
    residual: float
    fill_ratio: float
    border_fraction: float
    angular_coverage: float
    diameter_reasonable: bool
    cropped: bool
    confidence: float
    polarity: str


def theoretical_wafer_area_cm2(diameter_mm: float = 100.0) -> float:
    """Area of a complete circular wafer with ``diameter_mm``."""

    diameter = float(diameter_mm)
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError(f"diameter_mm must be positive and finite, got {diameter_mm}")
    radius_cm = diameter / 20.0
    return math.pi * radius_cm * radius_cm


def pixel_scale_from_diameter(
    diameter_mm: float, diameter_px: float
) -> tuple[float, float, float]:
    """Return ``(mm_per_pixel, cm_per_pixel, pixel_area_cm2)``."""

    physical, pixels = float(diameter_mm), float(diameter_px)
    if not math.isfinite(physical) or physical <= 0.0:
        raise ValueError("diameter_mm must be positive and finite")
    if not math.isfinite(pixels) or pixels <= 0.0:
        raise ValueError("diameter_px must be positive and finite")
    mm_per_pixel = physical / pixels
    cm_per_pixel = mm_per_pixel / 10.0
    return mm_per_pixel, cm_per_pixel, cm_per_pixel**2


def _normalized_preview(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"Wafer preview must be 2-D grayscale, got {data.shape}")
    if data.size == 0:
        raise ValueError("Wafer preview is empty")
    if data.dtype == np.uint8:
        return np.array(data, copy=False)
    finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data.reshape(-1)
    if finite.size == 0:
        raise ValueError("Wafer preview contains no finite pixels")
    low, high = np.percentile(finite, [1.0, 99.0])
    if not high > low:
        high = low + 1.0
    scaled = np.nan_to_num(data.astype(np.float32), nan=float(low))
    return np.rint(np.clip((scaled - low) * 255.0 / (high - low), 0, 255)).astype(np.uint8)


def _robust_circle_fit(contour: np.ndarray) -> tuple[float, float, float, float]:
    points = contour.reshape(-1, 2).astype(np.float64)
    if points.shape[0] < 5:
        raise ValueError("At least five contour points are required for circle fitting")
    if points.shape[0] > 5000:
        points = points[:: int(math.ceil(points.shape[0] / 5000))]
    (initial_x, initial_y), initial_radius = cv2.minEnclosingCircle(
        points.astype(np.float32).reshape(-1, 1, 2)
    )

    def residuals(parameters: np.ndarray) -> np.ndarray:
        center_x, center_y, radius = parameters
        distances = np.hypot(points[:, 0] - center_x, points[:, 1] - center_y)
        return distances - radius

    result = least_squares(
        residuals,
        np.array([initial_x, initial_y, max(initial_radius, 1.0)]),
        loss="soft_l1",
        f_scale=max(1.0, initial_radius * 0.01),
        max_nfev=300,
    )
    center_x, center_y, radius = map(float, result.x)
    radius = abs(radius)
    errors = residuals(np.array([center_x, center_y, radius]))
    normalized_rms = float(np.sqrt(np.mean(errors**2)) / max(radius, 1e-9))
    return center_x, center_y, radius, normalized_rms


def _angular_coverage(
    contour: np.ndarray, center_x: float, center_y: float, bins: int = 72
) -> float:
    points = contour.reshape(-1, 2).astype(np.float64)
    angles = (np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x) + 2 * np.pi) % (
        2 * np.pi
    )
    occupied = np.unique(np.floor(angles * bins / (2 * np.pi)).astype(int) % bins)
    return float(occupied.size / bins)


def _candidate_from_contour(
    contour: np.ndarray, shape: tuple[int, int], polarity: str
) -> _FitCandidate | None:
    height, width = shape
    area = float(cv2.contourArea(contour))
    image_area = float(height * width)
    if area < 0.08 * image_area:
        return None
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return None
    circularity = float(np.clip(4.0 * math.pi * area / (perimeter**2), 0.0, 1.0))
    try:
        center_x, center_y, radius, residual = _robust_circle_fit(contour)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not all(math.isfinite(value) for value in (center_x, center_y, radius, residual)):
        return None
    points = contour.reshape(-1, 2)
    border = (
        (points[:, 0] <= 1)
        | (points[:, 1] <= 1)
        | (points[:, 0] >= width - 2)
        | (points[:, 1] >= height - 2)
    )
    border_fraction = float(np.mean(border))
    coverage = _angular_coverage(contour, center_x, center_y)
    fill_ratio = area / max(math.pi * radius * radius, 1.0)
    diameter = 2.0 * radius
    diameter_reasonable = 0.45 * min(height, width) <= diameter <= 1.5 * max(height, width)
    extends = (
        center_x - radius < -2
        or center_y - radius < -2
        or center_x + radius > width + 1
        or center_y + radius > height + 1
    )
    cropped = bool(extends or border_fraction > 0.03)

    circle_score = float(np.clip((circularity - 0.35) / 0.60, 0.0, 1.0))
    residual_score = math.exp(-18.0 * residual)
    fill_score = math.exp(-3.0 * abs(1.0 - min(fill_ratio, 2.0)))
    coverage_score = float(np.clip((coverage - 0.45) / 0.50, 0.0, 1.0))
    diameter_score = 1.0 if diameter_reasonable else 0.15
    image_center_distance = math.hypot(center_x - width / 2, center_y - height / 2)
    center_score = math.exp(-2.0 * image_center_distance / max(width, height))
    confidence = (
        0.24 * circle_score
        + 0.26 * residual_score
        + 0.14 * fill_score
        + 0.16 * coverage_score
        + 0.10 * diameter_score
        + 0.10 * center_score
    )
    area_fraction = area / image_area
    # A threshold polarity that merely returns the rectangular image boundary
    # must not be mistaken for a wafer.
    if area_fraction > 0.985 and border_fraction > 0.25:
        confidence *= 0.30
    elif border_fraction > 0.30:
        confidence *= 0.65
    return _FitCandidate(
        contour=contour,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        area=area,
        circularity=circularity,
        residual=residual,
        fill_ratio=fill_ratio,
        border_fraction=border_fraction,
        angular_coverage=coverage,
        diameter_reasonable=diameter_reasonable,
        cropped=cropped,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        polarity=polarity,
    )


def _automatic_candidates(preview: np.ndarray) -> list[_FitCandidate]:
    height, width = preview.shape
    sigma = max(1.0, min(height, width) / 500.0)
    blurred = cv2.GaussianBlur(preview, (0, 0), sigmaX=sigma, sigmaY=sigma)
    otsu_value, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    LOGGER.debug("Wafer preview Otsu threshold: %.3f", otsu_value)
    kernel_size = max(3, int(round(min(height, width) / 300)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    candidates: list[_FitCandidate] = []
    for polarity, mask in (("bright-wafer", binary), ("dark-wafer", cv2.bitwise_not(binary))):
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            candidate = _candidate_from_contour(contour, (height, width), polarity)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _map_coordinate(value: float, preview_size: int, full_size: int) -> float:
    if preview_size <= 0:
        raise ValueError("preview_size must be positive")
    # OpenCV/Pillow-style resampling maps pixel centres, not array endpoints.
    # This affine transform has the same full/preview scale used for fitted
    # radii and contour area, avoiding a small but measurable area bias.
    return (value + 0.5) * full_size / preview_size - 0.5


def detect_wafer(
    preview: np.ndarray,
    *,
    full_shape: tuple[int, int] | None = None,
    diameter_mm: float = 100.0,
    center_x: float | None = None,
    center_y: float | None = None,
    radius_px: float | None = None,
    min_confidence: float = 0.55,
    allow_low_confidence: bool = False,
) -> WaferGeometry:
    """Detect wafer geometry or validate a complete manual specification.

    Manual values are interpreted in full-resolution pixels.  Supplying only a
    subset is rejected.  Automatic results below ``min_confidence`` raise
    :class:`WaferDetectionError` with instructions to use manual CLI values;
    callers must opt in explicitly with ``allow_low_confidence=True`` to inspect
    such a geometry without treating it as a trustworthy density calibration.
    """

    normalized = _normalized_preview(preview)
    preview_height, preview_width = normalized.shape
    if full_shape is None:
        full_height, full_width = preview_height, preview_width
    else:
        if len(full_shape) != 2:
            raise ValueError(f"full_shape must be (height, width), got {full_shape}")
        full_height, full_width = map(int, full_shape)
        if full_height <= 0 or full_width <= 0:
            raise ValueError(f"full_shape values must be positive, got {full_shape}")
    manual_values = (center_x, center_y, radius_px)
    if any(value is not None for value in manual_values):
        if not all(value is not None for value in manual_values):
            raise WaferDetectionError(
                "Manual wafer geometry requires --center-x, --center-y and --radius-px together"
            )
        assert center_x is not None and center_y is not None and radius_px is not None
        cropped = (
            center_x - radius_px < 0
            or center_y - radius_px < 0
            or center_x + radius_px > full_width - 1
            or center_y + radius_px > full_height - 1
        )
        warnings = (
            ["Manual circle intersects the image boundary; pixel-mask area will be clipped"]
            if cropped
            else []
        )
        return WaferGeometry(
            center_x=float(center_x),
            center_y=float(center_y),
            radius_px=float(radius_px),
            image_width=full_width,
            image_height=full_height,
            diameter_mm=float(diameter_mm),
            confidence=1.0,
            circularity=1.0,
            fit_residual=0.0,
            diameter_reasonable=True,
            is_cropped=cropped,
            detection_method="manual-cli",
            contour_polygon=None,
            warnings=warnings,
        )

    candidates = _automatic_candidates(normalized)
    if not candidates:
        raise WaferDetectionError(
            "Automatic wafer detection found no reasonable circular contour. "
            "Provide --center-x, --center-y and --radius-px or run the interactive review tool."
        )
    best = max(candidates, key=lambda item: item.confidence)
    scale_x = full_width / preview_width
    scale_y = full_height / preview_height
    center_full_x = _map_coordinate(best.center_x, preview_width, full_width)
    center_full_y = _map_coordinate(best.center_y, preview_height, full_height)
    radius_full = best.radius * (scale_x + scale_y) / 2.0
    perimeter = cv2.arcLength(best.contour, True)
    # Keep a fairly detailed polygon: this contour, unlike an overlay-only
    # outline, determines physical area.  A coarse 32-sided approximation can
    # bias a 100 mm wafer area by several tenths of a percent.
    epsilon = max(0.20, 0.00005 * perimeter)
    polygon_preview = cv2.approxPolyDP(best.contour, epsilon, True).reshape(-1, 2)
    polygon_full = [
        (
            _map_coordinate(float(point[0]), preview_width, full_width),
            _map_coordinate(float(point[1]), preview_height, full_height),
        )
        for point in polygon_preview
    ]
    warnings: list[str] = []
    if best.cropped:
        warnings.append(
            "Wafer contour/circle reaches the image boundary; the wafer is likely cropped and only visible pixels are analysed"
        )
    if not best.diameter_reasonable:
        warnings.append("Detected wafer diameter is not reasonable relative to image dimensions")
    if best.residual > 0.06:
        warnings.append(
            f"Circle-fit RMS residual is high ({best.residual:.3f} of radius)"
        )
    geometry = WaferGeometry(
        center_x=center_full_x,
        center_y=center_full_y,
        radius_px=radius_full,
        image_width=full_width,
        image_height=full_height,
        diameter_mm=float(diameter_mm),
        confidence=best.confidence,
        circularity=best.circularity,
        fit_residual=best.residual,
        diameter_reasonable=best.diameter_reasonable,
        is_cropped=best.cropped,
        contour_area_px=best.area * scale_x * scale_y,
        border_contact_fraction=best.border_fraction,
        angular_coverage=best.angular_coverage,
        detection_method=f"preview-otsu-largest-contour/{best.polarity}",
        contour_polygon=polygon_full,
        warnings=warnings,
    )
    LOGGER.info(
        "Wafer fit: center=(%.1f, %.1f), radius=%.1f px, confidence=%.3f, circularity=%.3f, residual=%.4f",
        geometry.center_x,
        geometry.center_y,
        geometry.radius_px,
        geometry.confidence,
        geometry.circularity,
        geometry.fit_residual,
    )
    if geometry.confidence < float(min_confidence) and not allow_low_confidence:
        raise WaferDetectionError(
            f"Automatic wafer detection confidence {geometry.confidence:.3f} is below "
            f"the required {float(min_confidence):.3f}. Density was not calculated. "
            "Inspect the preview and rerun with --center-x, --center-y and --radius-px.",
            geometry=geometry,
        )
    return geometry


def _validate_region(
    geometry: WaferGeometry, x: int, y: int, width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, region_width, region_height = map(int, (x, y, width, height))
    if region_width <= 0 or region_height <= 0:
        raise ValueError("Mask tile width and height must be positive")
    if (
        x0 < 0
        or y0 < 0
        or x0 + region_width > geometry.image_width
        or y0 + region_height > geometry.image_height
    ):
        raise ValueError("Mask tile lies outside geometry image bounds")
    return x0, y0, region_width, region_height


def create_full_wafer_mask_tile(
    geometry: WaferGeometry,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    use_contour: bool = True,
) -> np.ndarray:
    """Create a boolean full-wafer mask for one global image region."""

    x0, y0, region_width, region_height = _validate_region(
        geometry, x, y, width, height
    )
    if use_contour and geometry.contour_polygon:
        mask = np.zeros((region_height, region_width), dtype=np.uint8)
        polygon = np.rint(np.asarray(geometry.contour_polygon, dtype=np.float64)).astype(
            np.int32
        )
        polygon[:, 0] -= x0
        polygon[:, 1] -= y0
        cv2.fillPoly(mask, [polygon.reshape(-1, 1, 2)], 1)
        return mask.astype(bool)
    yy, xx = np.ogrid[y0 : y0 + region_height, x0 : x0 + region_width]
    return (xx - geometry.center_x) ** 2 + (yy - geometry.center_y) ** 2 <= geometry.radius_px**2


def create_full_wafer_mask(
    geometry: WaferGeometry,
    shape: tuple[int, int] | None = None,
    *,
    use_contour: bool = True,
) -> np.ndarray:
    """Create the complete segmented-contour (or fallback circle) wafer mask."""

    height, width = shape or (geometry.image_height, geometry.image_width)
    if (height, width) != (geometry.image_height, geometry.image_width):
        raise ValueError(
            "shape must equal the full-resolution dimensions stored in geometry"
        )
    return create_full_wafer_mask_tile(
        geometry, 0, 0, width, height, use_contour=use_contour
    )


def _edge_exclusion_from_full(full_mask: np.ndarray, edge_width_px: float) -> np.ndarray:
    if edge_width_px <= 0.0:
        return np.zeros_like(full_mask, dtype=bool)
    # Zero padding makes a cropped silhouette's image boundary a real boundary.
    padded = np.pad(full_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return full_mask & (distance <= float(edge_width_px))


def create_edge_exclusion_mask(
    full_wafer_mask: np.ndarray,
    geometry: WaferGeometry,
    exclude_edge_mm: float = 0.0,
) -> np.ndarray:
    """Return true pixels in the configured inward edge-exclusion band."""

    width_mm = float(exclude_edge_mm)
    if not math.isfinite(width_mm) or width_mm < 0.0:
        raise ValueError("exclude_edge_mm must be finite and non-negative")
    return _edge_exclusion_from_full(
        np.asarray(full_wafer_mask, dtype=bool), width_mm / geometry.mm_per_pixel
    )


def _parse_rectangle(region: Any) -> tuple[float, float, float, float]:
    if isinstance(region, Mapping):
        if "bbox" in region:
            return _parse_rectangle(region["bbox"])
        x0 = float(region.get("x0", region.get("x", 0.0)))
        y0 = float(region.get("y0", region.get("y", 0.0)))
        if "x1" in region and "y1" in region:
            return x0, y0, float(region["x1"]), float(region["y1"])
        return x0, y0, x0 + float(region["width"]), y0 + float(region["height"])
    values = list(region)
    if len(values) != 4:
        raise ValueError(f"Rectangle must contain four coordinates, got {region!r}")
    x0, y0, x1, y1 = map(float, values)
    return x0, y0, x1, y1


def _split_invalid_regions(
    invalid_regions: Sequence[Any] | None,
    rectangles: Sequence[Any] | None,
    polygons: Sequence[Sequence[Sequence[float]]] | None,
) -> tuple[list[Any], list[Sequence[Sequence[float]]]]:
    all_rectangles = list(rectangles or [])
    all_polygons: list[Sequence[Sequence[float]]] = list(polygons or [])
    for region in invalid_regions or []:
        if isinstance(region, Mapping):
            kind = str(region.get("type", "rectangle")).lower()
            if kind in {"polygon", "poly"}:
                points = region.get("points", region.get("polygon"))
                if points is None:
                    raise ValueError(f"Invalid polygon region lacks points: {region}")
                all_polygons.append(points)
            elif kind in {"rectangle", "rect", "bbox"}:
                all_rectangles.append(region)
            else:
                raise ValueError(f"Unknown invalid region type: {kind}")
        else:
            all_rectangles.append(region)
    return all_rectangles, all_polygons


def create_invalid_mask_tile(
    geometry: WaferGeometry,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    invalid_regions: Sequence[Any] | None = None,
    rectangles: Sequence[Any] | None = None,
    polygons: Sequence[Sequence[Sequence[float]]] | None = None,
    supplied_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize global invalid rectangles/polygons into one mask tile."""

    x0, y0, region_width, region_height = _validate_region(
        geometry, x, y, width, height
    )
    result = np.zeros((region_height, region_width), dtype=np.uint8)
    if supplied_mask is not None:
        supplied = np.asarray(supplied_mask)
        expected = (geometry.image_height, geometry.image_width)
        if supplied.shape != expected:
            raise ValueError(
                f"supplied invalid_mask shape {supplied.shape} does not match {expected}"
            )
        result |= supplied[y0 : y0 + region_height, x0 : x0 + region_width].astype(
            np.uint8
        )
    rectangle_list, polygon_list = _split_invalid_regions(
        invalid_regions, rectangles, polygons
    )
    for rectangle in rectangle_list:
        left, top, right, bottom = _parse_rectangle(rectangle)
        local_left = max(0, int(math.floor(left)) - x0)
        local_top = max(0, int(math.floor(top)) - y0)
        local_right = min(region_width, int(math.ceil(right)) - x0)
        local_bottom = min(region_height, int(math.ceil(bottom)) - y0)
        if local_right > local_left and local_bottom > local_top:
            result[local_top:local_bottom, local_left:local_right] = 1
    for polygon_values in polygon_list:
        polygon = np.asarray(polygon_values, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            raise ValueError(f"Invalid polygon coordinate array: {polygon.shape}")
        polygon = np.rint(polygon).astype(np.int32)
        polygon[:, 0] -= x0
        polygon[:, 1] -= y0
        cv2.fillPoly(result, [polygon.reshape(-1, 1, 2)], 1)
    return result.astype(bool)


def build_analysis_masks(
    geometry: WaferGeometry,
    *,
    exclude_edge_mm: float = 0.0,
    invalid_regions: Sequence[Any] | None = None,
    invalid_rectangles: Sequence[Any] | None = None,
    invalid_polygons: Sequence[Sequence[Sequence[float]]] | None = None,
    invalid_mask: np.ndarray | None = None,
    use_contour: bool = True,
) -> AnalysisMasks:
    """Build full-size wafer, edge, invalid and final valid masks."""

    full = create_full_wafer_mask(geometry, use_contour=use_contour)
    edge = create_edge_exclusion_mask(full, geometry, exclude_edge_mm)
    invalid = create_invalid_mask_tile(
        geometry,
        0,
        0,
        geometry.image_width,
        geometry.image_height,
        invalid_regions=invalid_regions,
        rectangles=invalid_rectangles,
        polygons=invalid_polygons,
        supplied_mask=invalid_mask,
    )
    valid = full & ~edge & ~invalid
    if not np.any(valid):
        raise ValueError("The final valid analysis mask is empty")
    return AnalysisMasks(full, edge, invalid, valid)


def _build_analysis_masks_tile(
    geometry: WaferGeometry,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    exclude_edge_mm: float,
    invalid_regions: Sequence[Any] | None,
    invalid_rectangles: Sequence[Any] | None,
    invalid_polygons: Sequence[Sequence[Sequence[float]]] | None,
    invalid_mask: np.ndarray | None,
    use_contour: bool,
) -> AnalysisMasks:
    edge_width_px = float(exclude_edge_mm) / geometry.mm_per_pixel
    if edge_width_px < 0.0 or not math.isfinite(edge_width_px):
        raise ValueError("exclude_edge_mm must be finite and non-negative")
    # Include enough silhouette halo for exact distance-to-edge classification.
    margin = int(math.ceil(edge_width_px)) + 2 if edge_width_px > 0 else 0
    expanded_x0 = max(0, x - margin)
    expanded_y0 = max(0, y - margin)
    expanded_x1 = min(geometry.image_width, x + width + margin)
    expanded_y1 = min(geometry.image_height, y + height + margin)
    expanded_full = create_full_wafer_mask_tile(
        geometry,
        expanded_x0,
        expanded_y0,
        expanded_x1 - expanded_x0,
        expanded_y1 - expanded_y0,
        use_contour=use_contour,
    )
    expanded_edge = _edge_exclusion_from_full(expanded_full, edge_width_px)
    local_x = x - expanded_x0
    local_y = y - expanded_y0
    selector = (slice(local_y, local_y + height), slice(local_x, local_x + width))
    full = expanded_full[selector]
    edge = expanded_edge[selector]
    invalid = create_invalid_mask_tile(
        geometry,
        x,
        y,
        width,
        height,
        invalid_regions=invalid_regions,
        rectangles=invalid_rectangles,
        polygons=invalid_polygons,
        supplied_mask=invalid_mask,
    )
    valid = full & ~edge & ~invalid
    return AnalysisMasks(full, edge, invalid, valid)


def build_valid_boundary_index(
    geometry: WaferGeometry,
    *,
    tile_size: int = 2048,
    exclude_edge_mm: float = 0.0,
    invalid_regions: Sequence[Any] | None = None,
    invalid_rectangles: Sequence[Any] | None = None,
    invalid_polygons: Sequence[Sequence[Sequence[float]]] | None = None,
    invalid_mask: np.ndarray | None = None,
    use_contour: bool = True,
    max_boundary_points: int = 2_000_000,
    max_region_raster_pixels: int = 8_000_000,
) -> ValidBoundaryIndex:
    """Index the exact final-mask boundary using bounded raster-mask tiles.

    The full wafer mask, edge exclusion, analytic invalid regions, and an
    optional supplied raster invalid mask are combined by the same functions
    used for area calculation.  Only invalid pixels touching a valid pixel are
    retained.  This is sufficient for exact Euclidean nearest-zero queries and
    bounds temporary raster memory by ``tile_size`` plus a one-pixel boundary
    halo and the configured edge-exclusion halo.

    A pathological raster invalid mask can have a boundary proportional to the
    complete image area.  ``max_boundary_points`` is an explicit safety limit:
    exceeding it aborts the analysis instead of silently approximating the
    boundary or allocating an unbounded KD-tree.  Analytic invalid polygons
    are rendered once in their translation-invariant bounding box so their
    OpenCV raster boundary exactly matches full-mask rendering; an independently
    bounded ``max_region_raster_pixels`` controls pathological polygons.
    """

    size = int(tile_size)
    point_limit = int(max_boundary_points)
    region_raster_limit = int(max_region_raster_pixels)
    if size <= 0:
        raise ValueError("tile_size must be positive")
    if point_limit <= 0:
        raise ValueError("max_boundary_points must be positive")
    if region_raster_limit <= 0:
        raise ValueError("max_region_raster_pixels must be positive")
    edge_width_px = float(exclude_edge_mm) / geometry.mm_per_pixel
    if edge_width_px < 0.0 or not math.isfinite(edge_width_px):
        raise ValueError("exclude_edge_mm must be finite and non-negative")
    edge_halo = int(math.ceil(edge_width_px)) + 2 if edge_width_px > 0.0 else 0

    chunks: list[np.ndarray] = []
    point_count = 0
    max_mask_tile_pixels = 0
    adjacency_kernel = np.ones((3, 3), dtype=np.uint8)

    def append_points(points_xy: np.ndarray) -> None:
        nonlocal point_count
        if not len(points_xy):
            return
        point_count += int(len(points_xy))
        if point_count > point_limit:
            raise RuntimeError(
                "Final valid-mask boundary is too complex for the bounded-memory "
                f"index ({point_count:,} points exceeds {point_limit:,}). "
                "Density was not calculated; simplify the supplied raster invalid mask "
                "or analyse it with a larger explicitly reviewed boundary budget."
            )
        chunks.append(np.asarray(points_xy, dtype=np.int32))

    # The padded full-mask distance transform treats the image exterior as
    # invalid.  Represent that same one-pixel exterior ring without allocating
    # a padded full-frame raster.
    append_points(
        np.column_stack(
            (
                np.concatenate(
                    (
                        np.arange(geometry.image_width, dtype=np.int32),
                        np.arange(geometry.image_width, dtype=np.int32),
                        np.full(geometry.image_height, -1, dtype=np.int32),
                        np.full(geometry.image_height, geometry.image_width, dtype=np.int32),
                    )
                ),
                np.concatenate(
                    (
                        np.full(geometry.image_width, -1, dtype=np.int32),
                        np.full(geometry.image_width, geometry.image_height, dtype=np.int32),
                        np.arange(geometry.image_height, dtype=np.int32),
                        np.arange(geometry.image_height, dtype=np.int32),
                    )
                ),
            )
        )
    )

    # Cache analytic invalid regions, but do not index their individual
    # boundaries.  They must first be unioned with the wafer silhouette, edge
    # exclusion, supplied raster mask, and one another.  Otherwise an internal
    # edge hidden by an overlapping exclusion could be mistaken for a boundary
    # of the *final* valid mask.
    #
    # Rasterizing a polygon independently inside each scan tile can change a
    # few OpenCV edge pixels when vertices lie outside that tile.  Render its
    # complete bounding box once, then copy intersecting slices into scan
    # tiles.  This is translation-invariant and matches full-mask fillPoly.
    rectangle_list, polygon_list = _split_invalid_regions(
        invalid_regions, invalid_rectangles, invalid_polygons
    )
    rectangle_cache: list[tuple[int, int, int, int]] = []
    for rectangle in rectangle_list:
        left, top, right, bottom = _parse_rectangle(rectangle)
        x0 = max(0, int(math.floor(left)))
        y0 = max(0, int(math.floor(top)))
        x1 = min(geometry.image_width, int(math.ceil(right)))
        y1 = min(geometry.image_height, int(math.ceil(bottom)))
        if x1 <= x0 or y1 <= y0:
            continue
        rectangle_cache.append((x0, y0, x1, y1))

    polygon_cache: list[tuple[int, int, np.ndarray]] = []
    cached_polygon_pixels = 0
    for polygon_values in polygon_list:
        polygon = np.asarray(polygon_values, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            raise ValueError(f"Invalid polygon coordinate array: {polygon.shape}")
        polygon = np.rint(polygon).astype(np.int32)
        polygon_x0 = int(np.min(polygon[:, 0]))
        polygon_y0 = int(np.min(polygon[:, 1]))
        polygon_x1 = int(np.max(polygon[:, 0])) + 1
        polygon_y1 = int(np.max(polygon[:, 1])) + 1
        polygon_width = polygon_x1 - polygon_x0
        polygon_height = polygon_y1 - polygon_y0
        polygon_pixels = polygon_width * polygon_height
        cached_polygon_pixels += polygon_pixels
        if cached_polygon_pixels > region_raster_limit:
            raise RuntimeError(
                "Invalid polygon rasters are too large for exact bounded rendering "
                f"({cached_polygon_pixels:,} pixels exceeds {region_raster_limit:,}). "
                "Density was not calculated; split the invalid polygon into smaller "
                "regions or use a reviewed raster-mask workflow."
            )
        local_polygon = polygon.copy()
        local_polygon[:, 0] -= polygon_x0
        local_polygon[:, 1] -= polygon_y0
        filled = np.zeros((polygon_height, polygon_width), dtype=np.uint8)
        cv2.fillPoly(filled, [local_polygon.reshape(-1, 1, 2)], 1)
        visible_x0 = max(0, polygon_x0)
        visible_y0 = max(0, polygon_y0)
        visible_x1 = min(geometry.image_width, polygon_x1)
        visible_y1 = min(geometry.image_height, polygon_y1)
        if visible_x1 <= visible_x0 or visible_y1 <= visible_y0:
            continue
        local_x0 = visible_x0 - polygon_x0
        local_y0 = visible_y0 - polygon_y0
        visible = filled[
            local_y0 : local_y0 + (visible_y1 - visible_y0),
            local_x0 : local_x0 + (visible_x1 - visible_x0),
        ].astype(bool, copy=True)
        polygon_cache.append((visible_x0, visible_y0, visible))

    contour: np.ndarray | None = None
    contour_yx: np.ndarray | None = None
    contour_outline = np.empty((0, 2), dtype=np.int32)
    segment_min_x = np.empty(0, dtype=np.int32)
    segment_max_x = np.empty(0, dtype=np.int32)
    segment_min_y = np.empty(0, dtype=np.int32)
    segment_max_y = np.empty(0, dtype=np.int32)
    if use_contour and geometry.contour_polygon:
        contour_float = np.asarray(geometry.contour_polygon, dtype=np.float64)
        if (
            contour_float.ndim != 2
            or contour_float.shape[0] < 3
            or contour_float.shape[1] != 2
            or not np.all(np.isfinite(contour_float))
        ):
            raise ValueError("Wafer contour must contain finite (x, y) coordinates")
        contour = np.rint(contour_float).astype(np.int32)
        contour_yx = contour[:, ::-1]
        segment_start = contour
        segment_end = np.roll(contour, -1, axis=0)
        segment_min_x = np.minimum(segment_start[:, 0], segment_end[:, 0])
        segment_max_x = np.maximum(segment_start[:, 0], segment_end[:, 0])
        segment_min_y = np.minimum(segment_start[:, 1], segment_end[:, 1])
        segment_max_y = np.maximum(segment_start[:, 1], segment_end[:, 1])

        # Full-image fillPoly can be reconstructed exactly as point-in-polygon
        # membership plus its LINE_8 outline.  Build that outline one segment
        # at a time; detected contours are dense, so each temporary bbox is
        # small even when the complete wafer spans a very large image.
        outline_chunks: list[np.ndarray] = []
        raw_outline_points = 0
        for start, end in zip(segment_start, segment_end, strict=True):
            line_x0 = int(min(start[0], end[0]))
            line_y0 = int(min(start[1], end[1]))
            line_x1 = int(max(start[0], end[0])) + 1
            line_y1 = int(max(start[1], end[1])) + 1
            line_pixels = (line_x1 - line_x0) * (line_y1 - line_y0)
            if line_pixels > region_raster_limit:
                raise RuntimeError(
                    "Wafer contour segment is too coarse for exact bounded rendering "
                    f"({line_pixels:,} pixels exceeds {region_raster_limit:,}). "
                    "Density was not calculated; provide a denser contour or manual "
                    "circle geometry."
                )
            line_patch = np.zeros(
                (line_y1 - line_y0, line_x1 - line_x0), dtype=np.uint8
            )
            cv2.line(
                line_patch,
                (int(start[0]) - line_x0, int(start[1]) - line_y0),
                (int(end[0]) - line_x0, int(end[1]) - line_y0),
                1,
                thickness=1,
                lineType=cv2.LINE_8,
            )
            rows, cols = np.nonzero(line_patch)
            if not len(rows):
                continue
            points = np.column_stack((cols + line_x0, rows + line_y0)).astype(
                np.int32, copy=False
            )
            in_image = (
                (points[:, 0] >= 0)
                & (points[:, 0] < geometry.image_width)
                & (points[:, 1] >= 0)
                & (points[:, 1] < geometry.image_height)
            )
            points = points[in_image]
            raw_outline_points += int(len(points))
            if raw_outline_points > point_limit * 2:
                raise RuntimeError(
                    "Wafer contour is too complex for the bounded-memory boundary "
                    "index. Density was not calculated; simplify the contour or use "
                    "a larger explicitly reviewed boundary budget."
                )
            if len(points):
                outline_chunks.append(points)
        if outline_chunks:
            contour_outline = np.unique(np.concatenate(outline_chunks, axis=0), axis=0)

    def exact_full_wafer_region(
        region_x0: int,
        region_y0: int,
        region_x1: int,
        region_y1: int,
    ) -> np.ndarray:
        """Render the full-mask raster exactly without a full-frame allocation."""

        region_width = region_x1 - region_x0
        region_height = region_y1 - region_y0
        if contour is None or contour_yx is None:
            yy, xx = np.ogrid[region_y0:region_y1, region_x0:region_x1]
            return (
                (xx - geometry.center_x) ** 2
                + (yy - geometry.center_y) ** 2
                <= geometry.radius_px**2
            )

        segment_intersects = np.any(
            (segment_max_x >= region_x0)
            & (segment_min_x < region_x1)
            & (segment_max_y >= region_y0)
            & (segment_min_y < region_y1)
        )
        if not segment_intersects:
            inside = bool(
                points_in_poly(
                    np.asarray([[region_y0, region_x0]], dtype=np.float64),
                    contour_yx,
                )[0]
            )
            return np.full((region_height, region_width), inside, dtype=bool)

        result = np.empty((region_height, region_width), dtype=bool)
        # Bound coordinate workspace independently of the output raster.  The
        # result itself is the ordinary tile mask accounted for by
        # max_mask_tile_pixels; point batches stay below about 262k pixels.
        batch_point_limit = min(region_raster_limit, 262_144)
        rows_per_batch = max(1, batch_point_limit // max(1, region_width))
        for local_y0 in range(0, region_height, rows_per_batch):
            local_y1 = min(region_height, local_y0 + rows_per_batch)
            yy, xx = np.indices((local_y1 - local_y0, region_width))
            coordinates_yx = np.column_stack(
                (
                    (yy + region_y0 + local_y0).ravel(),
                    (xx + region_x0).ravel(),
                )
            )
            result[local_y0:local_y1] = points_in_poly(
                coordinates_yx, contour_yx
            ).reshape(local_y1 - local_y0, region_width)

        if len(contour_outline):
            in_region = (
                (contour_outline[:, 0] >= region_x0)
                & (contour_outline[:, 0] < region_x1)
                & (contour_outline[:, 1] >= region_y0)
                & (contour_outline[:, 1] < region_y1)
            )
            outline = contour_outline[in_region]
            result[
                outline[:, 1] - region_y0,
                outline[:, 0] - region_x0,
            ] = True
        return result

    for core_y in range(0, geometry.image_height, size):
        core_height = min(size, geometry.image_height - core_y)
        for core_x in range(0, geometry.image_width, size):
            core_width = min(size, geometry.image_width - core_x)
            x0 = max(0, core_x - 1)
            y0 = max(0, core_y - 1)
            x1 = min(geometry.image_width, core_x + core_width + 1)
            y1 = min(geometry.image_height, core_y + core_height + 1)

            # _build_analysis_masks_tile internally adds the edge-exclusion
            # halo.  Track that largest temporary source mask for audit/tests.
            working_x0 = max(0, x0 - edge_halo)
            working_y0 = max(0, y0 - edge_halo)
            working_x1 = min(geometry.image_width, x1 + edge_halo)
            working_y1 = min(geometry.image_height, y1 + edge_halo)
            max_mask_tile_pixels = max(
                max_mask_tile_pixels,
                (working_x1 - working_x0) * (working_y1 - working_y0),
            )

            expanded_full = exact_full_wafer_region(
                working_x0,
                working_y0,
                working_x1,
                working_y1,
            )
            expanded_edge = _edge_exclusion_from_full(expanded_full, edge_width_px)
            selector = (
                slice(y0 - working_y0, y1 - working_y0),
                slice(x0 - working_x0, x1 - working_x0),
            )
            full = expanded_full[selector]
            edge = expanded_edge[selector]
            raster_invalid = create_invalid_mask_tile(
                geometry,
                x0,
                y0,
                x1 - x0,
                y1 - y0,
                invalid_regions=None,
                rectangles=None,
                polygons=None,
                supplied_mask=invalid_mask,
            )
            valid = full & ~edge & ~raster_invalid

            # Union all analytic exclusions into this halo tile before finding
            # invalid pixels adjacent to valid.  This removes boundaries hidden
            # by overlaps, by the edge-exclusion band, or by the wafer exterior.
            analytic_invalid = np.zeros_like(valid, dtype=bool)
            for rect_x0, rect_y0, rect_x1, rect_y1 in rectangle_cache:
                overlap_x0 = max(x0, rect_x0)
                overlap_y0 = max(y0, rect_y0)
                overlap_x1 = min(x1, rect_x1)
                overlap_y1 = min(y1, rect_y1)
                if overlap_x1 > overlap_x0 and overlap_y1 > overlap_y0:
                    analytic_invalid[
                        overlap_y0 - y0 : overlap_y1 - y0,
                        overlap_x0 - x0 : overlap_x1 - x0,
                    ] = True
            for polygon_x0, polygon_y0, polygon_mask in polygon_cache:
                polygon_y1 = polygon_y0 + polygon_mask.shape[0]
                polygon_x1 = polygon_x0 + polygon_mask.shape[1]
                overlap_x0 = max(x0, polygon_x0)
                overlap_y0 = max(y0, polygon_y0)
                overlap_x1 = min(x1, polygon_x1)
                overlap_y1 = min(y1, polygon_y1)
                if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
                    continue
                analytic_invalid[
                    overlap_y0 - y0 : overlap_y1 - y0,
                    overlap_x0 - x0 : overlap_x1 - x0,
                ] |= polygon_mask[
                    overlap_y0 - polygon_y0 : overlap_y1 - polygon_y0,
                    overlap_x0 - polygon_x0 : overlap_x1 - polygon_x0,
                ]
            valid &= ~analytic_invalid
            valid_u8 = valid.astype(np.uint8, copy=False)
            adjacent_to_valid = cv2.dilate(
                valid_u8,
                adjacency_kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
            invalid_boundary = ~valid & adjacent_to_valid

            local_x = core_x - x0
            local_y = core_y - y0
            core_boundary = invalid_boundary[
                local_y : local_y + core_height,
                local_x : local_x + core_width,
            ]
            rows, cols = np.nonzero(core_boundary)
            if len(rows):
                append_points(
                    np.column_stack((cols + core_x, rows + core_y))
                )

    points = np.concatenate(chunks, axis=0)
    return ValidBoundaryIndex(
        points,
        max_mask_tile_pixels=max_mask_tile_pixels,
        scan_tile_size=size,
    )


def iter_analysis_mask_tiles(
    geometry: WaferGeometry,
    *,
    tile_size: int = 2048,
    exclude_edge_mm: float = 0.0,
    invalid_regions: Sequence[Any] | None = None,
    invalid_rectangles: Sequence[Any] | None = None,
    invalid_polygons: Sequence[Sequence[Sequence[float]]] | None = None,
    invalid_mask: np.ndarray | None = None,
    use_contour: bool = True,
) -> Iterator[AnalysisMaskTile]:
    """Yield non-overlapping mask tiles without allocating full-size masks."""

    size = int(tile_size)
    if size <= 0:
        raise ValueError("tile_size must be positive")
    for y in range(0, geometry.image_height, size):
        height = min(size, geometry.image_height - y)
        for x in range(0, geometry.image_width, size):
            width = min(size, geometry.image_width - x)
            yield AnalysisMaskTile(
                x=x,
                y=y,
                masks=_build_analysis_masks_tile(
                    geometry,
                    x,
                    y,
                    width,
                    height,
                    exclude_edge_mm=exclude_edge_mm,
                    invalid_regions=invalid_regions,
                    invalid_rectangles=invalid_rectangles,
                    invalid_polygons=invalid_polygons,
                    invalid_mask=invalid_mask,
                    use_contour=use_contour,
                ),
            )


def calculate_area_statistics(
    geometry: WaferGeometry,
    masks: AnalysisMasks | None = None,
    *,
    tile_size: int = 2048,
    exclude_edge_mm: float = 0.0,
    invalid_regions: Sequence[Any] | None = None,
    invalid_rectangles: Sequence[Any] | None = None,
    invalid_polygons: Sequence[Sequence[Sequence[float]]] | None = None,
    invalid_mask: np.ndarray | None = None,
    use_contour: bool = True,
) -> AreaStatistics:
    """Count final mask pixels exactly, optionally in bounded-memory tiles."""

    full_count = edge_count = invalid_count = valid_count = 0
    tiles: Iterator[AnalysisMaskTile]
    if masks is not None:
        tiles = iter(
            [AnalysisMaskTile(x=0, y=0, masks=masks)]
        )
    else:
        tiles = iter_analysis_mask_tiles(
            geometry,
            tile_size=tile_size,
            exclude_edge_mm=exclude_edge_mm,
            invalid_regions=invalid_regions,
            invalid_rectangles=invalid_rectangles,
            invalid_polygons=invalid_polygons,
            invalid_mask=invalid_mask,
            use_contour=use_contour,
        )
    for tile in tiles:
        tile_masks = tile.masks
        full = np.asarray(tile_masks.full_wafer_mask, dtype=bool)
        edge = full & np.asarray(tile_masks.edge_exclusion_mask, dtype=bool)
        # Report invalid area after edge removal so exclusions reconcile exactly.
        invalid = full & ~edge & np.asarray(tile_masks.invalid_mask, dtype=bool)
        valid = full & ~edge & ~invalid
        full_count += int(np.count_nonzero(full))
        edge_count += int(np.count_nonzero(edge))
        invalid_count += int(np.count_nonzero(invalid))
        valid_count += int(np.count_nonzero(valid))
    if full_count != edge_count + invalid_count + valid_count:
        raise RuntimeError("Internal mask area reconciliation failed")
    if valid_count == 0:
        raise ValueError("The final valid analysis area is zero")
    return AreaStatistics(
        full_wafer_pixel_count=full_count,
        edge_excluded_pixel_count=edge_count,
        invalid_pixel_count=invalid_count,
        valid_pixel_count=valid_count,
        pixel_area_cm2=geometry.pixel_area_cm2,
        theoretical_area_cm2=geometry.theoretical_area_cm2,
        circle_fit_area_cm2=geometry.circle_fit_area_cm2,
    )


def mask_area_cm2(mask: np.ndarray, geometry: WaferGeometry) -> float:
    """Convert the number of true mask pixels to square centimetres."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {array.shape}")
    return float(np.count_nonzero(array)) * geometry.pixel_area_cm2


def image_to_wafer_coordinates(
    x_px: float | np.ndarray,
    y_px: float | np.ndarray,
    geometry: WaferGeometry,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert image coordinates to millimetres about the wafer centre.

    Image y increases downwards; wafer y is defined positive upwards.
    """

    x_value = np.asarray(x_px, dtype=float)
    y_value = np.asarray(y_px, dtype=float)
    x_mm = (x_value - geometry.center_x) * geometry.mm_per_pixel
    y_mm = -(y_value - geometry.center_y) * geometry.mm_per_pixel
    if x_mm.ndim == 0 and y_mm.ndim == 0:
        return float(x_mm), float(y_mm)
    return x_mm, y_mm


def distance_to_fitted_wafer_edge_mm(
    x_px: float | np.ndarray,
    y_px: float | np.ndarray,
    geometry: WaferGeometry,
) -> float | np.ndarray:
    """Signed inward distance to the fitted circle (negative outside)."""

    radial_px = np.hypot(
        np.asarray(x_px, dtype=float) - geometry.center_x,
        np.asarray(y_px, dtype=float) - geometry.center_y,
    )
    result = (geometry.radius_px - radial_px) * geometry.mm_per_pixel
    return float(result) if np.ndim(result) == 0 else result


# Concise aliases useful to callers/tests.
compute_pixel_scale = pixel_scale_from_diameter
calculate_valid_area = calculate_area_statistics


__all__ = [
    "AnalysisMaskTile",
    "AnalysisMasks",
    "AreaStatistics",
    "ValidBoundaryIndex",
    "WaferDetectionError",
    "WaferGeometry",
    "build_analysis_masks",
    "build_valid_boundary_index",
    "calculate_area_statistics",
    "calculate_valid_area",
    "compute_pixel_scale",
    "create_edge_exclusion_mask",
    "create_full_wafer_mask",
    "create_full_wafer_mask_tile",
    "create_invalid_mask_tile",
    "detect_wafer",
    "distance_to_fitted_wafer_edge_mm",
    "image_to_wafer_coordinates",
    "iter_analysis_mask_tiles",
    "mask_area_cm2",
    "pixel_scale_from_diameter",
    "theoretical_wafer_area_cm2",
]
