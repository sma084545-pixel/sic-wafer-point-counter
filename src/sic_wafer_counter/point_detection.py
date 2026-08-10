"""Candidate detection, conservative watershed splitting, and tile deduplication."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from skimage.feature import blob_dog, peak_local_max
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

from .preprocessing import as_float01

LOGGER = logging.getLogger(__name__)

FloatImage = NDArray[np.float32]
BoolImage = NDArray[np.bool_]
LabelImage = NDArray[np.int32]


def _float01_view_or_copy(image: NDArray[np.generic]) -> FloatImage:
    """Reuse normalized float32 data, otherwise return a normalized copy."""

    array = np.asarray(image)
    if array.dtype == np.float32 and array.ndim == 2 and array.size:
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


def _odd(value: int, *, name: str, allow_one: bool = True) -> int:
    minimum = 1 if allow_one else 3
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if value % 2 == 0:
        LOGGER.warning("%s=%d is even; using %d", name, value, value + 1)
        value += 1
    return value


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Parameters for black-hat/DoG candidate detection and splitting."""

    method: str = "blackhat"
    threshold_method: str = "otsu"
    threshold_offset: float = 0.0
    otsu_classes: int = 3
    threshold_quantile: float = 99.2
    adaptive_block_size: int = 51
    adaptive_c: float = -2.0
    min_area_px: int = 5
    max_area_px: int = 5000
    opening_kernel_px: int = 1
    closing_kernel_px: int = 3
    use_watershed: bool = True
    min_peak_distance_px: int = 5
    watershed_compactness: float = 0.0
    watershed_min_component_area_px: int = 20
    watershed_max_aspect_ratio: float = 4.0
    watershed_max_eccentricity: float = 0.92
    watershed_min_component_circularity: float = 0.20
    watershed_min_peak_rel_height: float = 0.35
    dog_min_sigma_px: float = 1.0
    dog_max_sigma_px: float = 6.0
    dog_sigma_ratio: float = 1.6
    dog_threshold: float = 0.04
    combine_mode: str = "union"
    blackhat_kernel_sizes_px: tuple[int, ...] = (9, 17, 31)
    connectivity: int = 2

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "DetectionConfig":
        """Build from a full YAML mapping or an already scoped section."""

        if mapping is None:
            return cls()
        values = _section(mapping, "detection")
        kwargs = {
            key: values[key]
            for key in cls.__dataclass_fields__.keys()
            if key in values
        }
        if "blackhat_kernel_sizes_px" in kwargs:
            kwargs["blackhat_kernel_sizes_px"] = tuple(
                int(item) for item in kwargs["blackhat_kernel_sizes_px"]
            )
        return cls(**kwargs)

    def validated(self) -> "DetectionConfig":
        """Normalize aliases and reject inconsistent parameters."""

        method = self.method.lower().strip().replace("+", "_")
        aliases = {
            "both": "combine",
            "combined": "combine",
            "blackhat_dog": "combine",
            "dog_blackhat": "combine",
            "log": "dog",  # accepted as a close blob-detector fallback
        }
        method = aliases.get(method, method)
        if method not in {"blackhat", "dog", "combine"}:
            raise ValueError("method must be blackhat, dog, or combine")
        threshold = self.threshold_method.lower().strip()
        if threshold == "percentile":
            threshold = "quantile"
        if threshold not in {"otsu", "adaptive", "quantile"}:
            raise ValueError("threshold_method must be otsu, adaptive, or quantile")
        combine = self.combine_mode.lower().strip()
        if combine not in {"union", "intersection"}:
            raise ValueError("combine_mode must be union or intersection")
        if not 0.0 <= float(self.threshold_quantile) <= 100.0:
            raise ValueError("threshold_quantile must lie in [0, 100]")
        if not 2 <= int(self.otsu_classes) <= 5:
            raise ValueError("otsu_classes must be between 2 and 5")
        if self.min_area_px < 1 or self.max_area_px < self.min_area_px:
            raise ValueError("Require 1 <= min_area_px <= max_area_px")
        if self.dog_min_sigma_px <= 0 or self.dog_max_sigma_px < self.dog_min_sigma_px:
            raise ValueError("Invalid DoG sigma range")
        if self.dog_sigma_ratio <= 1.0 or self.dog_threshold < 0:
            raise ValueError("dog_sigma_ratio must exceed 1 and threshold be non-negative")
        if self.min_peak_distance_px < 1:
            raise ValueError("min_peak_distance_px must be >= 1")
        if self.watershed_max_aspect_ratio < 1.0:
            raise ValueError("watershed_max_aspect_ratio must be >= 1")
        if not 0.0 <= self.watershed_max_eccentricity <= 1.0:
            raise ValueError("watershed_max_eccentricity must lie in [0, 1]")
        if not 0.0 <= self.watershed_min_component_circularity <= 1.0:
            raise ValueError("watershed_min_component_circularity must lie in [0, 1]")
        if not 0.0 <= self.watershed_min_peak_rel_height <= 1.0:
            raise ValueError("watershed_min_peak_rel_height must lie in [0, 1]")
        if self.connectivity not in {1, 2}:
            raise ValueError("connectivity must be 1 or 2")
        kernels = tuple(
            _odd(item, name="blackhat_kernel_sizes_px")
            for item in self.blackhat_kernel_sizes_px
        )
        if not kernels:
            raise ValueError("blackhat_kernel_sizes_px cannot be empty")
        return replace(
            self,
            method=method,
            threshold_method=threshold,
            otsu_classes=int(self.otsu_classes),
            combine_mode=combine,
            adaptive_block_size=_odd(
                self.adaptive_block_size, name="adaptive_block_size", allow_one=False
            ),
            opening_kernel_px=_odd(self.opening_kernel_px, name="opening_kernel_px"),
            closing_kernel_px=_odd(self.closing_kernel_px, name="closing_kernel_px"),
            blackhat_kernel_sizes_px=kernels,
        )


@dataclass(frozen=True, slots=True)
class CandidateRegion:
    """Compact representation of one candidate in global pixel coordinates."""

    candidate_id: int
    centroid_x_px: float
    centroid_y_px: float
    bbox: tuple[int, int, int, int]
    area_px: int
    tile_index: int | None = None
    core_distance_px: float = 0.0
    coords_yx: NDArray[np.int64] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Full-frame candidate detection result."""

    response: FloatImage
    candidate_mask_before_watershed: BoolImage
    candidate_mask: BoolImage
    labels: LabelImage
    pre_watershed_count: int
    post_watershed_count: int
    threshold_value: float | None
    blackhat_mask: BoolImage | None
    dog_mask: BoolImage | None
    config: DetectionConfig
    warnings: tuple[str, ...] = ()

    @property
    def before_count(self) -> int:
        """Compatibility name for the connected count before watershed."""

        return self.pre_watershed_count

    @property
    def after_count(self) -> int:
        """Compatibility name for the connected count after watershed."""

        return self.post_watershed_count

    @property
    def candidates(self) -> list[CandidateRegion]:
        """Describe labeled objects without copying their coordinate arrays."""

        return regions_from_labels(self.labels, include_coords=False)


@dataclass(frozen=True, slots=True)
class TiledDetectionResult:
    """Candidates detected in overlapping tiles and mapped to the source frame."""

    candidates: tuple[CandidateRegion, ...]
    image_shape: tuple[int, int]
    pre_watershed_count: int
    post_watershed_count: int
    tile_count: int
    threshold_value: float | None = None
    labels: LabelImage | None = None
    warnings: tuple[str, ...] = ()


def multiscale_blackhat(
    image: NDArray[np.generic], kernel_sizes_px: Sequence[int] = (9, 17, 31)
) -> FloatImage:
    """Enhance dark objects by taking the maximum black-hat response over scales."""

    gray = as_float01(image)
    return _multiscale_blackhat_float(gray, kernel_sizes_px)


def _multiscale_blackhat_float(
    gray: FloatImage, kernel_sizes_px: Sequence[int]
) -> FloatImage:
    """Black-hat implementation for an already normalized float image."""

    response = np.zeros(gray.shape, dtype=np.float32)
    for raw_size in kernel_sizes_px:
        size = _odd(int(raw_size), name="blackhat_kernel_sizes_px")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        current = cv2.morphologyEx(
            gray, cv2.MORPH_BLACKHAT, kernel, borderType=cv2.BORDER_REFLECT
        )
        np.maximum(response, current, out=response)
    return response


def estimate_response_threshold(
    response: FloatImage, valid_mask: BoolImage, config: DetectionConfig
) -> float | None:
    """Estimate one global Otsu/quantile threshold with bounded sample memory.

    Adaptive thresholding intentionally returns ``None`` because it has no
    single global cutoff.
    """

    if config.threshold_method == "adaptive":
        return None
    # Sample before boolean indexing.  On a 100-megapixel wafer this avoids a
    # second, potentially hundreds-of-MB 1-D copy just to estimate a histogram.
    sample_stride = max(1, int(math.ceil(math.sqrt(response.size / 2_000_000))))
    sampled_response = response[::sample_stride, ::sample_stride]
    sampled_mask = valid_mask[::sample_stride, ::sample_stride]
    values = sampled_response[sampled_mask]
    finite = values if np.isfinite(values).all() else values[np.isfinite(values)]
    valid_max = float(np.max(response, where=valid_mask, initial=0.0))
    if finite.size == 0 or not np.isfinite(valid_max) or valid_max <= 0.0:
        return None
    if config.threshold_method == "quantile":
        return float(np.percentile(finite, config.threshold_quantile)) + float(
            config.threshold_offset
        )

    # Three-class Otsu separates background / point targets / very dark lines
    # or blobs, and the first boundary is intentionally used.  This prevents a
    # few large high-response artifacts from raising a binary Otsu threshold
    # enough to hide legitimate low-contrast points.  otsu_classes=2 restores
    # classic binary Otsu behavior.
    try:
        thresholds = threshold_multiotsu(
            finite, classes=config.otsu_classes, nbins=256
        )
        return float(thresholds[0]) + float(config.threshold_offset)
    except ValueError:
        # Too few distinct levels for multi-Otsu: deterministic binary Otsu is
        # a safe fallback and handles constant-ish tiles gracefully.
        valid_u8 = np.rint(np.clip(finite, 0.0, 1.0) * 255.0).astype(np.uint8)
        otsu_u8, _ = cv2.threshold(
            valid_u8.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        return float(otsu_u8) / 255.0 + float(config.threshold_offset)


def _threshold_response(
    response: FloatImage,
    valid_mask: BoolImage,
    config: DetectionConfig,
    forced_threshold: float | None = None,
) -> tuple[BoolImage, float | None]:
    """Threshold a bright-target response inside the analysis mask."""

    if forced_threshold is not None:
        threshold_value = float(forced_threshold)
        if not np.isfinite(threshold_value):
            raise ValueError("forced_threshold must be finite")
        binary = response > threshold_value
    elif config.threshold_method == "adaptive":
        response_u8 = np.rint(np.clip(response, 0.0, 1.0) * 255.0).astype(np.uint8)
        binary_u8 = cv2.adaptiveThreshold(
            response_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            config.adaptive_block_size,
            float(config.adaptive_c),
        )
        binary = binary_u8.astype(bool)
        threshold_value: float | None = None
    else:
        threshold_value = estimate_response_threshold(response, valid_mask, config)
        binary = (
            np.zeros(response.shape, dtype=bool)
            if threshold_value is None
            else response > threshold_value
        )
    binary &= valid_mask
    return np.asarray(binary, dtype=bool), threshold_value


def _dog_mask(
    response: FloatImage, valid_mask: BoolImage, config: DetectionConfig
) -> BoolImage:
    """Convert scikit-image DoG blob centers/scales into a candidate mask."""

    work = response.copy()
    work[~valid_mask] = 0.0
    blobs = blob_dog(
        work,
        min_sigma=config.dog_min_sigma_px,
        max_sigma=config.dog_max_sigma_px,
        sigma_ratio=config.dog_sigma_ratio,
        threshold=config.dog_threshold,
        overlap=0.5,
        exclude_border=False,
    )
    mask = np.zeros(response.shape, dtype=np.uint8)
    for y, x, sigma in blobs:
        radius = max(1, int(math.ceil(math.sqrt(2.0) * float(sigma))))
        cv2.circle(mask, (int(round(x)), int(round(y))), radius, 1, thickness=-1)
    return np.asarray(mask.astype(bool) & valid_mask, dtype=bool)


def _clean_mask(mask: BoolImage, config: DetectionConfig) -> BoolImage:
    """Perform small opening/closing and remove isolated tiny components."""

    work = mask.astype(np.uint8)
    if config.opening_kernel_px > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.opening_kernel_px, config.opening_kernel_px),
        )
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
    if config.closing_kernel_px > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.closing_kernel_px, config.closing_kernel_px),
        )
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
    # ``remove_small_objects(min_size=...)`` is being renamed in newer
    # scikit-image releases.  A bincount over connected labels is simple,
    # deterministic, and avoids carrying that deprecation warning into every
    # scientific run log.
    component_labels = label(work.astype(bool), connectivity=config.connectivity)
    sizes = np.bincount(component_labels.ravel())
    keep = sizes >= config.min_area_px
    keep[0] = False
    return np.asarray(keep[component_labels], dtype=bool)


def _conservative_watershed(
    component_labels: LabelImage, config: DetectionConfig
) -> LabelImage:
    """Split compact, sufficiently large components one at a time.

    Elongated source components are deliberately *not* split.  This is crucial:
    splitting a scratch or scale bar along its length would turn one rejectable
    line into many superficially point-like false positives.
    """

    output = np.zeros(component_labels.shape, dtype=np.int32)
    next_label = 1
    for region in regionprops(component_labels):
        min_row, min_col, max_row, max_col = region.bbox
        component = component_labels[min_row:max_row, min_col:max_col] == region.label
        height, width = component.shape
        bbox_aspect = max(height, width) / max(1.0, min(height, width))
        perimeter = float(region.perimeter)
        circularity = (
            float(4.0 * math.pi * region.area / (perimeter * perimeter))
            if perimeter > 0.0
            else 0.0
        )
        eligible = (
            region.area >= config.watershed_min_component_area_px
            and region.area <= config.max_area_px
            and bbox_aspect <= config.watershed_max_aspect_ratio
            and region.eccentricity <= config.watershed_max_eccentricity
            and circularity >= config.watershed_min_component_circularity
        )
        local_labels: LabelImage | None = None
        if eligible:
            distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
            max_distance = float(distance.max(initial=0.0))
            if max_distance > 0.0:
                peaks = peak_local_max(
                    distance,
                    min_distance=config.min_peak_distance_px,
                    threshold_abs=max_distance * config.watershed_min_peak_rel_height,
                    labels=component.astype(np.uint8),
                    exclude_border=False,
                )
                # Area limits the number of plausible objects.  It is a second
                # guard against noisy distance ridges causing over-segmentation.
                max_peaks = max(1, int(math.ceil(region.area / config.min_area_px)))
                peaks = peaks[:max_peaks]
                if len(peaks) >= 2:
                    markers = np.zeros(component.shape, dtype=np.int32)
                    for marker_id, (row, col) in enumerate(peaks, start=1):
                        markers[row, col] = marker_id
                    local_labels = np.asarray(
                        watershed(
                            -distance,
                            markers=markers,
                            mask=component,
                            compactness=config.watershed_compactness,
                        ),
                        dtype=np.int32,
                    )

        target = output[min_row:max_row, min_col:max_col]
        if local_labels is None:
            target[component] = next_label
            next_label += 1
        else:
            for local_id in range(1, int(local_labels.max(initial=0)) + 1):
                pixels = local_labels == local_id
                if not pixels.any():
                    continue
                target[pixels] = next_label
                next_label += 1
    return output


def detect_candidates(
    image: NDArray[np.generic],
    valid_mask: NDArray[np.bool_] | None = None,
    config: DetectionConfig | Mapping[str, Any] | None = None,
    *,
    dark_response: NDArray[np.generic] | None = None,
    threshold_value: float | None = None,
) -> DetectionResult:
    """Detect dark point-like candidate regions in one image or tile.

    ``image`` should be the filtered grayscale image.  Supplying the
    preprocessing ``dark_response`` avoids recomputing black-hat enhancement;
    otherwise a multi-scale morphological black-hat is calculated from
    ``image``.  DoG operates on that same bright-target response.  A supplied
    ``threshold_value`` makes overlap tiles use one globally calibrated cutoff.
    """

    cfg = (
        config.validated()
        if isinstance(config, DetectionConfig)
        else DetectionConfig.from_mapping(config).validated()
    )
    source = np.asarray(image)
    if source.ndim != 2 or source.size == 0:
        raise ValueError(f"Expected a non-empty 2-D image, got {source.shape}")
    # When preprocessing already supplied the response, the grayscale array is
    # needed only for shape validation.  Avoid normalizing/copying another full
    # frame in that common large-image path.
    gray = as_float01(source) if dark_response is None else source
    if valid_mask is None:
        mask = np.ones(gray.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != gray.shape:
            raise ValueError(f"valid_mask {mask.shape} does not match image {gray.shape}")
        mask = mask.copy()
    if not mask.any():
        raise ValueError("valid_mask contains no valid pixels")

    if dark_response is None:
        response = _multiscale_blackhat_float(gray, cfg.blackhat_kernel_sizes_px)
    else:
        response = as_float01(np.asarray(dark_response))
        if response.shape != gray.shape:
            raise ValueError("dark_response shape must match image")
    response = np.asarray(response, dtype=np.float32)
    response[~mask] = 0.0

    blackhat_mask: BoolImage | None = None
    dog_mask: BoolImage | None = None
    requested_threshold = threshold_value
    threshold_value = None
    if cfg.method in {"blackhat", "combine"}:
        blackhat_mask, measured_threshold = _threshold_response(
            response, mask, cfg, forced_threshold=requested_threshold
        )
        threshold_value = measured_threshold
    if cfg.method in {"dog", "combine"}:
        dog_mask = _dog_mask(response, mask, cfg)
    if cfg.method == "blackhat":
        assert blackhat_mask is not None
        candidates = blackhat_mask
    elif cfg.method == "dog":
        assert dog_mask is not None
        candidates = dog_mask
    elif cfg.combine_mode == "intersection":
        assert blackhat_mask is not None and dog_mask is not None
        candidates = blackhat_mask & dog_mask
    else:
        assert blackhat_mask is not None and dog_mask is not None
        candidates = blackhat_mask | dog_mask

    candidates = _clean_mask(np.asarray(candidates, dtype=bool) & mask, cfg)
    pre_labels = np.asarray(
        label(candidates, connectivity=cfg.connectivity), dtype=np.int32
    )
    pre_count = int(pre_labels.max(initial=0))
    if cfg.use_watershed and pre_count:
        final_labels = _conservative_watershed(pre_labels, cfg)
    else:
        final_labels = pre_labels.copy()
    final_labels[~mask] = 0
    post_count = int(final_labels.max(initial=0))
    final_mask = final_labels > 0
    warnings: list[str] = []
    if pre_count and post_count > max(pre_count * 4, pre_count + 100):
        warning = (
            f"Watershed increased candidate count from {pre_count} to {post_count}; "
            "inspect possible over-segmentation"
        )
        LOGGER.warning(warning)
        warnings.append(warning)
    LOGGER.info(
        "Candidate detection (%s/%s): %d before watershed, %d after",
        cfg.method,
        cfg.threshold_method,
        pre_count,
        post_count,
    )
    return DetectionResult(
        response=response,
        candidate_mask_before_watershed=candidates,
        candidate_mask=np.asarray(final_mask, dtype=bool),
        labels=final_labels,
        pre_watershed_count=pre_count,
        post_watershed_count=post_count,
        threshold_value=threshold_value,
        blackhat_mask=blackhat_mask,
        dog_mask=dog_mask,
        config=cfg,
        warnings=tuple(warnings),
    )


def detection_from_probability(
    probability: NDArray[np.generic],
    valid_mask: NDArray[np.bool_],
    config: DetectionConfig | Mapping[str, Any] | None = None,
    *,
    probability_threshold: float,
    minimum_object_area_px: int | None = None,
) -> DetectionResult:
    """Convert a pixel-model probability map into auditable candidates.

    This deliberately reuses the detector's morphology cleanup and conservative
    watershed.  The probability model therefore proposes foreground pixels but
    cannot bypass the final valid mask, edge distance or feature filters.
    """

    cfg = (
        config.validated()
        if isinstance(config, DetectionConfig)
        else DetectionConfig.from_mapping(config).validated()
    )
    values = np.asarray(probability, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 2 or values.shape != mask.shape or not np.isfinite(values).all():
        raise ValueError("probability and valid_mask must be finite, equally sized 2-D arrays")
    threshold = float(probability_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("probability_threshold must lie strictly between 0 and 1")
    if not mask.any():
        raise ValueError("valid_mask contains no valid pixels")
    if minimum_object_area_px is not None:
        minimum = int(minimum_object_area_px)
        if minimum < 1:
            raise ValueError("minimum_object_area_px must be >= 1")
        cfg = replace(cfg, min_area_px=minimum).validated()
    response = values.copy()
    response[~mask] = 0.0
    candidates = _clean_mask((response >= threshold) & mask, cfg)
    pre_labels = np.asarray(label(candidates, connectivity=cfg.connectivity), dtype=np.int32)
    pre_count = int(pre_labels.max(initial=0))
    if cfg.use_watershed and pre_count:
        final_labels = _conservative_watershed(pre_labels, cfg)
    else:
        final_labels = pre_labels.copy()
    final_labels[~mask] = 0
    post_count = int(final_labels.max(initial=0))
    warnings: list[str] = []
    if pre_count and post_count > max(pre_count * 4, pre_count + 100):
        warnings.append(
            f"Pixel-model watershed increased candidates from {pre_count} to {post_count}; "
            "inspect possible over-segmentation"
        )
    return DetectionResult(
        response=response,
        candidate_mask_before_watershed=candidates,
        candidate_mask=final_labels > 0,
        labels=final_labels,
        pre_watershed_count=pre_count,
        post_watershed_count=post_count,
        threshold_value=threshold,
        blackhat_mask=None,
        dog_mask=None,
        config=cfg,
        warnings=tuple(warnings),
    )


def regions_from_labels(
    labels: NDArray[np.integer],
    *,
    coordinate_offset_xy: tuple[int, int] = (0, 0),
    tile_index: int | None = None,
    include_coords: bool = False,
    core_bounds_xyxy: tuple[int, int, int, int] | None = None,
) -> list[CandidateRegion]:
    """Convert a label image to candidate records in global coordinates."""

    offset_x, offset_y = coordinate_offset_xy
    result: list[CandidateRegion] = []
    for region in regionprops(np.asarray(labels, dtype=np.int32)):
        cy, cx = region.centroid
        gx, gy = float(cx + offset_x), float(cy + offset_y)
        min_row, min_col, max_row, max_col = region.bbox
        bbox = (
            int(min_col + offset_x),
            int(min_row + offset_y),
            int(max_col + offset_x),
            int(max_row + offset_y),
        )
        coords: NDArray[np.int64] | None = None
        if include_coords:
            coords = np.asarray(region.coords, dtype=np.int64).copy()
            coords[:, 0] += offset_y
            coords[:, 1] += offset_x
        core_distance = 0.0
        if core_bounds_xyxy is not None:
            x0, y0, x1, y1 = core_bounds_xyxy
            core_distance = float(min(gx - x0, x1 - gx, gy - y0, y1 - gy))
        result.append(
            CandidateRegion(
                candidate_id=int(region.label),
                centroid_x_px=gx,
                centroid_y_px=gy,
                bbox=bbox,
                area_px=int(region.area),
                tile_index=tile_index,
                core_distance_px=core_distance,
                coords_yx=coords,
            )
        )
    return result


def _bbox_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    width = max(0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0, min(ay1, by1) - max(ay0, by0))
    intersection = width * height
    if not intersection:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return float(intersection / union) if union else 0.0


def deduplicate_candidates(
    candidates: Sequence[CandidateRegion],
    *,
    max_centroid_distance_px: float = 3.0,
    min_bbox_iou: float = 0.30,
    min_pixel_overlap_fraction: float = 0.05,
) -> list[CandidateRegion]:
    """Deduplicate overlap-tile candidates using centroid distance or bbox IoU.

    Within a duplicate group, the record whose centroid lies deepest inside its
    tile core is retained.  This favors the least boundary-affected measurement.
    """

    count = len(candidates)
    if count < 2:
        return [replace(item, candidate_id=index) for index, item in enumerate(candidates, 1)]
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    def is_cross_tile(left: int, right: int) -> bool:
        left_tile, right_tile = candidates[left].tile_index, candidates[right].tile_index
        # None denotes a generic caller-provided list where tile provenance is
        # unavailable; explicit equal tile IDs can never be duplicates.
        return left_tile is None or right_tile is None or left_tile != right_tile

    centers = np.asarray(
        [(item.centroid_x_px, item.centroid_y_px) for item in candidates], dtype=float
    )
    if max_centroid_distance_px > 0:
        for left, right in cKDTree(centers).query_pairs(max_centroid_distance_px):
            if is_cross_tile(int(left), int(right)):
                union(int(left), int(right))

    # A sweep-line catches unusually large overlap duplicates whose centroid
    # shift exceeds the distance threshold without an all-pairs comparison.
    ordering = sorted(range(count), key=lambda index: candidates[index].bbox[0])
    active: list[int] = []
    pixel_overlap_pairs: list[tuple[int, int]] = []
    for index in ordering:
        min_x = candidates[index].bbox[0]
        active = [other for other in active if candidates[other].bbox[2] > min_x]
        for other in active:
            if not is_cross_tile(index, other):
                continue
            iou = _bbox_iou(candidates[index].bbox, candidates[other].bbox)
            if iou >= min_bbox_iou:
                union(index, other)
            elif iou > 0.0:
                pixel_overlap_pairs.append((index, other))
        active.append(index)

    # A scratch can span far more than one tile, giving two partial bboxes a low
    # IoU even though their halo pixels overlap.  Count exact shared candidate
    # pixels and merge only across tiles.  The fraction is relative to the
    # smaller region, so the default 5% works with a 128 px halo on a 2048 px
    # tile while remaining too strict for a merely adjacent watershed boundary.
    if min_pixel_overlap_fraction > 0:
        for left, right in pixel_overlap_pairs:
            left_coords = candidates[left].coords_yx
            right_coords = candidates[right].coords_yx
            if left_coords is None or right_coords is None:
                continue
            # Packed int64 coordinates make an exact vectorized intersection
            # without a global Python dictionary proportional to all dark pixels.
            left_keys = (left_coords[:, 0] << 32) | left_coords[:, 1]
            right_keys = (right_coords[:, 0] << 32) | right_coords[:, 1]
            overlap_count = np.intersect1d(
                left_keys, right_keys, assume_unique=True
            ).size
            denominator = min(candidates[left].area_px, candidates[right].area_px)
            if denominator and overlap_count / denominator >= min_pixel_overlap_fraction:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    retained: list[CandidateRegion] = []
    for members in groups.values():
        best = max(
            members,
            key=lambda index: (
                candidates[index].core_distance_px,
                candidates[index].area_px,
            ),
        )
        coordinate_arrays = [candidates[index].coords_yx for index in members]
        if len(members) > 1 and all(coords is not None for coords in coordinate_arrays):
            merged_coords = np.unique(
                np.concatenate(
                    [coords for coords in coordinate_arrays if coords is not None], axis=0
                ),
                axis=0,
            )
            min_row, min_col = np.min(merged_coords, axis=0)
            max_row, max_col = np.max(merged_coords, axis=0) + 1
            retained.append(
                replace(
                    candidates[best],
                    centroid_x_px=float(np.mean(merged_coords[:, 1])),
                    centroid_y_px=float(np.mean(merged_coords[:, 0])),
                    bbox=(int(min_col), int(min_row), int(max_col), int(max_row)),
                    area_px=int(len(merged_coords)),
                    coords_yx=merged_coords,
                )
            )
        else:
            retained.append(candidates[best])
    retained.sort(key=lambda item: (item.centroid_y_px, item.centroid_x_px))
    return [replace(item, candidate_id=index) for index, item in enumerate(retained, 1)]


def detect_candidates_tiled(
    image: NDArray[np.generic],
    valid_mask: NDArray[np.bool_] | None = None,
    config: DetectionConfig | Mapping[str, Any] | None = None,
    *,
    dark_response: NDArray[np.generic] | None = None,
    threshold_value: float | None = None,
    tile_size: int = 2048,
    tile_overlap: int = 128,
    build_label_image: bool = False,
    max_centroid_distance_px: float = 3.0,
    min_bbox_iou: float = 0.30,
    min_pixel_overlap_fraction: float = 0.05,
) -> TiledDetectionResult:
    """Detect candidates in overlap-expanded tiles using a unique core rule.

    The image passed here is already addressable as an array.  TIFF/BigTIFF
    backends may call the same per-tile :func:`detect_candidates` function while
    reading windows lazily.  Thus this function solves overlap ownership,
    global-coordinate mapping, and deduplication, but does not claim to make an
    already materialized NumPy array lazy.  When a full dark response is
    available, one global Otsu/quantile threshold is estimated and reused in all
    tiles; callers streaming lazy tiles can supply a preview-derived
    ``threshold_value`` for the same behavior.
    """

    gray = np.asarray(image)
    if gray.ndim != 2:
        raise ValueError("detect_candidates_tiled expects a 2-D image")
    height, width = gray.shape
    tile_size, tile_overlap = int(tile_size), int(tile_overlap)
    if tile_size < 32:
        raise ValueError("tile_size must be >= 32")
    if tile_overlap < 0 or tile_overlap * 2 >= tile_size:
        raise ValueError("Require 0 <= tile_overlap < tile_size / 2")
    mask = np.ones(gray.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if mask.shape != gray.shape:
        raise ValueError("valid_mask shape must match image")
    cfg = (
        config.validated()
        if isinstance(config, DetectionConfig)
        else DetectionConfig.from_mapping(config).validated()
    )
    response_full = None if dark_response is None else np.asarray(dark_response)
    if response_full is not None and response_full.shape != gray.shape:
        raise ValueError("dark_response shape must match image")
    global_threshold = threshold_value
    if (
        global_threshold is None
        and response_full is not None
        and cfg.method in {"blackhat", "combine"}
        and cfg.threshold_method != "adaptive"
    ):
        response_for_threshold = _float01_view_or_copy(response_full)
        global_threshold = estimate_response_threshold(
            response_for_threshold, mask, cfg
        )

    retained: list[CandidateRegion] = []
    pre_owned_count = 0
    tile_count = 0
    for core_y0 in range(0, height, tile_size):
        core_y1 = min(height, core_y0 + tile_size)
        for core_x0 in range(0, width, tile_size):
            core_x1 = min(width, core_x0 + tile_size)
            x0, y0 = max(0, core_x0 - tile_overlap), max(0, core_y0 - tile_overlap)
            x1, y1 = min(width, core_x1 + tile_overlap), min(height, core_y1 + tile_overlap)
            local_response = None
            if response_full is not None:
                local_response = response_full[y0:y1, x0:x1]
            detected = detect_candidates(
                gray[y0:y1, x0:x1],
                mask[y0:y1, x0:x1],
                cfg,
                dark_response=local_response,
                threshold_value=global_threshold,
            )
            core = (core_x0, core_y0, core_x1, core_y1)
            pre_regions = regions_from_labels(
                label(
                    detected.candidate_mask_before_watershed,
                    connectivity=cfg.connectivity,
                ),
                coordinate_offset_xy=(x0, y0),
            )
            pre_owned_count += sum(
                core_x0 <= item.centroid_x_px < core_x1
                and core_y0 <= item.centroid_y_px < core_y1
                for item in pre_regions
            )
            regions = regions_from_labels(
                detected.labels,
                coordinate_offset_xy=(x0, y0),
                tile_index=tile_count,
                include_coords=True,
                core_bounds_xyxy=core,
            )
            retained.extend(
                item
                for item in regions
                if core_x0 <= item.centroid_x_px < core_x1
                and core_y0 <= item.centroid_y_px < core_y1
            )
            tile_count += 1

    unique = deduplicate_candidates(
        retained,
        max_centroid_distance_px=max_centroid_distance_px,
        min_bbox_iou=min_bbox_iou,
        min_pixel_overlap_fraction=min_pixel_overlap_fraction,
    )
    global_labels: LabelImage | None = None
    if build_label_image:
        global_labels = np.zeros((height, width), dtype=np.int32)
        for item in unique:
            if item.coords_yx is not None:
                coords = item.coords_yx
                global_labels[coords[:, 0], coords[:, 1]] = item.candidate_id
    LOGGER.info(
        "Tiled detection: %d tiles, %d owned before watershed, %d unique after",
        tile_count,
        pre_owned_count,
        len(unique),
    )
    return TiledDetectionResult(
        candidates=tuple(unique),
        image_shape=(height, width),
        pre_watershed_count=pre_owned_count,
        post_watershed_count=len(unique),
        tile_count=tile_count,
        threshold_value=global_threshold,
        labels=global_labels,
    )


# Public aliases matching common terminology used by callers/tests.
detect_points = detect_candidates
detect_points_tiled = detect_candidates_tiled


__all__ = [
    "CandidateRegion",
    "DetectionConfig",
    "DetectionResult",
    "TiledDetectionResult",
    "deduplicate_candidates",
    "detect_candidates",
    "detect_candidates_tiled",
    "detection_from_probability",
    "detect_points",
    "detect_points_tiled",
    "estimate_response_threshold",
    "multiscale_blackhat",
    "regions_from_labels",
]
