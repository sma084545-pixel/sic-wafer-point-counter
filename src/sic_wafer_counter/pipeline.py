"""End-to-end orchestration for one auditable wafer analysis run."""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label as connected_label
from skimage.measure import regionprops

from . import __version__
from .candidate_classifier import apply_candidate_classifier, model_from_config
from .density import COUNTING_UNCERTAINTY_NOTE, DensityResult, calculate_density
from .feature_extraction import (
    DEFECT_COLUMNS,
    extract_candidate_features,
    features_to_records,
    valid_boundary_distance_transform,
)
from .image_io import ImageData, ImageTile, load_image
from .point_detection import (
    CandidateRegion,
    DetectionConfig,
    deduplicate_candidates,
    detect_candidates,
    estimate_response_threshold,
)
from .preprocessing import PreprocessingResult, preprocess_image
from .physical_parameters import resolve_physical_parameters
from .reporting import (
    SCIENTIFIC_LIMITATION_ZH,
    generate_html_report,
    write_analysis_outputs,
    write_summary_files,
)
from .utils import atomic_write_json, atomic_write_yaml, software_versions, utc_now_iso
from .wafer_detection import (
    AnalysisMasks,
    AreaStatistics,
    ValidBoundaryIndex,
    WaferDetectionError,
    WaferGeometry,
    build_analysis_masks,
    build_valid_boundary_index,
    calculate_area_statistics,
    create_edge_exclusion_mask,
    create_full_wafer_mask_tile,
    create_invalid_mask_tile,
    detect_wafer,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AnalysisResult:
    """In-memory handoff for a completed analysis."""

    summary: dict[str, Any]
    defects: pd.DataFrame
    output_files: dict[str, Path]
    geometry: WaferGeometry
    area_statistics: AreaStatistics


@dataclass(slots=True)
class _DetectionBundle:
    frame: pd.DataFrame
    raw_count: int
    post_watershed_count: int
    post_watershed_owned_count: int
    duplicate_count: int
    tile_count: int
    threshold_value: float | None
    candidate_mask: np.ndarray
    preprocessed_preview: np.ndarray
    warnings: list[str]


def _reference_profile_report(
    config: Mapping[str, Any],
    *,
    um_per_pixel: float,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve the paper morphology profile as a non-gating diagnostic."""

    raw = config.get("reference_profile", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("reference_profile configuration must be a mapping")
    enabled = bool(raw.get("enabled", False))
    major_um = float(raw.get("expected_major_axis_um", 50.0))
    minor_um = float(raw.get("expected_minor_axis_um", 30.0))
    minimum_pixels = float(raw.get("minimum_axis_pixels", 3.0))
    if not all(np.isfinite((major_um, minor_um, minimum_pixels))):
        raise ValueError("reference-profile sizes must be finite")
    if major_um <= 0 or minor_um <= 0 or minimum_pixels <= 0 or minor_um > major_um:
        raise ValueError("reference profile requires major >= minor > 0 and minimum_axis_pixels > 0")
    major_px = major_um / float(um_per_pixel)
    minor_px = minor_um / float(um_per_pixel)
    conditions_confirmed = bool(raw.get("imaging_conditions_confirmed", False))
    if not enabled:
        status = "disabled"
    elif not conditions_confirmed:
        status = "diagnostic_only_imaging_conditions_unconfirmed"
    elif minor_px < minimum_pixels:
        status = "below_stable_minor_axis_sampling"
    else:
        status = "sampling_sufficient_for_profile_review"
    report = {
        "name": str(raw.get("name", "rigaku_2021_cu_ka_008_tsd_image_feature")),
        "enabled": enabled,
        "imaging_conditions_confirmed": conditions_confirmed,
        "source_conditions": str(raw.get("source_conditions", "4H-SiC; Cu Kalpha; (008) reflection")),
        "expected_major_axis_um": major_um,
        "expected_minor_axis_um": minor_um,
        "expected_major_axis_px": major_px,
        "expected_minor_axis_px": minor_px,
        "minimum_axis_pixels": minimum_pixels,
        "status": status,
        "classification_gating": False,
        "interpretation": (
            "Approximate paper image-feature dimensions for diagnostic review only; "
            "not a universal TSD acceptance rule."
        ),
    }
    warnings: list[str] = []
    if enabled and not conditions_confirmed:
        warnings.append(
            "Rigaku 2021 morphology profile is diagnostic only because Cu Kalpha/(008) "
            "imaging-condition equivalence has not been confirmed"
        )
    if enabled and conditions_confirmed and minor_px < minimum_pixels:
        warnings.append(
            "The expected 30 um minor axis spans only "
            f"{minor_px:.2f} px, below the configured {minimum_pixels:g} px stable "
            "morphology-review threshold; TSD shape classification is not reliable"
        )
    return report, warnings


def _odd_scaled(value: int, scale: float, minimum: int = 1) -> int:
    result = max(minimum, int(round(float(value) / max(scale, 1.0))))
    if result % 2 == 0:
        result += 1
    return result


def _preview_preprocessing_config(
    config: Mapping[str, Any], source_shape: tuple[int, int], preview_shape: tuple[int, int]
) -> dict[str, Any]:
    """Scale full-resolution spatial kernels for a diagnostic preview."""

    adjusted = copy.deepcopy(dict(config))
    section = adjusted.setdefault("preprocessing", {})
    scale_y = source_shape[0] / preview_shape[0]
    scale_x = source_shape[1] / preview_shape[1]
    scale = (scale_x + scale_y) / 2.0
    for key, minimum in (
        ("background_kernel_px", 3),
        ("stripe_smoothing_px", 3),
        ("median_kernel", 1),
    ):
        if key in section:
            section[key] = _odd_scaled(int(section[key]), scale, minimum)
    if "gaussian_sigma" in section:
        section["gaussian_sigma"] = max(0.0, float(section["gaussian_sigma"]) / scale)
    if "background_gaussian_sigma" in section:
        section["background_gaussian_sigma"] = max(
            0.5, float(section["background_gaussian_sigma"]) / scale
        )
    return adjusted


def _scale_invalid_regions(
    regions: Sequence[Any], scale_x: float, scale_y: float
) -> list[Any]:
    """Map configured full-image invalid regions to preview coordinates."""

    scaled: list[Any] = []
    for value in regions:
        if isinstance(value, Mapping):
            region = copy.deepcopy(dict(value))
            kind = str(region.get("type", "rectangle")).lower()
            if kind in {"polygon", "poly"}:
                key = "points" if "points" in region else "polygon"
                region[key] = [
                    [float(point[0]) / scale_x, float(point[1]) / scale_y]
                    for point in region[key]
                ]
            elif "bbox" in region:
                x0, y0, x1, y1 = map(float, region["bbox"])
                region["bbox"] = [x0 / scale_x, y0 / scale_y, x1 / scale_x, y1 / scale_y]
            else:
                for key in ("x", "x0", "x1", "width"):
                    if key in region:
                        region[key] = float(region[key]) / scale_x
                for key in ("y", "y0", "y1", "height"):
                    if key in region:
                        region[key] = float(region[key]) / scale_y
            scaled.append(region)
        else:
            x0, y0, x1, y1 = map(float, value)
            scaled.append([x0 / scale_x, y0 / scale_y, x1 / scale_x, y1 / scale_y])
    return scaled


def _preview_geometry(geometry: WaferGeometry, preview_shape: tuple[int, int]) -> WaferGeometry:
    """Create a geometry in preview pixels for masks and QC graphics."""

    preview_height, preview_width = preview_shape
    scale_x = geometry.image_width / preview_width
    scale_y = geometry.image_height / preview_height
    contour = None
    if geometry.contour_polygon:
        contour = [(x / scale_x, y / scale_y) for x, y in geometry.contour_polygon]
    return WaferGeometry(
        center_x=geometry.center_x / scale_x,
        center_y=geometry.center_y / scale_y,
        radius_px=geometry.radius_px / ((scale_x + scale_y) / 2.0),
        image_width=preview_width,
        image_height=preview_height,
        diameter_mm=geometry.diameter_mm,
        confidence=geometry.confidence,
        circularity=geometry.circularity,
        fit_residual=geometry.fit_residual,
        diameter_reasonable=geometry.diameter_reasonable,
        is_cropped=geometry.is_cropped,
        contour_area_px=(
            geometry.contour_area_px / (scale_x * scale_y)
            if geometry.contour_area_px is not None
            else None
        ),
        border_contact_fraction=geometry.border_contact_fraction,
        angular_coverage=geometry.angular_coverage,
        detection_method=geometry.detection_method,
        contour_polygon=contour,
        warnings=list(geometry.warnings),
    )


def _load_invalid_mask(config: Mapping[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    wafer = config.get("wafer", {})
    if not isinstance(wafer, Mapping):
        return None
    path_value = wafer.get("invalid_mask_path")
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser().resolve()
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read invalid_mask_path: {path}")
    if mask.shape != shape:
        raise ValueError(f"Invalid mask shape {mask.shape} does not match image {shape}")
    return mask > 0


def _tile_masks(
    geometry: WaferGeometry,
    tile: ImageTile,
    *,
    exclude_edge_mm: float,
    invalid_regions: Sequence[Any],
    invalid_mask: np.ndarray | None,
) -> AnalysisMasks:
    """Build masks for an overlap tile; core ownership removes halo artifacts."""

    edge_width_px = float(exclude_edge_mm) / geometry.mm_per_pixel
    margin = int(math.ceil(edge_width_px)) + 2 if edge_width_px > 0 else 0
    expanded_x0 = max(0, tile.x - margin)
    expanded_y0 = max(0, tile.y - margin)
    expanded_x1 = min(geometry.image_width, tile.x + tile.width + margin)
    expanded_y1 = min(geometry.image_height, tile.y + tile.height + margin)
    expanded_full = create_full_wafer_mask_tile(
        geometry,
        expanded_x0,
        expanded_y0,
        expanded_x1 - expanded_x0,
        expanded_y1 - expanded_y0,
        use_contour=True,
    )
    expanded_edge = create_edge_exclusion_mask(expanded_full, geometry, exclude_edge_mm)
    local_x = tile.x - expanded_x0
    local_y = tile.y - expanded_y0
    selector = (
        slice(local_y, local_y + tile.height),
        slice(local_x, local_x + tile.width),
    )
    full = expanded_full[selector]
    edge = expanded_edge[selector]
    invalid = create_invalid_mask_tile(
        geometry,
        tile.x,
        tile.y,
        tile.width,
        tile.height,
        invalid_regions=invalid_regions,
        supplied_mask=invalid_mask,
    )
    return AnalysisMasks(full, edge, invalid, full & ~edge & ~invalid)


def _exact_tile_valid_boundary_distance(
    geometry: WaferGeometry,
    tile: ImageTile,
    labels: np.ndarray,
    *,
    exclude_edge_mm: float,
    invalid_regions: Sequence[Any],
    invalid_mask: np.ndarray | None,
    initial_margin_px: int,
    boundary_index: ValidBoundaryIndex | None = None,
) -> np.ndarray:
    """Return exact candidate distances without a full-frame distance raster.

    Only labelled pixels receive values in the returned tile-sized array;
    feature extraction reads exactly those positions.  The shared boundary
    index contains every invalid pixel adjacent to the final valid mask, so its
    nearest-neighbour result equals the Euclidean distance transform while the
    temporary raster memory remains tile-bounded.

    ``initial_margin_px`` is retained for compatibility with older callers;
    when an index is not supplied it determines a conservative scan-tile size.
    """

    label_image = np.asarray(labels)
    if label_image.shape != (tile.height, tile.width):
        raise ValueError("labels shape must match the tile dimensions")
    candidate_rows, candidate_cols = np.nonzero(label_image > 0)
    result = np.zeros(label_image.shape, dtype=np.float32)
    if not len(candidate_rows):
        return result
    if boundary_index is None:
        scan_size = max(
            32,
            min(
                2048,
                max(tile.width, tile.height, int(initial_margin_px) * 2),
            ),
        )
        boundary_index = build_valid_boundary_index(
            geometry,
            tile_size=scan_size,
            exclude_edge_mm=exclude_edge_mm,
            invalid_regions=invalid_regions,
            invalid_mask=invalid_mask,
            use_contour=True,
        )
    global_coordinates_yx = np.column_stack(
        (candidate_rows + tile.y, candidate_cols + tile.x)
    )
    result[candidate_rows, candidate_cols] = boundary_index.query_yx(
        global_coordinates_yx
    )
    return result


def _globalize_record(record: dict[str, Any], tile: ImageTile) -> dict[str, Any]:
    result = dict(record)
    result["centroid_x_px"] = float(result["centroid_x_px"]) + tile.x
    result["centroid_y_px"] = float(result["centroid_y_px"]) + tile.y
    bbox_value = result.get("bounding_box", "")
    bbox = json.loads(bbox_value) if isinstance(bbox_value, str) else list(bbox_value)
    bbox = [
        int(bbox[0]) + tile.x,
        int(bbox[1]) + tile.y,
        int(bbox[2]) + tile.x,
        int(bbox[3]) + tile.y,
    ]
    result["bounding_box"] = json.dumps(bbox, separators=(",", ":"))
    return result


def _inside_core(record: Mapping[str, Any], tile: ImageTile) -> bool:
    x0, y0, x1, y1 = tile.core_bounds
    x, y = float(record["centroid_x_px"]), float(record["centroid_y_px"])
    return x0 <= x < x1 and y0 <= y < y1


def _deduplicate_records(
    records: list[dict[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    if not records:
        return [], 0
    io_config = config.get("io", {})
    max_distance = float(io_config.get("dedup_centroid_distance_px", 3.0))
    min_iou = float(io_config.get("dedup_min_bbox_iou", 0.30))
    candidates: list[CandidateRegion] = []
    record_lookup: dict[tuple[Any, ...], dict[str, Any]] = {}

    def signature(candidate: CandidateRegion) -> tuple[Any, ...]:
        return (
            candidate.tile_index,
            candidate.centroid_x_px,
            candidate.centroid_y_px,
            candidate.bbox,
            candidate.area_px,
            candidate.core_distance_px,
        )

    for index, record in enumerate(records):
        bbox = json.loads(str(record["bounding_box"]))
        candidate = CandidateRegion(
            candidate_id=index + 1,
            centroid_x_px=float(record["centroid_x_px"]),
            centroid_y_px=float(record["centroid_y_px"]),
            bbox=tuple(map(int, bbox)),
            area_px=int(record["area_px"]),
            tile_index=int(record.get("_tile_index", 0)),
            core_distance_px=float(record.get("_core_distance_px", 0.0)),
        )
        candidates.append(candidate)
        record_lookup[signature(candidate)] = record
    unique = deduplicate_candidates(
        candidates,
        max_centroid_distance_px=max_distance,
        min_bbox_iou=min_iou,
    )
    retained = [record_lookup[signature(candidate)] for candidate in unique]
    retained.sort(key=lambda row: (float(row["centroid_y_px"]), float(row["centroid_x_px"])))
    for defect_id, record in enumerate(retained, start=1):
        record["defect_id"] = defect_id
        record.pop("_core_distance_px", None)
        record.pop("_tile_index", None)
    return retained, len(records) - len(retained)


def _candidate_preview_mask(
    frame: pd.DataFrame, source_shape: tuple[int, int], preview_shape: tuple[int, int]
) -> np.ndarray:
    mask = np.zeros(preview_shape, dtype=np.uint8)
    if frame.empty:
        return mask.astype(bool)
    scale_y = source_shape[0] / preview_shape[0]
    scale_x = source_shape[1] / preview_shape[1]
    for row in frame.itertuples(index=False):
        x = int(round(float(row.centroid_x_px) / scale_x))
        y = int(round(float(row.centroid_y_px) / scale_y))
        diameter = max(1.0, float(row.equivalent_diameter_px))
        radius = max(1, int(round(diameter / (scale_x + scale_y))))
        cv2.circle(mask, (x, y), radius, 1, thickness=-1)
    return mask.astype(bool)


def _detect_small(
    image: np.ndarray,
    masks: AnalysisMasks,
    geometry: WaferGeometry,
    config: Mapping[str, Any],
) -> _DetectionBundle:
    preprocessing = preprocess_image(image, masks.valid_analysis_mask, config)
    detection = detect_candidates(
        preprocessing.filtered,
        masks.valid_analysis_mask,
        config,
        dark_response=preprocessing.dark_response,
    )
    valid_boundary_distance = valid_boundary_distance_transform(
        masks.valid_analysis_mask
    )
    features = extract_candidate_features(
        detection,
        preprocessing.image,
        dark_response=preprocessing.dark_response,
        background=preprocessing.background,
        valid_mask=masks.valid_analysis_mask,
        config=config,
        geometry=geometry,
        wafer_diameter_mm=geometry.diameter_mm,
        valid_boundary_distance_px=valid_boundary_distance,
    )
    records = features_to_records(features)
    frame = pd.DataFrame.from_records(records, columns=DEFECT_COLUMNS)
    return _DetectionBundle(
        frame=frame,
        raw_count=detection.pre_watershed_count,
        post_watershed_count=detection.post_watershed_count,
        post_watershed_owned_count=detection.post_watershed_count,
        duplicate_count=0,
        tile_count=1,
        threshold_value=detection.threshold_value,
        candidate_mask=detection.candidate_mask,
        preprocessed_preview=preprocessing.dark_response,
        warnings=list(detection.warnings),
    )


def _detect_large(
    image_data: ImageData,
    geometry: WaferGeometry,
    config: Mapping[str, Any],
    *,
    exclude_edge_mm: float,
    invalid_regions: Sequence[Any],
    invalid_mask: np.ndarray | None,
    preview_masks: AnalysisMasks,
) -> _DetectionBundle:
    io_config = config.get("io", {})
    tile_size = int(io_config.get("tile_size", 2048))
    overlap = int(io_config.get("tile_overlap", 128))
    records: list[dict[str, Any]] = []
    raw_count = 0
    owned_after = 0
    tile_count = 0
    warnings: list[str] = []

    detection_config = DetectionConfig.from_mapping(config).validated()
    global_threshold: float | None = None
    if (
        detection_config.method in {"blackhat", "combine"}
        and detection_config.threshold_method != "adaptive"
    ):
        total_core_tiles = max(
            1,
            math.ceil(image_data.shape[0] / tile_size)
            * math.ceil(image_data.shape[1] / tile_size),
        )
        target_per_tile = max(2_000, 2_000_000 // total_core_tiles)
        response_samples: list[np.ndarray] = []
        LOGGER.info("Estimating one global response threshold from tiled samples")
        for sample_tile in image_data.iter_tiles(tile_size=tile_size, overlap=overlap):
            sample_masks = _tile_masks(
                geometry,
                sample_tile,
                exclude_edge_mm=exclude_edge_mm,
                invalid_regions=invalid_regions,
                invalid_mask=invalid_mask,
            )
            if not sample_masks.valid_analysis_mask.any():
                continue
            sample_preprocessing = preprocess_image(
                sample_tile.image, sample_masks.valid_analysis_mask, config
            )
            core_response = sample_preprocessing.dark_response[sample_tile.core_slice]
            core_mask = sample_masks.valid_analysis_mask[sample_tile.core_slice]
            stride = max(
                1,
                int(math.ceil(math.sqrt(core_response.size / target_per_tile))),
            )
            sampled_response = core_response[::stride, ::stride]
            sampled_mask = core_mask[::stride, ::stride]
            values = sampled_response[sampled_mask]
            if values.size:
                response_samples.append(np.asarray(values, dtype=np.float32))
        if response_samples:
            combined = np.concatenate(response_samples)
            if combined.size > 2_000_000:
                step = int(math.ceil(combined.size / 2_000_000))
                combined = combined[::step]
            sample_image = combined.reshape(1, -1)
            global_threshold = estimate_response_threshold(
                sample_image,
                np.ones(sample_image.shape, dtype=bool),
                detection_config,
            )
            LOGGER.info("Global tiled response threshold: %s", global_threshold)
        else:
            warnings.append("No valid dark-response samples were available for global thresholding")
    elif detection_config.threshold_method == "adaptive":
        warnings.append(
            "Adaptive thresholding is necessarily tile-local; inspect overlap consistency"
        )

    # Build one exact, bounded-memory index of the final valid boundary.  The
    # previous per-tile expansion could grow a central tile to the complete
    # wafer before its nearest boundary became visible.
    boundary_index = build_valid_boundary_index(
        geometry,
        tile_size=tile_size,
        exclude_edge_mm=exclude_edge_mm,
        invalid_regions=invalid_regions,
        invalid_mask=invalid_mask,
        use_contour=True,
    )

    for tile in image_data.iter_tiles(tile_size=tile_size, overlap=overlap):
        masks = _tile_masks(
            geometry,
            tile,
            exclude_edge_mm=exclude_edge_mm,
            invalid_regions=invalid_regions,
            invalid_mask=invalid_mask,
        )
        if not masks.valid_analysis_mask.any():
            continue
        tile_count += 1
        preprocessing = preprocess_image(tile.image, masks.valid_analysis_mask, config)
        detection = detect_candidates(
            preprocessing.filtered,
            masks.valid_analysis_mask,
            config,
            dark_response=preprocessing.dark_response,
            threshold_value=global_threshold,
        )
        warnings.extend(detection.warnings)
        boundary_distance = _exact_tile_valid_boundary_distance(
            geometry,
            tile,
            detection.labels,
            exclude_edge_mm=exclude_edge_mm,
            invalid_regions=invalid_regions,
            invalid_mask=invalid_mask,
            initial_margin_px=max(overlap, 32),
            boundary_index=boundary_index,
        )

        before_labels = connected_label(detection.candidate_mask_before_watershed)
        for region in regionprops(before_labels):
            cy, cx = region.centroid
            global_record = {
                "centroid_x_px": float(cx) + tile.x,
                "centroid_y_px": float(cy) + tile.y,
            }
            if _inside_core(global_record, tile):
                raw_count += 1

        features = extract_candidate_features(
            detection,
            preprocessing.image,
            dark_response=preprocessing.dark_response,
            background=preprocessing.background,
            valid_mask=masks.valid_analysis_mask,
            config=config,
            center_x_px=geometry.center_x - tile.x,
            center_y_px=geometry.center_y - tile.y,
            radius_px=geometry.radius_px,
            mm_per_pixel=geometry.mm_per_pixel,
            wafer_diameter_mm=geometry.diameter_mm,
            valid_boundary_distance_px=boundary_distance,
        )
        for record in features_to_records(features):
            global_record = _globalize_record(record, tile)
            if not _inside_core(global_record, tile):
                continue
            x0, y0, x1, y1 = tile.core_bounds
            x = float(global_record["centroid_x_px"])
            y = float(global_record["centroid_y_px"])
            global_record["_core_distance_px"] = min(x - x0, x1 - x, y - y0, y1 - y)
            global_record["_tile_index"] = tile_count
            records.append(global_record)
            owned_after += 1

    unique_records, duplicate_count = _deduplicate_records(records, config)
    frame = pd.DataFrame.from_records(unique_records, columns=DEFECT_COLUMNS)
    preview_config = _preview_preprocessing_config(config, image_data.shape, image_data.preview.shape)
    preview_preprocessing = preprocess_image(
        image_data.preview, preview_masks.valid_analysis_mask, preview_config
    )
    candidate_mask = _candidate_preview_mask(frame, image_data.shape, image_data.preview.shape)
    if duplicate_count:
        warnings.append(f"Removed {duplicate_count} overlap-tile duplicate candidates")
    warnings.append(
        "candidate_mask.png is a preview-resolution raster reconstructed from tiled detections"
    )
    return _DetectionBundle(
        frame=frame,
        raw_count=raw_count,
        post_watershed_count=len(frame),
        post_watershed_owned_count=owned_after,
        duplicate_count=duplicate_count,
        tile_count=tile_count,
        threshold_value=global_threshold,
        candidate_mask=candidate_mask,
        preprocessed_preview=preview_preprocessing.dark_response,
        warnings=warnings,
    )


def _failure_artifacts(
    output_dir: Path,
    input_path: Path,
    image_data: ImageData,
    error: WaferDetectionError,
) -> None:
    """Persist enough evidence to diagnose a refused automatic calibration."""

    cv2.imwrite(str(output_dir / "wafer_detection_preview.png"), image_data.preview)
    payload: dict[str, Any] = {
        "status": "failed",
        "input_path": str(input_path),
        "error": str(error),
        "density_calculated": False,
        "image_metadata": image_data.metadata.to_dict(),
        "warnings": [str(error)],
    }
    if error.geometry is not None:
        payload["low_confidence_geometry"] = error.geometry.to_dict()
    atomic_write_json(output_dir / "summary.json", payload)


def analyze_image(
    input_image: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    center_x: float | None = None,
    center_y: float | None = None,
    radius_px: float | None = None,
) -> AnalysisResult:
    """Run one complete analysis and write its audit bundle.

    Low-confidence automatic wafer detection raises :class:`WaferDetectionError`
    after a failure summary and preview have been saved.  No density is emitted
    in that case.
    """

    started = time.perf_counter()
    input_path = Path(input_image).expanduser().resolve()
    folder = Path(output_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    wafer_config = config.get("wafer", {})
    io_config = config.get("io", {})
    diameter_mm = float(wafer_config.get("diameter_mm", 100.0))
    exclude_edge_mm = float(wafer_config.get("exclude_edge_mm", 0.0))
    invalid_regions = list(wafer_config.get("invalid_regions", []))

    with load_image(input_path, config) as image_data:
        invalid_mask = _load_invalid_mask(config, image_data.shape)
        try:
            geometry = detect_wafer(
                image_data.preview,
                full_shape=image_data.shape,
                diameter_mm=diameter_mm,
                center_x=center_x,
                center_y=center_y,
                radius_px=radius_px,
                min_confidence=float(wafer_config.get("detection_min_confidence", 0.55)),
            )
        except WaferDetectionError as error:
            _failure_artifacts(folder, input_path, image_data, error)
            raise

        physical_resolution = resolve_physical_parameters(
            config, mm_per_pixel=geometry.mm_per_pixel
        )
        analysis_config = physical_resolution.config
        resolved_parameters_path = atomic_write_yaml(
            folder / "resolved_physical_parameters.yaml", physical_resolution.report
        )

        use_tiled = image_data.gray is None
        full_masks: AnalysisMasks | None = None
        if use_tiled:
            area = calculate_area_statistics(
                geometry,
                tile_size=int(analysis_config.get("io", {}).get("tile_size", 2048)),
                exclude_edge_mm=exclude_edge_mm,
                invalid_regions=invalid_regions,
                invalid_mask=invalid_mask,
                use_contour=True,
            )
            preview_geometry = _preview_geometry(geometry, image_data.preview.shape)
            scale_x = geometry.image_width / image_data.preview.shape[1]
            scale_y = geometry.image_height / image_data.preview.shape[0]
            preview_invalid_regions = _scale_invalid_regions(invalid_regions, scale_x, scale_y)
            preview_invalid_mask = None
            if invalid_mask is not None:
                preview_invalid_mask = cv2.resize(
                    invalid_mask.astype(np.uint8),
                    (image_data.preview.shape[1], image_data.preview.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            preview_masks = build_analysis_masks(
                preview_geometry,
                exclude_edge_mm=exclude_edge_mm,
                invalid_regions=preview_invalid_regions,
                invalid_mask=preview_invalid_mask,
                use_contour=True,
            )
            detected = _detect_large(
                image_data,
                geometry,
                analysis_config,
                exclude_edge_mm=exclude_edge_mm,
                invalid_regions=invalid_regions,
                invalid_mask=invalid_mask,
                preview_masks=preview_masks,
            )
            output_image = image_data.preview
            output_masks = preview_masks
        else:
            full_masks = build_analysis_masks(
                geometry,
                exclude_edge_mm=exclude_edge_mm,
                invalid_regions=invalid_regions,
                invalid_mask=invalid_mask,
                use_contour=True,
            )
            area = calculate_area_statistics(geometry, masks=full_masks)
            detected = _detect_small(
                image_data.require_full(), full_masks, geometry, analysis_config
            )
            output_image = image_data.require_full()
            output_masks = full_masks

        classifier_model = model_from_config(analysis_config)
        classifier_warnings: list[str] = []
        if classifier_model is None:
            classifier_report: dict[str, Any] = {
                "status": "disabled",
                "physical_identity_validated": False,
            }
        else:
            detected.frame, classifier_report, classifier_warnings = (
                apply_candidate_classifier(detected.frame, classifier_model)
            )
        accepted_count = int(detected.frame["accepted"].sum()) if not detected.frame.empty else 0
        density: DensityResult = calculate_density(accepted_count, area.valid_area_cm2)
        reference_profile, reference_warnings = _reference_profile_report(
            analysis_config, um_per_pixel=geometry.um_per_pixel
        )
        warnings = list(dict.fromkeys(
            image_data.metadata.warnings
            + image_data.metadata.limitations
            + geometry.warnings
            + list(physical_resolution.warnings)
            + detected.warnings
            + reference_warnings
            + classifier_warnings
        ))
        summary: dict[str, Any] = {
            "status": "completed",
            "input_file_name": input_path.name,
            "input_path": str(input_path),
            "image_size": [image_data.metadata.width, image_data.metadata.height],
            "image_width_px": image_data.metadata.width,
            "image_height_px": image_data.metadata.height,
            "image_dtype": image_data.metadata.dtype,
            "source_dtype": image_data.metadata.dtype,
            "analysis_dtype": image_data.metadata.analysis_dtype,
            "normalization_low_value": image_data.metadata.normalization_low_value,
            "normalization_high_value": image_data.metadata.normalization_high_value,
            "low_clipped_fraction": image_data.metadata.low_clipped_fraction,
            "high_clipped_fraction": image_data.metadata.high_clipped_fraction,
            "white_is_zero": image_data.metadata.white_is_zero,
            "analysis_quantized_to_uint8": image_data.metadata.analysis_quantized_to_uint8,
            "image_metadata": image_data.metadata.to_dict(),
            "wafer_diameter_mm": diameter_mm,
            "wafer_center_px": [geometry.center_x, geometry.center_y],
            "center_x_px": geometry.center_x,
            "center_y_px": geometry.center_y,
            "wafer_radius_px": geometry.radius_px,
            "detected_wafer_diameter_px": geometry.diameter_px,
            "mm_per_pixel": geometry.mm_per_pixel,
            "cm_per_pixel": geometry.cm_per_pixel,
            "um_per_pixel": geometry.um_per_pixel,
            "pixel_area_cm2": geometry.pixel_area_cm2,
            "wafer_detection": geometry.to_dict(),
            "theoretical_area_cm2": area.theoretical_area_cm2,
            "fitted_wafer_area_cm2": area.circle_fit_area_cm2,
            **area.to_dict(),
            "exclude_edge_mm": exclude_edge_mm,
            "raw_candidate_count": detected.raw_count,
            "post_watershed_candidate_count": detected.post_watershed_count,
            "post_watershed_owned_before_dedup_count": detected.post_watershed_owned_count,
            "tile_overlap_duplicates_removed": detected.duplicate_count,
            "processed_tile_count": detected.tile_count,
            "detection_threshold_value": detected.threshold_value,
            "accepted_count": accepted_count,
            "rejected_count": int(len(detected.frame) - accepted_count),
            "decision_basis": (
                "trained_candidate_classifier"
                if classifier_model is not None
                else "configured_image_rules"
            ),
            "candidate_classifier": classifier_report,
            "point_density_cm2": density.density_cm2,
            "density_unit": "cm^-2",
            "counting_uncertainty_cm2": density.standard_uncertainty_cm2,
            "poisson_95_ci_lower_cm2": density.density_ci_lower_cm2,
            "poisson_95_ci_upper_cm2": density.density_ci_upper_cm2,
            "poisson_count_95_ci": [density.count_ci_lower, density.count_ci_upper],
            "counting_uncertainty_scope": COUNTING_UNCERTAINTY_NOTE,
            "scientific_interpretation_limit": SCIENTIFIC_LIMITATION_ZH,
            "real_annotation_validation_status": "not validated on real SiC data",
            "paper_reference_alignment": {
                "automatic_xrt_marker": "red rectangle",
                "independent_reference_marker": "yellow circle",
                "independent_reference_data_supplied": False,
                "independent_reference_status": "not provided; no DIC/KOH agreement claim",
                "source": "Reimann and Kranert, Rigaku Journal 37(2), 2021, Fig. 5",
            },
            "reference_morphology_profile": reference_profile,
            "uncertainty_budget_summary": {
                "counting": "Poisson/Garwood interval reported",
                "classification": (
                    str(classifier_report.get("validation_status"))
                    if classifier_model is not None
                    else "not quantified: no real SiC expert-label validation"
                ),
                "parameter_sensitivity": "not quantified: no calibration-wafer sensitivity run",
                "area_calibration": "not quantified: no diameter/pixel/mask-boundary uncertainty supplied",
                "spatial_heterogeneity": "descriptive spatial outputs only; no multi-wafer inference",
            },
            "filter_parameters": copy.deepcopy(dict(config.get("filters", {}))),
            "resolved_physical_parameters_file": resolved_parameters_path.name,
            "software_version": __version__,
            "software_versions": software_versions(),
            "processing_mode": "overlap-tiled" if use_tiled else "full-array",
            "runtime_seconds": time.perf_counter() - started,
            "generated_at_utc": utc_now_iso(),
            "warnings": warnings,
        }

        def crop_reader(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
            return image_data.source.read_region(
                x0, y0, x1 - x0, y1 - y0, normalize=False
            )

        def comparison_crop_reader(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
            return image_data.source.read_region(
                x0, y0, x1 - x0, y1 - y0, normalize=True
            )

        output_files = write_analysis_outputs(
            folder,
            summary,
            detected.frame,
            config,
            input_image_path=input_path,
            original_image=output_image,
            full_wafer_mask=output_masks.full_wafer_mask,
            valid_analysis_mask=output_masks.valid_analysis_mask,
            preprocessed_image=detected.preprocessed_preview,
            candidate_mask=detected.candidate_mask,
            source_shape=image_data.shape,
            crop_reader=crop_reader,
            comparison_crop_reader=comparison_crop_reader,
        )
        if classifier_model is not None:
            output_files["candidate_classifier"] = atomic_write_json(
                folder / "candidate_classifier.json",
                classifier_model,
                sort_keys=True,
            )
        output_files["resolved_physical_parameters"] = resolved_parameters_path
        # Include report/PNG/CSV generation in the stated wall-clock runtime,
        # then refresh the two summary files and HTML headline deterministically.
        summary["runtime_seconds"] = time.perf_counter() - started
        output_files.update(write_summary_files(summary, folder))
        if bool(config.get("output", {}).get("generate_html_report", True)):
            output_files["report_html"] = generate_html_report(summary, folder)
        LOGGER.info(
            "Completed %s: n=%d, S=%.6f cm^2, rho=%.6g cm^-2",
            input_path.name,
            accepted_count,
            area.valid_area_cm2,
            density.density_cm2,
        )
        return AnalysisResult(summary, detected.frame, output_files, geometry, area)


__all__ = ["AnalysisResult", "analyze_image"]
