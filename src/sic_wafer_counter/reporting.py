"""Auditable tabular, JSON, crop, and HTML outputs.

The reporter never upgrades an image-processing classification into a physical
claim.  It preserves every candidate (including rejection reasons), the exact
configuration, and review crops so results can be traced and corrected later.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np
import pandas as pd
import tifffile
import yaml
from jinja2 import BaseLoader, Environment, select_autoescape

from . import __version__
from .density import calculate_density
from .validation import spatial_heterogeneity_indicator
from .visualization import (
    normalize_for_display,
    save_area_normalized_distributions,
    save_binary_mask,
    save_distribution_plots,
    save_grayscale_image,
    save_overlays,
)

LOGGER = logging.getLogger(__name__)

SCIENTIFIC_LIMITATION_ZH = (
    "在当前图像判定和筛选标准下检测到的点状缺陷数量为 n。只有在点状图像特征与位错已经通过"
    "人工或独立实验确认一一对应时，n/S 才能作为位错密度报告。"
)
SPATIAL_INTERPRETATION_LIMIT_ZH = (
    "径向和方位角结果是基于当前单片晶圆的描述性面积归一化统计。外圈计数更多并不自动表示边缘密度更高；"
    "工艺或生长机制解释需要多片晶圆和实验元数据。方位角以图像正 x 轴为参考，未提供统一 notch/flat 或晶向坐标时，"
    "不得把它解释为晶体学方向。多重分箱比较及低计数区间均应谨慎解释；泊松区间不包含系统误差。"
)
UNCERTAINTY_NOTE_ZH = (
    "该统计不确定度只反映有限计数造成的随机误差，不包含图像分割、漏检、误检以及物理判定错误"
    "造成的系统误差。"
)

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
    "crop_path",
)


def _json_safe(value: Any) -> Any:
    """Convert numpy, pandas, Path, and non-finite values to strict JSON data."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    try:
        if not isinstance(value, (str, bytes, bool)) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted", "accept"}


def defects_to_frame(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]] | None,
) -> pd.DataFrame:
    """Normalize candidate records and guarantee the audit schema exists."""

    if defects is None:
        frame = pd.DataFrame()
    elif isinstance(defects, pd.DataFrame):
        frame = defects.copy()
    else:
        materialized = list(defects)
        normalized = [
            item.to_record() if hasattr(item, "to_record") and callable(item.to_record) else item
            for item in materialized
        ]
        frame = pd.DataFrame(normalized)

    # Compatibility with regionprops-style or early pipeline field names.
    aliases = {
        "label": "defect_id",
        "candidate_id": "defect_id",
        "center_x_px": "centroid_x_px",
        "center_y_px": "centroid_y_px",
        "equivalent_diameter_area": "equivalent_diameter_px",
        "axis_major_length": "major_axis_length_px",
        "axis_minor_length": "minor_axis_length_px",
    }
    for source, target in aliases.items():
        if target not in frame and source in frame:
            frame[target] = frame[source]
    if "defect_id" not in frame:
        frame["defect_id"] = np.arange(1, len(frame) + 1, dtype=int)
    if "accepted" not in frame:
        frame["accepted"] = True
    frame["accepted"] = frame["accepted"].map(_truthy).astype(bool)
    if "rejection_reason" not in frame:
        frame["rejection_reason"] = ""
    frame.loc[frame["accepted"], "rejection_reason"] = frame.loc[
        frame["accepted"], "rejection_reason"
    ].fillna("")
    missing_reason = frame["rejection_reason"].isna() | frame["rejection_reason"].fillna("").astype(str).str.strip().eq("")
    frame.loc[~frame["accepted"] & missing_reason, "rejection_reason"] = "unspecified"
    if "bounding_box" in frame:
        frame["bounding_box"] = frame["bounding_box"].map(
            lambda value: json.dumps(list(value), separators=(",", ":"))
            if isinstance(value, (tuple, list, np.ndarray))
            else value
        )

    for column in DEFECT_COLUMNS:
        if column not in frame:
            if column == "accepted":
                frame[column] = False
            elif column in {"rejection_reason", "crop_path", "bounding_box"}:
                frame[column] = ""
            else:
                frame[column] = np.nan
    primary = list(DEFECT_COLUMNS)
    extras = [column for column in frame.columns if column not in primary]
    return frame.loc[:, primary + extras]


def _summary_value(summary: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Read the first present key, including dotted nested paths."""

    for key in keys:
        current: Any = summary
        found = True
        for part in key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                found = False
                break
            current = current[part]
        if found and current is not None:
            return current
    return default


def _canonical_summary(
    summary: Mapping[str, Any],
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None,
    input_image_path: str | Path | None,
) -> dict[str, Any]:
    """Add stable required fields while preserving caller-specific metadata."""

    result: dict[str, Any] = dict(summary)
    accepted_count = int(frame["accepted"].sum())
    rejected_count = int(len(frame) - accepted_count)
    valid_area = _summary_value(
        result,
        "valid_analysis_area_cm2",
        "valid_area_cm2",
        "areas.valid_analysis_area_cm2",
        "areas.valid_analysis_cm2",
        "areas.valid_cm2",
    )
    density = _summary_value(
        result, "point_density_cm2", "density_cm2", "density.density_cm2", "density.rho_cm2"
    )
    uncertainty = _summary_value(
        result,
        "counting_uncertainty_cm2",
        "sigma_rho_cm2",
        "density.counting_uncertainty_cm2",
        "density.sigma_rho_cm2",
        "density.sigma_cm2",
    )
    if valid_area is not None:
        valid_area = float(valid_area)
        if valid_area > 0:
            recalculated = calculate_density(accepted_count, valid_area)
            if density is not None and not math.isclose(
                float(density), recalculated.density_cm2, rel_tol=1e-9, abs_tol=1e-12
            ):
                warnings = result.get("warnings", [])
                if isinstance(warnings, str):
                    warnings = [warnings]
                else:
                    warnings = list(warnings)
                warnings.append(
                    "Summary density disagreed with accepted candidate table; top-level density was recomputed from n/S."
                )
                result["warnings"] = warnings
            density = recalculated.density_cm2
            uncertainty = recalculated.standard_uncertainty_cm2
            result["poisson_95_ci_lower_cm2"] = recalculated.density_ci_lower_cm2
            result["poisson_95_ci_upper_cm2"] = recalculated.density_ci_upper_cm2
            result["poisson_count_ci_lower"] = recalculated.count_ci_lower
            result["poisson_count_ci_upper"] = recalculated.count_ci_upper

    image_path = Path(input_image_path) if input_image_path else None
    inferred_name = image_path.name if image_path else _summary_value(
        result, "input_file_name", "image.file_name", "image_metadata.file_name", "input.file_name", default=""
    )
    result.setdefault("input_file_name", inferred_name)
    if image_path:
        result.setdefault("input_path", str(image_path.resolve()))
    result.setdefault("software_version", __version__)
    result.setdefault("python_version", platform.python_version())
    result.setdefault("generated_at_utc", datetime.now(timezone.utc).isoformat())
    raw_count = _summary_value(
        result,
        "raw_candidate_count",
        "pre_watershed_candidate_count",
        "detection.pre_watershed_count",
        default=len(frame),
    )
    post_count = _summary_value(
        result,
        "post_watershed_candidate_count",
        "detection.post_watershed_count",
        default=len(frame),
    )
    result.setdefault("raw_candidate_count", int(raw_count))
    result.setdefault("post_watershed_candidate_count", int(post_count))
    result["accepted_count"] = accepted_count
    result["rejected_count"] = rejected_count
    if valid_area is not None:
        result.setdefault("valid_analysis_area_cm2", valid_area)
    if density is not None:
        result["point_density_cm2"] = float(density)
        result.setdefault("density_unit", "cm^-2")
    if uncertainty is not None:
        result["counting_uncertainty_cm2"] = float(uncertainty)

    # Stable top-level aliases keep summary.csv and downstream review scripts
    # usable even when the pipeline also stores richer nested dataclass output.
    canonical_aliases: dict[str, tuple[str, ...]] = {
        "wafer_diameter_mm": ("wafer.diameter_mm", "geometry.diameter_mm"),
        "center_x_px": ("wafer.center_x_px", "geometry.center_x_px"),
        "center_y_px": ("wafer.center_y_px", "geometry.center_y_px"),
        "wafer_radius_px": ("wafer.radius_px", "geometry.radius_px"),
        "mm_per_pixel": ("wafer.mm_per_pixel", "geometry.mm_per_pixel"),
        "theoretical_area_cm2": (
            "areas.theoretical_complete_wafer_area_cm2",
            "wafer.theoretical_area_cm2",
        ),
        "fitted_wafer_area_cm2": (
            "areas.circle_fit_area_cm2",
            "wafer.circle_fit_area_cm2",
        ),
        "edge_excluded_area_cm2": ("areas.edge_excluded_area_cm2",),
        "other_invalid_area_cm2": ("areas.other_invalid_area_cm2",),
        "runtime_seconds": ("runtime.elapsed_seconds", "elapsed_seconds"),
        "poisson_95_ci_lower_cm2": (
            "poisson_density_ci_lower_cm2",
            "density.poisson_density_ci_lower_cm2",
            "density.density_ci_lower_cm2",
        ),
        "poisson_95_ci_upper_cm2": (
            "poisson_density_ci_upper_cm2",
            "density.poisson_density_ci_upper_cm2",
            "density.density_ci_upper_cm2",
        ),
    }
    for target, sources in canonical_aliases.items():
        if target not in result:
            value = _summary_value(result, *sources)
            if value is not None:
                result[target] = value
    if "wafer_center_px" not in result and "center_x_px" in result and "center_y_px" in result:
        result["wafer_center_px"] = [result["center_x_px"], result["center_y_px"]]
    width = _summary_value(
        result, "image_width_px", "image.width", "image_metadata.width", "input.width"
    )
    height = _summary_value(
        result, "image_height_px", "image.height", "image_metadata.height", "input.height"
    )
    if width is not None and height is not None:
        result.setdefault("image_width_px", int(width))
        result.setdefault("image_height_px", int(height))
        result.setdefault("image_size", [int(width), int(height)])
    result.setdefault("warnings", [])
    result.setdefault("filter_parameters", dict((config or {}).get("filters", {})))
    result["scientific_interpretation_limit"] = SCIENTIFIC_LIMITATION_ZH
    result["counting_uncertainty_scope"] = UNCERTAINTY_NOTE_ZH
    return _json_safe(result)


def _flatten_summary(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_summary(value, full_key))
        elif isinstance(value, (list, tuple, set)):
            flat[full_key] = json.dumps(_json_safe(value), ensure_ascii=False)
        else:
            flat[full_key] = _json_safe(value)
    return flat


def write_summary_files(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write machine-readable JSON and a one-row spreadsheet-friendly CSV."""

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(summary)
    json_path = folder / "summary.json"
    json_path.write_text(
        json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    csv_path = folder / "summary.csv"
    pd.DataFrame([_flatten_summary(safe)]).to_csv(csv_path, index=False)
    return {"summary_json": json_path, "summary_csv": csv_path}


def save_candidate_crops(
    original_image: np.ndarray | None,
    defects: pd.DataFrame,
    output_dir: str | Path,
    *,
    half_size_px: int = 32,
    source_shape: tuple[int, int] | None = None,
    crop_reader: Callable[[int, int, int, int], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Save a raw-image crop for every candidate and return updated records.

    ``crop_reader`` receives full-resolution ``x0, y0, x1, y1`` coordinates and
    is the preferred path for tiled BigTIFF/pyvips inputs.  A low-resolution
    preview must not be supplied as ``original_image`` with global coordinates.
    """

    if half_size_px < 2:
        raise ValueError("half_size_px must be at least 2")
    folder = Path(output_dir)
    crop_folder = folder / "candidate_crops"
    crop_folder.mkdir(parents=True, exist_ok=True)
    frame = defects.copy()
    if crop_reader is None and original_image is None:
        raise ValueError("Either original_image or crop_reader is required")
    if source_shape is not None:
        height, width = int(source_shape[0]), int(source_shape[1])
    elif original_image is not None:
        height, width = np.asarray(original_image).shape[:2]
    else:
        raise ValueError("source_shape is required when crop_reader has no backing array")
    crop_paths: list[str] = []
    preview_paths: list[str] = []
    for row_number, (_, record) in enumerate(frame.iterrows(), start=1):
        x_value = record.get("centroid_x_px", np.nan)
        y_value = record.get("centroid_y_px", np.nan)
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            crop_paths.append("")
            preview_paths.append("")
            continue
        x = int(round(float(x_value)))
        y = int(round(float(y_value)))
        x0, x1 = max(0, x - half_size_px), min(width, x + half_size_px + 1)
        y0, y1 = max(0, y - half_size_px), min(height, y + half_size_px + 1)
        if x0 >= x1 or y0 >= y1:
            crop_paths.append("")
            preview_paths.append("")
            continue
        if crop_reader is not None:
            try:
                raw_crop = np.asarray(crop_reader(x0, y0, x1, y1))
            except Exception as exc:  # reader backends expose different error classes
                LOGGER.warning(
                    "Candidate crop reader failed for (%d, %d, %d, %d): %s",
                    x0,
                    y0,
                    x1,
                    y1,
                    exc,
                )
                crop_paths.append("")
                preview_paths.append("")
                continue
        else:
            raw_crop = np.asarray(original_image)[y0:y1, x0:x1]  # type: ignore[index]
        if raw_crop.size == 0:
            LOGGER.warning("Candidate crop reader returned an empty array for (%d, %d, %d, %d)", x0, y0, x1, y1)
            crop_paths.append("")
            preview_paths.append("")
            continue
        crop = normalize_for_display(raw_crop, 0.5, 99.5)
        defect_id = record.get("defect_id", row_number)
        try:
            stem = f"candidate_{int(defect_id):06d}"
        except (TypeError, ValueError):
            stem = f"candidate_{row_number:06d}"
        raw_destination = crop_folder / f"{stem}.tif"
        preview_destination = crop_folder / f"{stem}_preview.png"
        try:
            # TIFF preserves uint16/floating source values for later audit.
            tifffile.imwrite(raw_destination, raw_crop)
            preview_ok = cv2.imwrite(str(preview_destination), crop)
        except (OSError, ValueError, tifffile.TiffFileError) as exc:
            LOGGER.warning("Could not save candidate crop %s: %s", raw_destination, exc)
            crop_paths.append("")
            preview_paths.append("")
        else:
            if not preview_ok:
                LOGGER.warning("Could not save candidate preview %s", preview_destination)
                preview_paths.append("")
            else:
                preview_paths.append(str(preview_destination.relative_to(folder)))
            crop_paths.append(str(raw_destination.relative_to(folder)))
    frame["crop_path"] = crop_paths
    frame["crop_preview_path"] = preview_paths
    return frame


def save_defect_comparison_details(
    original_image: np.ndarray | None,
    defects: pd.DataFrame,
    output_path: str | Path,
    *,
    half_size_px: int = 48,
    max_candidates: int = 12,
    source_shape: tuple[int, int] | None = None,
    crop_reader: Callable[[int, int, int, int], np.ndarray] | None = None,
) -> Path:
    """Save a deterministic raw-versus-decision candidate audit montage.

    The montage samples both accepted and rejected candidates without changing
    the underlying count.  Each card shows the normalized raw crop beside the
    same crop with the automatic decision marker.  Full-resolution TIFF values
    remain available through ``candidate_crops`` when that export is enabled;
    this PNG is a compact visual comparison for reports and browser runs.
    """

    if half_size_px < 8:
        raise ValueError("half_size_px must be at least 8")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if max_candidates > 60:
        raise ValueError("max_candidates must not exceed 60")
    if crop_reader is None and original_image is None:
        raise ValueError("Either original_image or crop_reader is required")

    frame = defects_to_frame(defects)
    accepted = frame.loc[frame["accepted"].map(_truthy)].copy()
    rejected = frame.loc[~frame["accepted"].map(_truthy)].copy()

    def sample_rows(group: pd.DataFrame, count: int) -> pd.DataFrame:
        if count <= 0 or group.empty:
            return group.iloc[0:0]
        ordered = group.sort_values("defect_id", kind="stable")
        if len(ordered) <= count:
            return ordered
        positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
        return ordered.iloc[positions]

    accepted_slots = min(len(accepted), (max_candidates + 1) // 2)
    rejected_slots = min(len(rejected), max_candidates - accepted_slots)
    remaining = max_candidates - accepted_slots - rejected_slots
    if remaining:
        accepted_slots += min(remaining, len(accepted) - accepted_slots)
        remaining = max_candidates - accepted_slots - rejected_slots
        rejected_slots += min(remaining, len(rejected) - rejected_slots)
    selected = pd.concat(
        [sample_rows(accepted, accepted_slots), sample_rows(rejected, rejected_slots)],
        ignore_index=True,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    card_width, card_height = 388, 224
    columns = 3
    rows = max(1, int(math.ceil(max(1, len(selected)) / columns)))
    header_height = 72
    canvas = np.full((header_height + rows * card_height, columns * card_width, 3), 248, np.uint8)
    cv2.putText(
        canvas,
        "Defect recognition detail comparison",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (30, 42, 48),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Left: raw local grayscale   Right: automatic decision   Green=accepted, red=rejected",
        (20, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (70, 82, 88),
        1,
        cv2.LINE_AA,
    )
    if selected.empty:
        cv2.putText(
            canvas,
            "No candidates were available for comparison.",
            (20, header_height + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
    else:
        if source_shape is not None:
            source_height, source_width = map(int, source_shape[:2])
        elif original_image is not None:
            source_height, source_width = np.asarray(original_image).shape[:2]
        else:  # guarded above; keeps type checkers and alternate readers clear
            raise ValueError("source_shape is required for crop_reader-only comparison")

        tile_size = 156
        for position, (_, record) in enumerate(selected.iterrows()):
            x = int(round(float(record["centroid_x_px"])))
            y = int(round(float(record["centroid_y_px"])))
            x0, x1 = max(0, x - half_size_px), min(source_width, x + half_size_px + 1)
            y0, y1 = max(0, y - half_size_px), min(source_height, y + half_size_px + 1)
            if crop_reader is not None:
                try:
                    raw_crop = np.asarray(crop_reader(x0, y0, x1, y1))
                except Exception as exc:  # backend readers expose different error classes
                    LOGGER.warning("Comparison crop reader failed for defect %s: %s", record.get("defect_id"), exc)
                    continue
            else:
                raw_crop = np.asarray(original_image)[y0:y1, x0:x1]  # type: ignore[index]
            if raw_crop.size == 0:
                continue
            display = normalize_for_display(raw_crop, 0.5, 99.5)
            display = cv2.resize(display, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
            raw_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            marked = raw_bgr.copy()
            accepted_value = _truthy(record.get("accepted"))
            marker_color = (40, 190, 40) if accepted_value else (45, 45, 225)
            scale_x = tile_size / float(max(1, x1 - x0))
            scale_y = tile_size / float(max(1, y1 - y0))
            marker_radius = max(
                9,
                min(
                    30,
                    int(round(float(record.get("equivalent_diameter_px", 8.0)) * 0.75 * math.sqrt(scale_x * scale_y))),
                ),
            )
            center = (
                int(round((x - x0) * scale_x)),
                int(round((y - y0) * scale_y)),
            )
            if accepted_value:
                cv2.circle(marked, center, marker_radius, marker_color, 3, cv2.LINE_AA)
            else:
                cv2.line(marked, (center[0] - marker_radius, center[1] - marker_radius),
                         (center[0] + marker_radius, center[1] + marker_radius), marker_color, 3, cv2.LINE_AA)
                cv2.line(marked, (center[0] - marker_radius, center[1] + marker_radius),
                         (center[0] + marker_radius, center[1] - marker_radius), marker_color, 3, cv2.LINE_AA)

            row, column = divmod(position, columns)
            top = header_height + row * card_height
            left = column * card_width
            canvas[top + 10:top + 10 + tile_size, left + 10:left + 10 + tile_size] = raw_bgr
            canvas[top + 10:top + 10 + tile_size, left + 176:left + 176 + tile_size] = marked
            cv2.rectangle(canvas, (left + 4, top + 4), (left + card_width - 5, top + card_height - 5),
                          (216, 222, 224), 1)
            defect_id = record.get("defect_id", position + 1)
            state = "ACCEPT" if accepted_value else "REJECT"
            reason = "" if accepted_value else str(record.get("rejection_reason", "unspecified"))
            label = f"ID {defect_id}  {state}" + (f"  {reason}" if reason else "")
            cv2.putText(canvas, label[:55], (left + 10, top + 188), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, marker_color, 1, cv2.LINE_AA)
            metrics = (
                f"d_eq={float(record.get('equivalent_diameter_mm', np.nan)):.4g} mm  "
                f"contrast={float(record.get('contrast', np.nan)):.3g}"
            )
            cv2.putText(canvas, metrics, (left + 10, top + 208), cv2.FONT_HERSHEY_SIMPLEX,
                        0.39, (65, 72, 76), 1, cv2.LINE_AA)

    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Could not write defect comparison montage: {destination}")
    return destination


def write_defect_tables(defects: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Write complete, accepted, and rejected candidate tables."""

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    frame = defects_to_frame(defects)
    outputs = {
        "defects_all": folder / "defects_all.csv",
        "defects_accepted": folder / "defects_accepted.csv",
        "defects_rejected": folder / "defects_rejected.csv",
    }
    frame.to_csv(outputs["defects_all"], index=False)
    frame.loc[frame["accepted"]].to_csv(outputs["defects_accepted"], index=False)
    frame.loc[~frame["accepted"]].to_csv(outputs["defects_rejected"], index=False)
    return outputs


def write_analysis_config(config: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Freeze the effective configuration alongside results."""

    path = Path(output_dir) / "analysis_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_json_safe(config), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


_REPORT_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SiC 晶圆点状目标分析报告</title>
  <style>
    :root { --ink:#202833; --muted:#64727f; --line:#dbe1e6; --accent:#176b87; --warn:#8b4b13; }
    body { max-width:1100px; margin:2rem auto; padding:0 1.25rem; color:var(--ink); font:15px/1.58 system-ui,sans-serif; }
    h1,h2 { line-height:1.25; } h1 { color:var(--accent); }
    .lead,.caveat { padding:1rem 1.2rem; border-left:5px solid var(--accent); background:#eef7fa; }
    .caveat { border-color:var(--warn); background:#fff7e9; }
    table { width:100%; border-collapse:collapse; margin:1rem 0; }
    th,td { padding:.55rem .65rem; border:1px solid var(--line); text-align:left; vertical-align:top; }
    th { width:34%; background:#f5f7f8; } code { overflow-wrap:anywhere; }
    .gallery { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:1rem; }
    figure { margin:0; border:1px solid var(--line); padding:.65rem; } img { width:100%; height:auto; }
    .warning { color:var(--warn); } footer { margin-top:2rem; color:var(--muted); font-size:.9rem; }
  </style>
</head>
<body>
  <h1>SiC 晶圆点状目标分析报告</h1>
  <p class="lead">本报告统计的是满足当前图像规则的黑色点状目标，不自动等同于真实位错。</p>
  <h2>核心结果</h2>
  <table>
    {% for label, value in headline_rows %}<tr><th>{{ label }}</th><td>{{ value }}</td></tr>{% endfor %}
  </table>
  {% if warnings %}<h2>警告</h2><ul class="warning">{% for item in warnings %}<li>{{ item }}</li>{% endfor %}</ul>{% endif %}
  <h2>质量复核图</h2>
  <div class="gallery">
    {% for filename, caption in images %}{% if filename %}<figure><a href="{{ filename }}"><img src="{{ filename }}" alt="{{ caption }}"></a><figcaption>{{ caption }}</figcaption></figure>{% endif %}{% endfor %}
  </div>
  <h2>可追溯输出</h2>
  <ul>
    <li><a href="defects_all.csv">全部候选及拒绝原因</a></li>
    <li><a href="defects_accepted.csv">自动接受目标</a></li>
    <li><a href="defects_rejected.csv">自动拒绝目标</a></li>
    {% if candidate_crops_available %}<li><a href="candidate_crops/">原始数值 TIFF 候选裁剪及显示预览</a></li>{% endif %}
    <li><a href="analysis_config.yaml">本次实际参数</a></li>
    <li><a href="radial_density.csv">径向密度（有效面积归一化）</a></li>
    <li><a href="angular_density.csv">方位角密度（有效面积归一化）</a></li>
    <li><a href="regional_density.csv">中心/中间/边缘密度</a></li>
    <li><a href="summary.json">完整机器可读摘要</a></li>
  </ul>
  <h2>科学解释边界</h2>
  <p class="caveat">{{ scientific_limit }}<br><br>{{ uncertainty_note }}</p>
  <h2>空间分布解释限制</h2>
  <p class="caveat">{{ spatial_limit }}</p>
  <h2>全部摘要字段</h2>
  <table>{% for key, value in flat_summary.items() %}<tr><th><code>{{ key }}</code></th><td>{{ value }}</td></tr>{% endfor %}</table>
  <footer>sic-wafer-point-counter {{ software_version }} · 报告生成时间 {{ generated_at }}</footer>
</body>
</html>
"""


def generate_html_report(
    summary: Mapping[str, Any],
    output_dir: str | Path,
    *,
    image_files: Mapping[str, str | Path] | None = None,
) -> Path:
    """Render a self-contained-index HTML report linking all audit artifacts."""

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    flat = _flatten_summary(summary)
    label_keys = (
        ("输入文件", ("input_file_name",)),
        ("图像尺寸", ("image_size", "image_dimensions_px")),
        ("晶圆实际直径", ("wafer_diameter_mm", "wafer.diameter_mm")),
        ("检测圆心 (px)", ("wafer_center_px", "wafer.center_px")),
        ("检测半径 (px)", ("wafer_radius_px", "wafer.radius_px")),
        ("标定", ("mm_per_pixel", "wafer.mm_per_pixel")),
        ("理论完整面积 (cm²)", ("theoretical_area_cm2", "areas.theoretical_complete_wafer_area_cm2", "areas.theoretical_cm2")),
        ("拟合圆面积 (cm²)", ("fitted_wafer_area_cm2", "areas.circle_fit_area_cm2", "areas.fitted_cm2")),
        ("最终有效面积 (cm²)", ("valid_analysis_area_cm2", "valid_area_cm2", "areas.valid_analysis_area_cm2", "areas.valid_cm2")),
        ("原始/分水岭后候选", ("candidate_count_display",)),
        ("接受/拒绝", ("acceptance_count_display",)),
        ("点状目标密度", ("density_display",)),
        ("计数统计不确定度", ("uncertainty_display",)),
        ("95% 计数区间", ("confidence_interval_display",)),
        ("中心/中间/边缘密度", ("regional_density_display",)),
        ("真实标注验证状态", ("real_annotation_validation_status",)),
        ("不确定度预算", ("uncertainty_budget_display",)),
        ("运行时间", ("runtime_seconds",)),
    )
    display_summary = dict(summary)
    display_summary["candidate_count_display"] = (
        f"{_summary_value(summary, 'raw_candidate_count', default='—')} / "
        f"{_summary_value(summary, 'post_watershed_candidate_count', default='—')}"
    )
    display_summary["acceptance_count_display"] = (
        f"{_summary_value(summary, 'accepted_count', default='—')} / "
        f"{_summary_value(summary, 'rejected_count', default='—')}"
    )
    rho = _summary_value(
        summary, "point_density_cm2", "density_cm2", "density.density_cm2", "density.rho_cm2"
    )
    sigma = _summary_value(
        summary,
        "counting_uncertainty_cm2",
        "sigma_rho_cm2",
        "density.counting_uncertainty_cm2",
        "density.sigma_rho_cm2",
        "density.sigma_cm2",
    )
    display_summary["density_display"] = f"{float(rho):.6g} cm^-2" if rho is not None else "—"
    display_summary["uncertainty_display"] = f"± {float(sigma):.6g} cm^-2" if sigma is not None else "—"
    low = _summary_value(
        summary,
        "poisson_95_ci_lower_cm2",
        "poisson_density_ci_lower_cm2",
        "density.poisson_density_ci_lower_cm2",
        "density.ci95_lower_cm2",
    )
    high = _summary_value(
        summary,
        "poisson_95_ci_upper_cm2",
        "poisson_density_ci_upper_cm2",
        "density.poisson_density_ci_upper_cm2",
        "density.ci95_upper_cm2",
    )
    display_summary["confidence_interval_display"] = (
        f"[{float(low):.6g}, {float(high):.6g}] cm^-2" if low is not None and high is not None else "—"
    )
    regional = summary.get("regional_density")
    if isinstance(regional, list):
        parts = []
        for item in regional:
            if isinstance(item, Mapping):
                name, density = item.get("region"), item.get("density_cm2")
                if name is not None and density is not None:
                    try:
                        parts.append(f"{name}: {float(density):.6g} cm^-2")
                    except (TypeError, ValueError):
                        parts.append(f"{name}: NA")
        display_summary["regional_density_display"] = "; ".join(parts) or "—"
    uncertainty_budget = summary.get("uncertainty_budget_summary")
    if isinstance(uncertainty_budget, Mapping):
        display_summary["uncertainty_budget_display"] = "; ".join(
            f"{key}: {value}" for key, value in uncertainty_budget.items()
        )
    rows: list[tuple[str, Any]] = []
    for label, keys in label_keys:
        value = _summary_value(display_summary, *keys, default="—")
        if label == "标定" and value != "—":
            value = f"{float(value):.8g} mm/px"
        rows.append((label, value))

    default_images = {
        "overlay_accepted.png": "自动接受目标（绿色圆圈和编号）",
        "overlay_all_candidates.png": "全部候选（接受：绿色；拒绝：红色叉）",
        "defect_comparison_details.png": "缺陷识别对照细节（原始局部与自动判定并排）",
        "wafer_mask.png": "完整晶圆掩膜",
        "valid_analysis_mask.png": "最终有效分析掩膜",
        "preprocessed_preview.png": "预处理响应预览",
        "candidate_mask.png": "候选二值掩膜",
        "defect_size_histogram.png": "目标尺寸分布",
        "radial_distribution.png": "径向计数（非面积归一化，仅供描述）",
        "angular_distribution.png": "方位角计数（非面积归一化，仅供描述）",
        "radial_density.png": "径向密度（实际有效面积归一化）",
        "angular_density.png": "方位角密度（图像正 x 轴参考）",
        "wafer_position_scatter.png": "二维位置分布",
        "density_heatmap.png": "分区密度/计数热图",
    }
    images: list[tuple[str, str]] = []
    if image_files:
        for caption, candidate in image_files.items():
            candidate_path = Path(candidate)
            try:
                relative = candidate_path.relative_to(folder)
            except ValueError:
                relative = candidate_path
            if candidate_path.exists() or (folder / relative).exists():
                images.append((str(relative), caption))
    else:
        images = [(name, caption) for name, caption in default_images.items() if (folder / name).exists()]

    environment = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html", "xml"]))
    report = environment.from_string(_REPORT_TEMPLATE).render(
        headline_rows=rows,
        warnings=_summary_value(summary, "warnings", default=[]),
        images=images,
        scientific_limit=SCIENTIFIC_LIMITATION_ZH,
        uncertainty_note=UNCERTAINTY_NOTE_ZH,
        spatial_limit=SPATIAL_INTERPRETATION_LIMIT_ZH,
        flat_summary=flat,
        software_version=_summary_value(summary, "software_version", default=__version__),
        generated_at=_summary_value(summary, "generated_at_utc", default=""),
        candidate_crops_available=(folder / "candidate_crops").is_dir(),
    )
    path = folder / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


def write_analysis_outputs(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    defects: pd.DataFrame | Iterable[Mapping[str, Any]] | None,
    config: Mapping[str, Any],
    *,
    input_image_path: str | Path | None = None,
    original_image: np.ndarray | None = None,
    full_wafer_mask: np.ndarray | None = None,
    valid_analysis_mask: np.ndarray | None = None,
    preprocessed_image: np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
    logger_messages: Iterable[str] | None = None,
    source_shape: tuple[int, int] | None = None,
    crop_reader: Callable[[int, int, int, int], np.ndarray] | None = None,
) -> dict[str, Path]:
    """Write the complete required output bundle for one analysis run.

    Missing optional arrays only suppress their corresponding PNG; tabular and
    metadata artifacts are always written.  This is useful for rerunning reports
    from a reviewed CSV without repeating expensive image processing.
    """

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    frame = defects_to_frame(defects)
    output_config = dict(config.get("output", {}))
    io_config = dict(config.get("io", {}))
    max_size = int(io_config.get("max_overlay_size", 6000))

    if output_config.get("save_candidate_crops", True) and (
        original_image is not None or crop_reader is not None
    ):
        preview_mismatch = (
            crop_reader is None
            and original_image is not None
            and source_shape is not None
            and tuple(np.asarray(original_image).shape[:2]) != tuple(source_shape[:2])
        )
        if preview_mismatch:
            LOGGER.warning(
                "Candidate crops skipped: original_image is a preview; provide crop_reader for full-resolution crops"
            )
        else:
            frame = save_candidate_crops(
                original_image,
                frame,
                folder,
                half_size_px=int(output_config.get("crop_half_size_px", 32)),
                source_shape=source_shape,
                crop_reader=crop_reader,
            )

    canonical = _canonical_summary(summary, frame, config, input_image_path)
    outputs: dict[str, Path] = {}
    outputs.update(write_summary_files(canonical, folder))
    outputs.update(write_defect_tables(frame, folder))
    outputs["analysis_config"] = write_analysis_config(config, folder)

    if original_image is not None:
        center = _summary_value(canonical, "wafer_center_px", "wafer.center_px")
        if isinstance(center, Mapping):
            center = (center.get("x"), center.get("y"))
        if not (isinstance(center, (list, tuple)) and len(center) >= 2):
            cx = _summary_value(canonical, "center_x_px", "wafer.center_x_px")
            cy = _summary_value(canonical, "center_y_px", "wafer.center_y_px")
            center = (cx, cy) if cx is not None and cy is not None else None
        radius = _summary_value(canonical, "wafer_radius_px", "radius_px", "wafer.radius_px")
        outputs.update(
            save_overlays(
                original_image,
                frame,
                folder,
                draw_labels=bool(output_config.get("draw_labels", True)),
                draw_rejected=bool(output_config.get("draw_rejected", True)),
                max_size=max_size,
                wafer_center_px=tuple(center[:2]) if center is not None else None,
                wafer_radius_px=float(radius) if radius is not None else None,
                source_shape=source_shape,
            )
        )
        if bool(output_config.get("generate_defect_comparison", True)):
            outputs["defect_comparison_details"] = save_defect_comparison_details(
                original_image,
                frame,
                folder / "defect_comparison_details.png",
                half_size_px=int(output_config.get("comparison_crop_half_size_px", 48)),
                max_candidates=int(output_config.get("comparison_max_candidates", 12)),
                source_shape=source_shape,
                crop_reader=crop_reader,
            )
    if full_wafer_mask is not None:
        outputs["wafer_mask"] = save_binary_mask(full_wafer_mask, folder / "wafer_mask.png", max_size=max_size)
    if valid_analysis_mask is not None:
        outputs["valid_analysis_mask"] = save_binary_mask(
            valid_analysis_mask, folder / "valid_analysis_mask.png", max_size=max_size
        )
    if preprocessed_image is not None and output_config.get("save_intermediates", True):
        outputs["preprocessed_preview"] = save_grayscale_image(
            preprocessed_image, folder / "preprocessed_preview.png", max_size=max_size
        )
    if candidate_mask is not None and output_config.get("save_intermediates", True):
        outputs["candidate_mask"] = save_binary_mask(
            candidate_mask, folder / "candidate_mask.png", max_size=max_size
        )

    center_for_plot: tuple[float, float] | None = None
    center_value = _summary_value(canonical, "wafer_center_px", "wafer.center_px")
    if isinstance(center_value, Mapping):
        center_for_plot = (float(center_value["x"]), float(center_value["y"]))
    elif isinstance(center_value, (tuple, list)) and len(center_value) >= 2:
        center_for_plot = (float(center_value[0]), float(center_value[1]))
    elif _summary_value(canonical, "center_x_px") is not None:
        center_for_plot = (
            float(_summary_value(canonical, "center_x_px")),
            float(_summary_value(canonical, "center_y_px")),
        )
    mm_per_pixel = _summary_value(canonical, "mm_per_pixel", "wafer.mm_per_pixel")
    diameter_mm = float(_summary_value(canonical, "wafer_diameter_mm", default=100.0))
    plot_center = center_for_plot
    plot_mm_per_pixel: float | tuple[float, float] | None = (
        float(mm_per_pixel) if mm_per_pixel is not None else None
    )
    if (
        valid_analysis_mask is not None
        and source_shape is not None
        and tuple(np.asarray(valid_analysis_mask).shape[:2]) != tuple(source_shape[:2])
        and center_for_plot is not None
        and mm_per_pixel is not None
    ):
        mask_height, mask_width = np.asarray(valid_analysis_mask).shape[:2]
        source_height, source_width = source_shape[:2]
        preview_per_source_x = mask_width / float(source_width)
        preview_per_source_y = mask_height / float(source_height)
        source_per_preview_x = 1.0 / preview_per_source_x
        source_per_preview_y = 1.0 / preview_per_source_y
        plot_center = (
            center_for_plot[0] * preview_per_source_x,
            center_for_plot[1] * preview_per_source_y,
        )
        # Preserve x/y scales separately; this also keeps the physical preview
        # pixel area exact when integer resizing made the ratios differ slightly.
        plot_mm_per_pixel = (
            float(mm_per_pixel) * source_per_preview_x,
            float(mm_per_pixel) * source_per_preview_y,
        )
        LOGGER.info(
            "Density heatmap valid mask is a preview (%dx%d of %dx%d); remapped geometry for area bins",
            mask_width,
            mask_height,
            source_width,
            source_height,
        )
    outputs.update(
        save_distribution_plots(
            frame,
            folder,
            valid_mask=valid_analysis_mask,
            center_px=plot_center,
            mm_per_pixel=plot_mm_per_pixel,
            wafer_radius_mm=diameter_mm / 2.0,
            generate_heatmap=bool(output_config.get("generate_heatmap", True)),
        )
    )
    if (
        valid_analysis_mask is not None
        and plot_center is not None
        and plot_mm_per_pixel is not None
    ):
        spatial_config = config.get("spatial", {})
        if not isinstance(spatial_config, Mapping):
            raise ValueError("spatial configuration must be a mapping")
        spatial_outputs, spatial_tables = save_area_normalized_distributions(
            frame,
            folder,
            valid_mask=valid_analysis_mask,
            center_px=plot_center,
            mm_per_pixel=plot_mm_per_pixel,
            wafer_radius_mm=diameter_mm / 2.0,
            radial_bins=int(spatial_config.get("radial_bin_count", 6)),
            radial_mode=str(spatial_config.get("radial_bin_mode", "equal_area")),
            angular_sectors=int(spatial_config.get("angular_sector_count", 12)),
            regions=spatial_config.get("regions"),
        )
        outputs.update(spatial_outputs)
        spatial_summary = {
            "radial_bin_mode": str(spatial_config.get("radial_bin_mode", "equal_area")),
            "angle_reference": "image_positive_x",
            "regional_density": _json_safe(
                spatial_tables["regional"].to_dict(orient="records")
            ),
            "spatial_heterogeneity": spatial_heterogeneity_indicator(
                spatial_tables["regional"]["count"].tolist(),
                spatial_tables["regional"]["valid_area_cm2"].tolist(),
            ),
        }
        canonical["spatial_density"] = spatial_summary
        canonical["regional_density"] = spatial_summary["regional_density"]
        # ``summary`` is normally the mutable dictionary constructed by the
        # pipeline; update it too so its final post-report write includes the
        # exact tables used for the figures.
        if isinstance(summary, dict):
            summary["spatial_density"] = spatial_summary
            summary["regional_density"] = spatial_summary["regional_density"]
        outputs.update(write_summary_files(canonical, folder))

    log_path = folder / "run.log"
    log_lines = list(logger_messages or [])
    if not log_lines:
        log_lines = [
            f"{canonical.get('generated_at_utc', '')} reporting: output bundle completed",
            f"Python: {sys.version.split()[0]}; software: {__version__}",
        ]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(str(line) for line in log_lines) + "\n")
    outputs["run_log"] = log_path

    if output_config.get("generate_html_report", True):
        outputs["report_html"] = generate_html_report(canonical, folder)
    return outputs


# Short aliases retained for callers that build the output set in stages.
write_summary = write_summary_files
write_results = write_analysis_outputs
