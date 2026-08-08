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
from .paper_alignment import references_from_config
from .validation import spatial_heterogeneity_indicator
from .visualization import (
    normalize_for_display,
    save_area_normalized_distributions,
    save_binary_mask,
    save_distribution_plots,
    save_grayscale_image,
    save_overlays,
    save_xrt_detection_overlay,
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
PAPER_ALIGNMENT_NOTE_NOT_PROVIDED_ZH = (
    "论文式红色矩形仅表示程序自动接受的 XRT 点状候选。黄色圆圈在原论文中表示独立 DIC 坑位；"
    "本分析没有导入、配准并核验独立 DIC/KOH 数据，因此不会生成黄色验证圈，也不能声称完成 TSD 物理验证。"
)


def _paper_alignment_note(summary: Mapping[str, Any]) -> str:
    """Describe the independent-reference status without overstating evidence."""

    reference = summary.get("independent_reference")
    if not isinstance(reference, Mapping) or reference.get("status") == "not provided":
        return PAPER_ALIGNMENT_NOTE_NOT_PROVIDED_ZH
    method = str(reference.get("method", "independent reference"))
    count = int(reference.get("confirmed_registered_count", 0))
    coverage = reference.get("agreement", {}).get("reference_coverage_complete", False)
    precision_note = (
        "已声明参考覆盖完整，可按匹配容差计算 precision/recall/F1。"
        if coverage
        else "参考覆盖未声明完整，因此只报告匹配数和相对于已登记参考点的召回率，不计算 precision/F1。"
    )
    return (
        f"黄色圆圈来自经过源图与独立参考图 SHA-256 核验、并已配准的 {method} 坐标，共 {count} 个；"
        "红框仍只表示程序自动接受的 XRT 点状候选，参考点不会改变自动 n 或 rho。"
        f"{precision_note} 单次配准对照不能证明该图像规则在其他晶圆上普遍等同于 TSD。"
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
    "equivalent_diameter_um",
    "major_axis_length_px",
    "minor_axis_length_px",
    "major_axis_length_um",
    "minor_axis_length_um",
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
            crop_values = np.asarray(raw_crop)
            if (
                np.issubdtype(crop_values.dtype, np.floating)
                and np.isfinite(crop_values).all()
                and float(np.min(crop_values)) >= 0.0
                and float(np.max(crop_values)) <= 1.0
            ):
                # The pipeline supplies globally normalized float32 crops here,
                # preserving WhiteIsZero direction and between-candidate contrast.
                display = np.rint(crop_values.astype(np.float32) * 255.0).astype(np.uint8)
            else:
                display = normalize_for_display(crop_values, 0.5, 99.5)
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


def _true_candidate_bbox(record: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    """Parse the segmentation bounding box without inventing a display-sized box."""

    raw = record.get("bounding_box")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite((x0, y0, x1, y1))) or x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _globally_normalized_crop_to_uint8(crop: np.ndarray) -> np.ndarray:
    """Convert one globally normalized scientific crop without local restretching."""

    values = np.asarray(crop)
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"Expected a non-empty 2-D grayscale field, got {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(
            "XRT detail crop_reader must return global-normalized scientific float [0, 1]"
        )
    if not np.isfinite(values).all():
        raise ValueError("XRT detail crop_reader returned non-finite values")
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(
            "XRT detail crop_reader did not use the global scientific normalization window"
        )
    return np.rint(np.clip(values, 0.0, 1.0).astype(np.float32) * 255.0).astype(np.uint8)


def _representative_candidate_indices(frame: pd.DataFrame, limit: int) -> list[int]:
    """Select deterministic, spatially spread accepted-candidate anchors."""

    if frame.empty or limit <= 0:
        return []
    x_values = pd.to_numeric(frame["centroid_x_px"], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(frame["centroid_y_px"], errors="coerce").to_numpy(dtype=float)
    coordinates = np.column_stack((x_values, y_values))
    if not np.isfinite(coordinates).all():
        raise ValueError("Accepted candidates require finite centroid coordinates for detail fields")

    # Start near the robust spatial middle, then spread subsequent fields by a
    # deterministic farthest-point rule. Stable input ordering resolves ties.
    median = np.median(coordinates, axis=0)
    first = int(np.argmin(np.sum((coordinates - median) ** 2, axis=1)))
    selected = [first]
    nearest_distance = np.sum((coordinates - coordinates[first]) ** 2, axis=1)
    while len(selected) < min(limit, len(frame)):
        nearest_distance[np.asarray(selected, dtype=int)] = -1.0
        next_index = int(np.argmax(nearest_distance))
        selected.append(next_index)
        candidate_distance = np.sum((coordinates - coordinates[next_index]) ** 2, axis=1)
        nearest_distance = np.minimum(nearest_distance, candidate_distance)
    return selected


def save_xrt_detection_detail_montage(
    original_image: np.ndarray | None,
    defects: pd.DataFrame,
    output_path: str | Path,
    *,
    mm_per_pixel: float,
    field_size_mm: float = 4.0,
    max_fields: int = 6,
    scale_bar_mm: float = 1.0,
    source_shape: tuple[int, int] | None = None,
    crop_reader: Callable[[int, int, int, int], np.ndarray] | None = None,
    independent_reference_points: pd.DataFrame | None = None,
    independent_reference_label: str = "NOT PROVIDED",
) -> Path:
    """Save full-resolution local XRT fields with automatic red boxes.

    ``crop_reader`` must return scientific floating-point values using the image
    source's single global normalization
    window. Every accepted candidate whose true segmentation ``bounding_box``
    intersects a selected field is drawn as a red rectangle. Yellow circles are
    drawn only for coordinates already verified by :mod:`paper_alignment`; this
    function never infers or invents them. The artifact never changes candidate
    acceptance or density.
    """

    if not np.isfinite(mm_per_pixel) or mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be finite and positive")
    if not np.isfinite(field_size_mm) or field_size_mm <= 0:
        raise ValueError("field_size_mm must be finite and positive")
    if not isinstance(max_fields, int) or not 1 <= max_fields <= 6:
        raise ValueError("max_fields must be an integer from 1 to 6")
    if not np.isfinite(scale_bar_mm) or scale_bar_mm <= 0 or scale_bar_mm >= field_size_mm:
        raise ValueError("scale_bar_mm must be positive and smaller than field_size_mm")
    if crop_reader is None and original_image is None:
        raise ValueError("A global-normalized crop_reader or full scientific image is required")

    if source_shape is not None:
        source_height, source_width = map(int, source_shape[:2])
    elif original_image is not None:
        source_height, source_width = np.asarray(original_image).shape[:2]
    else:
        raise ValueError("source_shape is required for crop_reader-only detail fields")
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"Invalid source_shape: {(source_height, source_width)}")
    if crop_reader is None and original_image is not None:
        actual_shape = tuple(np.asarray(original_image).shape[:2])
        if actual_shape != (source_height, source_width):
            raise ValueError("Full-resolution crop_reader is required when original_image is a preview")

    frame = defects_to_frame(defects)
    accepted = frame.loc[frame["accepted"].map(_truthy)].copy()
    accepted["_x"] = pd.to_numeric(accepted["centroid_x_px"], errors="coerce")
    accepted["_y"] = pd.to_numeric(accepted["centroid_y_px"], errors="coerce")
    accepted["_id_numeric"] = pd.to_numeric(accepted.get("defect_id"), errors="coerce")
    accepted["_id_text"] = accepted.get(
        "defect_id", pd.Series(index=accepted.index, dtype=object)
    ).astype(str)
    accepted = accepted.sort_values(
        ["_id_numeric", "_id_text", "_y", "_x"], kind="stable", na_position="last"
    ).reset_index(drop=True)
    references = (
        independent_reference_points.copy()
        if independent_reference_points is not None
        else pd.DataFrame(columns=("x_px", "y_px"))
    )
    if not references.empty:
        missing_reference_columns = {"x_px", "y_px"} - set(references.columns)
        if missing_reference_columns:
            raise ValueError(
                "Independent reference points are missing columns: "
                f"{sorted(missing_reference_columns)}"
            )
        references["x_px"] = pd.to_numeric(references["x_px"], errors="coerce")
        references["y_px"] = pd.to_numeric(references["y_px"], errors="coerce")
        if references[["x_px", "y_px"]].isna().any().any():
            raise ValueError("Independent reference points require finite x_px and y_px")

    bboxes: list[tuple[float, float, float, float]] = []
    for _, record in accepted.iterrows():
        bbox = _true_candidate_bbox(record)
        if bbox is None:
            raise ValueError(
                f"Accepted candidate {record.get('defect_id', '?')} has no valid true bounding_box"
            )
        bboxes.append(bbox)

    field_size_px = max(1, int(round(float(field_size_mm) / float(mm_per_pixel))))
    field_width_px = min(field_size_px, source_width)
    field_height_px = min(field_size_px, source_height)

    def field_bounds(x: float, y: float) -> tuple[int, int, int, int]:
        x0 = max(0, min(source_width - field_width_px, int(round(x - field_width_px / 2.0))))
        y0 = max(0, min(source_height - field_height_px, int(round(y - field_height_px / 2.0))))
        return x0, y0, x0 + field_width_px, y0 + field_height_px

    selected_fields: list[tuple[int, int, int, int]] = []
    for _, reference in references.sort_values(["y_px", "x_px"], kind="stable").iterrows():
        bounds = field_bounds(float(reference["x_px"]), float(reference["y_px"]))
        if bounds not in selected_fields:
            selected_fields.append(bounds)
        if len(selected_fields) >= max_fields:
            break
    for index in _representative_candidate_indices(accepted, max_fields):
        bounds = field_bounds(float(accepted.iloc[index]["_x"]), float(accepted.iloc[index]["_y"]))
        if bounds not in selected_fields and len(selected_fields) < max_fields:
            selected_fields.append(bounds)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tile_size = 420
    card_footer = 42
    card_gap = 12
    header_height = 82
    columns = min(3, max(1, len(selected_fields)))
    rows = max(1, int(math.ceil(max(1, len(selected_fields)) / columns)))
    card_width = tile_size
    card_height = tile_size + card_footer
    canvas_width = columns * card_width + (columns + 1) * card_gap
    canvas_height = header_height + rows * card_height + (rows + 1) * card_gap
    canvas = np.full((canvas_height, canvas_width, 3), 248, dtype=np.uint8)
    cv2.putText(
        canvas,
        "XRT point-target detection fields",
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        (27, 35, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"RED = automatic candidates | YELLOW = {str(independent_reference_label)[:32]}",
        (18, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (58, 68, 73),
        1,
        cv2.LINE_AA,
    )

    if not selected_fields:
        cv2.putText(
            canvas,
            "No automatically accepted candidates were available for field selection.",
            (18, header_height + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (70, 78, 82),
            1,
            cv2.LINE_AA,
        )
    else:
        for field_number, (x0, y0, x1, y1) in enumerate(selected_fields, start=1):
            if crop_reader is not None:
                crop = np.asarray(crop_reader(x0, y0, x1, y1))
            else:
                crop = np.asarray(original_image)[y0:y1, x0:x1]  # type: ignore[index]
            display = _globally_normalized_crop_to_uint8(crop)
            interpolation = cv2.INTER_AREA if max(display.shape) > tile_size else cv2.INTER_LINEAR
            tile = cv2.resize(display, (tile_size, tile_size), interpolation=interpolation)
            tile_bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            scale_x = tile_size / float(x1 - x0)
            scale_y = tile_size / float(y1 - y0)

            field_box_count = 0
            for record_number in range(len(accepted)):
                bx0, by0, bx1, by1 = bboxes[record_number]
                if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
                    continue
                left = int(round((max(bx0, x0) - x0) * scale_x))
                top = int(round((max(by0, y0) - y0) * scale_y))
                right = int(round((min(bx1, x1) - x0) * scale_x)) - 1
                bottom = int(round((min(by1, y1) - y0) * scale_y)) - 1
                left, right = np.clip((left, right), 0, tile_size - 1).astype(int)
                top, bottom = np.clip((top, bottom), 0, tile_size - 1).astype(int)
                if right >= left and bottom >= top:
                    cv2.rectangle(tile_bgr, (left, top), (right, bottom), (0, 0, 255), 2, cv2.LINE_AA)
                    field_box_count += 1

            field_reference_count = 0
            for _, reference in references.iterrows():
                reference_x = float(reference["x_px"])
                reference_y = float(reference["y_px"])
                if not (x0 <= reference_x < x1 and y0 <= reference_y < y1):
                    continue
                point = (
                    int(round((reference_x - x0) * scale_x)),
                    int(round((reference_y - y0) * scale_y)),
                )
                cv2.circle(tile_bgr, point, 7, (0, 255, 255), 2, cv2.LINE_AA)
                field_reference_count += 1

            bar_length = max(1, int(round((float(scale_bar_mm) / float(mm_per_pixel)) * scale_x)))
            bar_length = min(bar_length, tile_size - 48)
            bar_x1 = tile_size - 18
            bar_x0 = bar_x1 - bar_length
            bar_y = tile_size - 24
            cv2.line(tile_bgr, (bar_x0, bar_y), (bar_x1, bar_y), (0, 0, 0), 7, cv2.LINE_AA)
            scale_label = f"{float(scale_bar_mm):g} mm"
            (label_width, _), _ = cv2.getTextSize(
                scale_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_origin = (bar_x1 - label_width, bar_y - 10)
            cv2.putText(
                tile_bgr, scale_label, label_origin, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 3, cv2.LINE_AA,
            )
            cv2.putText(
                tile_bgr, scale_label, label_origin, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

            row, column = divmod(field_number - 1, columns)
            left = card_gap + column * (card_width + card_gap)
            top = header_height + card_gap + row * (card_height + card_gap)
            canvas[top:top + tile_size, left:left + tile_size] = tile_bgr
            actual_width_mm = (x1 - x0) * float(mm_per_pixel)
            actual_height_mm = (y1 - y0) * float(mm_per_pixel)
            footer = (
                f"Field {field_number:02d} | {actual_width_mm:.3g} x {actual_height_mm:.3g} mm"
                f" | red boxes: {field_box_count} | yellow refs: {field_reference_count}"
            )
            cv2.putText(
                canvas, footer, (left + 5, top + tile_size + 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (45, 53, 57), 1, cv2.LINE_AA,
            )
            cv2.rectangle(
                canvas,
                (left - 1, top - 1),
                (left + card_width, top + card_height - 1),
                (208, 216, 218),
                1,
            )

    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Could not write XRT detection detail montage: {destination}")
    return destination


def save_paper_aligned_result_figure(
    detection_field_path: str | Path,
    density_heatmap_path: str | Path,
    output_path: str | Path,
    *,
    independent_reference_status: str,
) -> Path:
    """Combine the paper-semantics detection field and quantitative density map.

    This is a presentation artifact only. Its density panel is the existing
    area-normalized heatmap, and it never changes candidate classification,
    count, valid area, or density.
    """

    detection = cv2.imread(str(detection_field_path), cv2.IMREAD_COLOR)
    heatmap = cv2.imread(str(density_heatmap_path), cv2.IMREAD_COLOR)
    if detection is None or heatmap is None:
        raise OSError("Paper-aligned figure requires readable detection and heatmap PNG files")

    target_width = max(detection.shape[1], heatmap.shape[1], 900)

    def fit_width(image: np.ndarray) -> np.ndarray:
        scale = target_width / float(image.shape[1])
        height = max(1, int(round(image.shape[0] * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        return cv2.resize(image, (target_width, height), interpolation=interpolation)

    detection = fit_width(detection)
    heatmap = fit_width(heatmap)
    header_height = 92
    gap = 24
    footer_height = 72
    canvas = np.full(
        (
            header_height + detection.shape[0] + gap + heatmap.shape[0] + footer_height,
            target_width,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        "Paper-aligned XRT point-target analysis",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (28, 36, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Automatic image-rule candidates (top) and actual-area-normalized wafer density (bottom)",
        (24, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (62, 72, 76),
        1,
        cv2.LINE_AA,
    )
    y_detection = header_height
    canvas[y_detection:y_detection + detection.shape[0]] = detection
    y_heatmap = y_detection + detection.shape[0] + gap
    canvas[y_heatmap:y_heatmap + heatmap.shape[0]] = heatmap
    footer_y = y_heatmap + heatmap.shape[0] + 31
    cv2.putText(
        canvas,
        f"Independent registered reference: {independent_reference_status}",
        (24, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (60, 68, 72),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Density = accepted point-like targets / final valid-mask area; this figure does not by itself prove TSD identity.",
        (24, footer_y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (60, 68, 72),
        1,
        cv2.LINE_AA,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Could not write paper-aligned result figure: {destination}")
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
    {% if density_heatmap_grid_available %}<li><a href="density_heatmap_grid.csv">整片二维密度逐格数据（数量、实际有效面积、密度与泊松区间）</a></li>{% endif %}
    {% if independent_reference_available %}<li><a href="independent_reference_points.csv">独立参考原始登记表（保留可能与不确定标注）</a></li><li><a href="independent_reference_matches.csv">自动候选与独立参考匹配审计</a></li>{% endif %}
    <li><a href="summary.json">完整机器可读摘要</a></li>
  </ul>
  <h2>科学解释边界</h2>
  <p class="caveat">{{ scientific_limit }}<br><br>{{ paper_alignment_note }}<br><br>{{ uncertainty_note }}</p>
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

    reference_available = (folder / "independent_reference_points.csv").is_file()
    default_images = {
        "overlay_accepted.png": "自动接受目标（绿色圆圈和编号）",
        "overlay_all_candidates.png": "全部候选（接受：绿色；拒绝：红色叉）",
        "overlay_xrt_red_boxes.png": (
            "论文语义对齐图：红框为自动 XRT 点状候选，黄圈为已核验独立参考"
            if reference_available
            else "论文语义对齐的自动 XRT 点状候选红框图（无独立 DIC/KOH 黄圈）"
        ),
        "xrt_detection_detail_montage.png": (
            "论文风格局部视场：红框为自动接受点状候选；黄圈为已核验独立参考"
            if reference_available
            else "论文风格局部视场：红框为自动接受点状候选；独立参考未提供"
        ),
        "paper_detection_field.png": "单视场论文语义对照：红框为自动候选，黄色圈仅来自已核验独立参考",
        "paper_aligned_result_figure.png": "论文式综合成果：点状候选对照视场与按实际有效面积归一化的整片密度图",
        "defect_comparison_details.png": "原始局部与自动判定复核（非 DIC/KOH 独立验证）",
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
        "density_heatmap.png": "按最终有效掩膜逐格面积归一化的点状目标密度（cm^-2）",
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
        paper_alignment_note=_paper_alignment_note(summary),
        uncertainty_note=UNCERTAINTY_NOTE_ZH,
        spatial_limit=SPATIAL_INTERPRETATION_LIMIT_ZH,
        flat_summary=flat,
        software_version=_summary_value(summary, "software_version", default=__version__),
        generated_at=_summary_value(summary, "generated_at_utc", default=""),
        candidate_crops_available=(folder / "candidate_crops").is_dir(),
        density_heatmap_grid_available=(folder / "density_heatmap_grid.csv").is_file(),
        independent_reference_available=(folder / "independent_reference_points.csv").is_file(),
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
    comparison_crop_reader: Callable[[int, int, int, int], np.ndarray] | None = None,
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

    outputs: dict[str, Path] = {}
    canonical = _canonical_summary(summary, frame, config, input_image_path)
    effective_source_shape = source_shape
    if effective_source_shape is None and original_image is not None:
        effective_source_shape = tuple(np.asarray(original_image).shape[:2])
    scale_for_reference = _summary_value(canonical, "mm_per_pixel", "wafer.mm_per_pixel")
    registered_references, reference_summary, reference_matches = references_from_config(
        config,
        source_image_path=input_image_path,
        source_shape=effective_source_shape,
        automatic_candidates=frame,
        mm_per_pixel=(
            float(scale_for_reference) if scale_for_reference is not None else None
        ),
    )
    canonical["independent_reference"] = _json_safe(reference_summary)
    paper_reference_alignment = dict(canonical.get("paper_reference_alignment", {}))
    paper_reference_alignment.update(
        {
            "automatic_xrt_marker": "red rectangle",
            "independent_reference_marker": "yellow circle",
            "independent_reference_data_supplied": registered_references is not None,
            "independent_reference_status": reference_summary.get("status"),
            "independent_reference_method": reference_summary.get("method"),
            "physical_identity_claim": False,
        }
    )
    canonical["paper_reference_alignment"] = paper_reference_alignment
    if isinstance(summary, dict):
        summary["independent_reference"] = _json_safe(reference_summary)
        summary["paper_reference_alignment"] = paper_reference_alignment
    if registered_references is not None:
        outputs["independent_reference_points"] = folder / "independent_reference_points.csv"
        registered_references.all_rows.to_csv(
            outputs["independent_reference_points"], index=False
        )
        outputs["independent_reference_matches"] = folder / "independent_reference_matches.csv"
        reference_matches.to_csv(outputs["independent_reference_matches"], index=False)

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
        scale_value = _summary_value(canonical, "mm_per_pixel", "wafer.mm_per_pixel")
        if bool(output_config.get("generate_xrt_red_box_overlay", True)):
            outputs["overlay_xrt_red_boxes"] = save_xrt_detection_overlay(
                original_image,
                frame,
                folder / "overlay_xrt_red_boxes.png",
                max_size=max_size,
                source_shape=source_shape,
                mm_per_pixel=float(scale_value),
                scale_bar_mm=float(output_config.get("xrt_overlay_scale_bar_mm", 10.0)),
                draw_labels=bool(output_config.get("xrt_overlay_draw_labels", False)),
                independent_reference_points=(
                    registered_references.points if registered_references is not None else None
                ),
            )
        if bool(output_config.get("generate_xrt_detection_detail_montage", True)):
            outputs["xrt_detection_detail_montage"] = save_xrt_detection_detail_montage(
                original_image,
                frame,
                folder / "xrt_detection_detail_montage.png",
                mm_per_pixel=float(scale_value),
                field_size_mm=float(
                    output_config.get("xrt_detection_detail_field_size_mm", 4.0)
                ),
                max_fields=int(output_config.get("xrt_detection_detail_max_fields", 6)),
                scale_bar_mm=float(
                    output_config.get("xrt_detection_detail_scale_bar_mm", 1.0)
                ),
                source_shape=source_shape,
                crop_reader=comparison_crop_reader,
                independent_reference_points=(
                    registered_references.points if registered_references is not None else None
                ),
                independent_reference_label=(
                    f"registered {reference_summary.get('method')} observations"
                    if registered_references is not None
                    else "NOT PROVIDED"
                ),
            )
        if bool(output_config.get("generate_paper_aligned_figure", True)):
            outputs["paper_detection_field"] = save_xrt_detection_detail_montage(
                original_image,
                frame,
                folder / "paper_detection_field.png",
                mm_per_pixel=float(scale_value),
                field_size_mm=float(output_config.get("paper_detail_field_size_mm", 6.0)),
                max_fields=1,
                scale_bar_mm=float(output_config.get("paper_detail_scale_bar_mm", 2.0)),
                source_shape=source_shape,
                crop_reader=comparison_crop_reader,
                independent_reference_points=(
                    registered_references.points if registered_references is not None else None
                ),
                independent_reference_label=(
                    f"registered {reference_summary.get('method')} observations"
                    if registered_references is not None
                    else "NOT PROVIDED"
                ),
            )
        if bool(output_config.get("generate_defect_comparison", True)):
            outputs["defect_comparison_details"] = save_defect_comparison_details(
                original_image,
                frame,
                folder / "defect_comparison_details.png",
                half_size_px=int(output_config.get("comparison_crop_half_size_px", 48)),
                max_candidates=int(output_config.get("comparison_max_candidates", 12)),
                source_shape=source_shape,
                crop_reader=comparison_crop_reader or crop_reader,
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
    spatial_config = config.get("spatial", {})
    if not isinstance(spatial_config, Mapping):
        raise ValueError("spatial configuration must be a mapping")
    heatmap_requested = bool(output_config.get("generate_heatmap", True))
    heatmap_ready = (
        valid_analysis_mask is not None
        and plot_center is not None
        and plot_mm_per_pixel is not None
    )
    if heatmap_requested and not heatmap_ready:
        LOGGER.warning(
            "Density heatmap skipped: final valid mask and physical calibration are required; "
            "a count-only plot is not substituted"
        )
    outputs.update(
        save_distribution_plots(
            frame,
            folder,
            valid_mask=valid_analysis_mask,
            center_px=plot_center,
            mm_per_pixel=plot_mm_per_pixel,
            wafer_radius_mm=diameter_mm / 2.0,
            generate_heatmap=heatmap_requested and heatmap_ready,
            heatmap_bins=int(spatial_config.get("heatmap_bin_count", 40)),
            heatmap_colormap=str(spatial_config.get("heatmap_colormap", "turbo")),
            heatmap_vmin_cm2=spatial_config.get("heatmap_vmin_cm2", 0.0),
            heatmap_vmax_cm2=spatial_config.get("heatmap_vmax_cm2"),
            heatmap_clip_percentile=float(spatial_config.get("heatmap_clip_percentile", 99.5)),
            heatmap_min_valid_fraction=float(
                spatial_config.get("heatmap_min_valid_fraction", 0.05)
            ),
            heatmap_grid_interval_mm=float(
                spatial_config.get("heatmap_grid_interval_mm", 10.0)
            ),
            heatmap_reported_mean_density_cm2=float(
                _summary_value(canonical, "point_density_cm2")
            ),
        )
    )
    if heatmap_ready:
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
        heatmap_summary: dict[str, Any] = {
            "status": "not generated",
            "normalization": "count / actual valid_analysis_mask area in each cell",
        }
        grid_path = outputs.get("density_heatmap_grid")
        if grid_path is not None and Path(grid_path).is_file():
            grid_frame = pd.read_csv(grid_path)
            heatmap_area = float(grid_frame["valid_area_cm2"].sum())
            heatmap_count = int(grid_frame["count"].sum())
            expected_area = float(
                _summary_value(canonical, "valid_analysis_area_cm2", default=heatmap_area)
            )
            area_relative_error = (
                abs(heatmap_area - expected_area) / expected_area if expected_area > 0 else 0.0
            )
            finite_density = pd.to_numeric(grid_frame["density_cm2"], errors="coerce")
            display_rows = grid_frame["valid_area_fraction"] >= float(
                spatial_config.get("heatmap_min_valid_fraction", 0.05)
            )
            display_density = finite_density[display_rows].dropna()
            configured_vmax = spatial_config.get("heatmap_vmax_cm2")
            displayed_vmax = (
                float(configured_vmax)
                if configured_vmax is not None
                else (
                    float(
                        np.percentile(
                            display_density,
                            float(spatial_config.get("heatmap_clip_percentile", 99.5)),
                        )
                    )
                    if not display_density.empty
                    else None
                )
            )
            heatmap_summary = {
                "status": "generated",
                "normalization": "count / actual valid_analysis_mask area in each cell",
                "bin_count_per_axis": int(spatial_config.get("heatmap_bin_count", 40)),
                "cell_size_mm": float(grid_frame.iloc[0]["x_right_mm"] - grid_frame.iloc[0]["x_left_mm"]),
                "valid_area_cm2": heatmap_area,
                "valid_area_relative_error_vs_primary": area_relative_error,
                "count": heatmap_count,
                "reported_whole_wafer_mean_density_cm2": float(
                    _summary_value(canonical, "point_density_cm2")
                ),
                "grid_derived_mean_density_cm2": (
                    float(heatmap_count / heatmap_area) if heatmap_area > 0 else None
                ),
                "count_matches_accepted_n": heatmap_count
                == int(_summary_value(canonical, "accepted_count", default=heatmap_count)),
                "minimum_display_valid_fraction": float(
                    spatial_config.get("heatmap_min_valid_fraction", 0.05)
                ),
                "colormap": str(spatial_config.get("heatmap_colormap", "turbo")),
                "display_vmin_cm2": spatial_config.get("heatmap_vmin_cm2", 0.0),
                "display_vmax_cm2": displayed_vmax,
                "display_scale_clipped": bool(
                    displayed_vmax is not None
                    and not display_density.empty
                    and float(display_density.max()) > displayed_vmax
                ),
                "quantitative_grid_file": Path(grid_path).name,
            }
            tolerance = float(spatial_config.get("heatmap_area_tolerance_fraction", 0.005))
            if area_relative_error > tolerance:
                warning = (
                    "Density heatmap grid area differs from the primary valid analysis area "
                    f"by {area_relative_error:.3%}; inspect preview-mask geometry before interpretation"
                )
                LOGGER.warning(warning)
                warning_list = list(canonical.get("warnings", []))
                if warning not in warning_list:
                    warning_list.append(warning)
                canonical["warnings"] = warning_list
                if isinstance(summary, dict):
                    summary["warnings"] = warning_list
        spatial_summary = {
            "radial_bin_mode": str(spatial_config.get("radial_bin_mode", "equal_area")),
            "angle_reference": "image_positive_x",
            "density_heatmap": heatmap_summary,
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

    if bool(output_config.get("generate_paper_aligned_figure", True)):
        detection_path = outputs.get("paper_detection_field")
        heatmap_path = outputs.get("density_heatmap")
        if detection_path is not None and heatmap_path is not None:
            reference_status = (
                f"registered {reference_summary.get('method')} coordinates supplied"
                if registered_references is not None
                else "not provided; yellow markers intentionally absent"
            )
            outputs["paper_aligned_result_figure"] = save_paper_aligned_result_figure(
                detection_path,
                heatmap_path,
                folder / "paper_aligned_result_figure.png",
                independent_reference_status=reference_status,
            )
            paper_summary = {
                "status": "generated",
                "detection_panel": Path(detection_path).name,
                "density_panel": Path(heatmap_path).name,
                "combined_figure": "paper_aligned_result_figure.png",
                "red_box_meaning": "automatically accepted image-rule candidate",
                "yellow_circle_meaning": (
                    "registered independent observation"
                    if registered_references is not None
                    else "not drawn because independent reference was not provided"
                ),
                "density_normalization": "count / actual valid_analysis_mask area per cell",
                "physical_identity_claim": False,
            }
            canonical["paper_aligned_outputs"] = paper_summary
            if isinstance(summary, dict):
                summary["paper_aligned_outputs"] = paper_summary
            outputs.update(write_summary_files(canonical, folder))
        else:
            LOGGER.warning(
                "Paper-aligned combined figure skipped: a full-resolution detection field "
                "and an actual-area density heatmap are both required"
            )

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
