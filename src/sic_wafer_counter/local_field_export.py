"""Reference-style local-field exports for every valid wafer grid cell.

Each field receives a marked PNG, a typed Excel position table, and an
unchanged numeric TIFF crop.  The first file in the package is a global Excel
workbook with whole-wafer metrics, an area-normalized field-density matrix, and
an index linking every candidate to its field.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import cv2
import numpy as np
import pandas as pd
import tifffile

from .visualization import normalize_for_display
from .xlsx_export import SheetSpec, write_xlsx


SCIENTIFIC_NOTE = "未提供独立 DIC/KOH：不得称已确认 TSD/TED/BPD"

CANDIDATE_COLUMNS = (
    "field_id",
    "defect_id",
    "x_mm",
    "y_mm",
    "local_x_px",
    "local_y_px",
    "accepted",
    "rejection_reason",
    "area_px",
    "area_mm2",
    "equivalent_diameter_um",
    "major_axis_length_um",
    "minor_axis_length_um",
    "aspect_ratio",
    "eccentricity",
    "circularity",
    "solidity",
    "contrast",
    "distance_to_valid_boundary_mm",
)


@dataclass(frozen=True, slots=True)
class LocalFieldExportResult:
    """Traceable summary of one generated local-field package."""

    global_workbook: Path
    field_count: int
    candidate_count: int
    accepted_count: int
    mask_area_cm2: float | None
    primary_area_relative_error: float | None
    field_size_mm: float

    def to_dict(self, output_dir: Path) -> dict[str, Any]:
        return {
            "status": "generated",
            "global_workbook": str(self.global_workbook.relative_to(output_dir)),
            "field_count": self.field_count,
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "field_size_mm": self.field_size_mm,
            "mask_area_cm2": self.mask_area_cm2,
            "primary_area_relative_error": self.primary_area_relative_error,
            "field_outputs": ["01_marked.png", "02_positions.xlsx", "03_raw_original.tif"],
            "automatic_marker_semantics": "red=accepted image-rule candidate; green=rejected",
            "physical_identity_claim": False,
        }


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted"}


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bbox(record: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    raw = record.get("bounding_box")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        values = tuple(_finite(value) for value in raw)
        if all(value is not None for value in values):
            return tuple(float(value) for value in values)  # type: ignore[arg-type,return-value]
    x = _finite(record.get("centroid_x_px"))
    y = _finite(record.get("centroid_y_px"))
    diameter = _finite(record.get("equivalent_diameter_px"))
    if x is None or y is None:
        return None
    half = max(2.0, (diameter or 6.0) / 2.0)
    return x - half, y - half, x + half, y + half


def _field_key(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    *,
    radius_mm: float,
    field_size_mm: float,
    count: int,
) -> np.ndarray:
    x_index = np.floor((x_mm + radius_mm) / field_size_mm).astype(np.int64)
    y_index = np.floor((radius_mm - y_mm) / field_size_mm).astype(np.int64)
    np.clip(x_index, 0, count - 1, out=x_index)
    np.clip(y_index, 0, count - 1, out=y_index)
    return y_index * count + x_index


def _mask_area_by_field(
    valid_mask: np.ndarray | None,
    *,
    source_shape: tuple[int, int],
    center_px: tuple[float, float],
    mm_per_pixel: float,
    radius_mm: float,
    field_size_mm: float,
    count: int,
) -> tuple[np.ndarray, float | None]:
    field_counts = np.zeros(count * count, dtype=np.int64)
    if valid_mask is None:
        return field_counts, None
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("valid_analysis_mask must be two-dimensional")
    source_height, source_width = source_shape
    mask_height, mask_width = mask.shape
    x_source_per_mask = source_width / float(mask_width)
    y_source_per_mask = source_height / float(mask_height)
    block_rows = 512
    for y0 in range(0, mask_height, block_rows):
        block = mask[y0 : y0 + block_rows]
        local_y, local_x = np.nonzero(block)
        if not local_x.size:
            continue
        source_x = (local_x.astype(np.float64) + 0.5) * x_source_per_mask
        source_y = (local_y.astype(np.float64) + y0 + 0.5) * y_source_per_mask
        x_mm = (source_x - center_px[0]) * mm_per_pixel
        y_mm = -(source_y - center_px[1]) * mm_per_pixel
        keys = _field_key(
            x_mm,
            y_mm,
            radius_mm=radius_mm,
            field_size_mm=field_size_mm,
            count=count,
        )
        field_counts += np.bincount(keys, minlength=count * count)
    pixel_area_cm2 = (
        mm_per_pixel * x_source_per_mask / 10.0
    ) * (
        mm_per_pixel * y_source_per_mask / 10.0
    )
    return field_counts, float(pixel_area_cm2)


def _marked_field(
    image: np.ndarray,
    records: pd.DataFrame,
    *,
    source_x0: int,
    source_y0: int,
    bounds_mm: tuple[float, float, float, float],
    destination: Path,
) -> None:
    source = np.asarray(image)
    if (
        np.issubdtype(source.dtype, np.floating)
        and source.size
        and float(np.nanmin(source)) >= 0.0
        and float(np.nanmax(source)) <= 1.0
    ):
        gray = np.clip(source * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        gray = normalize_for_display(source, 0.5, 99.5)
    marked = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for record in records.to_dict(orient="records"):
        box = _bbox(record)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        local = (
            int(round(x0 - source_x0)),
            int(round(y0 - source_y0)),
            int(round(x1 - source_x0)),
            int(round(y1 - source_y0)),
        )
        lx0 = max(0, min(marked.shape[1] - 1, local[0]))
        ly0 = max(0, min(marked.shape[0] - 1, local[1]))
        lx1 = max(0, min(marked.shape[1] - 1, local[2]))
        ly1 = max(0, min(marked.shape[0] - 1, local[3]))
        if _truthy(record.get("accepted")):
            cv2.rectangle(marked, (lx0, ly0), (lx1, ly1), (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(marked, (lx0, ly0), (lx1, ly1), (0, 180, 0), 1, cv2.LINE_AA)
            cv2.line(marked, (lx0, ly0), (lx1, ly1), (0, 180, 0), 1, cv2.LINE_AA)
            cv2.line(marked, (lx0, ly1), (lx1, ly0), (0, 180, 0), 1, cv2.LINE_AA)

    minimum_display_side = 240
    if min(marked.shape[:2]) < minimum_display_side:
        scale = minimum_display_side / float(min(marked.shape[:2]))
        marked = cv2.resize(
            marked,
            (int(round(marked.shape[1] * scale)), int(round(marked.shape[0] * scale))),
            interpolation=cv2.INTER_NEAREST,
        )

    top, left, bottom, right = 46, 82, 60, 24
    canvas = np.full(
        (marked.shape[0] + top + bottom, marked.shape[1] + left + right, 3),
        255,
        dtype=np.uint8,
    )
    canvas[top : top + marked.shape[0], left : left + marked.shape[1]] = marked
    cv2.rectangle(
        canvas,
        (left, top),
        (left + marked.shape[1] - 1, top + marked.shape[0] - 1),
        (0, 210, 235),
        1,
    )
    x_left, x_right, y_bottom, y_top = bounds_mm
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "RED accepted | GREEN rejected | image rules",
        (left, 25),
        font,
        0.38,
        (32, 48, 54),
        1,
        cv2.LINE_AA,
    )
    for fraction in (0.0, 0.5, 1.0):
        px = left + int(round(fraction * (marked.shape[1] - 1)))
        value = x_left + fraction * (x_right - x_left)
        cv2.line(canvas, (px, top + marked.shape[0]), (px, top + marked.shape[0] + 5), (30, 30, 30), 1)
        cv2.putText(canvas, f"{value:.2f}", (px - 20, top + marked.shape[0] + 23), font, 0.36, (20, 20, 20), 1, cv2.LINE_AA)
        py = top + int(round(fraction * (marked.shape[0] - 1)))
        y_value = y_top - fraction * (y_top - y_bottom)
        cv2.line(canvas, (left - 5, py), (left, py), (30, 30, 30), 1)
        cv2.putText(canvas, f"{y_value:.2f}", (4, py + 4), font, 0.34, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "x (mm)", (left + max(0, marked.shape[1] // 2 - 22), canvas.shape[0] - 8), font, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "y(mm)", (4, 16), font, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Could not save marked local field {destination}")


def _candidate_rows(frame: pd.DataFrame) -> Iterable[list[Any]]:
    for record in frame.to_dict(orient="records"):
        yield [record.get(column) for column in CANDIDATE_COLUMNS]


def _field_workbook(
    path: Path,
    field: Mapping[str, Any],
    candidates: pd.DataFrame,
) -> Path:
    summary_rows = [
        ["局部视场分析汇总", field["field_id"], None],
        ["标记说明", "红框=自动接受；绿框/叉=自动拒绝", SCIENTIFIC_NOTE],
        [None, None, None],
        ["指标", "值", "单位/说明"],
        ["视场中心 X", field["center_x_mm"], "mm"],
        ["视场中心 Y", field["center_y_mm"], "mm"],
        ["左/右边界", f'{field["x_left_mm"]:.4f} / {field["x_right_mm"]:.4f}', "mm"],
        ["下/上边界", f'{field["y_bottom_mm"]:.4f} / {field["y_top_mm"]:.4f}', "mm"],
        ["最终有效掩膜面积", field.get("valid_area_cm2"), "cm^-2 计算所用面积为 cm^2"],
        ["自动接受数量", field["accepted_count"], "当前图像规则"],
        ["自动拒绝数量", field["rejected_count"], "保留拒绝原因"],
        ["点状目标密度", field.get("density_cm2"), "cm^-2；非物理类别确认"],
        ["原始数值图", "03_raw_original.tif", "未改写像素值"],
        ["标记图", "01_marked.png", "红=接受；绿=拒绝"],
    ]
    candidate_rows = [list(CANDIDATE_COLUMNS), *_candidate_rows(candidates)]
    return write_xlsx(
        path,
        [
            SheetSpec(
                "视场概览",
                rows=lambda rows=summary_rows: iter(rows),
                column_widths=(26, 32, 48),
                title_rows=frozenset({1}),
                header_rows=frozenset({4}),
                freeze_rows=4,
            ),
            SheetSpec(
                "候选位置",
                rows=lambda rows=candidate_rows: iter(rows),
                column_widths=(18, 12, 14, 14, 13, 13, 12, 26, 12, 13, 18, 18, 18, 13, 13, 13, 13, 13, 22),
                header_rows=frozenset({1}),
                freeze_rows=1,
                auto_filter_ref=f"A1:S{max(1, len(candidate_rows))}",
            ),
        ],
    )


def _global_workbook(
    path: Path,
    *,
    summary: Mapping[str, Any],
    fields: list[dict[str, Any]],
    candidates: pd.DataFrame,
    count: int,
    radius_mm: float,
    field_size_mm: float,
    mask_area_cm2: float | None,
    area_relative_error: float | None,
) -> Path:
    overview_rows = [
        ["SiC 晶圆点状目标分析：全局概览", None, None],
        ["标记说明", "红框=自动接受；绿框/叉=自动拒绝", SCIENTIFIC_NOTE],
        [None, None, None],
        ["指标", "值", "单位/说明"],
        ["输入文件", summary.get("input_file_name"), ""],
        ["晶圆实际直径", summary.get("wafer_diameter_mm"), "mm"],
        ["最终有效分析面积 S", summary.get("valid_analysis_area_cm2"), "cm^2"],
        ["自动接受数量 n", summary.get("accepted_count"), "当前图像规则"],
        ["自动拒绝数量", summary.get("rejected_count"), "保留拒绝原因"],
        ["整片点状目标密度 rho", summary.get("point_density_cm2"), "cm^-2；rho=n/S"],
        ["泊松计数不确定度", summary.get("counting_uncertainty_cm2"), "cm^-2；不含系统误差"],
        ["局部视场边长", field_size_mm, "mm"],
        ["导出的有效局部视场", len(fields), "个；含零计数视场"],
        ["局部视场有效面积合计", mask_area_cm2, "cm^2；来自同一最终有效掩膜"],
        ["局部面积合计相对主 S 差异", area_relative_error, "应仅来自掩膜分辨率映射"],
        ["真实专家标注验证", summary.get("real_annotation_validation_status"), "未验证时不得称 TSD/TED/BPD"],
        ["软件版本", summary.get("software_version"), ""],
    ]
    field_columns = [
        "field_id", "center_x_mm", "center_y_mm", "x_left_mm", "x_right_mm",
        "y_bottom_mm", "y_top_mm", "valid_area_cm2", "accepted_count",
        "rejected_count", "candidate_count", "density_cm2", "output_folder",
    ]
    field_rows = [field_columns] + [[field.get(column) for column in field_columns] for field in fields]
    candidate_rows = [list(CANDIDATE_COLUMNS), *_candidate_rows(candidates)]

    by_key = {int(field["field_key"]): field for field in fields}
    matrix_rows: list[list[Any]] = []
    x_centers = [(-radius_mm + index * field_size_mm) + field_size_mm / 2.0 for index in range(count)]
    matrix_rows.append(["Y\\X (mm)", *x_centers, "行汇总密度"])
    for y_index in range(count):
        y_top = radius_mm - y_index * field_size_mm
        y_center = y_top - field_size_mm / 2.0
        row: list[Any] = [y_center]
        row_count = 0
        row_area = 0.0
        for x_index in range(count):
            field = by_key.get(y_index * count + x_index)
            row.append(field.get("density_cm2") if field else None)
            if field:
                row_count += int(field["accepted_count"])
                row_area += float(field.get("valid_area_cm2") or 0.0)
        row.append(row_count / row_area if row_area > 0 else None)
        matrix_rows.append(row)
    total_row: list[Any] = ["列汇总密度"]
    for x_index in range(count):
        column_fields = [by_key.get(y_index * count + x_index) for y_index in range(count)]
        column_count = sum(int(field["accepted_count"]) for field in column_fields if field)
        column_area = sum(float(field.get("valid_area_cm2") or 0.0) for field in column_fields if field)
        total_row.append(column_count / column_area if column_area > 0 else None)
    total_count = sum(int(field["accepted_count"]) for field in fields)
    total_area = sum(float(field.get("valid_area_cm2") or 0.0) for field in fields)
    total_row.append(total_count / total_area if total_area > 0 else None)
    matrix_rows.append(total_row)

    return write_xlsx(
        path,
        [
            SheetSpec(
                "全局概览",
                rows=lambda rows=overview_rows: iter(rows),
                column_widths=(30, 34, 48),
                title_rows=frozenset({1}),
                header_rows=frozenset({4}),
                freeze_rows=4,
            ),
            SheetSpec(
                "视场密度矩阵",
                rows=lambda rows=matrix_rows: iter(rows),
                column_widths=(15, *([12] * count), 16),
                header_rows=frozenset({1, len(matrix_rows)}),
                freeze_rows=1,
                auto_filter_ref=f"A1:{_excel_column(count + 1)}{len(matrix_rows)}",
            ),
            SheetSpec(
                "局部视场索引",
                rows=lambda rows=field_rows: iter(rows),
                column_widths=(18, 13, 13, 13, 13, 13, 13, 16, 14, 14, 14, 16, 38),
                header_rows=frozenset({1}),
                freeze_rows=1,
                auto_filter_ref=f"A1:M{max(1, len(field_rows))}",
            ),
            SheetSpec(
                "全部候选位置",
                rows=lambda rows=candidate_rows: iter(rows),
                column_widths=(18, 12, 14, 14, 13, 13, 12, 26, 12, 13, 18, 18, 18, 13, 13, 13, 13, 13, 22),
                header_rows=frozenset({1}),
                freeze_rows=1,
                auto_filter_ref=f"A1:S{max(1, len(candidate_rows))}",
            ),
        ],
    )


def _excel_column(index: int) -> str:
    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def export_local_fields(
    defects: pd.DataFrame,
    output_dir: str | Path,
    summary: Mapping[str, Any],
    *,
    source_shape: tuple[int, int],
    center_px: tuple[float, float],
    mm_per_pixel: float,
    valid_analysis_mask: np.ndarray | None,
    raw_reader: Callable[[int, int, int, int], np.ndarray],
    display_reader: Callable[[int, int, int, int], np.ndarray],
    field_size_mm: float = 4.0,
    max_band_bytes: int = 134_217_728,
) -> LocalFieldExportResult:
    """Export every valid local field, including zero-count fields."""

    diameter_mm = float(summary.get("wafer_diameter_mm", 100.0))
    radius_mm = diameter_mm / 2.0
    if not math.isfinite(field_size_mm) or field_size_mm < 1.0 or field_size_mm > diameter_mm:
        raise ValueError("local field size must be finite and between 1 mm and wafer diameter")
    count = int(math.ceil(diameter_mm / field_size_mm))
    source_height, source_width = map(int, source_shape[:2])
    frame = defects.copy()
    x_values = pd.to_numeric(frame.get("x_mm"), errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(frame.get("y_mm"), errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("All local-field candidates require finite x_mm and y_mm")
    keys = _field_key(
        x_values,
        y_values,
        radius_mm=radius_mm,
        field_size_mm=field_size_mm,
        count=count,
    )
    frame["_field_key"] = keys
    frame["field_id"] = ""
    frame["local_x_px"] = np.nan
    frame["local_y_px"] = np.nan
    area_counts, mask_pixel_area_cm2 = _mask_area_by_field(
        valid_analysis_mask,
        source_shape=(source_height, source_width),
        center_px=center_px,
        mm_per_pixel=mm_per_pixel,
        radius_mm=radius_mm,
        field_size_mm=field_size_mm,
        count=count,
    )
    candidate_keys = set(int(value) for value in keys)
    valid_keys = set(int(value) for value in np.flatnonzero(area_counts > 0))
    exported_keys = sorted(valid_keys | candidate_keys)
    root = Path(output_dir) / "local_fields"
    root.mkdir(parents=True, exist_ok=True)
    fields: list[dict[str, Any]] = []
    keys_by_y: dict[int, list[int]] = {}
    for key in exported_keys:
        keys_by_y.setdefault(key // count, []).append(key)

    field_sequence = 0
    for y_index in sorted(keys_by_y):
        y_top = radius_mm - y_index * field_size_mm
        y_bottom = max(-radius_mm, y_top - field_size_mm)
        source_y0 = max(0, int(math.floor(center_px[1] - y_top / mm_per_pixel)))
        source_y1 = min(source_height, int(math.ceil(center_px[1] - y_bottom / mm_per_pixel)))
        if source_y1 <= source_y0:
            continue
        estimated_band_bytes = source_width * (source_y1 - source_y0) * 8
        raw_band = display_band = None
        if estimated_band_bytes <= max_band_bytes:
            raw_band = np.asarray(raw_reader(0, source_y0, source_width, source_y1 - source_y0))
            display_band = np.asarray(display_reader(0, source_y0, source_width, source_y1 - source_y0))
        for key in sorted(keys_by_y[y_index]):
            x_index = key % count
            x_left = -radius_mm + x_index * field_size_mm
            x_right = min(radius_mm, x_left + field_size_mm)
            source_x0 = max(0, int(math.floor(center_px[0] + x_left / mm_per_pixel)))
            source_x1 = min(source_width, int(math.ceil(center_px[0] + x_right / mm_per_pixel)))
            if source_x1 <= source_x0:
                continue
            if raw_band is None or display_band is None:
                raw_crop = np.asarray(raw_reader(source_x0, source_y0, source_x1 - source_x0, source_y1 - source_y0))
                display_crop = np.asarray(display_reader(source_x0, source_y0, source_x1 - source_x0, source_y1 - source_y0))
            else:
                raw_crop = raw_band[:, source_x0:source_x1]
                display_crop = display_band[:, source_x0:source_x1]
            field_sequence += 1
            center_x = (x_left + x_right) / 2.0
            center_y = (y_bottom + y_top) / 2.0
            field_id = f"field_{field_sequence:04d}_X_{center_x:.3f}_Y_{center_y:.3f}"
            field_dir = root / field_id
            field_dir.mkdir(parents=True, exist_ok=True)
            candidates = frame.loc[frame["_field_key"] == key].copy()
            candidates["field_id"] = field_id
            candidates["local_x_px"] = pd.to_numeric(candidates["centroid_x_px"], errors="coerce") - source_x0
            candidates["local_y_px"] = pd.to_numeric(candidates["centroid_y_px"], errors="coerce") - source_y0
            frame.loc[candidates.index, "field_id"] = field_id
            frame.loc[candidates.index, "local_x_px"] = candidates["local_x_px"]
            frame.loc[candidates.index, "local_y_px"] = candidates["local_y_px"]
            accepted_count = int(candidates.get("accepted", pd.Series(dtype=object)).map(_truthy).sum())
            rejected_count = int(len(candidates) - accepted_count)
            valid_area = (
                float(area_counts[key]) * mask_pixel_area_cm2
                if mask_pixel_area_cm2 is not None
                else None
            )
            density = accepted_count / valid_area if valid_area and valid_area > 0 else None
            record = {
                "field_key": key,
                "field_id": field_id,
                "center_x_mm": center_x,
                "center_y_mm": center_y,
                "x_left_mm": x_left,
                "x_right_mm": x_right,
                "y_bottom_mm": y_bottom,
                "y_top_mm": y_top,
                "source_x0_px": source_x0,
                "source_y0_px": source_y0,
                "source_x1_px": source_x1,
                "source_y1_px": source_y1,
                "valid_area_cm2": valid_area,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "candidate_count": int(len(candidates)),
                "density_cm2": density,
                "output_folder": str(field_dir.relative_to(Path(output_dir))),
            }
            fields.append(record)
            tifffile.imwrite(field_dir / "03_raw_original.tif", raw_crop)
            _marked_field(
                display_crop,
                candidates,
                source_x0=source_x0,
                source_y0=source_y0,
                bounds_mm=(x_left, x_right, y_bottom, y_top),
                destination=field_dir / "01_marked.png",
            )
            _field_workbook(field_dir / "02_positions.xlsx", record, candidates)

    mask_area = (
        float(area_counts.sum()) * mask_pixel_area_cm2
        if mask_pixel_area_cm2 is not None
        else None
    )
    primary_area = _finite(summary.get("valid_analysis_area_cm2"))
    area_error = (
        abs(mask_area - primary_area) / primary_area
        if mask_area is not None and primary_area is not None and primary_area > 0
        else None
    )
    global_workbook = _global_workbook(
        root / "00_global_overview.xlsx",
        summary=summary,
        fields=fields,
        candidates=frame,
        count=count,
        radius_mm=radius_mm,
        field_size_mm=field_size_mm,
        mask_area_cm2=mask_area,
        area_relative_error=area_error,
    )
    return LocalFieldExportResult(
        global_workbook=global_workbook,
        field_count=len(fields),
        candidate_count=len(frame),
        accepted_count=int(frame.get("accepted", pd.Series(dtype=object)).map(_truthy).sum()),
        mask_area_cm2=mask_area,
        primary_area_relative_error=area_error,
        field_size_mm=field_size_mm,
    )


__all__ = ["LocalFieldExportResult", "export_local_fields"]
