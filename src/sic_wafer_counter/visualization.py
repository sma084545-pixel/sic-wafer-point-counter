"""Visual quality-control outputs for wafer analyses.

The functions in this module deliberately accept either a pandas ``DataFrame``
or an iterable of dictionaries.  Keeping the drawing layer loosely coupled to
the detector makes it useful for automated results and manually reviewed CSVs.
Large arrays are reduced once, immediately before drawing/saving, so creating a
set of review images does not require several full-resolution RGB copies.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd
from .density import poisson_count_interval

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _as_frame(records: pd.DataFrame | Iterable[Mapping[str, Any]] | None) -> pd.DataFrame:
    """Return a defensive DataFrame copy for drawing and plotting."""

    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    materialized = list(records)
    normalized = [
        item.to_record() if hasattr(item, "to_record") and callable(item.to_record) else item
        for item in materialized
    ]
    return pd.DataFrame(normalized)


def _truthy(value: Any) -> bool:
    """Interpret common CSV boolean representations without ``bool('False')``."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted", "accept"}


def _finite_percentiles(
    image: np.ndarray,
    low_percentile: float,
    high_percentile: float,
    mask: np.ndarray | None = None,
    max_samples: int = 2_000_000,
) -> tuple[float, float]:
    """Estimate robust display limits with bounded temporary memory."""

    source = np.asarray(image)
    if source.ndim == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    if mask is not None and mask.shape == source.shape:
        values = source[np.asarray(mask, dtype=bool)]
    else:
        # A strided sample avoids materialising a flattened copy of a BigTIFF.
        stride = max(1, int(np.sqrt(source.size / max_samples)))
        values = source[::stride, ::stride].reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    if values.size > max_samples:
        step = max(1, values.size // max_samples)
        values = values[::step]
    lo, hi = np.percentile(values, [low_percentile, high_percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def normalize_for_display(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Robustly map an 8/16-bit or floating image to an 8-bit gray preview."""

    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        else:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D gray image, received shape {array.shape}")
    lo, hi = _finite_percentiles(array, low_percentile, high_percentile, mask)
    scaled = np.clip((array.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0.0, 255.0)
    scaled[~np.isfinite(scaled)] = 0.0
    return scaled.astype(np.uint8)


def _resize_for_output(
    image: np.ndarray,
    max_size: int | None,
    interpolation: int = cv2.INTER_AREA,
) -> tuple[np.ndarray, float]:
    """Resize while retaining aspect ratio; return image and output/input scale."""

    if not max_size or max(image.shape[:2]) <= max_size:
        return image, 1.0
    scale = float(max_size) / float(max(image.shape[:2]))
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=interpolation), scale


def save_grayscale_image(
    image: np.ndarray,
    output_path: str | Path,
    *,
    max_size: int | None = 6000,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> Path:
    """Save a normalized gray preview without modifying the source array."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reduce native dtype first: a 60k x 60k uint16 source must not create a
    # same-size float display-normalization copy merely to save a 6k preview.
    reduced, _ = _resize_for_output(np.asarray(image), max_size)
    preview = normalize_for_display(reduced, low_percentile, high_percentile)
    if not cv2.imwrite(str(destination), preview):
        raise OSError(f"Could not write image: {destination}")
    return destination


def save_binary_mask(
    mask: np.ndarray,
    output_path: str | Path,
    *,
    max_size: int | None = 6000,
) -> Path:
    """Save a boolean/label mask as an unambiguous black-and-white PNG."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    display = (np.asarray(mask) > 0).astype(np.uint8) * 255
    display, _ = _resize_for_output(display, max_size, cv2.INTER_NEAREST)
    if not cv2.imwrite(str(destination), display):
        raise OSError(f"Could not write mask: {destination}")
    return destination


def _column(frame: pd.DataFrame, alternatives: Sequence[str], default: Any = np.nan) -> pd.Series:
    """Select the first available column from a list of schema alternatives."""

    for name in alternatives:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def create_overlay(
    image: np.ndarray,
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    accepted_only: bool = False,
    draw_labels: bool = True,
    draw_rejected: bool = True,
    max_size: int | None = 6000,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    wafer_center_px: tuple[float, float] | None = None,
    wafer_radius_px: float | None = None,
    source_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build a BGR review overlay with accepted circles and rejected crosses.

    Coordinates and sizes are scaled together when ``max_size`` limits output
    resolution.  If ``image`` is a low-resolution preview but defect coordinates
    refer to the original image, pass the original ``(height, width)`` as
    ``source_shape``.  Labels remain readable after reduction, and their
    placement is clipped by OpenCV rather than generating another full canvas.
    """

    input_height, input_width = np.asarray(image).shape[:2]
    if source_shape is None:
        coordinate_scale_x = coordinate_scale_y = 1.0
    else:
        source_height, source_width = source_shape[:2]
        if source_height <= 0 or source_width <= 0:
            raise ValueError(f"Invalid source_shape: {source_shape}")
        coordinate_scale_x = input_width / float(source_width)
        coordinate_scale_y = input_height / float(source_height)
    size_coordinate_scale = math.sqrt(coordinate_scale_x * coordinate_scale_y)
    reduced, scale = _resize_for_output(np.asarray(image), max_size)
    gray = normalize_for_display(reduced, low_percentile, high_percentile)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    frame = _as_frame(defects)
    if frame.empty:
        return overlay

    accepted_values = _column(frame, ("accepted", "reviewed_accepted"), True).map(_truthy)
    if accepted_only:
        frame = frame.loc[accepted_values].copy()
        accepted_values = pd.Series(True, index=frame.index)

    x_values = pd.to_numeric(
        _column(frame, ("centroid_x_px", "center_x_px", "x_px")), errors="coerce"
    )
    y_values = pd.to_numeric(
        _column(frame, ("centroid_y_px", "center_y_px", "y_px")), errors="coerce"
    )
    diameter_values = pd.to_numeric(
        _column(frame, ("equivalent_diameter_px", "diameter_px"), 8.0), errors="coerce"
    ).fillna(8.0)
    id_values = _column(frame, ("defect_id", "candidate_id", "label"), "?")

    base_thickness = max(1, int(round(2 * max(0.6, min(scale, 1.5)))))
    font_scale = max(0.35, min(0.85, 0.52 * max(0.8, np.sqrt(scale))))
    for position, index in enumerate(frame.index):
        x = x_values.loc[index]
        y = y_values.loc[index]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        point = (
            int(round(float(x) * coordinate_scale_x * scale)),
            int(round(float(y) * coordinate_scale_y * scale)),
        )
        radius = max(
            4,
            int(
                round(
                    max(6.0, float(diameter_values.loc[index]) * 0.75)
                    * size_coordinate_scale
                    * scale
                )
            ),
        )
        is_accepted = bool(accepted_values.loc[index])
        if is_accepted:
            # OpenCV uses BGR: vivid green remains legible on gray backgrounds.
            cv2.circle(overlay, point, radius, (40, 230, 40), base_thickness, cv2.LINE_AA)
        elif draw_rejected and not accepted_only:
            arm = max(4, radius)
            cv2.line(
                overlay,
                (point[0] - arm, point[1] - arm),
                (point[0] + arm, point[1] + arm),
                (40, 40, 240),
                base_thickness,
                cv2.LINE_AA,
            )
            cv2.line(
                overlay,
                (point[0] - arm, point[1] + arm),
                (point[0] + arm, point[1] - arm),
                (40, 40, 240),
                base_thickness,
                cv2.LINE_AA,
            )
        else:
            continue
        if draw_labels:
            label = str(id_values.loc[index] if pd.notna(id_values.loc[index]) else position + 1)
            color = (40, 255, 255) if is_accepted else (60, 150, 255)
            cv2.putText(
                overlay,
                label,
                (point[0] + radius + 2, point[1] - radius - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                max(1, base_thickness),
                cv2.LINE_AA,
            )

    if wafer_center_px is not None and wafer_radius_px is not None:
        center = (
            int(round(wafer_center_px[0] * coordinate_scale_x * scale)),
            int(round(wafer_center_px[1] * coordinate_scale_y * scale)),
        )
        cv2.circle(
            overlay,
            center,
            max(1, int(round(wafer_radius_px * size_coordinate_scale * scale))),
            (255, 180, 30),
            max(1, base_thickness),
            cv2.LINE_AA,
        )
    return overlay


def save_overlay(
    image: np.ndarray,
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Create and save a numbered candidate overlay."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlay = create_overlay(image, defects, **kwargs)
    if not cv2.imwrite(str(destination), overlay):
        raise OSError(f"Could not write overlay: {destination}")
    return destination


def save_overlays(
    image: np.ndarray,
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Path]:
    """Write accepted-only and all-candidate overlays with stable filenames."""

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    accepted_kwargs = dict(kwargs)
    accepted_kwargs["accepted_only"] = True
    all_kwargs = dict(kwargs)
    all_kwargs["accepted_only"] = False
    return {
        "overlay_accepted": save_overlay(
            image, defects, folder / "overlay_accepted.png", **accepted_kwargs
        ),
        "overlay_all_candidates": save_overlay(
            image, defects, folder / "overlay_all_candidates.png", **all_kwargs
        ),
    }


def _candidate_bbox(record: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    """Return an auditable ``(x0, y0, x1, y1)`` candidate box when available."""

    raw = record.get("bounding_box")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            values = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            values = ()
        if len(values) == 4 and all(np.isfinite(values)):
            return values  # type: ignore[return-value]

    try:
        x = float(record.get("centroid_x_px"))
        y = float(record.get("centroid_y_px"))
        diameter = float(record.get("equivalent_diameter_px", 8.0))
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite((x, y, diameter))):
        return None
    half = max(1.0, diameter / 2.0)
    return x - half, y - half, x + half, y + half


def create_xrt_detection_overlay(
    image: np.ndarray,
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    max_size: int | None = 6000,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    source_shape: tuple[int, int] | None = None,
    mm_per_pixel: float | None = None,
    scale_bar_mm: float | None = 10.0,
    draw_labels: bool = False,
    independent_reference_points: pd.DataFrame | Iterable[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    """Create a paper-aligned XRT overlay without fabricating validation data.

    Automatically accepted XRT point-like targets are drawn as red rectangles,
    matching the visual semantics of Fig. 5 in Reimann & Kranert (2021). Yellow
    circles are reserved for *independently supplied* registered reference
    points (for example DIC or KOH observations); the analysis pipeline never
    invents those points. This function changes presentation only and never
    changes candidate acceptance or density.
    """

    input_height, input_width = np.asarray(image).shape[:2]
    if source_shape is None:
        source_height, source_width = input_height, input_width
    else:
        source_height, source_width = map(int, source_shape[:2])
        if source_height <= 0 or source_width <= 0:
            raise ValueError(f"Invalid source_shape: {source_shape}")
    coordinate_scale_x = input_width / float(source_width)
    coordinate_scale_y = input_height / float(source_height)
    reduced, resize_scale = _resize_for_output(np.asarray(image), max_size)
    gray = normalize_for_display(reduced, low_percentile, high_percentile)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    output_scale_x = coordinate_scale_x * resize_scale
    output_scale_y = coordinate_scale_y * resize_scale

    frame = _as_frame(defects)
    if not frame.empty:
        accepted = _column(frame, ("accepted", "reviewed_accepted"), True).map(_truthy)
        frame = frame.loc[accepted].copy()
        thickness = max(1, int(round(2 * max(0.6, min(resize_scale, 1.5)))))
        for position, (_, row) in enumerate(frame.iterrows(), start=1):
            box = _candidate_bbox(row)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            left = int(round(x0 * output_scale_x))
            top = int(round(y0 * output_scale_y))
            right = int(round(x1 * output_scale_x))
            bottom = int(round(y1 * output_scale_y))
            # Preserve visibility after a whole-wafer overlay is reduced. The
            # candidate CSV retains the true segmentation bounding box.
            center_x, center_y = (left + right) // 2, (top + bottom) // 2
            half_width = max(3, int(math.ceil(abs(right - left) / 2.0)))
            half_height = max(3, int(math.ceil(abs(bottom - top) / 2.0)))
            left, right = center_x - half_width, center_x + half_width
            top, bottom = center_y - half_height, center_y + half_height
            cv2.rectangle(
                overlay, (left, top), (right, bottom), (0, 0, 255), thickness, cv2.LINE_AA
            )
            if draw_labels:
                cv2.putText(
                    overlay,
                    str(row.get("defect_id", position)),
                    (right + 2, max(10, top - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

    # Yellow markers have the independent-observation meaning used in the
    # paper, so they appear only when a caller explicitly supplies coordinates.
    references = _as_frame(independent_reference_points)
    if not references.empty:
        x_values = pd.to_numeric(
            _column(references, ("x_px", "centroid_x_px", "center_x_px")), errors="coerce"
        )
        y_values = pd.to_numeric(
            _column(references, ("y_px", "centroid_y_px", "center_y_px")), errors="coerce"
        )
        for index in references.index:
            x_value, y_value = x_values.loc[index], y_values.loc[index]
            if np.isfinite(x_value) and np.isfinite(y_value):
                point = (
                    int(round(float(x_value) * output_scale_x)),
                    int(round(float(y_value) * output_scale_y)),
                )
                cv2.circle(overlay, point, 6, (0, 255, 255), 2, cv2.LINE_AA)

    if scale_bar_mm is not None:
        if mm_per_pixel is None or not np.isfinite(mm_per_pixel) or mm_per_pixel <= 0:
            raise ValueError("A positive mm_per_pixel is required for a physical scale bar")
        if not np.isfinite(scale_bar_mm) or scale_bar_mm <= 0:
            raise ValueError("scale_bar_mm must be finite and positive")
        bar_px = int(round((float(scale_bar_mm) / float(mm_per_pixel)) * output_scale_x))
        bar_px = max(1, min(bar_px, max(1, overlay.shape[1] // 3)))
        margin = max(14, int(round(min(overlay.shape[:2]) * 0.018)))
        bar_x1 = overlay.shape[1] - margin
        bar_x0 = max(margin, bar_x1 - bar_px)
        bar_y = overlay.shape[0] - margin
        line_width = max(3, int(round(min(overlay.shape[:2]) / 450.0)))
        cv2.line(
            overlay, (bar_x0, bar_y), (bar_x1, bar_y), (0, 0, 0), line_width + 2, cv2.LINE_AA
        )
        cv2.line(
            overlay,
            (bar_x0, bar_y),
            (bar_x1, bar_y),
            (255, 255, 255),
            max(1, line_width // 2),
            cv2.LINE_AA,
        )
        label = f"{float(scale_bar_mm):g} mm"
        font_scale = max(0.42, min(0.9, overlay.shape[1] / 3500.0))
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        origin = (max(margin, bar_x1 - text_width), max(text_height + 3, bar_y - 8))
        cv2.putText(
            overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            overlay,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def save_xrt_detection_overlay(
    image: np.ndarray,
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Save the red-box automatic-XRT overlay with optional real references."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlay = create_xrt_detection_overlay(image, defects, **kwargs)
    if not cv2.imwrite(str(destination), overlay):
        raise OSError(f"Could not write XRT detection overlay: {destination}")
    return destination


def _accepted_frame(defects: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    frame = _as_frame(defects)
    if frame.empty:
        return frame
    accepted = _column(frame, ("accepted", "reviewed_accepted"), True).map(_truthy)
    return frame.loc[accepted].copy()


def _save_empty_plot(path: Path, title: str, message: str = "No accepted candidates") -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_size_histogram(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]], output_path: str | Path
) -> Path:
    """Save an accepted-target equivalent-diameter histogram."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _accepted_frame(defects)
    unit = "mm" if "equivalent_diameter_mm" in frame else "px"
    key = "equivalent_diameter_mm" if unit == "mm" else "equivalent_diameter_px"
    values = pd.to_numeric(frame.get(key, pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return _save_empty_plot(path, "Point-like target size distribution")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(values, bins="auto", color="#2678b2", edgecolor="white")
    ax.set(xlabel=f"Equivalent diameter ({unit})", ylabel="Count", title="Accepted target size distribution")
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_radial_distribution(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]], output_path: str | Path
) -> Path:
    """Save a descriptive radial count histogram, not an areal-density claim."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _accepted_frame(defects)
    values = pd.to_numeric(frame.get("radial_distance_mm", pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return _save_empty_plot(path, "Radial count distribution (not density)")
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(values, bins="auto", color="#2f9e72", edgecolor="white")
    ax.set(xlabel="Radial distance (mm)", ylabel="Count", title="Accepted target radial count distribution (not density)")
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _pixel_scales(mm_per_pixel: float | tuple[float, float]) -> tuple[float, float, float]:
    """Return x/y physical scales and one exact pixel area in cm²."""

    if isinstance(mm_per_pixel, tuple):
        scale_x, scale_y = map(float, mm_per_pixel)
    else:
        scale_x = scale_y = float(mm_per_pixel)
    if not (math.isfinite(scale_x) and math.isfinite(scale_y) and scale_x > 0 and scale_y > 0):
        raise ValueError("mm_per_pixel must contain positive finite scale(s)")
    return scale_x, scale_y, (scale_x / 10.0) * (scale_y / 10.0)


def _polar_mask_histograms(
    valid_mask: np.ndarray,
    *,
    center_px: tuple[float, float],
    mm_per_pixel: float | tuple[float, float],
    radial_edges_mm: np.ndarray | None = None,
    angular_edges_deg: np.ndarray | None = None,
    row_chunk: int = 512,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Count final-valid pixels in polar bins with bounded temporary arrays."""

    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("valid_mask must be two-dimensional")
    scale_x, scale_y, _ = _pixel_scales(mm_per_pixel)
    radial_counts = (
        np.zeros(len(radial_edges_mm) - 1, dtype=np.int64)
        if radial_edges_mm is not None
        else None
    )
    angular_counts = (
        np.zeros(len(angular_edges_deg) - 1, dtype=np.int64)
        if angular_edges_deg is not None
        else None
    )
    bounded_rows = max(1, min(row_chunk, 2_000_000 // max(1, mask.shape[1])))
    for y0 in range(0, mask.shape[0], bounded_rows):
        rows, cols = np.nonzero(mask[y0 : y0 + bounded_rows])
        if not len(rows):
            continue
        y = rows.astype(np.float64) + y0
        x_mm = (cols.astype(np.float64) - float(center_px[0])) * scale_x
        y_mm = -(y - float(center_px[1])) * scale_y
        if radial_counts is not None:
            radii = np.hypot(x_mm, y_mm)
            radial_counts += np.histogram(radii, bins=radial_edges_mm)[0]
        if angular_counts is not None:
            angles = np.degrees(np.arctan2(y_mm, x_mm)) % 360.0
            angular_counts += np.histogram(angles, bins=angular_edges_deg)[0]
    return radial_counts, angular_counts


def _equal_area_radial_edges(
    valid_mask: np.ndarray,
    *,
    center_px: tuple[float, float],
    mm_per_pixel: float | tuple[float, float],
    wafer_radius_mm: float,
    bins: int,
) -> np.ndarray:
    """Approximate equal-valid-area radial edges from an actual-mask histogram."""

    if bins < 1:
        raise ValueError("bins must be at least one")
    fine_edges = np.linspace(0.0, float(wafer_radius_mm), 4097)
    counts, _ = _polar_mask_histograms(
        valid_mask,
        center_px=center_px,
        mm_per_pixel=mm_per_pixel,
        radial_edges_mm=fine_edges,
    )
    assert counts is not None
    if counts.sum() == 0:
        return np.linspace(0.0, float(wafer_radius_mm), bins + 1)
    cumulative = np.cumsum(counts)
    target = np.linspace(0, cumulative[-1], bins + 1)
    indices = np.searchsorted(cumulative, target, side="left")
    edges = fine_edges[np.clip(indices, 0, len(fine_edges) - 1)]
    edges[0], edges[-1] = 0.0, float(wafer_radius_mm)
    # Pixel grids can produce duplicate quantile edges.  Keep bin count stable
    # and make zero-area bins explicit rather than silently merging them.
    return np.maximum.accumulate(edges)


def _max_valid_radius_mm(
    valid_mask: np.ndarray,
    *,
    center_px: tuple[float, float],
    mm_per_pixel: float | tuple[float, float],
    row_chunk: int = 512,
) -> float:
    """Find the outermost final-valid pixel without a full coordinate grid."""

    mask = np.asarray(valid_mask, dtype=bool)
    scale_x, scale_y, _ = _pixel_scales(mm_per_pixel)
    maximum = 0.0
    bounded_rows = max(1, min(row_chunk, 2_000_000 // max(1, mask.shape[1])))
    for y0 in range(0, mask.shape[0], bounded_rows):
        rows, cols = np.nonzero(mask[y0 : y0 + bounded_rows])
        if not len(rows):
            continue
        x_mm = (cols.astype(np.float64) - float(center_px[0])) * scale_x
        y_mm = -((rows.astype(np.float64) + y0) - float(center_px[1])) * scale_y
        maximum = max(maximum, float(np.hypot(x_mm, y_mm).max(initial=0.0)))
    return maximum


def _density_frame(
    edges: np.ndarray,
    counts: np.ndarray,
    valid_pixels: np.ndarray,
    *,
    pixel_area_cm2: float,
    names: tuple[str, str],
) -> pd.DataFrame:
    """Create a standard area-normalized density table with Garwood intervals."""

    rows: list[dict[str, Any]] = []
    for index, count in enumerate(np.asarray(counts, dtype=int)):
        area = float(valid_pixels[index]) * pixel_area_cm2
        lower, upper = poisson_count_interval(int(count))
        density = float(count / area) if area > 0 else float("nan")
        rows.append(
            {
                names[0]: float(edges[index]),
                names[1]: float(edges[index + 1]),
                "valid_pixel_count": int(valid_pixels[index]),
                "valid_area_cm2": area,
                "count": int(count),
                "density_cm2": density,
                "poisson_lower_cm2": float(lower / area) if area > 0 else float("nan"),
                "poisson_upper_cm2": float(upper / area) if area > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def calculate_radial_density(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    valid_mask: np.ndarray,
    center_px: tuple[float, float],
    mm_per_pixel: float | tuple[float, float],
    wafer_radius_mm: float,
    bins: int = 6,
    mode: str = "equal_area",
) -> pd.DataFrame:
    """Calculate radial density from actual valid-mask annular areas."""

    normalized_mode = str(mode).lower().strip()
    if normalized_mode not in {"equal_area", "equal_width"}:
        raise ValueError("radial bin mode must be equal_area or equal_width")
    analysis_outer_radius_mm = max(
        float(wafer_radius_mm),
        _max_valid_radius_mm(
            valid_mask, center_px=center_px, mm_per_pixel=mm_per_pixel
        ),
    )
    edges = (
        _equal_area_radial_edges(
            valid_mask, center_px=center_px, mm_per_pixel=mm_per_pixel,
            wafer_radius_mm=analysis_outer_radius_mm, bins=bins,
        )
        if normalized_mode == "equal_area"
        else np.linspace(0.0, analysis_outer_radius_mm, bins + 1)
    )
    frame = _accepted_frame(defects)
    radii = pd.to_numeric(frame.get("radial_distance_mm", pd.Series(dtype=float)), errors="coerce")
    counts = np.histogram(radii[np.isfinite(radii)], bins=edges)[0]
    valid_pixels, _ = _polar_mask_histograms(
        valid_mask, center_px=center_px, mm_per_pixel=mm_per_pixel, radial_edges_mm=edges
    )
    assert valid_pixels is not None
    _, _, pixel_area_cm2 = _pixel_scales(mm_per_pixel)
    table = _density_frame(
        edges, counts, valid_pixels, pixel_area_cm2=pixel_area_cm2,
        names=("r_inner_mm", "r_outer_mm"),
    )
    table["bin_mode"] = normalized_mode
    table["analysis_outer_radius_mm"] = analysis_outer_radius_mm
    return table


def calculate_angular_density(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    valid_mask: np.ndarray,
    center_px: tuple[float, float],
    mm_per_pixel: float | tuple[float, float],
    sectors: int = 12,
) -> pd.DataFrame:
    """Calculate sector density with actual valid-mask area and image +x reference."""

    if int(sectors) < 1:
        raise ValueError("sectors must be at least one")
    edges = np.linspace(0.0, 360.0, int(sectors) + 1)
    frame = _accepted_frame(defects)
    angles = pd.to_numeric(frame.get("polar_angle_deg", pd.Series(dtype=float)), errors="coerce") % 360.0
    counts = np.histogram(angles[np.isfinite(angles)], bins=edges)[0]
    _, valid_pixels = _polar_mask_histograms(
        valid_mask, center_px=center_px, mm_per_pixel=mm_per_pixel, angular_edges_deg=edges
    )
    assert valid_pixels is not None
    _, _, pixel_area_cm2 = _pixel_scales(mm_per_pixel)
    table = _density_frame(
        edges, counts, valid_pixels, pixel_area_cm2=pixel_area_cm2,
        names=("angle_start_deg", "angle_end_deg"),
    )
    table["angle_reference"] = "image_positive_x"
    return table


def calculate_regional_density(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    valid_mask: np.ndarray,
    center_px: tuple[float, float],
    mm_per_pixel: float | tuple[float, float],
    wafer_radius_mm: float,
    regions: Mapping[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """Calculate configured normalized-radius region densities from valid pixels."""

    ranges = regions or {"center": (0.0, 0.33), "middle": (0.33, 0.67), "edge": (0.67, 1.0)}
    rows: list[dict[str, Any]] = []
    frame = _accepted_frame(defects)
    radial = pd.to_numeric(frame.get("radial_distance_mm", pd.Series(dtype=float)), errors="coerce")
    _, _, pixel_area_cm2 = _pixel_scales(mm_per_pixel)
    analysis_outer_radius_mm = max(
        float(wafer_radius_mm),
        _max_valid_radius_mm(
            valid_mask, center_px=center_px, mm_per_pixel=mm_per_pixel
        ),
    )
    for name, interval in ranges.items():
        if len(interval) != 2:
            raise ValueError(f"Region {name!r} must contain [inner, outer] normalized radius")
        inner, outer = map(float, interval)
        if not (0.0 <= inner <= outer <= 1.0):
            raise ValueError(f"Region {name!r} must lie within normalized radius [0, 1]")
        lower_mm, upper_mm = (
            inner * analysis_outer_radius_mm,
            outer * analysis_outer_radius_mm,
        )
        # A one-bin ``histogram`` includes both endpoints.  Nudge a non-final
        # upper edge inward so exact boundary pixels belong only to the next
        # region, matching the candidate counting rule below.
        histogram_upper = (
            np.nextafter(upper_mm, -np.inf) if not math.isclose(outer, 1.0) else upper_mm
        )
        edges = np.asarray([lower_mm, histogram_upper], dtype=float)
        valid_pixels, _ = _polar_mask_histograms(
            valid_mask, center_px=center_px, mm_per_pixel=mm_per_pixel, radial_edges_mm=edges
        )
        assert valid_pixels is not None
        include = (radial >= lower_mm) & (
            (radial <= upper_mm) if math.isclose(outer, 1.0) else (radial < upper_mm)
        )
        count = int(include.sum())
        area = float(valid_pixels[0]) * pixel_area_cm2
        lower, upper = poisson_count_interval(count)
        rows.append(
            {
                "region": str(name),
                "normalized_r_inner": inner,
                "normalized_r_outer": outer,
                "analysis_outer_radius_mm": analysis_outer_radius_mm,
                "valid_pixel_count": int(valid_pixels[0]),
                "valid_area_cm2": area,
                "count": count,
                "density_cm2": float(count / area) if area > 0 else float("nan"),
                "poisson_lower_cm2": float(lower / area) if area > 0 else float("nan"),
                "poisson_upper_cm2": float(upper / area) if area > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _save_density_plot(
    table: pd.DataFrame,
    path: Path,
    *,
    x_column: str,
    title: str,
    xlabel: str,
) -> Path:
    """Draw density points with exact Poisson count intervals."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if table.empty:
        return _save_empty_plot(path, title)
    x = pd.to_numeric(table[x_column], errors="coerce").to_numpy(dtype=float)
    density = pd.to_numeric(table["density_cm2"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(table["poisson_lower_cm2"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(table["poisson_upper_cm2"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(density) & np.isfinite(low) & np.isfinite(high)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    if finite.any():
        ax.errorbar(
            x[finite], density[finite],
            yerr=np.vstack((density[finite] - low[finite], high[finite] - density[finite])),
            marker="o", linestyle="-", capsize=3, color="#2f9e72",
        )
    else:
        ax.text(0.5, 0.5, "No valid area in requested bins", ha="center", transform=ax.transAxes)
    ax.set(xlabel=xlabel, ylabel=r"Density (cm$^{-2}$)", title=title)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_area_normalized_distributions(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    valid_mask: np.ndarray,
    center_px: tuple[float, float],
    mm_per_pixel: float | tuple[float, float],
    wafer_radius_mm: float,
    radial_bins: int = 6,
    radial_mode: str = "equal_area",
    angular_sectors: int = 12,
    regions: Mapping[str, Sequence[float]] | None = None,
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    """Write radial/angular/regional density CSVs and figures from actual areas."""

    folder = Path(output_dir)
    radial = calculate_radial_density(
        defects, valid_mask=valid_mask, center_px=center_px, mm_per_pixel=mm_per_pixel,
        wafer_radius_mm=wafer_radius_mm, bins=radial_bins, mode=radial_mode,
    )
    angular = calculate_angular_density(
        defects, valid_mask=valid_mask, center_px=center_px, mm_per_pixel=mm_per_pixel,
        sectors=angular_sectors,
    )
    regional = calculate_regional_density(
        defects, valid_mask=valid_mask, center_px=center_px, mm_per_pixel=mm_per_pixel,
        wafer_radius_mm=wafer_radius_mm, regions=regions,
    )
    paths = {
        "radial_density_csv": folder / "radial_density.csv",
        "angular_density_csv": folder / "angular_density.csv",
        "regional_density_csv": folder / "regional_density.csv",
    }
    radial.to_csv(paths["radial_density_csv"], index=False)
    angular.to_csv(paths["angular_density_csv"], index=False)
    regional.to_csv(paths["regional_density_csv"], index=False)
    paths["radial_density_plot"] = _save_density_plot(
        radial, folder / "radial_density.png", x_column="r_outer_mm",
        title="Accepted target radial density", xlabel="Outer radial-bin boundary (mm)",
    )
    paths["angular_density_plot"] = _save_density_plot(
        angular, folder / "angular_density.png", x_column="angle_start_deg",
        title="Accepted target angular density (image +x reference)", xlabel="Angle from image positive x (degrees)",
    )
    return paths, {"radial": radial, "angular": angular, "regional": regional}


def save_angular_distribution(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]], output_path: str | Path
) -> Path:
    """Save a descriptive angular count histogram, not an areal-density claim."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _accepted_frame(defects)
    values = pd.to_numeric(frame.get("polar_angle_deg", pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return _save_empty_plot(path, "Angular count distribution (not density)")
    # Feature extraction reports [0, 360); normalize to the plotted symmetric
    # interval so targets in quadrants III/IV are not silently dropped.
    values = (values + 180.0) % 360.0 - 180.0
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(values, bins=np.linspace(-180, 180, 25), color="#9a5bc5", edgecolor="white")
    ax.set(xlabel="Polar angle (degrees)", ylabel="Count", title="Accepted target angular count distribution (not density)")
    ax.set_xlim(-180, 180)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_wafer_scatter(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    wafer_radius_mm: float = 50.0,
) -> Path:
    """Save a wafer-centred two-dimensional target position plot."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _accepted_frame(defects)
    x = pd.to_numeric(frame.get("x_mm", pd.Series(dtype=float)), errors="coerce")
    y = pd.to_numeric(frame.get("y_mm", pd.Series(dtype=float)), errors="coerce")
    finite = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    boundary = plt.Circle((0, 0), wafer_radius_mm, fill=False, color="black", lw=1.2)
    ax.add_patch(boundary)
    if finite.any():
        ax.scatter(x[finite], y[finite], s=12, alpha=0.75, c="#d94841")
    else:
        ax.text(0.5, 0.5, "No accepted coordinates", ha="center", transform=ax.transAxes)
    limit = wafer_radius_mm * 1.05
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x (mm)", ylabel="y (mm)", title="Accepted target positions")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _valid_area_histogram(
    valid_mask: np.ndarray,
    center_px: tuple[float, float],
    mm_per_pixel: float | tuple[float, float],
    edges: np.ndarray,
    row_chunk: int = 512,
) -> np.ndarray:
    """Bin valid pixels without building full-size coordinate grids."""

    if isinstance(mm_per_pixel, tuple):
        mm_per_pixel_x, mm_per_pixel_y = map(float, mm_per_pixel)
    else:
        mm_per_pixel_x = mm_per_pixel_y = float(mm_per_pixel)
    area_pixels = np.zeros((len(edges) - 1, len(edges) - 1), dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    # Bound the temporary coordinate arrays created by np.nonzero on extremely
    # wide masks (roughly two million pixels per chunk).
    bounded_rows = max(1, min(row_chunk, 2_000_000 // max(1, mask.shape[1])))
    for y0 in range(0, mask.shape[0], bounded_rows):
        y1 = min(mask.shape[0], y0 + bounded_rows)
        rows, cols = np.nonzero(mask[y0:y1])
        if not rows.size:
            continue
        global_rows = rows + y0
        x_mm = (cols.astype(np.float64) - center_px[0]) * mm_per_pixel_x
        y_mm = -(global_rows.astype(np.float64) - center_px[1]) * mm_per_pixel_y
        counts, _, _ = np.histogram2d(x_mm, y_mm, bins=(edges, edges))
        area_pixels += counts
    return area_pixels


def calculate_density_grid(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    valid_mask: np.ndarray,
    center_px: tuple[float, float],
    mm_per_pixel: float | tuple[float, float],
    wafer_radius_mm: float = 50.0,
    bins: int = 40,
) -> pd.DataFrame:
    """Return an auditable whole-wafer grid normalized by actual valid area."""

    if bins < 2 or bins > 500:
        raise ValueError("heatmap bins must lie between 2 and 500")
    if not np.isfinite(wafer_radius_mm) or wafer_radius_mm <= 0:
        raise ValueError("wafer_radius_mm must be finite and positive")
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("valid_mask must be a non-empty two-dimensional mask")
    if isinstance(mm_per_pixel, tuple):
        scale_x, scale_y = map(float, mm_per_pixel)
    else:
        scale_x = scale_y = float(mm_per_pixel)
    if not all(np.isfinite((scale_x, scale_y))) or scale_x <= 0 or scale_y <= 0:
        raise ValueError("mm_per_pixel must contain positive finite values")

    frame = _accepted_frame(defects)
    x = pd.to_numeric(frame.get("x_mm", pd.Series(dtype=float)), errors="coerce")
    y = pd.to_numeric(frame.get("y_mm", pd.Series(dtype=float)), errors="coerce")
    finite = np.isfinite(x) & np.isfinite(y)
    active_columns = np.flatnonzero(np.any(mask, axis=0))
    active_rows = np.flatnonzero(np.any(mask, axis=1))
    x_extremes = (
        (float(active_columns[0]) - center_px[0]) * scale_x,
        (float(active_columns[-1]) - center_px[0]) * scale_x,
    )
    y_extremes = (
        -(float(active_rows[0]) - center_px[1]) * scale_y,
        -(float(active_rows[-1]) - center_px[1]) * scale_y,
    )
    # Small contour residuals can extend beyond the fitted 50 mm radius. The
    # grid expands just enough to include every valid pixel, preserving area.
    extent_mm = max(
        float(wafer_radius_mm),
        *(abs(value) for value in (*x_extremes, *y_extremes)),
    )
    extent_mm = float(np.nextafter(extent_mm, math.inf))
    edges = np.linspace(-extent_mm, extent_mm, int(bins) + 1)
    counts, _, _ = np.histogram2d(x[finite], y[finite], bins=(edges, edges))
    area_pixels = _valid_area_histogram(mask, center_px, (scale_x, scale_y), edges)
    pixel_area_cm2 = (scale_x / 10.0) * (scale_y / 10.0)
    area_cm2 = area_pixels * pixel_area_cm2
    density = np.divide(
        counts, area_cm2, out=np.full_like(counts, np.nan), where=area_cm2 > 0
    )
    full_cell_area_cm2 = ((edges[1] - edges[0]) / 10.0) ** 2

    rows: list[dict[str, Any]] = []
    for x_index in range(int(bins)):
        for y_index in range(int(bins)):
            count = int(counts[x_index, y_index])
            area = float(area_cm2[x_index, y_index])
            if area > 0:
                count_low, count_high = poisson_count_interval(count)
                lower = float(count_low / area)
                upper = float(count_high / area)
                value: float | None = float(density[x_index, y_index])
            else:
                lower = upper = value = None
            rows.append(
                {
                    "x_bin": x_index,
                    "y_bin": y_index,
                    "x_left_mm": float(edges[x_index]),
                    "x_right_mm": float(edges[x_index + 1]),
                    "y_bottom_mm": float(edges[y_index]),
                    "y_top_mm": float(edges[y_index + 1]),
                    "x_center_mm": float((edges[x_index] + edges[x_index + 1]) / 2.0),
                    "y_center_mm": float((edges[y_index] + edges[y_index + 1]) / 2.0),
                    "valid_pixel_count": int(area_pixels[x_index, y_index]),
                    "valid_area_cm2": area,
                    "valid_area_fraction": float(np.clip(area / full_cell_area_cm2, 0.0, 1.0)),
                    "count": count,
                    "density_cm2": value,
                    "poisson_lower_cm2": lower,
                    "poisson_upper_cm2": upper,
                }
            )
    return pd.DataFrame.from_records(rows)


def _plot_valid_mask_boundary(
    ax: Any,
    valid_mask: np.ndarray,
    center_px: tuple[float, float],
    mm_per_pixel: float | tuple[float, float],
) -> None:
    """Draw the actual valid-mask outline using bounded contour memory."""

    mask = np.asarray(valid_mask, dtype=np.uint8)
    stride = max(1, int(math.ceil(max(mask.shape) / 2000.0)))
    sampled = mask[::stride, ::stride]
    contours, _ = cv2.findContours(sampled, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if isinstance(mm_per_pixel, tuple):
        scale_x, scale_y = map(float, mm_per_pixel)
    else:
        scale_x = scale_y = float(mm_per_pixel)
    minimum_area = max(4.0, sampled.size * 0.00001)
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue
        points = contour[:, 0, :].astype(np.float64) * stride
        x_mm = (points[:, 0] - center_px[0]) * scale_x
        y_mm = -(points[:, 1] - center_px[1]) * scale_y
        ax.plot(x_mm, y_mm, color="white", linewidth=0.8, alpha=0.9)


def save_density_heatmap(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    valid_mask: np.ndarray,
    center_px: tuple[float, float],
    mm_per_pixel: float | tuple[float, float],
    wafer_radius_mm: float = 50.0,
    bins: int = 40,
    colormap: str = "turbo",
    vmin_cm2: float | None = 0.0,
    vmax_cm2: float | None = None,
    clip_percentile: float = 99.5,
    min_valid_fraction: float = 0.05,
    grid_interval_mm: float = 10.0,
    grid_csv_path: str | Path | None = None,
) -> Path:
    """Save an area-normalized point-like target density map and grid table.

    Every quantitative cell uses ``count / actual valid-mask area``. Cells
    with no valid area are NA. A configurable minimum valid fraction affects
    display only; all cells remain in the CSV so edge behavior is auditable.
    """

    if not 0.0 <= float(min_valid_fraction) <= 1.0:
        raise ValueError("heatmap min_valid_fraction must lie in [0, 1]")
    if not 0.0 < float(clip_percentile) <= 100.0:
        raise ValueError("heatmap clip_percentile must lie in (0, 100]")
    if grid_interval_mm <= 0 or not np.isfinite(grid_interval_mm):
        raise ValueError("heatmap grid_interval_mm must be finite and positive")
    grid = calculate_density_grid(
        defects,
        valid_mask=valid_mask,
        center_px=center_px,
        mm_per_pixel=mm_per_pixel,
        wafer_radius_mm=wafer_radius_mm,
        bins=bins,
    )
    csv_path = Path(grid_csv_path) if grid_csv_path is not None else Path(output_path).with_name(
        "density_heatmap_grid.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(csv_path, index=False)

    density = grid["density_cm2"].to_numpy(dtype=float).reshape(bins, bins)
    area_fraction = grid["valid_area_fraction"].to_numpy(dtype=float).reshape(bins, bins)
    display = density.copy()
    display[area_fraction < float(min_valid_fraction)] = np.nan
    finite = display[np.isfinite(display)]
    if finite.size:
        colour_min = 0.0 if vmin_cm2 is None else float(vmin_cm2)
        colour_max = (
            float(np.percentile(finite, clip_percentile))
            if vmax_cm2 is None
            else float(vmax_cm2)
        )
        raw_max = float(np.max(finite))
        if not np.isfinite(colour_max) or colour_max <= colour_min:
            colour_max = max(raw_max, colour_min + 1.0)
    else:
        colour_min = 0.0 if vmin_cm2 is None else float(vmin_cm2)
        colour_max = 1.0 if vmax_cm2 is None else float(vmax_cm2)
        raw_max = colour_max
    if colour_max <= colour_min:
        raise ValueError("heatmap vmax_cm2 must be greater than vmin_cm2")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = np.concatenate(
        (
            grid["x_left_mm"].drop_duplicates().to_numpy(dtype=float),
            [float(grid["x_right_mm"].max())],
        )
    )
    plot_extent_mm = float(max(abs(edges[0]), abs(edges[-1])))
    palette = plt.get_cmap(colormap).copy()
    palette.set_bad(color=(0.92, 0.92, 0.92, 1.0))
    fig, ax = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)
    rendered = ax.imshow(
        display.T,
        origin="lower",
        extent=(edges[0], edges[-1], edges[0], edges[-1]),
        cmap=palette,
        interpolation="nearest",
        vmin=colour_min,
        vmax=colour_max,
    )
    _plot_valid_mask_boundary(ax, valid_mask, center_px, mm_per_pixel)
    total_area = float(grid["valid_area_cm2"].sum())
    total_count = int(grid["count"].sum())
    mean_density = float(total_count / total_area) if total_area > 0 else float("nan")
    cell_size_mm = float(edges[1] - edges[0])
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=(
            "Accepted point-like target density\n"
            f"mean={mean_density:.4g} cm$^{{-2}}$, n={total_count}; "
            f"{cell_size_mm:.3g} mm cells, actual valid area"
        ),
        aspect="equal",
    )
    ticks = np.arange(
        math.ceil(-plot_extent_mm / grid_interval_mm) * grid_interval_mm,
        plot_extent_mm + grid_interval_mm * 0.5,
        grid_interval_mm,
    )
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(color="white", alpha=0.26, linewidth=0.5)
    colourbar = fig.colorbar(
        rendered,
        ax=ax,
        label=r"Detected point-like target density (cm$^{-2}$)",
        extend="max" if raw_max > colour_max else "neither",
    )
    colourbar.ax.tick_params(labelsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

def save_distribution_plots(
    defects: pd.DataFrame | Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    valid_mask: np.ndarray | None = None,
    center_px: tuple[float, float] | None = None,
    mm_per_pixel: float | tuple[float, float] | None = None,
    wafer_radius_mm: float = 50.0,
    generate_heatmap: bool = True,
    heatmap_bins: int = 40,
    heatmap_colormap: str = "turbo",
    heatmap_vmin_cm2: float | None = 0.0,
    heatmap_vmax_cm2: float | None = None,
    heatmap_clip_percentile: float = 99.5,
    heatmap_min_valid_fraction: float = 0.05,
    heatmap_grid_interval_mm: float = 10.0,
) -> dict[str, Path]:
    """Write the optional diagnostic plot set requested by the report."""

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    outputs = {
        "size_histogram": save_size_histogram(defects, folder / "defect_size_histogram.png"),
        "radial_distribution": save_radial_distribution(defects, folder / "radial_distribution.png"),
        "angular_distribution": save_angular_distribution(defects, folder / "angular_distribution.png"),
        "wafer_scatter": save_wafer_scatter(
            defects, folder / "wafer_position_scatter.png", wafer_radius_mm=wafer_radius_mm
        ),
    }
    if generate_heatmap:
        outputs["density_heatmap"] = save_density_heatmap(
            defects,
            folder / "density_heatmap.png",
            valid_mask=valid_mask,
            center_px=center_px,
            mm_per_pixel=mm_per_pixel,
            wafer_radius_mm=wafer_radius_mm,
            bins=heatmap_bins,
            colormap=heatmap_colormap,
            vmin_cm2=heatmap_vmin_cm2,
            vmax_cm2=heatmap_vmax_cm2,
            clip_percentile=heatmap_clip_percentile,
            min_valid_fraction=heatmap_min_valid_fraction,
            grid_interval_mm=heatmap_grid_interval_mm,
            grid_csv_path=folder / "density_heatmap_grid.csv",
        )
        outputs["density_heatmap_grid"] = folder / "density_heatmap_grid.csv"
    return outputs


# Backwards-friendly aliases used by small external scripts.
save_mask = save_binary_mask
save_preprocessed_preview = save_grayscale_image
