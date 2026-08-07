"""Resolve optional physical-unit measurement rules once per wafer analysis.

The computer-vision routines intentionally keep using pixels.  This module is
the single, auditable boundary between user-facing micrometre rules and those
pixel-domain routines.  Keeping this conversion in one place prevents subtle
differences between full-frame and tiled processing.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PhysicalParameterResolution:
    """Resolved configuration plus a serialisable measurement conversion log."""

    config: dict[str, Any]
    report: dict[str, Any]
    warnings: tuple[str, ...]


def _finite_nonnegative(value: Any, *, name: str, positive: bool = False) -> float:
    """Validate one user-facing physical scalar without accepting NaN/Inf."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite numeric value, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not math.isfinite(number) or number < 0.0 or (positive and number <= 0.0):
        comparator = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {comparator}, got {value!r}")
    return number


def _as_physical_list(value: Any, *, name: str) -> list[float]:
    """Validate a non-empty physical length sequence."""

    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a non-empty list of physical lengths")
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return [_finite_nonnegative(item, name=name, positive=True) for item in value]


def _odd_positive(value_px: float) -> int:
    """Round a physical morphology length to a positive odd pixel kernel."""

    resolved = max(1, int(round(value_px)))
    return resolved if resolved % 2 else resolved + 1


def _positive_pixel_count(value_px: float) -> int:
    """Round a physical integer pixel quantity while retaining a valid minimum."""

    return max(1, int(round(value_px)))


def resolve_physical_parameters(
    config: Mapping[str, Any], *, mm_per_pixel: float
) -> PhysicalParameterResolution:
    """Resolve optional ``*_um``/``*_um2`` rules to existing pixel parameters.

    Physical values win over legacy pixel values.  The returned configuration
    contains only the legacy names consumed by existing CV routines; the report
    preserves both the original physical input and converted pixel value.
    """

    scale = _finite_nonnegative(mm_per_pixel, name="mm_per_pixel", positive=True)
    um_per_pixel = scale * 1000.0
    resolved = copy.deepcopy(dict(config))
    warnings: list[str] = []
    entries: dict[str, dict[str, Any]] = {}

    for section_name in ("preprocessing", "detection", "filters"):
        value = resolved.get(section_name, {})
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{section_name} configuration must be a mapping")
        resolved[section_name] = dict(value)

    def convert_length(
        *, section: str, physical: str, legacy: str, kernel: bool = False,
        integer: bool = False, direct_mm: bool = False, positive: bool = False,
    ) -> None:
        values = resolved[section]
        label = f"{section}.{physical}"
        if physical not in values:
            entries[f"{section}.{legacy}"] = {
                "source": "legacy_px" if not direct_mm else "legacy_mm",
                "input_physical": None,
                "input_legacy": values.get(legacy),
                "resolved_pixel_value": (
                    None if direct_mm else values.get(legacy)
                ),
                "resolved_value": values.get(legacy),
            }
            return
        raw = _finite_nonnegative(values[physical], name=label, positive=positive)
        if legacy in values:
            warnings.append(
                f"{label} takes precedence over legacy {section}.{legacy}"
            )
        pixel_value = raw / um_per_pixel
        if kernel:
            converted: float | int = _odd_positive(pixel_value)
        elif integer:
            converted = _positive_pixel_count(pixel_value)
        elif direct_mm:
            converted = raw / 1000.0
        else:
            converted = pixel_value
        values[legacy] = converted
        values.pop(physical, None)
        entries[f"{section}.{legacy}"] = {
            "source": "physical",
            "input_physical": raw,
            "physical_unit": "um",
            "resolved_pixel_value": pixel_value,
            "resolved_value": converted,
        }

    def convert_area(*, physical: str, legacy: str) -> None:
        values = resolved["filters"]
        label = f"filters.{physical}"
        if physical not in values:
            entries[f"filters.{legacy}"] = {
                "source": "legacy_px",
                "input_physical": None,
                "input_legacy": values.get(legacy),
                "resolved_pixel_value": values.get(legacy),
                "resolved_value": values.get(legacy),
            }
            return
        raw = _finite_nonnegative(values[physical], name=label)
        if legacy in values:
            warnings.append(f"{label} takes precedence over legacy filters.{legacy}")
        converted = raw / (um_per_pixel * um_per_pixel)
        values[legacy] = converted
        values.pop(physical, None)
        entries[f"filters.{legacy}"] = {
            "source": "physical",
            "input_physical": raw,
            "physical_unit": "um2",
            "resolved_pixel_value": converted,
            "resolved_value": converted,
        }

    convert_length(
        section="preprocessing", physical="background_kernel_um",
        legacy="background_kernel_px", kernel=True, positive=True,
    )
    convert_length(
        section="preprocessing", physical="gaussian_sigma_um",
        legacy="gaussian_sigma",
    )

    detection = resolved["detection"]
    physical_kernels = "blackhat_kernel_sizes_um"
    legacy_kernels = "blackhat_kernel_sizes_px"
    if physical_kernels in detection:
        lengths = _as_physical_list(
            detection[physical_kernels], name=f"detection.{physical_kernels}"
        )
        if legacy_kernels in detection:
            warnings.append(
                "detection.blackhat_kernel_sizes_um takes precedence over legacy "
                "detection.blackhat_kernel_sizes_px"
            )
        pixel_lengths = [item / um_per_pixel for item in lengths]
        detection[legacy_kernels] = [_odd_positive(item) for item in pixel_lengths]
        detection.pop(physical_kernels, None)
        entries[f"detection.{legacy_kernels}"] = {
            "source": "physical",
            "input_physical": lengths,
            "physical_unit": "um",
            "resolved_pixel_value": pixel_lengths,
            "resolved_value": detection[legacy_kernels],
        }
    else:
        entries[f"detection.{legacy_kernels}"] = {
            "source": "legacy_px",
            "input_physical": None,
            "input_legacy": detection.get(legacy_kernels),
            "resolved_pixel_value": detection.get(legacy_kernels),
            "resolved_value": detection.get(legacy_kernels),
        }

    convert_length(
        section="detection", physical="min_peak_distance_um",
        legacy="min_peak_distance_px", integer=True, positive=True,
    )
    convert_length(
        section="detection", physical="dog_min_sigma_um",
        legacy="dog_min_sigma_px", positive=True,
    )
    convert_length(
        section="detection", physical="dog_max_sigma_um",
        legacy="dog_max_sigma_px", positive=True,
    )

    convert_length(
        section="filters", physical="min_equivalent_diameter_um",
        legacy="min_equivalent_diameter_px",
    )
    convert_length(
        section="filters", physical="max_equivalent_diameter_um",
        legacy="max_equivalent_diameter_px",
    )
    convert_area(physical="min_area_um2", legacy="min_area_px")
    convert_area(physical="max_area_um2", legacy="max_area_px")
    convert_length(
        section="filters", physical="local_background_ring_um",
        legacy="local_background_ring_px", integer=True,
    )
    convert_length(
        section="filters", physical="min_edge_distance_um",
        legacy="min_edge_distance_mm", direct_mm=True,
    )

    filters = resolved["filters"]
    min_diameter = float(filters.get("min_equivalent_diameter_px", 0.0))
    max_diameter = float(filters.get("max_equivalent_diameter_px", float("inf")))
    min_area = float(filters.get("min_area_px", 0.0))
    max_area = float(filters.get("max_area_px", float("inf")))
    if min_diameter > max_diameter:
        raise ValueError("min_equivalent_diameter must not exceed max_equivalent_diameter")
    if min_area > max_area:
        raise ValueError("min_area must not exceed max_area")
    dog_min = float(detection.get("dog_min_sigma_px", 0.0))
    dog_max = float(detection.get("dog_max_sigma_px", 0.0))
    if dog_min <= 0.0 or dog_min > dog_max:
        raise ValueError("dog_min_sigma must be positive and not exceed dog_max_sigma")

    report = {
        "mm_per_pixel": scale,
        "um_per_pixel": um_per_pixel,
        "parameters": entries,
        "warnings": warnings,
    }
    return PhysicalParameterResolution(
        config=resolved, report=report, warnings=tuple(warnings)
    )


__all__ = ["PhysicalParameterResolution", "resolve_physical_parameters"]
