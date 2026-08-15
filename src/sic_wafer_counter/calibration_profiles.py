"""Auditable, opt-in measurement profiles derived from supplied reference data.

Profiles are deliberately separate from the generic defaults.  A profile may
be useful for matching a documented acquisition setup, but one-wafer internal
calibration is not evidence of cross-wafer accuracy or physical defect class.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .utils import deep_merge


GENERIC_PROFILE_ID = "generic_transparent_rules"
STANDARD_R40_B2_PROFILE_ID = "cn4n82006412_r40_b2_standard_20260815"

_PROFILES: dict[str, dict[str, Any]] = {
    GENERIC_PROFILE_ID: {
        "profile_id": GENERIC_PROFILE_ID,
        "display_name": "通用透明规则（未针对当前设备校准）",
        "status": "generic_unvalidated_defaults",
        "overrides": {},
        "real_sic_accuracy_claim_allowed": False,
    },
    STANDARD_R40_B2_PROFILE_ID: {
        "profile_id": STANDARD_R40_B2_PROFILE_ID,
        "display_name": "R40-b2 标准红框校准（2026-08-15）",
        "status": "single_wafer_spatial_internal_calibration",
        "source_wafer_id": "CN4N82006412-SiC-0008",
        "source_image_sha256": "71441da15a97bb309d3584670058b7be6a559aeb3d15926ee1dacdbf0623f280",
        "reference_map_sha256": "a4422d2851d34cef32f103cf8329cf925b8bc6186702a70b6c66035968336631",
        "reference_semantics": "user-supplied red-box image-target annotations",
        "high_confidence_reference_count": 21350,
        "ignored_overlap_region_count": 1321,
        "selection_note": (
            "threshold_offset was selected by a small local sweep on six source-resolution "
            "fields and checked on fifteen disjoint fields from the same wafer"
        ),
        "overrides": {
            "detection": {
                "method": "blackhat",
                "threshold_method": "otsu",
                "threshold_offset": 0.030,
                "use_watershed": False,
            }
        },
        "real_sic_accuracy_claim_allowed": False,
        "scientific_limit": (
            "This profile reduces over-counting relative to supplied image annotations. "
            "It has no independent-wafer or DIC/KOH validation and does not confirm TSD/TED/BPD."
        ),
    },
}


def available_profiles() -> tuple[dict[str, Any], ...]:
    """Return public metadata without mutable override dictionaries."""

    values: list[dict[str, Any]] = []
    for profile in _PROFILES.values():
        item = copy.deepcopy(profile)
        item.pop("overrides", None)
        values.append(item)
    return tuple(values)


def profile_metadata(profile_id: str) -> dict[str, Any]:
    """Return one validated profile's public metadata."""

    identifier = str(profile_id).strip() or GENERIC_PROFILE_ID
    if identifier not in _PROFILES:
        raise ValueError(f"未知分析校准配置：{identifier}")
    result = copy.deepcopy(_PROFILES[identifier])
    result.pop("overrides", None)
    return result


def apply_calibration_profile(
    config: Mapping[str, Any], profile_id: str | None
) -> dict[str, Any]:
    """Apply an opt-in profile and attach its provenance to the run config."""

    identifier = str(profile_id or GENERIC_PROFILE_ID).strip() or GENERIC_PROFILE_ID
    if identifier not in _PROFILES:
        raise ValueError(f"未知分析校准配置：{identifier}")
    profile = _PROFILES[identifier]
    result = deep_merge(config, profile.get("overrides", {}))
    result["analysis_profile"] = profile_metadata(identifier)
    return result


__all__ = [
    "GENERIC_PROFILE_ID",
    "STANDARD_R40_B2_PROFILE_ID",
    "apply_calibration_profile",
    "available_profiles",
    "profile_metadata",
]
