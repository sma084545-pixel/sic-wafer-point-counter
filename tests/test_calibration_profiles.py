"""Opt-in standard-reference calibration profile tests."""

from __future__ import annotations

import pytest

from sic_wafer_counter.calibration_profiles import (
    GENERIC_PROFILE_ID,
    STANDARD_R40_B2_PROFILE_ID,
    apply_calibration_profile,
    available_profiles,
)
from sic_wafer_counter.web import _analysis_config_from_form


def test_r40_b2_profile_is_opt_in_auditable_and_does_not_mutate_defaults(
    default_config,
) -> None:
    original_offset = default_config["detection"]["threshold_offset"]
    calibrated = apply_calibration_profile(default_config, STANDARD_R40_B2_PROFILE_ID)
    assert default_config["detection"]["threshold_offset"] == original_offset
    assert calibrated["detection"]["threshold_method"] == "otsu"
    assert calibrated["detection"]["threshold_offset"] == pytest.approx(0.030)
    assert calibrated["detection"]["use_watershed"] is False
    profile = calibrated["analysis_profile"]
    assert profile["status"] == "single_wafer_spatial_internal_calibration"
    assert profile["source_wafer_id"] == "CN4N82006412-SiC-0008"
    assert profile["real_sic_accuracy_claim_allowed"] is False
    assert len(profile["source_image_sha256"]) == 64
    assert "overrides" not in profile


def test_generic_profile_preserves_transparent_rules_and_unknown_profile_fails(
    default_config,
) -> None:
    generic = apply_calibration_profile(default_config, GENERIC_PROFILE_ID)
    assert generic["detection"] == default_config["detection"]
    assert generic["analysis_profile"]["status"] == "generic_unvalidated_defaults"
    assert {item["profile_id"] for item in available_profiles()} == {
        GENERIC_PROFILE_ID,
        STANDARD_R40_B2_PROFILE_ID,
    }
    with pytest.raises(ValueError, match="未知分析校准配置"):
        apply_calibration_profile(default_config, "unknown")


def test_local_web_form_applies_calibrated_parameters_after_generic_controls() -> None:
    configured, manual = _analysis_config_from_form(
        {
            "analysis_profile": STANDARD_R40_B2_PROFILE_ID,
            "wafer_diameter_mm": "100",
            "exclude_edge_mm": "0",
            "threshold_method": "quantile",
            "use_watershed": "true",
        }
    )
    assert manual == (None, None, None)
    assert configured["detection"]["threshold_method"] == "otsu"
    assert configured["detection"]["threshold_offset"] == pytest.approx(0.030)
    assert configured["detection"]["use_watershed"] is False
