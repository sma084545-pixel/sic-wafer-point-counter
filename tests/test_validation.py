"""Real-annotation validation rules: labels, leakage, metrics, and bootstrap."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from sic_wafer_counter.validation import (
    ANNOTATION_SCHEMA_VERSION,
    annotation_provenance_status,
    annotation_template,
    assign_wafer_splits,
    bootstrap_classification_bias,
    evaluate_automatic_detections,
    reviewer_agreement,
    run_local_parameter_sensitivity,
    write_unvalidated_bundle,
)


def _automatic() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"image_id": "image_a", "wafer_id": "wafer_a", "centroid_x_px": 10, "centroid_y_px": 10, "x_mm": 0.0, "y_mm": 0.0, "accepted": True},
            {"image_id": "image_a", "wafer_id": "wafer_a", "centroid_x_px": 20, "centroid_y_px": 10, "x_mm": 1.0, "y_mm": 0.0, "accepted": True},
            {"image_id": "image_a", "wafer_id": "wafer_a", "centroid_x_px": 30, "centroid_y_px": 10, "x_mm": 2.0, "y_mm": 0.0, "accepted": True},
        ]
    )


def _reviewed() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"image_id": "image_a", "wafer_id": "wafer_a", "x_px": 10, "y_px": 10, "x_mm": 0.0, "y_mm": 0.0, "label": "confirmed_point"},
            {"image_id": "image_a", "wafer_id": "wafer_a", "x_px": 20, "y_px": 10, "x_mm": 1.0, "y_mm": 0.0, "label": "scratch"},
            {"image_id": "image_a", "wafer_id": "wafer_a", "x_px": 30, "y_px": 10, "x_mm": 2.0, "y_mm": 0.0, "label": "uncertain"},
            {"image_id": "image_a", "wafer_id": "wafer_a", "x_px": 50, "y_px": 10, "x_mm": 4.0, "y_mm": 0.0, "label": "confirmed_point"},
        ]
    )


def test_matching_metrics_use_physical_distance_and_preserve_uncertain() -> None:
    result = evaluate_automatic_detections(
        _automatic(),
        _reviewed(),
        matching_tolerance_um=100.0,
        annotation_coverage_complete=True,
        area_cm2_by_wafer={"wafer_a": 2.0},
    )
    assert result.metrics["TP"] == 1
    assert result.metrics["FP"] == 1
    assert result.metrics["FN"] == 1
    assert result.metrics["precision"] == pytest.approx(0.5)
    assert result.metrics["recall"] == pytest.approx(0.5)
    assert result.metrics["F1"] == pytest.approx(0.5)
    assert result.metrics["false_positives_per_cm2"] == pytest.approx(0.5)
    assert (result.matches["match_class"] == "indeterminate").sum() == 1


def test_incomplete_annotation_coverage_refuses_precision_claim() -> None:
    result = evaluate_automatic_detections(
        _automatic(),
        _reviewed(),
        matching_tolerance_um=100.0,
        annotation_coverage_complete=False,
        area_cm2_by_wafer={"wafer_a": 2.0},
    )
    assert result.metrics["validation_status"] == "partial_annotation_coverage"
    assert result.metrics["precision"] is None
    assert result.metrics["F1"] is None
    assert result.metrics["indeterminate_automatic_count"] >= 1


def test_zero_automatic_count_has_no_division_error() -> None:
    result = evaluate_automatic_detections(
        _automatic().iloc[0:0],
        _reviewed().iloc[[0]],
        matching_tolerance_um=100.0,
        annotation_coverage_complete=True,
    )
    assert result.metrics["TP"] == 0
    assert result.metrics["FN"] == 1
    assert result.metrics["precision"] is None
    assert result.metrics["recall"] == 0.0


def test_known_cohen_kappa_and_raw_agreement() -> None:
    rows = []
    labels = [("a", "confirmed_point", "confirmed_point"), ("b", "scratch", "scratch"), ("c", "confirmed_point", "scratch")]
    for key, first, second in labels:
        for reviewer, label in (("r1", first), ("r2", second)):
            rows.append(
                {
                    "image_id": "image_a", "wafer_id": "wafer_a", "candidate_id": key,
                    "x_px": 1.0, "y_px": 1.0, "x_mm": 0.0, "y_mm": 0.0,
                    "label": label, "reviewer_id": reviewer,
                }
            )
    summary, disagreements = reviewer_agreement(pd.DataFrame(rows))
    assert summary["raw_agreement"] == pytest.approx(2.0 / 3.0)
    assert summary["cohen_kappa"] == pytest.approx(0.4)
    assert len(disagreements) == 1


def test_wafer_splits_prevent_patch_level_leakage() -> None:
    with pytest.raises(ValueError, match="leakage"):
        assign_wafer_splits(
            ["wafer_a", "wafer_b"],
            calibration_wafers=["wafer_a"],
            validation_wafers=["wafer_a"],
            locked_test_wafers=["wafer_b"],
        )
    splits = assign_wafer_splits(
        ["wafer_a", "wafer_b", "wafer_c"],
        calibration_wafers=["wafer_a"],
        validation_wafers=["wafer_b"],
        locked_test_wafers=["wafer_c"],
    )
    assert set(splits["split"]) == {"calibration", "validation", "locked_test"}


def test_locked_test_is_excluded_from_parameter_sensitivity() -> None:
    calls: list[tuple[str, float, tuple[str, ...]]] = []

    def evaluator(name: str, value: float, wafers: tuple[str, ...]):
        calls.append((name, value, wafers))
        assert wafers == ("wafer_a",)
        return {"accepted_count": 10, "rho_cm2": 1.0}

    table = run_local_parameter_sensitivity(
        {"filters.min_circularity": 0.25},
        calibration_wafer_ids=["wafer_a"],
        locked_test_wafer_ids=["wafer_b"],
        evaluator=evaluator,
    )
    assert len(table) == 5
    assert len(calls) == 5
    with pytest.raises(ValueError, match="leakage"):
        run_local_parameter_sensitivity(
            {"filters.min_circularity": 0.25},
            calibration_wafer_ids=["wafer_a"],
            locked_test_wafer_ids=["wafer_a"],
            evaluator=evaluator,
        )


def test_wafer_bootstrap_is_seeded_and_zero_counts_work() -> None:
    per_wafer = pd.DataFrame(
        [
            {"automatic_count": 0, "reviewed_confirmed_count": 0, "valid_area_cm2": 1.0},
            {"automatic_count": 0, "reviewed_confirmed_count": 0, "valid_area_cm2": 2.0},
        ]
    )
    first = bootstrap_classification_bias(per_wafer, seed=11, samples=200)
    second = bootstrap_classification_bias(per_wafer, seed=11, samples=200)
    assert first == second
    assert first["bias_mean_cm2"] == pytest.approx(0.0)


def test_unvalidated_bundle_makes_no_real_performance_claim(tmp_path) -> None:
    paths = write_unvalidated_bundle(tmp_path / "validation")
    summary = pd.read_json(paths["validation_summary"], typ="series")
    assert summary["validation_status"] == "not validated on real SiC data"
    assert bool(summary["performance_metrics_reported"]) is False
    template = annotation_template()
    assert tuple(template.columns) == (
        "image_id", "wafer_id", "candidate_id", "x_px", "y_px", "x_mm", "y_mm",
        "label", "reviewer_id", "review_confidence", "notes", "source_image_sha256",
        "annotation_schema_version",
    )
    assert template.iloc[0]["annotation_schema_version"] == ANNOTATION_SCHEMA_VERSION


def test_annotation_provenance_reports_verified_missing_and_mismatch() -> None:
    annotations = pd.DataFrame(
        [
            {"image_id": "verified", "source_image_sha256": "abc"},
            {"image_id": "missing", "source_image_sha256": ""},
            {"image_id": "mismatch", "source_image_sha256": "abc"},
        ]
    )
    status = annotation_provenance_status(
        annotations, {"verified": "ABC", "mismatch": "different"}
    ).set_index("image_id")["status"]
    assert status["verified"] == "verified"
    assert status["missing"] == "missing_declared_hash"
    assert status["mismatch"] == "hash_mismatch"
