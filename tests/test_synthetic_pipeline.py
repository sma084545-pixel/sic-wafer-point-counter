"""End-to-end synthetic-data validation without ground-truth leakage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from sic_wafer_counter.feature_extraction import DEFECT_COLUMNS
from sic_wafer_counter.pipeline import analyze_image


def _metrics(accepted: pd.DataFrame, truth: pd.DataFrame, tolerance_px: float = 10.0) -> tuple[float, float, float]:
    expected = truth[truth["should_count"].astype(bool)].reset_index(drop=True)
    if accepted.empty or expected.empty:
        return 0.0, 0.0, 0.0
    distance = cdist(
        accepted[["centroid_x_px", "centroid_y_px"]],
        expected[["center_x_px", "center_y_px"]],
    )
    rows, cols = linear_sum_assignment(distance)
    matches = int((distance[rows, cols] <= tolerance_px).sum())
    precision = matches / len(accepted)
    recall = matches / len(expected)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _run(kind: str, generate_synthetic, default_config, tmp_path: Path):
    generated = generate_synthetic(kind)
    result = analyze_image(generated["image_path"], tmp_path / f"result_{kind}", default_config)
    truth = pd.read_csv(generated["ground_truth_path"])
    accepted = result.defects[result.defects["accepted"]].copy()
    return generated, result, truth, accepted


def test_clean_synthetic_count_is_exact(generate_synthetic, default_config, tmp_path) -> None:
    generated, result, truth, accepted = _run("clean", generate_synthetic, default_config, tmp_path)
    precision, recall, f1 = _metrics(accepted, truth)
    print(f"clean P/R/F1={precision:.3f}/{recall:.3f}/{f1:.3f}")
    assert generated["true_valid_count"] == 96
    assert result.summary["accepted_count"] == 96
    assert precision == recall == f1 == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["noisy", "difficult"])
def test_noisy_and_difficult_report_precision_recall_f1(kind, generate_synthetic, default_config, tmp_path) -> None:
    _, result, truth, accepted = _run(kind, generate_synthetic, default_config, tmp_path)
    precision, recall, f1 = _metrics(accepted, truth)
    print(f"{kind} P/R/F1={precision:.3f}/{recall:.3f}/{f1:.3f}")
    assert result.summary["accepted_count"] > 0
    assert precision >= 0.90
    assert recall >= 0.85
    assert f1 >= 0.90


def test_outputs_are_complete_and_reproducible(generate_synthetic, default_config, tmp_path) -> None:
    first = generate_synthetic("clean", seed=987)
    first_hash = hashlib.sha256(Path(first["image_path"]).read_bytes()).hexdigest()
    first_result = analyze_image(first["image_path"], tmp_path / "out_first", default_config)
    second = generate_synthetic("clean", seed=987)
    second_hash = hashlib.sha256(Path(second["image_path"]).read_bytes()).hexdigest()
    second_result = analyze_image(second["image_path"], tmp_path / "out_second", default_config)
    assert first_hash == second_hash
    assert first_result.summary["accepted_count"] == second_result.summary["accepted_count"]
    assert np.array_equal(
        first_result.defects[["centroid_x_px", "centroid_y_px", "accepted"]].to_numpy(),
        second_result.defects[["centroid_x_px", "centroid_y_px", "accepted"]].to_numpy(),
    )

    folder = tmp_path / "out_first"
    for filename in (
        "summary.json", "summary.csv", "defects_all.csv", "defects_accepted.csv",
        "defects_rejected.csv", "overlay_accepted.png", "overlay_all_candidates.png",
        "wafer_mask.png", "valid_analysis_mask.png", "preprocessed_preview.png",
        "candidate_mask.png", "analysis_config.yaml", "run.log", "report.html",
        "resolved_physical_parameters.yaml",
        "radial_density.csv", "radial_density.png", "angular_density.csv",
        "angular_density.png", "regional_density.csv",
    ):
        assert (folder / filename).is_file(), filename
    summary = json.loads((folder / "summary.json").read_text())
    required_summary = {
        "input_file_name", "image_size", "wafer_diameter_mm", "wafer_center_px",
        "wafer_radius_px", "mm_per_pixel", "theoretical_area_cm2",
        "fitted_wafer_area_cm2", "valid_analysis_area_cm2", "raw_candidate_count",
        "post_watershed_candidate_count", "accepted_count", "rejected_count",
        "point_density_cm2", "counting_uncertainty_cm2", "poisson_95_ci_lower_cm2",
        "poisson_95_ci_upper_cm2", "filter_parameters", "software_version",
        "runtime_seconds", "warnings",
        "um_per_pixel", "resolved_physical_parameters_file",
        "source_dtype", "analysis_dtype", "normalization_low_value",
        "normalization_high_value", "low_clipped_fraction", "high_clipped_fraction",
        "white_is_zero", "analysis_quantized_to_uint8",
        "spatial_density", "regional_density",
    }
    assert required_summary <= set(summary)
    assert "spatial_heterogeneity" in summary["spatial_density"]
    defects = pd.read_csv(folder / "defects_all.csv")
    assert set(DEFECT_COLUMNS) <= set(defects.columns)
