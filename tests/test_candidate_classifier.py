"""Portable candidate-classifier training and inference tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sic_wafer_counter.candidate_classifier import (
    apply_candidate_classifier,
    train_candidate_classifier,
    validate_model,
)


def _candidate(label: str, index: int, *, split: str, wafer_id: str) -> dict[str, object]:
    target = label == "target"
    scale = float(index % 10) * 0.01
    return {
        "run_id": wafer_id,
        "wafer_id": wafer_id,
        "defect_id": index,
        "label": label,
        "split": split,
        "area_mm2": (0.000020 if target else 0.000180) + scale * 0.000001,
        "equivalent_diameter_um": (5.0 if target else 18.0) + scale,
        "major_axis_length_um": (6.0 if target else 32.0) + scale,
        "minor_axis_length_um": (4.5 if target else 5.0) + scale,
        "aspect_ratio": (1.25 if target else 6.2) + scale,
        "eccentricity": 0.35 if target else 0.97,
        "circularity": 0.88 if target else 0.18,
        "solidity": 0.96 if target else 0.58,
        "contrast": 0.42 if target else 0.05,
        "mean_dark_response": 0.61 if target else 0.12,
        "accepted": True,
        "rejection_reason": "",
    }


def test_classifier_trains_with_held_out_wafer_and_preserves_hard_geometry() -> None:
    rows = [
        _candidate(label, index, split="calibration", wafer_id="wafer-train")
        for label in ("target", "artifact")
        for index in range(1, 7)
    ]
    rows += [
        _candidate(label, index + 100, split="validation", wafer_id="wafer-validation")
        for label in ("target", "artifact")
        for index in range(1, 4)
    ]
    model = train_candidate_classifier(pd.DataFrame(rows))
    assert validate_model(model)["model_sha256"] == model["model_sha256"]
    assert model["training_sample_count"] == 12
    assert model["validation"]["status"] == "held_out_validation_metrics_available"
    assert model["validation"]["physical_identity_validated"] is False

    candidates = pd.DataFrame(
        [
            _candidate("target", 201, split="calibration", wafer_id="new-wafer"),
            _candidate("artifact", 202, split="calibration", wafer_id="new-wafer"),
            {
                **_candidate("target", 203, split="calibration", wafer_id="new-wafer"),
                "accepted": False,
                "rejection_reason": "near_wafer_edge",
            },
        ]
    ).drop(columns=["label", "split", "run_id", "wafer_id"])
    scored, report, warnings = apply_candidate_classifier(candidates, model)
    assert scored.loc[0, "classifier_decision"] == "target"
    assert bool(scored.loc[0, "accepted"])
    assert scored.loc[1, "classifier_decision"] == "artifact"
    assert not bool(scored.loc[1, "accepted"])
    assert scored.loc[2, "classifier_decision"] == "blocked_by_geometry"
    assert not bool(scored.loc[2, "accepted"])
    assert scored.loc[2, "rejection_reason"] == "near_wafer_edge"
    assert report["model_sha256"] == model["model_sha256"]
    assert not warnings


def test_classifier_rejects_wafer_split_leakage_and_ignores_uncertain_labels() -> None:
    rows = [
        _candidate(label, index, split="calibration", wafer_id="same-wafer")
        for label in ("target", "artifact")
        for index in range(1, 6)
    ]
    rows.append(_candidate("target", 99, split="validation", wafer_id="same-wafer"))
    with pytest.raises(ValueError, match="同一晶圆/图像组"):
        train_candidate_classifier(pd.DataFrame(rows))

    clean_rows = [
        _candidate(label, index, split="calibration", wafer_id="train-wafer")
        for label in ("target", "artifact")
        for index in range(1, 6)
    ]
    clean_rows.append(
        _candidate("uncertain", 100, split="calibration", wafer_id="train-wafer")
    )
    model = train_candidate_classifier(pd.DataFrame(clean_rows))
    assert model["training_sample_count"] == 10
    assert model["excluded_uncertain_count"] == 1


def test_classifier_rejects_tampered_portable_model() -> None:
    rows = [
        _candidate(label, index, split="calibration", wafer_id="train-wafer")
        for label in ("target", "artifact")
        for index in range(1, 6)
    ]
    model = train_candidate_classifier(pd.DataFrame(rows))
    tampered = dict(model)
    tampered["coefficients"] = list(model["coefficients"])
    tampered["coefficients"][0] = float(tampered["coefficients"][0]) + 1.0
    with pytest.raises(ValueError, match="model_sha256"):
        validate_model(tampered)
