"""Tests for Rigaku-semantic independent-reference overlays and matching."""

from __future__ import annotations

from pathlib import Path
import copy

import cv2
import numpy as np
import pandas as pd
import pytest

from sic_wafer_counter.paper_alignment import (
    REFERENCE_COLUMNS,
    compare_automatic_to_registered_reference,
    file_sha256,
    load_registered_reference_points,
    reference_template,
    references_from_config,
)
from sic_wafer_counter.reporting import save_xrt_detection_detail_montage
from sic_wafer_counter.pipeline import analyze_image


def _images(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.bin"
    reference = tmp_path / "reference.bin"
    source.write_bytes(b"source-xrt-image")
    reference.write_bytes(b"independent-dic-image")
    return source, reference


def _reference_frame(source: Path, reference: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "reference_id": "DIC-001",
                "x_px": 100.0,
                "y_px": 100.0,
                "label": "confirmed_point",
                "reference_method": "DIC",
                "registration_status": "registered",
                "registration_rmse_um": 5.0,
                "source_image_sha256": file_sha256(source),
                "reference_image_sha256": file_sha256(reference),
                "notes": "registered independently",
                "reference_schema_version": "1.0",
            },
            {
                "reference_id": "DIC-002",
                "x_px": 140.0,
                "y_px": 140.0,
                "label": "uncertain",
                "reference_method": "DIC",
                "registration_status": "registered",
                "registration_rmse_um": 5.0,
                "source_image_sha256": file_sha256(source),
                "reference_image_sha256": file_sha256(reference),
                "notes": "not a confirmed physical reference",
                "reference_schema_version": "1.0",
            },
        ],
        columns=REFERENCE_COLUMNS,
    )


def test_template_never_fabricates_a_confirmed_reference() -> None:
    template = reference_template()
    assert template.iloc[0]["label"] == "uncertain"
    assert template.iloc[0]["registration_status"] == "not_registered"
    assert "confirmed_point" not in set(template["label"])

    _, status, _ = references_from_config(
        {"independent_reference": {"enabled": "false"}},
        source_image_path=None,
        source_shape=None,
        automatic_candidates=pd.DataFrame(),
        mm_per_pixel=None,
    )
    assert status["status"] == "not provided"
    with pytest.raises(ValueError, match="must be a boolean"):
        references_from_config(
            {"independent_reference": {"enabled": "maybe"}},
            source_image_path=None,
            source_shape=None,
            automatic_candidates=pd.DataFrame(),
            mm_per_pixel=None,
        )


def test_registered_reference_requires_both_image_hashes_and_filters_uncertain(
    tmp_path: Path,
) -> None:
    source, reference = _images(tmp_path)
    csv_path = tmp_path / "references.csv"
    _reference_frame(source, reference).to_csv(csv_path, index=False)
    loaded = load_registered_reference_points(
        csv_path,
        source_image_path=source,
        reference_image_path=reference,
        source_shape=(200, 200),
    )
    assert loaded.summary["confirmed_registered_count"] == 1
    assert loaded.summary["possible_or_uncertain_count"] == 1
    assert loaded.points["reference_id"].tolist() == ["DIC-001"]

    changed_source = tmp_path / "changed.bin"
    changed_source.write_bytes(b"different source")
    with pytest.raises(ValueError, match="source_image_sha256"):
        load_registered_reference_points(
            csv_path,
            source_image_path=changed_source,
            reference_image_path=reference,
            source_shape=(200, 200),
        )

    changed_reference = tmp_path / "changed_reference.bin"
    changed_reference.write_bytes(b"different reference")
    with pytest.raises(ValueError, match="reference_image_sha256"):
        load_registered_reference_points(
            csv_path,
            source_image_path=source,
            reference_image_path=changed_reference,
            source_shape=(200, 200),
        )

    malformed = _reference_frame(source, reference)
    malformed.loc[0, "registration_status"] = "registred"
    malformed.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="registration_status"):
        load_registered_reference_points(
            csv_path,
            source_image_path=source,
            reference_image_path=reference,
            source_shape=(200, 200),
        )


def test_matching_uses_physical_tolerance_and_does_not_invent_precision() -> None:
    automatic = pd.DataFrame(
        [
            {"defect_id": 1, "centroid_x_px": 10.0, "centroid_y_px": 10.0, "accepted": True},
            {"defect_id": 2, "centroid_x_px": 50.0, "centroid_y_px": 50.0, "accepted": True},
        ]
    )
    references = pd.DataFrame(
        [
            {"reference_id": "R1", "x_px": 11.0, "y_px": 10.0},
            {"reference_id": "R2", "x_px": 90.0, "y_px": 90.0},
        ]
    )
    incomplete, audit = compare_automatic_to_registered_reference(
        automatic,
        references,
        mm_per_pixel=0.01,
        matching_tolerance_um=20.0,
        reference_coverage_complete=False,
    )
    assert incomplete["matched_count"] == 1
    assert incomplete["recall_vs_registered_reference"] == pytest.approx(0.5)
    assert incomplete["precision_vs_registered_reference"] is None
    assert incomplete["f1_vs_registered_reference"] is None
    assert "unmatched_automatic_indeterminate" in set(audit["match_status"])

    complete, _ = compare_automatic_to_registered_reference(
        automatic,
        references,
        mm_per_pixel=0.01,
        matching_tolerance_um=20.0,
        reference_coverage_complete=True,
    )
    assert complete["precision_vs_registered_reference"] == pytest.approx(0.5)
    assert complete["f1_vs_registered_reference"] == pytest.approx(0.5)


def test_matching_scales_by_local_neighbours_instead_of_a_full_distance_matrix() -> None:
    count = 2_000
    positions = np.arange(count, dtype=float) * 10.0
    automatic = pd.DataFrame(
        {
            "defect_id": np.arange(1, count + 1),
            "centroid_x_px": positions,
            "centroid_y_px": np.zeros(count),
            "accepted": True,
        }
    )
    references = pd.DataFrame(
        {
            "reference_id": [f"R{index:04d}" for index in range(count)],
            "x_px": positions + 0.1,
            "y_px": np.zeros(count),
        }
    )
    metrics, audit = compare_automatic_to_registered_reference(
        automatic,
        references,
        mm_per_pixel=0.001,
        matching_tolerance_um=1.5,
        reference_coverage_complete=True,
    )
    assert metrics["matched_count"] == count
    assert metrics["precision_vs_registered_reference"] == pytest.approx(1.0)
    assert metrics["recall_vs_registered_reference"] == pytest.approx(1.0)
    assert len(audit) == count


def test_detail_field_draws_yellow_only_from_supplied_verified_points(tmp_path: Path) -> None:
    scientific = np.full((200, 200), 0.5, dtype=np.float32)
    defects = pd.DataFrame(
        [
            {
                "defect_id": 1,
                "centroid_x_px": 100.0,
                "centroid_y_px": 100.0,
                "bounding_box": "[96,96,104,104]",
                "accepted": True,
            }
        ]
    )
    references = pd.DataFrame([{"reference_id": "R1", "x_px": 100.0, "y_px": 100.0}])
    path = save_xrt_detection_detail_montage(
        scientific,
        defects,
        tmp_path / "field.png",
        mm_per_pixel=0.1,
        field_size_mm=4.0,
        max_fields=1,
        scale_bar_mm=1.0,
        independent_reference_points=references,
        independent_reference_label="registered DIC observations",
    )
    rendered = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert rendered is not None
    b, g, r = (rendered[:, :, channel].astype(np.int16) for channel in range(3))
    yellow = (r > 180) & (g > 180) & (b < 100)
    assert np.count_nonzero(yellow) > 20


def test_pipeline_reference_overlay_is_auditable_and_does_not_change_n(
    generate_synthetic,
    default_config,
    tmp_path: Path,
) -> None:
    generated = generate_synthetic("clean", seed=2468)
    source_path = Path(generated["image_path"])
    truth = pd.read_csv(generated["ground_truth_path"])
    truth_point = truth.loc[truth["should_count"].astype(bool)].iloc[0]
    reference_image = tmp_path / "independent_dic.png"
    assert cv2.imwrite(str(reference_image), np.full((64, 64), 180, dtype=np.uint8))
    reference_csv = tmp_path / "registered.csv"
    pd.DataFrame(
        [
            {
                "reference_id": "DIC-001",
                "x_px": truth_point["center_x_px"],
                "y_px": truth_point["center_y_px"],
                "label": "confirmed_point",
                "reference_method": "DIC",
                "registration_status": "registered",
                "registration_rmse_um": 10.0,
                "source_image_sha256": file_sha256(source_path),
                "reference_image_sha256": file_sha256(reference_image),
                "notes": "identity-registered synthetic fixture",
                "reference_schema_version": "1.0",
            }
        ],
        columns=REFERENCE_COLUMNS,
    ).to_csv(reference_csv, index=False)

    baseline = analyze_image(source_path, tmp_path / "baseline", default_config)
    configured = copy.deepcopy(default_config)
    configured["independent_reference"] = {
        "enabled": True,
        "csv_path": str(reference_csv),
        "reference_image_path": str(reference_image),
        "matching_tolerance_um": 2000.0,
        "coverage_complete": False,
    }
    configured["output"]["generate_heatmap"] = True
    referenced = analyze_image(source_path, tmp_path / "referenced", configured)

    assert referenced.summary["accepted_count"] == baseline.summary["accepted_count"] == 96
    reference_summary = referenced.summary["independent_reference"]
    assert reference_summary["automatic_count_changed_by_reference"] is False
    assert reference_summary["agreement"]["matched_count"] == 1
    assert reference_summary["agreement"]["precision_vs_registered_reference"] is None
    assert referenced.summary["paper_reference_alignment"][
        "independent_reference_data_supplied"
    ] is True
    assert referenced.summary["paper_reference_alignment"][
        "physical_identity_claim"
    ] is False
    heatmap_summary = referenced.summary["spatial_density"]["density_heatmap"]
    assert heatmap_summary["reported_whole_wafer_mean_density_cm2"] == pytest.approx(
        referenced.summary["point_density_cm2"]
    )
    for name in (
        "independent_reference_points.csv",
        "independent_reference_matches.csv",
        "paper_detection_field.png",
        "paper_aligned_result_figure.png",
    ):
        assert (tmp_path / "referenced" / name).is_file()
    overlay = cv2.imread(
        str(tmp_path / "referenced" / "overlay_xrt_red_boxes.png"),
        cv2.IMREAD_COLOR,
    )
    assert overlay is not None
    b, g, r = (overlay[:, :, channel].astype(np.int16) for channel in range(3))
    yellow = (r > 180) & (g > 180) & (b < 100)
    assert np.count_nonzero(yellow) > 0
