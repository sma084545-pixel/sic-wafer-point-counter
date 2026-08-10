"""Reference-style local field image/Excel/raw TIFF export tests."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest
import tifffile

from sic_wafer_counter.local_field_export import export_local_fields


def test_local_field_package_exports_all_valid_fields_and_preserves_raw_pixels(
    tmp_path: Path,
) -> None:
    source = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)
    yy, xx = np.ogrid[:100, :100]
    valid_mask = (xx - 50) ** 2 + (yy - 50) ** 2 <= 50**2
    valid_area = float(valid_mask.sum()) * 0.01
    defects = pd.DataFrame(
        [
            {
                "defect_id": 1,
                "centroid_x_px": 40,
                "centroid_y_px": 40,
                "x_mm": -10.0,
                "y_mm": 10.0,
                "bounding_box": json.dumps([37, 37, 44, 44]),
                "accepted": True,
                "rejection_reason": "",
                "rule_accepted": False,
                "rule_rejection_reason": "too_elongated",
                "classifier_applied": True,
                "classifier_probability": 0.91,
                "classifier_decision": "target",
                "decision_basis": "trained_candidate_classifier",
                "pixel_model_applied": True,
                "pixel_model_probability_mean": 0.87,
                "pixel_model_probability_max": 0.97,
                "pixel_model_threshold": 0.5,
                "pixel_model_sha256": "b" * 64,
                "pixel_segmentation_decision": "foreground_component",
                "area_px": 20,
                "equivalent_diameter_um": 5000.0,
            },
            {
                "defect_id": 2,
                "centroid_x_px": 60,
                "centroid_y_px": 60,
                "x_mm": 10.0,
                "y_mm": -10.0,
                "bounding_box": json.dumps([57, 57, 64, 64]),
                "accepted": False,
                "rejection_reason": "classifier_artifact",
                "rule_accepted": True,
                "rule_rejection_reason": "",
                "classifier_applied": True,
                "classifier_probability": 0.08,
                "classifier_decision": "artifact",
                "decision_basis": "trained_candidate_classifier",
                "pixel_model_applied": True,
                "pixel_model_probability_mean": 0.62,
                "pixel_model_probability_max": 0.91,
                "pixel_model_threshold": 0.5,
                "pixel_model_sha256": "b" * 64,
                "pixel_segmentation_decision": "foreground_component",
                "area_px": 22,
                "equivalent_diameter_um": 5200.0,
            },
        ]
    )

    def raw_reader(x: int, y: int, width: int, height: int) -> np.ndarray:
        return source[y : y + height, x : x + width]

    def display_reader(x: int, y: int, width: int, height: int) -> np.ndarray:
        crop = source[y : y + height, x : x + width].astype(np.float32)
        return crop / float(source.max())

    result = export_local_fields(
        defects,
        tmp_path,
        {
            "input_file_name": "reference_like.tif",
            "wafer_diameter_mm": 100.0,
            "valid_analysis_area_cm2": valid_area,
            "accepted_count": 1,
            "rejected_count": 1,
            "point_density_cm2": 1.0 / valid_area,
            "counting_uncertainty_cm2": 1.0 / valid_area,
            "decision_basis": "trained_candidate_classifier",
            "pixel_classifier": {
                "status": "applied",
                "model_sha256": "b" * 64,
                "probability_threshold": 0.5,
                "validation": {"status": "not_evaluated_yet"},
                "training_sources": [
                    {
                        "image_sha256": "c" * 64,
                        "wafer_id": "wafer-A",
                        "split": "calibration",
                        "roi_xywh": [0, 0, 100, 100],
                    }
                ],
            },
            "candidate_classifier": {"model_sha256": "a" * 64},
            "real_annotation_validation_status": "not validated on real SiC data",
            "software_version": "test",
        },
        source_shape=source.shape,
        center_px=(50.0, 50.0),
        mm_per_pixel=1.0,
        valid_analysis_mask=valid_mask,
        raw_reader=raw_reader,
        display_reader=display_reader,
        field_size_mm=25.0,
    )

    assert result.field_count == 16
    assert result.candidate_count == 2
    assert result.accepted_count == 1
    assert result.mask_area_cm2 == pytest.approx(valid_area)
    assert result.primary_area_relative_error == pytest.approx(0.0, abs=1e-15)
    assert result.global_workbook.name == "00_global_overview.xlsx"
    assert zipfile.is_zipfile(result.global_workbook)
    with zipfile.ZipFile(result.global_workbook) as workbook:
        workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")
        assert "全局概览" in workbook_xml
        assert "视场密度矩阵" in workbook_xml
        assert "局部视场索引" in workbook_xml
        assert "全部候选位置" in workbook_xml
        overview_xml = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
        assert "最终有效分析面积 S" in overview_xml
        assert "trained_candidate_classifier" in overview_xml
        assert "分类器模型 SHA-256" in overview_xml
        assert "像素模型 SHA-256" in overview_xml
        assert "b" * 64 in overview_xml
        assert "not_evaluated_yet" in overview_xml
        assert "未提供独立 DIC/KOH" in overview_xml

    field_dirs = sorted(path for path in (tmp_path / "local_fields").iterdir() if path.is_dir())
    assert len(field_dirs) == 16
    assert all((folder / "01_marked.png").is_file() for folder in field_dirs)
    assert all(zipfile.is_zipfile(folder / "02_positions.xlsx") for folder in field_dirs)
    assert all((folder / "03_raw_original.tif").is_file() for folder in field_dirs)

    marked = [cv2.imread(str(folder / "01_marked.png")) for folder in field_dirs]
    red_pixels = sum(int(np.count_nonzero(image[:, :, 2] > image[:, :, 1] + 80)) for image in marked)
    green_pixels = sum(int(np.count_nonzero(image[:, :, 1] > image[:, :, 2] + 50)) for image in marked)
    assert red_pixels > 10
    assert green_pixels > 10

    candidate_one_field = next(folder for folder in field_dirs if "X_-12.500_Y_12.500" in folder.name)
    with zipfile.ZipFile(candidate_one_field / "02_positions.xlsx") as workbook:
        field_overview = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
        candidate_sheet = workbook.read("xl/worksheets/sheet2.xml").decode("utf-8")
        assert "trained_candidate_classifier" in field_overview
        assert "classifier_probability" in candidate_sheet
        assert "rule_rejection_reason" in candidate_sheet
        assert "pixel_model_probability_mean" in candidate_sheet
        assert "pixel_model_sha256" in candidate_sheet
    raw = tifffile.imread(candidate_one_field / "03_raw_original.tif")
    assert np.array_equal(raw, source[25:50, 25:50])


def test_local_field_package_rejects_submillimetre_grid_explosion(tmp_path: Path) -> None:
    source = np.zeros((10, 10), dtype=np.uint8)
    defects = pd.DataFrame(
        [{"defect_id": 1, "centroid_x_px": 5, "centroid_y_px": 5, "x_mm": 0, "y_mm": 0, "accepted": True}]
    )

    def reader(x: int, y: int, width: int, height: int) -> np.ndarray:
        return source[y : y + height, x : x + width]

    try:
        export_local_fields(
            defects,
            tmp_path,
            {"wafer_diameter_mm": 100.0},
            source_shape=source.shape,
            center_px=(5.0, 5.0),
            mm_per_pixel=10.0,
            valid_analysis_mask=np.ones_like(source, dtype=bool),
            raw_reader=reader,
            display_reader=reader,
            field_size_mm=0.5,
        )
    except ValueError as exc:
        assert "between 1 mm" in str(exc)
    else:
        raise AssertionError("submillimetre local field grid should be rejected")
