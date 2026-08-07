"""Scientific-output tests for the full-resolution XRT detail montage."""

from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from sic_wafer_counter.reporting import (
    generate_html_report,
    save_xrt_detection_detail_montage,
)


ROOT = Path(__file__).resolve().parents[1]


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "defect_id": 2,
                "centroid_x_px": 108,
                "centroid_y_px": 100,
                "bounding_box": "[106,97,111,103]",
                "accepted": True,
            },
            {
                "defect_id": 1,
                "centroid_x_px": 100,
                "centroid_y_px": 100,
                "bounding_box": "[97,97,103,103]",
                "accepted": True,
            },
            {
                "defect_id": 3,
                "centroid_x_px": 90,
                "centroid_y_px": 110,
                "bounding_box": "[87,107,93,113]",
                "accepted": False,
                "rejection_reason": "too_elongated",
            },
        ]
    )


def test_detail_montage_uses_global_full_resolution_crops_and_true_boxes(
    tmp_path: Path,
) -> None:
    scientific = np.full((240, 240), 0.5, dtype=np.float32)
    calls: list[tuple[int, int, int, int]] = []

    def reader(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        calls.append((x0, y0, x1, y1))
        return scientific[y0:y1, x0:x1].copy()

    first = save_xrt_detection_detail_montage(
        np.zeros((60, 60), dtype=np.float32),
        _records(),
        tmp_path / "first.png",
        mm_per_pixel=0.1,
        field_size_mm=4.0,
        max_fields=1,
        scale_bar_mm=1.0,
        source_shape=scientific.shape,
        crop_reader=reader,
    )
    second = save_xrt_detection_detail_montage(
        np.zeros((60, 60), dtype=np.float32),
        _records().sample(frac=1.0, random_state=42),
        tmp_path / "second.png",
        mm_per_pixel=0.1,
        field_size_mm=4.0,
        max_fields=1,
        scale_bar_mm=1.0,
        source_shape=scientific.shape,
        crop_reader=reader,
    )

    # 4 mm / 0.1 mm px^-1 is an exact 40 px full-resolution field. Sorting by
    # candidate ID makes both runs choose the same median-nearest anchor.
    assert calls == [(80, 80, 120, 120), (80, 80, 120, 120)]
    assert first.read_bytes() == second.read_bytes()

    rendered = cv2.imread(str(first), cv2.IMREAD_COLOR)
    assert rendered is not None and rendered.shape == (568, 444, 3)
    b, g, r = (rendered[:, :, channel].astype(np.int16) for channel in range(3))
    red = (r > 180) & (r > g + 80) & (r > b + 80)
    yellow = (r > 180) & (g > 180) & (b < 100)

    # Both accepted candidates' actual segmentation bounds are red. The nearby
    # rejected candidate's true box is inside the same field but stays unmarked.
    assert np.count_nonzero(red[265:342, 182:260]) > 80
    assert np.count_nonzero(red[265:342, 278:344]) > 80
    assert np.count_nonzero(red[370:452, 78:156]) == 0
    assert np.count_nonzero(yellow) == 0

    # A constant 0.5 global-normalized crop remains mid-gray; it is not locally
    # contrast-stretched. The 1 mm scale bar spans about 105 display pixels.
    assert np.all(np.abs(rendered[120, 40].astype(int) - 128) <= 1)
    dark_bar = np.all(rendered[490, 280:430] < 20, axis=1)
    black_positions = np.flatnonzero(dark_bar)
    assert 104 <= int(black_positions[-1] - black_positions[0] + 1) <= 114

    source = inspect.getsource(save_xrt_detection_detail_montage)
    assert all(term not in source for term in ("DIC", "KOH", "TSD"))


def test_detail_montage_rejects_non_scientific_or_invented_geometry(tmp_path: Path) -> None:
    records = _records().iloc[[0]].copy()

    def quantized_reader(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        return np.full((y1 - y0, x1 - x0), 128, dtype=np.uint8)

    with pytest.raises(ValueError, match="global-normalized scientific float"):
        save_xrt_detection_detail_montage(
            None,
            records,
            tmp_path / "quantized.png",
            mm_per_pixel=0.1,
            source_shape=(240, 240),
            crop_reader=quantized_reader,
        )

    missing_bbox = records.drop(columns=["bounding_box"])
    with pytest.raises(ValueError, match="true bounding_box"):
        save_xrt_detection_detail_montage(
            np.full((240, 240), 0.5, dtype=np.float32),
            missing_bbox,
            tmp_path / "invented.png",
            mm_per_pixel=0.1,
        )

    with pytest.raises(ValueError, match="1 to 6"):
        save_xrt_detection_detail_montage(
            np.full((240, 240), 0.5, dtype=np.float32),
            records,
            tmp_path / "too_many.png",
            mm_per_pixel=0.1,
            max_fields=7,
        )


def test_detail_montage_configuration_and_html_caption_are_traceable(tmp_path: Path) -> None:
    source_config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    packaged_config = yaml.safe_load(
        (ROOT / "src" / "sic_wafer_counter" / "resources" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )
    keys = (
        "generate_xrt_detection_detail_montage",
        "xrt_detection_detail_field_size_mm",
        "xrt_detection_detail_max_fields",
        "xrt_detection_detail_scale_bar_mm",
    )
    assert {key: source_config["output"][key] for key in keys} == {
        key: packaged_config["output"][key] for key in keys
    } == {
        "generate_xrt_detection_detail_montage": True,
        "xrt_detection_detail_field_size_mm": 4.0,
        "xrt_detection_detail_max_fields": 6,
        "xrt_detection_detail_scale_bar_mm": 1.0,
    }

    cv2.imwrite(str(tmp_path / "xrt_detection_detail_montage.png"), np.zeros((8, 8), dtype=np.uint8))
    report = generate_html_report(
        {"software_version": "test", "warnings": []},
        tmp_path,
    ).read_text(encoding="utf-8")
    assert "xrt_detection_detail_montage.png" in report
    assert "红框为自动接受点状候选；独立参考未提供" in report
