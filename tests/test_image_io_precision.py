"""Scientific grayscale precision and global-normalization regression tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from sic_wafer_counter.image_io import (
    load_image,
    normalize_to_float32,
    normalize_to_uint8,
)
from sic_wafer_counter.visualization import save_grayscale_image


def _config(*, threshold: int = 1_000_000) -> dict[str, object]:
    return {
        "io": {
            "preview_max_size": 64,
            "normalization_low_percentile": 0.0,
            "normalization_high_percentile": 100.0,
            "large_image_threshold_pixels": threshold,
            "respect_tiff_photometric": True,
        }
    }


@pytest.mark.parametrize(
    ("dtype", "image"),
    [
        (np.uint8, np.array([[0, 64], [128, 255]], dtype=np.uint8)),
        (np.uint16, np.array([[0, 16_000], [32_000, 65_535]], dtype=np.uint16)),
        (np.float32, np.array([[0.0, 0.2], [0.7, 1.0]], dtype=np.float32)),
    ],
)
def test_all_supported_analysis_inputs_are_float32(
    dtype: np.dtype, image: np.ndarray, tmp_path: Path
) -> None:
    path = tmp_path / f"input_{np.dtype(dtype).name}.tif"
    tifffile.imwrite(path, image)
    with load_image(path, _config(), lazy=False) as data:
        full = data.require_full()
        assert full.dtype == np.float32
        assert full.min() == pytest.approx(0.0)
        assert full.max() == pytest.approx(1.0)
        assert data.preview.dtype == np.uint8
        assert data.metadata.analysis_dtype == "float32"
        assert data.metadata.analysis_quantized_to_uint8 is False


def test_uint16_low_contrast_is_retained_before_display_quantization(tmp_path: Path) -> None:
    """Two values that collapse to one display byte stay distinct scientifically."""

    image = np.full((32, 32), 30_000, dtype=np.uint16)
    image[0, 0], image[0, 1] = 0, 65_535
    image[12, 12], image[12, 13] = 30_000, 30_001
    path = tmp_path / "low_contrast_uint16.tif"
    tifffile.imwrite(path, image)
    with load_image(path, _config(), lazy=False) as data:
        full = data.require_full()
        assert full[12, 12] != full[12, 13]
        display = normalize_to_uint8(
            image,
            data.metadata.normalization_low_value,
            data.metadata.normalization_high_value,
        )
        assert display[12, 12] == display[12, 13]


def test_full_and_tile_reads_share_one_float_normalization_window(tmp_path: Path) -> None:
    image = np.arange(11 * 13, dtype=np.uint16).reshape(11, 13) * 211
    path = tmp_path / "tile_consistency.tif"
    tifffile.imwrite(path, image)
    with load_image(path, _config(), lazy=False) as data:
        full = data.require_full()
        assert full.dtype == np.float32
        covered = np.zeros(full.shape, dtype=bool)
        for tile in data.iter_tiles(tile_size=4, overlap=1):
            expected = full[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
            assert tile.image.dtype == np.float32
            assert np.array_equal(tile.image, expected)
            covered[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] = True
        assert covered.all()


def test_white_is_zero_is_applied_once_to_analysis_path(tmp_path: Path) -> None:
    path = tmp_path / "white_is_zero.tif"
    tifffile.imwrite(
        path,
        np.array([[0, 65_535], [0, 65_535]], dtype=np.uint16),
        photometric="miniswhite",
    )
    with load_image(path, _config(), lazy=False) as data:
        full = data.require_full()
        assert data.metadata.white_is_zero is True
        assert full[0, 0] == pytest.approx(1.0)
        assert full[0, 1] == pytest.approx(0.0)


def test_normalization_never_mutates_the_input_and_display_remains_uint8(tmp_path: Path) -> None:
    image = np.array([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32)
    before = image.copy()
    analysis = normalize_to_float32(image, 0.0, 1.0)
    assert np.array_equal(image, before, equal_nan=True)
    output = save_grayscale_image(analysis, tmp_path / "preview.png", max_size=None)
    display = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert display is not None
    assert display.dtype == np.uint8


def test_lazy_path_does_not_materialize_full_analysis_array(tmp_path: Path) -> None:
    image = np.arange(80 * 80, dtype=np.uint16).reshape(80, 80)
    path = tmp_path / "lazy_uint16.tif"
    tifffile.imwrite(path, image)
    with load_image(path, _config(threshold=1), lazy=None) as data:
        assert data.metadata.lazy is True
        assert data.gray is None
        tile = next(data.iter_tiles(tile_size=32, overlap=4))
        assert tile.image.dtype == np.float32
        assert data.gray is None
