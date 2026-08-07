"""Memory-conscious grayscale image input for wafer topographs.

TIFF/BigTIFF files are opened with :mod:`tifffile` first.  Uncompressed,
contiguous TIFF data are exposed through a read-only memory map; compressed
data use optional pyvips random access when available, otherwise tifffile must
decode the full array and the limitation is recorded in the metadata.  Large
images can remain preview-only and be normalized one overlapping tile at a
time using a single set of preview-derived percentile limits.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from .utils import validate_percentiles


LOGGER = logging.getLogger(__name__)
TIFF_SUFFIXES = {".tif", ".tiff", ".btf", ".bigtif", ".bigtiff"}


class ImageReadError(RuntimeError):
    """Raised when an image cannot be decoded into a supported grayscale form."""


@dataclasses.dataclass(slots=True)
class ImageMetadata:
    """Auditable metadata describing image loading and normalization."""

    path: str
    width: int
    height: int
    shape: tuple[int, ...]
    dtype: str
    channels: int
    loader: str
    is_tiff: bool
    is_bigtiff: bool
    photometric: str | None
    white_is_zero: bool
    original_min: float
    original_max: float
    extrema_exact: bool
    normalization_method: str
    normalization_low_percentile: float
    normalization_high_percentile: float
    normalization_low_value: float
    normalization_high_value: float
    analysis_dtype: str = "float32"
    analysis_quantized_to_uint8: bool = False
    low_clipped_fraction: float = 0.0
    high_clipped_fraction: float = 0.0
    clipping_fraction_estimated: bool = True
    preview_width: int = 0
    preview_height: int = 0
    preview_scale_x: float = 1.0
    preview_scale_y: float = 1.0
    lazy: bool = False
    random_access: bool = False
    warnings: list[str] = dataclasses.field(default_factory=list)
    limitations: list[str] = dataclasses.field(default_factory=list)

    @property
    def pixel_count(self) -> int:
        """Number of spatial pixels in the source image."""

        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        """Return plain JSON-compatible metadata."""

        result = dataclasses.asdict(self)
        result["shape"] = list(self.shape)
        result["pixel_count"] = self.pixel_count
        return result


@dataclasses.dataclass(slots=True)
class ImageTile:
    """An overlapping normalized tile with global and non-overlap core bounds.

    ``x``/``y`` locate the full tile (including halo) in source coordinates.
    Detection may use all pixels, then retain only detections whose centroids
    lie in ``core_*``.  This prevents cut objects without double counting.
    """

    image: np.ndarray
    x: int
    y: int
    width: int
    height: int
    core_x: int
    core_y: int
    core_width: int
    core_height: int

    @property
    def origin_x(self) -> int:
        return self.x

    @property
    def origin_y(self) -> int:
        return self.y

    @property
    def global_bounds(self) -> tuple[int, int, int, int]:
        """Return full tile bounds as ``(x0, y0, x1, y1)``."""

        return self.x, self.y, self.x + self.width, self.y + self.height

    @property
    def core_bounds(self) -> tuple[int, int, int, int]:
        """Return unique core bounds as ``(x0, y0, x1, y1)``."""

        return (
            self.core_x,
            self.core_y,
            self.core_x + self.core_width,
            self.core_y + self.core_height,
        )

    @property
    def core_slice(self) -> tuple[slice, slice]:
        """Slice selecting the unique core from ``image``."""

        local_x = self.core_x - self.x
        local_y = self.core_y - self.y
        return (
            slice(local_y, local_y + self.core_height),
            slice(local_x, local_x + self.core_width),
        )

    @property
    def core_image(self) -> np.ndarray:
        """View of the non-overlapping core region."""

        return self.image[self.core_slice]


@dataclasses.dataclass(slots=True)
class ImageData:
    """Loaded image result; ``gray`` is ``None`` for a lazy large image."""

    gray: np.ndarray | None
    preview: np.ndarray
    metadata: ImageMetadata
    source: "ImageSource"

    @property
    def image(self) -> np.ndarray | None:
        """Compatibility alias for ``gray``."""

        return self.gray

    @property
    def shape(self) -> tuple[int, int]:
        return self.metadata.height, self.metadata.width

    @property
    def preview_scale_x(self) -> float:
        return self.metadata.preview_scale_x

    @property
    def preview_scale_y(self) -> float:
        return self.metadata.preview_scale_y

    def require_full(self) -> np.ndarray:
        """Read and cache the complete normalized image on explicit request."""

        if self.gray is None:
            LOGGER.warning(
                "Materializing full normalized image (%d x %d); tile processing is safer",
                self.metadata.width,
                self.metadata.height,
            )
            self.gray = self.source.read_full(normalize=True)
            self.metadata.lazy = False
        return self.gray

    def iter_tiles(self, tile_size: int = 2048, overlap: int = 128) -> Iterator[ImageTile]:
        """Yield normalized tiles from the underlying image source."""

        yield from self.source.iter_tiles(tile_size=tile_size, overlap=overlap)

    def close(self) -> None:
        """Release decoder objects; numpy memory maps remain OS-managed views."""

        self.source.close()

    def __enter__(self) -> "ImageData":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _finite_values(array: np.ndarray) -> np.ndarray:
    flat = np.asarray(array).reshape(-1)
    if np.issubdtype(flat.dtype, np.floating):
        return flat[np.isfinite(flat)]
    return flat


def _grayscale(array: np.ndarray, *, color_order: str = "RGB") -> np.ndarray:
    """Return a 2-D grayscale view/array while preserving numerical range."""

    data = np.asarray(array)
    data = np.squeeze(data)
    if data.ndim == 2:
        return data
    if data.ndim != 3 or data.shape[-1] not in (2, 3, 4):
        raise ImageReadError(f"Unsupported image region shape for grayscale: {data.shape}")
    if data.shape[-1] == 2:
        # Common gray + alpha layout.  Alpha is deliberately ignored.
        return data[..., 0]
    rgb = data[..., :3]
    if color_order.upper() == "BGR":
        rgb = rgb[..., ::-1]
    # Float calculation avoids OpenCV's restrictions on endian/dtype variants.
    gray = (
        rgb[..., 0].astype(np.float32) * 0.2126
        + rgb[..., 1].astype(np.float32) * 0.7152
        + rgb[..., 2].astype(np.float32) * 0.0722
    )
    return gray


def normalize_to_float32(
    image: np.ndarray,
    low_value: float,
    high_value: float,
    *,
    white_is_zero: bool = False,
) -> np.ndarray:
    """Percentile-window a grayscale array to scientific ``float32`` ``[0, 1]``.

    The source is never modified.  Full-frame and tiled analysis reads use the
    same global low/high window, and WhiteIsZero inversion happens exactly once
    at this boundary.
    """

    low, high = float(low_value), float(high_value)
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        raise ValueError(f"Normalization limits must be finite and ascending: {low}, {high}")
    values = np.asarray(image, dtype=np.float32).copy()
    np.nan_to_num(values, copy=False, nan=low, posinf=high, neginf=low)
    values -= np.float32(low)
    values *= np.float32(1.0 / (high - low))
    np.clip(values, 0.0, 1.0, out=values)
    if white_is_zero:
        np.subtract(1.0, values, out=values)
    return np.ascontiguousarray(values, dtype=np.float32)


def normalize_to_uint8(
    image: np.ndarray,
    low_value: float,
    high_value: float,
    *,
    white_is_zero: bool = False,
) -> np.ndarray:
    """Map an image to uint8 for preview/display only, preserving input values."""

    values = normalize_to_float32(
        image, low_value, high_value, white_is_zero=white_is_zero
    )
    return np.rint(values * 255.0).astype(np.uint8)


def make_preview(image: np.ndarray, max_size: int = 2000) -> tuple[np.ndarray, float, float]:
    """Resize a 2-D image so its longest dimension is at most ``max_size``.

    Returns ``(preview, source_pixels_per_preview_x,
    source_pixels_per_preview_y)`` for exact coordinate mapping.
    """

    gray = np.asarray(image)
    if gray.ndim != 2:
        raise ValueError(f"make_preview expects a 2-D image, got {gray.shape}")
    if int(max_size) <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")
    height, width = gray.shape
    factor = min(1.0, float(max_size) / max(height, width))
    if factor >= 1.0:
        preview = np.array(gray, copy=True)
    else:
        target = (
            max(1, int(round(width * factor))),
            max(1, int(round(height * factor))),
        )
        preview = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)
    scale_x = width / preview.shape[1]
    scale_y = height / preview.shape[0]
    return preview, scale_x, scale_y


class ImageSource:
    """Random-access source with global percentile normalization.

    The class is safe to use as a context manager.  TIFF memory maps are always
    opened with ``mode='r'`` so a read-only input directory is sufficient.
    """

    _VIPS_DTYPES: dict[str, np.dtype[Any]] = {
        "uchar": np.dtype("uint8"),
        "char": np.dtype("int8"),
        "ushort": np.dtype("uint16"),
        "short": np.dtype("int16"),
        "uint": np.dtype("uint32"),
        "int": np.dtype("int32"),
        "float": np.dtype("float32"),
        "double": np.dtype("float64"),
    }

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        normalization_low_percentile: float = 1.0,
        normalization_high_percentile: float = 99.0,
        normalization_sample_max_size: int = 2000,
        respect_tiff_photometric: bool = True,
        prefer_pyvips: bool = True,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise ImageReadError(f"Input image does not exist: {self.path}")
        self.low_percentile, self.high_percentile = validate_percentiles(
            normalization_low_percentile, normalization_high_percentile
        )
        if normalization_sample_max_size <= 0:
            raise ValueError("normalization_sample_max_size must be positive")
        self.sample_max_size = int(normalization_sample_max_size)
        self.respect_tiff_photometric = bool(respect_tiff_photometric)
        self.prefer_pyvips = bool(prefer_pyvips)
        self._array: np.ndarray | None = None
        self._vips_image: Any | None = None
        self._color_order = "RGB"
        self._warnings: list[str] = []
        self._limitations: list[str] = []
        self._loader = ""
        self._is_tiff = self.path.suffix.lower() in TIFF_SUFFIXES
        self._is_bigtiff = False
        self._photometric: str | None = None
        self._white_is_zero = False
        self._random_access = False
        self._source_shape: tuple[int, ...] = ()
        self._dtype = np.dtype("uint8")
        self._channels = 1
        self._height = 0
        self._width = 0

        try:
            if self._is_tiff:
                self._open_tiff()
            else:
                self._open_raster()
            # Percentiles are measured on directly decimated source samples,
            # never on interpolation results (which would bias scientific TIFF
            # histograms and previously shifted the real sample's 99th percentile).
            sample = self._raw_normalization_sample(self.sample_max_size)
            finite = _finite_values(sample)
            if finite.size == 0:
                raise ImageReadError("Image contains no finite grayscale samples")
            low_value, high_value = np.percentile(
                finite, [self.low_percentile, self.high_percentile]
            )
            low_float, high_float = float(low_value), float(high_value)
            if not high_float > low_float:
                sample_min = float(np.min(finite))
                sample_max = float(np.max(finite))
                if sample_max > sample_min:
                    low_float, high_float = sample_min, sample_max
                    self._warnings.append(
                        "Configured percentile window collapsed; sample min/max were used"
                    )
                else:
                    # A uniform image is valid input but cannot support detection.
                    low_float, high_float = sample_min, sample_min + 1.0
                    self._warnings.append(
                        "Image sample is uniform; normalization uses a synthetic unit range"
                    )
            self._normalization_low_value = low_float
            self._normalization_high_value = high_float
            original_min, original_max, extrema_exact = self._calculate_extrema(sample)
        except Exception:
            self.close()
            raise

        self.metadata = ImageMetadata(
            path=str(self.path),
            width=self._width,
            height=self._height,
            shape=self._source_shape,
            dtype=str(self._dtype),
            channels=self._channels,
            loader=self._loader,
            is_tiff=self._is_tiff,
            is_bigtiff=self._is_bigtiff,
            photometric=self._photometric,
            white_is_zero=self._white_is_zero,
            original_min=original_min,
            original_max=original_max,
            extrema_exact=extrema_exact,
            normalization_method="global_percentile_linear_to_float32",
            normalization_low_percentile=self.low_percentile,
            normalization_high_percentile=self.high_percentile,
            normalization_low_value=self._normalization_low_value,
            normalization_high_value=self._normalization_high_value,
            analysis_dtype="float32",
            analysis_quantized_to_uint8=False,
            low_clipped_fraction=float(np.mean(finite <= low_float)),
            high_clipped_fraction=float(np.mean(finite >= high_float)),
            clipping_fraction_estimated=True,
            random_access=self._random_access,
            warnings=self._warnings,
            limitations=self._limitations,
        )
        LOGGER.info(
            "Opened %s: %dx%d %s via %s; normalization %.3g..%.3g%s",
            self.path.name,
            self._width,
            self._height,
            self._dtype,
            self._loader,
            self._normalization_low_value,
            self._normalization_high_value,
            " (WhiteIsZero inverted)" if self._white_is_zero else "",
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self._height, self._width

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _open_tiff(self) -> None:
        try:
            with tifffile.TiffFile(self.path) as tif:
                if not tif.series:
                    raise ImageReadError("TIFF has no image series")
                series = tif.series[0]
                self._source_shape = tuple(int(value) for value in series.shape)
                self._dtype = np.dtype(series.dtype)
                self._is_bigtiff = bool(tif.is_bigtiff)
                page = tif.pages[0]
                photometric = getattr(page, "photometric", None)
                self._photometric = (
                    getattr(photometric, "name", str(photometric))
                    if photometric is not None
                    else None
                )
                normalized_name = (self._photometric or "").upper()
                self._white_is_zero = self.respect_tiff_photometric and (
                    "MINISWHITE" in normalized_name
                    or "WHITEISZERO" in normalized_name
                    or normalized_name == "0"
                )
                compression = getattr(page, "compression", None)
                compression_name = getattr(compression, "name", str(compression))
                rows_per_strip = getattr(page, "rowsperstrip", None)
                if rows_per_strip and int(rows_per_strip) >= int(page.imagelength):
                    if str(compression_name).upper() not in {"NONE", "1"}:
                        self._limitations.append(
                            "Compressed single-strip TIFF may require decoding a full strip "
                            "for each requested tile; true bounded-memory tile reads are not guaranteed"
                        )
        except (OSError, ValueError, tifffile.TiffFileError) as exc:
            raise ImageReadError(f"Could not inspect TIFF {self.path}: {exc}") from exc

        try:
            # Explicit read-only mode is essential for read-only source directories.
            mapped = tifffile.memmap(self.path, mode="r", series=0)
            self._set_numpy_backend(mapped, loader="tifffile.memmap(read-only)")
            self._random_access = True
            return
        except (OSError, ValueError, TypeError, tifffile.TiffFileError) as exc:
            self._warnings.append(f"TIFF memory mapping unavailable: {exc}")

        if self.prefer_pyvips and self._try_open_pyvips():
            self._random_access = True
            self._limitations.append(
                "pyvips supplies region access, but memory use still depends on TIFF strip/tile layout"
            )
            return

        try:
            decoded = tifffile.imread(self.path, series=0)
        except (OSError, ValueError, tifffile.TiffFileError) as exc:
            raise ImageReadError(f"Could not decode TIFF {self.path}: {exc}") from exc
        self._set_numpy_backend(decoded, loader="tifffile.asarray(full fallback)")
        self._random_access = True
        self._limitations.append(
            "TIFF was not memory-mappable and pyvips was unavailable; the full decoded array is resident in memory"
        )

    def _try_open_pyvips(self) -> bool:
        try:
            import pyvips  # type: ignore[import-not-found]
        except (ImportError, OSError) as exc:
            self._warnings.append(f"Optional pyvips backend unavailable: {exc}")
            return False
        try:
            image = pyvips.Image.new_from_file(str(self.path), access="random")
        except Exception as exc:  # pyvips raises several backend-specific classes.
            self._warnings.append(f"pyvips could not open TIFF: {exc}")
            return False
        if image.format not in self._VIPS_DTYPES:
            self._warnings.append(f"Unsupported pyvips pixel format: {image.format}")
            return False
        self._vips_image = image
        self._array = None
        self._height, self._width = int(image.height), int(image.width)
        self._channels = int(image.bands)
        self._dtype = self._VIPS_DTYPES[image.format]
        self._source_shape = (
            (self._height, self._width)
            if self._channels == 1
            else (self._height, self._width, self._channels)
        )
        self._loader = "pyvips(random access)"
        self._color_order = "RGB"
        return True

    def _open_raster(self) -> None:
        image = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            self._color_order = "BGR"
            self._set_numpy_backend(image, loader="opencv")
            self._random_access = True
            self._limitations.append(
                "PNG/JPEG/BMP input is fully decoded in memory; tile iteration limits processing copies but not source residency"
            )
            return
        try:
            from PIL import Image

            with Image.open(self.path) as pil_image:
                image = np.asarray(pil_image)
            self._color_order = "RGB"
            self._set_numpy_backend(image, loader="Pillow fallback")
            self._random_access = True
            self._limitations.append(
                "This raster input is fully decoded in memory; tile iteration limits processing copies but not source residency"
            )
        except (OSError, ValueError) as exc:
            raise ImageReadError(f"Could not decode raster image {self.path}: {exc}") from exc

    def _set_numpy_backend(self, array: np.ndarray, *, loader: str) -> None:
        data = np.asarray(array)
        original_shape = tuple(int(value) for value in data.shape)
        data = np.squeeze(data)
        while data.ndim > 3:
            self._warnings.append(
                f"Multidimensional image {data.shape}: using the first non-spatial plane"
            )
            data = data[0]
        if data.ndim == 3 and data.shape[-1] not in (2, 3, 4):
            if data.shape[0] in (2, 3, 4):
                data = np.moveaxis(data, 0, -1)
            else:
                self._warnings.append(
                    f"Image shape {data.shape} is not a recognized colour layout; using first plane"
                )
                data = data[0]
        if data.ndim not in (2, 3):
            raise ImageReadError(f"Unsupported image dimensions: {original_shape}")
        self._array = data
        self._vips_image = None
        self._height, self._width = int(data.shape[0]), int(data.shape[1])
        self._channels = 1 if data.ndim == 2 else int(data.shape[-1])
        self._dtype = data.dtype
        self._source_shape = original_shape
        self._loader = loader

    def _vips_to_numpy(self, image: Any) -> np.ndarray:
        dtype = self._VIPS_DTYPES.get(image.format)
        if dtype is None:
            raise ImageReadError(f"Unsupported pyvips pixel format: {image.format}")
        raw = image.write_to_memory()
        array = np.frombuffer(raw, dtype=dtype)
        shape = (int(image.height), int(image.width), int(image.bands))
        return array.reshape(shape)

    def _raw_preview(self, max_size: int) -> np.ndarray:
        factor = min(1.0, float(max_size) / max(self._height, self._width))
        if self._array is not None:
            # Decimate before conversion/copy to avoid materializing a full second image.
            step = max(1, int(math.floor(1.0 / max(factor, np.finfo(float).eps))))
            sampled = self._array[::step, ::step, ...] if self._array.ndim == 3 else self._array[::step, ::step]
            gray = _grayscale(sampled, color_order=self._color_order)
            # OpenCV treats byte-swapped integer buffers as native-endian and
            # silently produces corrupt values.  Big-endian scientific TIFFs
            # therefore need an explicit native-endian copy before resize.
            if not gray.dtype.isnative:
                gray = gray.astype(gray.dtype.newbyteorder("="), copy=True)
            if max(gray.shape) > max_size:
                target_factor = max_size / max(gray.shape)
                target = (
                    max(1, int(round(gray.shape[1] * target_factor))),
                    max(1, int(round(gray.shape[0] * target_factor))),
                )
                gray = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)
            return np.asarray(gray)
        if self._vips_image is None:
            raise ImageReadError("Image source is closed")
        image = self._vips_image
        if factor < 1.0:
            image = image.resize(factor)
        return _grayscale(self._vips_to_numpy(image), color_order="RGB")

    def _raw_normalization_sample(self, max_size: int) -> np.ndarray:
        """Return a bounded, uninterpolated sample for percentile calibration."""

        if self._array is not None:
            step = max(1, int(math.ceil(max(self._height, self._width) / max_size)))
            sampled = (
                self._array[::step, ::step, ...]
                if self._array.ndim == 3
                else self._array[::step, ::step]
            )
            gray = _grayscale(sampled, color_order=self._color_order)
            if not gray.dtype.isnative:
                gray = gray.astype(gray.dtype.newbyteorder("="), copy=True)
            return np.asarray(gray)
        # pyvips has no cheap strided view; its shrink operation is still the
        # bounded-memory choice, and the sampled-extrema flag records this path.
        return self._raw_preview(max_size)

    def _calculate_extrema(self, sample: np.ndarray) -> tuple[float, float, bool]:
        # Numpy/memmap backends can be scanned exactly in bounded row chunks.
        if self._array is not None:
            minimum = math.inf
            maximum = -math.inf
            for y in range(0, self._height, 2048):
                region = self.read_region_raw(0, y, self._width, min(2048, self._height - y))
                finite = _finite_values(region)
                if finite.size:
                    minimum = min(minimum, float(np.min(finite)))
                    maximum = max(maximum, float(np.max(finite)))
            if math.isfinite(minimum) and math.isfinite(maximum):
                return minimum, maximum, True
        # Calling min/max through pyvips can trigger a full decode; use the same
        # preview sample and state explicitly that the extrema are sampled.
        finite = _finite_values(sample)
        self._warnings.append("Original min/max are preview-sampled, not exact")
        return float(np.min(finite)), float(np.max(finite)), False

    def read_region_raw(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Read an unnormalized grayscale region using global pixel coordinates."""

        x0, y0, region_width, region_height = map(int, (x, y, width, height))
        if region_width <= 0 or region_height <= 0:
            raise ValueError("Region width and height must be positive")
        if x0 < 0 or y0 < 0 or x0 + region_width > self._width or y0 + region_height > self._height:
            raise ValueError(
                f"Region {(x0, y0, region_width, region_height)} is outside image "
                f"bounds {(self._width, self._height)}"
            )
        if self._array is not None:
            region = self._array[y0 : y0 + region_height, x0 : x0 + region_width, ...]
            return _grayscale(region, color_order=self._color_order)
        if self._vips_image is not None:
            crop = self._vips_image.crop(x0, y0, region_width, region_height)
            return _grayscale(self._vips_to_numpy(crop), color_order="RGB")
        raise ImageReadError("Image source is closed")

    def read_region(
        self, x: int, y: int, width: int, height: int, *, normalize: bool = True
    ) -> np.ndarray:
        """Read a grayscale region, normalized to scientific float32 by default."""

        raw = self.read_region_raw(x, y, width, height)
        if not normalize:
            return raw
        return normalize_to_float32(
            raw,
            self._normalization_low_value,
            self._normalization_high_value,
            white_is_zero=self._white_is_zero,
        )

    def read_full(self, *, normalize: bool = True) -> np.ndarray:
        """Materialize the complete grayscale image (prefer tiles for large data)."""

        return self.read_region(0, 0, self._width, self._height, normalize=normalize)

    def get_preview(self, max_size: int = 2000) -> np.ndarray:
        """Return a normalized low-resolution preview and update scale metadata."""

        if int(max_size) <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        raw = self._raw_preview(int(max_size))
        preview = normalize_to_uint8(
            raw,
            self._normalization_low_value,
            self._normalization_high_value,
            white_is_zero=self._white_is_zero,
        )
        self.metadata.preview_height, self.metadata.preview_width = preview.shape
        self.metadata.preview_scale_x = self._width / preview.shape[1]
        self.metadata.preview_scale_y = self._height / preview.shape[0]
        return preview

    def iter_tiles(
        self,
        tile_size: int = 2048,
        overlap: int = 128,
        *,
        normalize: bool = True,
    ) -> Iterator[ImageTile]:
        """Yield halo-overlapped tiles with unique, non-overlapping cores."""

        size, halo = int(tile_size), int(overlap)
        if size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if halo < 0 or halo >= size:
            raise ValueError(
                f"tile_overlap must satisfy 0 <= overlap < tile_size, got {overlap}, {tile_size}"
            )
        for core_y in range(0, self._height, size):
            core_height = min(size, self._height - core_y)
            for core_x in range(0, self._width, size):
                core_width = min(size, self._width - core_x)
                x0 = max(0, core_x - halo)
                y0 = max(0, core_y - halo)
                x1 = min(self._width, core_x + core_width + halo)
                y1 = min(self._height, core_y + core_height + halo)
                tile = self.read_region(
                    x0, y0, x1 - x0, y1 - y0, normalize=normalize
                )
                yield ImageTile(
                    image=tile,
                    x=x0,
                    y=y0,
                    width=x1 - x0,
                    height=y1 - y0,
                    core_x=core_x,
                    core_y=core_y,
                    core_width=core_width,
                    core_height=core_height,
                )

    def close(self) -> None:
        """Release decoder references."""

        self._vips_image = None
        # Do not explicitly close numpy.memmap's private mmap while exported
        # views may exist.  Dropping our reference is safe and deterministic.
        self._array = None

    def __enter__(self) -> "ImageSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _io_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not config:
        return {}
    nested = config.get("io")
    return nested if isinstance(nested, Mapping) else config


def load_image(
    path: str | os.PathLike[str],
    config: Mapping[str, Any] | None = None,
    *,
    preview_max_size: int | None = None,
    normalization_low_percentile: float | None = None,
    normalization_high_percentile: float | None = None,
    respect_tiff_photometric: bool | None = None,
    large_image_threshold_pixels: int | None = None,
    lazy: bool | None = None,
    prefer_pyvips: bool = True,
) -> ImageData:
    """Open an image and return normalized full data or a lazy preview.

    When ``lazy`` is ``None`` (the default), images larger than the configured
    threshold remain preview-only.  Their tiles are normalized with the same
    global preview percentile values, ensuring repeatable inter-tile contrast.
    """

    options = _io_config(config)
    max_size = int(
        preview_max_size
        if preview_max_size is not None
        else options.get("preview_max_size", 2000)
    )
    low = float(
        normalization_low_percentile
        if normalization_low_percentile is not None
        else options.get("normalization_low_percentile", 1.0)
    )
    high = float(
        normalization_high_percentile
        if normalization_high_percentile is not None
        else options.get("normalization_high_percentile", 99.0)
    )
    respect = bool(
        respect_tiff_photometric
        if respect_tiff_photometric is not None
        else options.get("respect_tiff_photometric", True)
    )
    threshold = int(
        large_image_threshold_pixels
        if large_image_threshold_pixels is not None
        else options.get("large_image_threshold_pixels", 25_000_000)
    )
    if threshold <= 0:
        raise ValueError(
            f"large_image_threshold_pixels must be positive, got {threshold}"
        )
    source = ImageSource(
        path,
        normalization_low_percentile=low,
        normalization_high_percentile=high,
        normalization_sample_max_size=max_size,
        respect_tiff_photometric=respect,
        prefer_pyvips=prefer_pyvips,
    )
    try:
        preview = source.get_preview(max_size)
        lazy_mode = source.metadata.pixel_count > threshold if lazy is None else bool(lazy)
        gray = None if lazy_mode else source.read_full(normalize=True)
        source.metadata.lazy = lazy_mode
        if lazy_mode:
            source.metadata.warnings.append(
                "Full normalized image was not materialized; analyze using iter_tiles()"
            )
        return ImageData(gray=gray, preview=preview, metadata=source.metadata, source=source)
    except Exception:
        source.close()
        raise


def read_image(
    path: str | os.PathLike[str],
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ImageData:
    """Compatibility wrapper for :func:`load_image`."""

    return load_image(path, config=config, **kwargs)


def read_grayscale(
    path: str | os.PathLike[str],
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Convenience function that explicitly materializes a float32 analysis image."""

    kwargs["lazy"] = False
    data = load_image(path, config=config, **kwargs)
    try:
        return np.array(data.require_full(), copy=True)
    finally:
        data.close()


def iter_tiles(
    source: ImageSource | ImageData,
    tile_size: int = 2048,
    overlap: int = 128,
) -> Iterator[ImageTile]:
    """Top-level tile iterator accepting either source or loaded image data."""

    reader = source.source if isinstance(source, ImageData) else source
    yield from reader.iter_tiles(tile_size=tile_size, overlap=overlap)


__all__ = [
    "ImageData",
    "ImageMetadata",
    "ImageReadError",
    "ImageSource",
    "ImageTile",
    "iter_tiles",
    "load_image",
    "make_preview",
    "normalize_to_float32",
    "normalize_to_uint8",
    "read_grayscale",
    "read_image",
]
