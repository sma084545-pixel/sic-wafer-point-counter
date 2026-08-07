"""Configurable, non-destructive preprocessing for wafer images.

All public arrays in this module use floating point intensities in ``[0, 1]``.
The input array is never modified in-place.  The dark response is positive where
the image is darker than its estimated local background.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

LOGGER = logging.getLogger(__name__)

FloatImage = NDArray[np.float32]
BoolImage = NDArray[np.bool_]


def _section(mapping: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return a named config section, or the mapping itself if already scoped."""

    value = mapping.get(name)
    return value if isinstance(value, Mapping) else mapping


def _odd_at_least_one(value: int, *, name: str) -> int:
    """Validate and normalize a morphology/filter kernel size."""

    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    if value % 2 == 0:
        LOGGER.warning("%s=%d is even; using %d", name, value, value + 1)
        value += 1
    return value


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Parameters controlling background correction and mild denoising."""

    median_kernel: int = 3
    gaussian_sigma: float = 1.0
    background_method: str = "morphological"
    background_kernel_px: int = 101
    background_gaussian_sigma: float = 31.0
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    suppress_stripes: bool = False
    stripe_axis: str = "horizontal"
    stripe_smoothing_px: int = 51

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "PreprocessingConfig":
        """Build a config from either the full YAML mapping or its section."""

        if mapping is None:
            return cls()
        values = _section(mapping, "preprocessing")
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: values[key] for key in allowed if key in values})

    def validated(self) -> "PreprocessingConfig":
        """Return a normalized copy and raise on unsafe parameter values."""

        method = self.background_method.lower().strip()
        if method not in {"morphological", "gaussian"}:
            raise ValueError("background_method must be 'morphological' or 'gaussian'")
        axis = self.stripe_axis.lower().strip()
        if axis not in {"horizontal", "vertical"}:
            raise ValueError("stripe_axis must be 'horizontal' or 'vertical'")
        if self.gaussian_sigma < 0 or self.background_gaussian_sigma <= 0:
            raise ValueError("Gaussian sigmas must be non-negative (background > 0)")
        if self.clahe_clip_limit <= 0 or int(self.clahe_grid_size) < 1:
            raise ValueError("CLAHE parameters must be positive")
        return PreprocessingConfig(
            median_kernel=_odd_at_least_one(self.median_kernel, name="median_kernel"),
            gaussian_sigma=float(self.gaussian_sigma),
            background_method=method,
            background_kernel_px=_odd_at_least_one(
                self.background_kernel_px, name="background_kernel_px"
            ),
            background_gaussian_sigma=float(self.background_gaussian_sigma),
            use_clahe=bool(self.use_clahe),
            clahe_clip_limit=float(self.clahe_clip_limit),
            clahe_grid_size=int(self.clahe_grid_size),
            suppress_stripes=bool(self.suppress_stripes),
            stripe_axis=axis,
            stripe_smoothing_px=_odd_at_least_one(
                self.stripe_smoothing_px, name="stripe_smoothing_px"
            ),
        )


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Images produced by :func:`preprocess_image`.

    ``image`` is the normalized, immutable-in-practice source copy, ``filtered``
    includes optional CLAHE/stripe correction and denoising, ``background`` is
    the large-scale estimate, and ``dark_response`` equals
    ``max(background - filtered, 0)`` inside the valid mask.
    """

    image: FloatImage
    filtered: FloatImage
    background: FloatImage
    dark_response: FloatImage
    valid_mask: BoolImage
    config: PreprocessingConfig

    @property
    def preprocessed(self) -> FloatImage:
        """Compatibility alias for the filtered image."""

        return self.filtered


def as_float01(image: NDArray[np.generic]) -> FloatImage:
    """Return a finite 2-D image scaled to ``[0, 1]`` without changing input.

    Integer inputs use their data type's full non-negative range.  Floating
    inputs already in ``[0, 1]`` are copied; other floating inputs are robustly
    linearly scaled from their finite minimum and maximum.  Percentile
    normalization belongs in :mod:`image_io`, where its exact parameters and
    source metadata can be recorded.
    """

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("Cannot preprocess an empty image")

    if np.issubdtype(array.dtype, np.bool_):
        result = array.astype(np.float32, copy=True)
    elif np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        result = array.astype(np.float32, copy=True)
        # Signed camera arrays are uncommon, but supporting them avoids a
        # surprising negative normalization range.
        if info.min < 0:
            result = (result - float(info.min)) / float(info.max - info.min)
        elif info.max:
            result /= float(info.max)
    elif np.issubdtype(array.dtype, np.floating):
        result = array.astype(np.float32, copy=True)
        finite = np.isfinite(result)
        if not finite.any():
            raise ValueError("Image contains no finite pixels")
        low = float(np.min(result[finite]))
        high = float(np.max(result[finite]))
        result[~finite] = low
        if low < 0.0 or high > 1.0:
            if high > low:
                result = (result - low) / (high - low)
            else:
                result.fill(0.0)
    else:
        raise TypeError(f"Unsupported image dtype: {array.dtype}")
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def _masked_fill(image: FloatImage, valid_mask: BoolImage) -> FloatImage:
    """Fill invalid pixels with a stable wafer intensity before neighborhood ops."""

    filled = image.copy()
    stride = max(1, int(np.ceil(np.sqrt(image.size / 2_000_000))))
    sampled_image = image[::stride, ::stride]
    sampled_mask = valid_mask[::stride, ::stride]
    valid_values = sampled_image[sampled_mask]
    if valid_values.size == 0:
        # A very sparse mask can fall between sample points; use the exact path
        # only in that unusual case.
        valid_values = image[valid_mask]
    if valid_values.size == 0:
        raise ValueError("valid_mask contains no valid pixels")
    filled[~valid_mask] = float(np.median(valid_values))
    return filled


def _apply_clahe(image: FloatImage, config: PreprocessingConfig) -> FloatImage:
    """Apply OpenCV CLAHE using an explicit 8-bit working copy."""

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
    )
    image_u8 = np.rint(image * 255.0).astype(np.uint8)
    return (clahe.apply(image_u8).astype(np.float32) / 255.0).astype(np.float32)


def _suppress_stripes(
    image: FloatImage, valid_mask: BoolImage, axis: str, smoothing_px: int
) -> FloatImage:
    """Remove a slowly varying row/column offset while preserving global level.

    Horizontal stripes are estimated from per-row masked medians; vertical
    stripes use per-column medians.  Only the one-dimensional stripe profile is
    smoothed, which has a much smaller memory footprint than another full image.
    """

    masked = np.where(valid_mask, image, np.nan)
    reduction_axis = 1 if axis == "horizontal" else 0
    with np.errstate(all="ignore"):
        profile = np.nanmedian(masked, axis=reduction_axis)
    fallback = float(np.nanmedian(profile))
    if not np.isfinite(fallback):
        return image.copy()
    profile = np.where(np.isfinite(profile), profile, fallback)
    smooth = ndi.median_filter(profile, size=smoothing_px, mode="nearest")
    # The smooth profile is the illumination trend to preserve; deviations from
    # it are the row/column stripe offsets to subtract.
    bias = profile - smooth
    if axis == "horizontal":
        corrected = image - bias[:, np.newaxis]
    else:
        corrected = image - bias[np.newaxis, :]
    return np.asarray(np.clip(corrected, 0.0, 1.0), dtype=np.float32)


def estimate_background(image: FloatImage, config: PreprocessingConfig) -> FloatImage:
    """Estimate the slowly varying background of a normalized image."""

    if config.background_method == "gaussian":
        return np.asarray(
            ndi.gaussian_filter(
                image,
                sigma=config.background_gaussian_sigma,
                mode="reflect",
            ),
            dtype=np.float32,
        )

    # A closing fills dark structures smaller than the elliptical kernel and is
    # therefore an appropriate background estimator for dark point targets.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.background_kernel_px, config.background_kernel_px),
    )
    return np.asarray(
        cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REFLECT),
        dtype=np.float32,
    )


def preprocess_image(
    image: NDArray[np.generic],
    valid_mask: NDArray[np.bool_] | None = None,
    config: PreprocessingConfig | Mapping[str, Any] | None = None,
) -> PreprocessingResult:
    """Denoise and enhance dark targets without overwriting ``image``.

    Parameters
    ----------
    image:
        A two-dimensional 8/16-bit or floating grayscale image.
    valid_mask:
        Pixels that may contribute to analysis.  Invalid pixels are filled only
        for neighborhood calculations and are zeroed in the final response.
    config:
        :class:`PreprocessingConfig`, a full YAML mapping, or its
        ``preprocessing`` section.
    """

    cfg = (
        config.validated()
        if isinstance(config, PreprocessingConfig)
        else PreprocessingConfig.from_mapping(config).validated()
    )
    normalized = as_float01(image)
    if valid_mask is None:
        mask = np.ones(normalized.shape, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != normalized.shape:
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match image {normalized.shape}"
            )
        mask = mask.copy()
    working = _masked_fill(normalized, mask)

    if cfg.use_clahe:
        working = _apply_clahe(working, cfg)
    if cfg.suppress_stripes:
        working = _suppress_stripes(
            working, mask, cfg.stripe_axis, cfg.stripe_smoothing_px
        )
    if cfg.median_kernel > 1:
        working = np.asarray(
            ndi.median_filter(working, size=cfg.median_kernel, mode="reflect"),
            dtype=np.float32,
        )
    if cfg.gaussian_sigma > 0:
        working = np.asarray(
            ndi.gaussian_filter(working, sigma=cfg.gaussian_sigma, mode="reflect"),
            dtype=np.float32,
        )

    background = estimate_background(working, cfg)
    dark_response = np.maximum(background - working, 0.0).astype(np.float32)
    dark_response[~mask] = 0.0
    LOGGER.debug(
        "Preprocessed image %s: method=%s, dark response max=%.5f",
        normalized.shape,
        cfg.background_method,
        float(dark_response.max(initial=0.0)),
    )
    return PreprocessingResult(
        image=normalized,
        filtered=np.asarray(working, dtype=np.float32),
        background=np.asarray(background, dtype=np.float32),
        dark_response=dark_response,
        valid_mask=mask,
        config=cfg,
    )


# Concise alias useful in notebooks while keeping the descriptive API above.
preprocess = preprocess_image


__all__ = [
    "PreprocessingConfig",
    "PreprocessingResult",
    "as_float01",
    "estimate_background",
    "preprocess_image",
    "preprocess",
]
