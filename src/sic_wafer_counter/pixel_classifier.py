"""Portable pixel-level supervised learning for grayscale XRT images.

This module implements the scientific core of the interactive training loop.
Experts paint target/background pixels on the source image; deterministic
multi-scale image features are then classified by a small randomized decision
forest.  The JSON model is intentionally independent of scikit-learn so the
same model and inference code run in CPython and the Pyodide browser worker.

The predicted class is an *image class*, not a crystallographic identity.  A
positive probability must still pass the wafer mask, edge and morphology rules
before it contributes to the reported point-target count.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.measure import label, regionprops

from .utils import utc_now_iso
from . import __version__


MODEL_SCHEMA_VERSION = "1.0"
PROJECT_SCHEMA_VERSION = "1.0"
MODEL_TYPE = "portable_random_forest_pixel_classifier"
TARGET_LABEL = 1
BACKGROUND_LABEL = 2
IGNORE_LABEL = 3
UNLABELED = 0
LABEL_NAMES = {
    UNLABELED: "unlabeled",
    TARGET_LABEL: "target",
    BACKGROUND_LABEL: "background_or_artifact",
    IGNORE_LABEL: "ignore_or_uncertain",
}
ALLOWED_SPLITS = frozenset({"calibration", "validation", "locked_test"})


class PixelClassifierError(ValueError):
    """Raised when pixel labels or a portable model are incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class PixelTrainingSample:
    """One source-resolution labelled ROI used for train or holdout scoring."""

    image: NDArray[np.float32]
    labels: NDArray[np.uint8]
    image_sha256: str
    image_name: str
    wafer_id: str
    split: str
    roi_xywh: tuple[int, int, int, int]

    def validated(self) -> "PixelTrainingSample":
        image = np.asarray(self.image, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.uint8)
        if image.ndim != 2 or image.size == 0 or image.shape != labels.shape:
            raise PixelClassifierError("训练 ROI 必须是同尺寸的非空二维图像与标签掩膜")
        if not np.isfinite(image).all() or float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise PixelClassifierError("训练 ROI 必须是有限的 float32 [0,1] 科研灰度")
        invalid = set(int(value) for value in np.unique(labels)) - set(LABEL_NAMES)
        if invalid:
            raise PixelClassifierError(f"标签掩膜包含未知类别：{sorted(invalid)}")
        split = str(self.split).strip().lower()
        if split not in ALLOWED_SPLITS:
            raise PixelClassifierError("split 必须为 calibration、validation 或 locked_test")
        digest = str(self.image_sha256).strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise PixelClassifierError("训练图像 SHA-256 无效")
        x, y, width, height = (int(value) for value in self.roi_xywh)
        if x < 0 or y < 0 or width != image.shape[1] or height != image.shape[0]:
            raise PixelClassifierError("ROI 原图坐标与标签尺寸不一致")
        return PixelTrainingSample(
            image=image.copy(),
            labels=labels.copy(),
            image_sha256=digest,
            image_name=str(self.image_name),
            wafer_id=str(self.wafer_id).strip(),
            split=split,
            roi_xywh=(x, y, width, height),
        )


def default_feature_config() -> dict[str, Any]:
    """Return the portable, position-free grayscale feature definition."""

    return {
        "schema_version": "1.0",
        "gaussian_sigmas_px": [0.8, 1.6, 3.2],
        "local_radii_px": [2, 4],
        "hessian_sigma_px": 1.6,
        "structure_sigma_px": 1.6,
        "boundary_mode": "reflect",
        "absolute_coordinates_used": False,
    }


def _validated_feature_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    config = default_feature_config()
    if value:
        config.update(dict(value))
    sigmas = [float(item) for item in config.get("gaussian_sigmas_px", [])]
    radii = [int(item) for item in config.get("local_radii_px", [])]
    if not sigmas or any(not math.isfinite(item) or item <= 0 for item in sigmas):
        raise PixelClassifierError("gaussian_sigmas_px 必须为正的有限数组")
    if sorted(sigmas) != sigmas:
        raise PixelClassifierError("gaussian_sigmas_px 必须递增")
    if not radii or any(item < 1 for item in radii):
        raise PixelClassifierError("local_radii_px 必须为正整数数组")
    hessian = float(config.get("hessian_sigma_px", 1.6))
    structure = float(config.get("structure_sigma_px", 1.6))
    if not all(math.isfinite(item) and item > 0 for item in (hessian, structure)):
        raise PixelClassifierError("Hessian/structure sigma 必须为正的有限数")
    if str(config.get("boundary_mode", "reflect")) != "reflect":
        raise PixelClassifierError("当前模型只支持 reflect 边界语义")
    if bool(config.get("absolute_coordinates_used", False)):
        raise PixelClassifierError("像素模型禁止使用绝对坐标特征")
    config["gaussian_sigmas_px"] = sigmas
    config["local_radii_px"] = radii
    config["hessian_sigma_px"] = hessian
    config["structure_sigma_px"] = structure
    config["absolute_coordinates_used"] = False
    return config


def feature_names(config: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Return stable feature names in the exact tensor-channel order."""

    cfg = _validated_feature_config(config)
    sigmas = cfg["gaussian_sigmas_px"]
    names: list[str] = ["intensity"]
    names.extend(f"gaussian_sigma_{sigma:g}_px" for sigma in sigmas)
    names.extend(
        f"dog_sigma_{left:g}_to_{right:g}_px"
        for left, right in zip(sigmas[:-1], sigmas[1:])
    )
    for radius in cfg["local_radii_px"]:
        names.extend((f"local_mean_radius_{radius}_px", f"local_std_radius_{radius}_px"))
    names.extend(
        (
            "sobel_gradient_magnitude",
            f"laplacian_gaussian_sigma_{sigmas[0]:g}_px",
            f"laplacian_gaussian_sigma_{sigmas[1 if len(sigmas) > 1 else 0]:g}_px",
            f"hessian_trace_sigma_{cfg['hessian_sigma_px']:g}_px",
            f"hessian_determinant_sigma_{cfg['hessian_sigma_px']:g}_px",
            f"structure_coherence_sigma_{cfg['structure_sigma_px']:g}_px",
            f"dark_response_sigma_{sigmas[-1]:g}_px",
        )
    )
    return tuple(names)


def extract_pixel_features(
    image: NDArray[np.generic],
    config: Mapping[str, Any] | None = None,
) -> NDArray[np.float32]:
    """Calculate deterministic multi-scale features without spatial position.

    The returned array has shape ``(height, width, channels)``.  Input is never
    modified and must already use the run-wide float32 normalization window.
    """

    cfg = _validated_feature_config(config)
    source = np.asarray(image)
    if source.ndim != 2 or source.size == 0:
        raise PixelClassifierError("像素特征输入必须是非空二维灰度图")
    gray = np.asarray(source, dtype=np.float32).copy()
    if not np.isfinite(gray).all() or float(gray.min()) < 0.0 or float(gray.max()) > 1.0:
        raise PixelClassifierError("像素特征输入必须是有限的 float32 [0,1]")
    channels: list[NDArray[np.float32]] = [gray]
    gaussians = [
        np.asarray(ndi.gaussian_filter(gray, sigma=sigma, mode="reflect"), dtype=np.float32)
        for sigma in cfg["gaussian_sigmas_px"]
    ]
    channels.extend(gaussians)
    channels.extend(
        np.asarray(left - right, dtype=np.float32)
        for left, right in zip(gaussians[:-1], gaussians[1:])
    )
    for radius in cfg["local_radii_px"]:
        size = radius * 2 + 1
        mean = ndi.uniform_filter(gray, size=size, mode="reflect")
        second = ndi.uniform_filter(gray * gray, size=size, mode="reflect")
        variance = np.maximum(second - mean * mean, 0.0)
        channels.extend(
            (np.asarray(mean, dtype=np.float32), np.asarray(np.sqrt(variance), dtype=np.float32))
        )
    sobel_x = ndi.sobel(gray, axis=1, mode="reflect") / 8.0
    sobel_y = ndi.sobel(gray, axis=0, mode="reflect") / 8.0
    channels.append(np.asarray(np.hypot(sobel_x, sobel_y), dtype=np.float32))
    first_sigma = cfg["gaussian_sigmas_px"][0]
    second_sigma = cfg["gaussian_sigmas_px"][1 if len(gaussians) > 1 else 0]
    channels.append(
        np.asarray(ndi.gaussian_laplace(gray, sigma=first_sigma, mode="reflect"), dtype=np.float32)
    )
    channels.append(
        np.asarray(ndi.gaussian_laplace(gray, sigma=second_sigma, mode="reflect"), dtype=np.float32)
    )
    hessian_sigma = cfg["hessian_sigma_px"]
    hxx = ndi.gaussian_filter(gray, sigma=hessian_sigma, order=(0, 2), mode="reflect")
    hyy = ndi.gaussian_filter(gray, sigma=hessian_sigma, order=(2, 0), mode="reflect")
    hxy = ndi.gaussian_filter(gray, sigma=hessian_sigma, order=(1, 1), mode="reflect")
    channels.append(np.asarray(hxx + hyy, dtype=np.float32))
    channels.append(np.asarray(hxx * hyy - hxy * hxy, dtype=np.float32))
    structure_sigma = cfg["structure_sigma_px"]
    jxx = ndi.gaussian_filter(sobel_x * sobel_x, structure_sigma, mode="reflect")
    jyy = ndi.gaussian_filter(sobel_y * sobel_y, structure_sigma, mode="reflect")
    jxy = ndi.gaussian_filter(sobel_x * sobel_y, structure_sigma, mode="reflect")
    delta = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    coherence = delta / np.maximum(jxx + jyy, 1e-8)
    channels.append(np.asarray(np.clip(coherence, 0.0, 1.0), dtype=np.float32))
    channels.append(np.asarray(np.maximum(gaussians[-1] - gray, 0.0), dtype=np.float32))
    tensor = np.stack(channels, axis=-1).astype(np.float32, copy=False)
    expected = feature_names(cfg)
    if tensor.shape[-1] != len(expected) or not np.isfinite(tensor).all():
        raise PixelClassifierError("像素特征计算产生了非有限值或通道数量不一致")
    return tensor


def _canonical_value(value: Any) -> Any:
    """Normalize JSON values so browser number round-trips keep one digest.

    JavaScript has only one numeric type and serializes ``1.0`` as ``1``.
    Treating those representations as different would reject an otherwise
    byte-for-byte equivalent portable model after local-to-browser transfer.
    """

    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise PixelClassifierError("模型/项目 JSON 不允许 NaN 或 Inf")
        if number == 0.0:
            return 0
        return int(number) if number.is_integer() else number
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def model_sha256(model: Mapping[str, Any]) -> str:
    """Hash all model content except the digest field itself."""

    payload = {key: value for key, value in model.items() if key != "model_sha256"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def training_project_sha256(project: Mapping[str, Any]) -> str:
    """Hash a training project with browser-stable canonical JSON numbers."""

    payload = {key: value for key, value in project.items() if key != "project_sha256"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _leaf(probability: float, count: int) -> dict[str, Any]:
    return {"feature": -1, "probability": float(probability), "sample_count": int(count)}


def _gini(positive: int, total: int) -> float:
    if total <= 0:
        return 0.0
    probability = positive / total
    return 2.0 * probability * (1.0 - probability)


def _build_tree(
    matrix: NDArray[np.float32],
    labels: NDArray[np.uint8],
    indices: NDArray[np.int64],
    *,
    rng: np.random.Generator,
    max_depth: int,
    min_samples_leaf: int,
    max_features: int,
    depth: int = 0,
) -> list[dict[str, Any]]:
    """Build one randomized CART tree and return flat, index-addressed nodes."""

    nodes: list[dict[str, Any]] = []

    def recurse(active: NDArray[np.int64], level: int) -> int:
        node_index = len(nodes)
        positives = int(np.sum(labels[active]))
        probability = (positives + 1.0) / (len(active) + 2.0)
        nodes.append(_leaf(probability, len(active)))
        if (
            level >= max_depth
            or len(active) < min_samples_leaf * 2
            or positives == 0
            or positives == len(active)
        ):
            return node_index
        feature_count = matrix.shape[1]
        candidate_features = np.sort(
            rng.choice(feature_count, size=min(max_features, feature_count), replace=False)
        )
        best: tuple[float, int, float, NDArray[np.bool_]] | None = None
        for feature in candidate_features:
            values = matrix[active, int(feature)]
            if float(values.max()) - float(values.min()) <= 1e-12:
                continue
            quantiles = np.unique(np.quantile(values, (0.15, 0.30, 0.50, 0.70, 0.85)))
            for threshold in quantiles:
                left = values <= threshold
                left_count = int(np.sum(left))
                right_count = len(active) - left_count
                if left_count < min_samples_leaf or right_count < min_samples_leaf:
                    continue
                left_positive = int(np.sum(labels[active[left]]))
                right_positive = positives - left_positive
                impurity = (
                    left_count * _gini(left_positive, left_count)
                    + right_count * _gini(right_positive, right_count)
                ) / len(active)
                proposal = (float(impurity), int(feature), float(threshold), left)
                if best is None or proposal[:3] < best[:3]:
                    best = proposal
        if best is None:
            return node_index
        _, feature, threshold, left_selector = best
        left_index = recurse(active[left_selector], level + 1)
        right_index = recurse(active[~left_selector], level + 1)
        nodes[node_index] = {
            "feature": feature,
            "threshold": threshold,
            "left": left_index,
            "right": right_index,
            "probability": float(probability),
            "sample_count": int(len(active)),
        }
        return node_index

    recurse(indices, depth)
    return nodes


def _labelled_feature_rows(
    features: NDArray[np.float32], labels: NDArray[np.uint8]
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, int]]:
    """Collect labelled rows from one ROI without requiring both classes.

    An expert may deliberately paint target pixels in one ROI and background
    pixels in another.  Class balancing therefore belongs at the complete
    calibration-set level, not inside each ROI.
    """

    flat_features = features.reshape(-1, features.shape[-1])
    flat_labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    target = flat_features[flat_labels == TARGET_LABEL]
    background = flat_features[flat_labels == BACKGROUND_LABEL]
    return (
        np.asarray(target, dtype=np.float32),
        np.asarray(background, dtype=np.float32),
        {
            "labelled_target_pixels": int(len(target)),
            "labelled_background_pixels": int(len(background)),
            "ignored_pixels": int(np.sum(flat_labels == IGNORE_LABEL)),
        },
    )


def _assert_no_wafer_leakage(samples: Sequence[PixelTrainingSample]) -> None:
    by_wafer: dict[str, set[str]] = {}
    by_image: dict[str, set[str]] = {}
    for sample in samples:
        key = sample.wafer_id or sample.image_sha256
        by_wafer.setdefault(key, set()).add(sample.split)
        by_image.setdefault(sample.image_sha256, set()).add(sample.split)
    leaked = [f"wafer:{key}" for key, splits in by_wafer.items() if len(splits) > 1]
    leaked.extend(
        f"image:{key[:16]}" for key, splits in by_image.items() if len(splits) > 1
    )
    if leaked:
        raise PixelClassifierError(
            "同一 wafer_id/原图不能跨 calibration、validation 与 locked_test："
            + ", ".join(leaked[:5])
        )


def train_pixel_classifier(
    samples: Sequence[PixelTrainingSample],
    *,
    feature_config: Mapping[str, Any] | None = None,
    n_trees: int = 32,
    max_depth: int = 8,
    min_samples_leaf: int = 8,
    maximum_pixels_per_class: int = 20_000,
    bootstrap_pixels_per_class: int = 4_000,
    random_seed: int = 1729,
    probability_threshold: float = 0.5,
    minimum_object_area_px: int = 5,
) -> dict[str, Any]:
    """Train a class-balanced portable forest from source-resolution ROIs."""

    training_started = time.perf_counter()
    validated = [sample.validated() for sample in samples]
    if not validated:
        raise PixelClassifierError("尚无像素训练样本")
    _assert_no_wafer_leakage(validated)
    if not 1 <= int(n_trees) <= 256 or not 1 <= int(max_depth) <= 20:
        raise PixelClassifierError("n_trees/max_depth 超出安全范围")
    if int(min_samples_leaf) < 2 or int(maximum_pixels_per_class) < 20:
        raise PixelClassifierError("最小叶样本与每类采样上限无效")
    if not 0.0 < float(probability_threshold) < 1.0:
        raise PixelClassifierError("概率阈值必须位于 0 与 1 之间")
    if int(minimum_object_area_px) < 1:
        raise PixelClassifierError("最小目标面积必须至少为 1 px")
    cfg = _validated_feature_config(feature_config)
    calibration = [sample for sample in validated if sample.split == "calibration"]
    if not calibration:
        raise PixelClassifierError("至少需要一个 calibration ROI")
    target_blocks: list[NDArray[np.float32]] = []
    background_blocks: list[NDArray[np.float32]] = []
    total_counts = {
        "labelled_target_pixels": 0,
        "labelled_background_pixels": 0,
        "ignored_pixels": 0,
        "sampled_per_class": 0,
    }
    for sample in calibration:
        features = extract_pixel_features(sample.image, cfg)
        target, background, counts = _labelled_feature_rows(features, sample.labels)
        if len(target):
            target_blocks.append(target)
        if len(background):
            background_blocks.append(background)
        for key in ("labelled_target_pixels", "labelled_background_pixels", "ignored_pixels"):
            total_counts[key] += counts[key]
    if not target_blocks or not background_blocks:
        raise PixelClassifierError("calibration 集合至少需要目标前景和背景/伪影两类像素")
    target_all = np.concatenate(target_blocks, axis=0)
    background_all = np.concatenate(background_blocks, axis=0)
    sample_count = min(len(target_all), len(background_all), int(maximum_pixels_per_class))
    if sample_count < 20:
        raise PixelClassifierError("calibration 集合每类至少需要 20 个可用标注像素")
    selection_rng = np.random.default_rng(int(random_seed))
    target_selected = selection_rng.choice(len(target_all), sample_count, replace=False)
    background_selected = selection_rng.choice(len(background_all), sample_count, replace=False)
    matrix = np.concatenate(
        (target_all[target_selected], background_all[background_selected]), axis=0
    ).astype(np.float32, copy=False)
    binary_labels = np.concatenate(
        (np.ones(sample_count, dtype=np.uint8), np.zeros(sample_count, dtype=np.uint8))
    )
    order = selection_rng.permutation(len(binary_labels))
    matrix = matrix[order]
    binary_labels = binary_labels[order]
    total_counts["sampled_per_class"] = int(sample_count)
    target_indices = np.flatnonzero(binary_labels == 1)
    background_indices = np.flatnonzero(binary_labels == 0)
    rng = np.random.default_rng(int(random_seed))
    tree_sample_count = min(
        int(bootstrap_pixels_per_class), len(target_indices), len(background_indices)
    )
    if tree_sample_count < int(min_samples_leaf) * 2:
        raise PixelClassifierError("标注像素不足以建立当前叶大小的随机森林")
    maximum_features = max(1, int(round(math.sqrt(matrix.shape[1]))))
    trees: list[list[dict[str, Any]]] = []
    for _ in range(int(n_trees)):
        selected_target = rng.choice(target_indices, tree_sample_count, replace=True)
        selected_background = rng.choice(background_indices, tree_sample_count, replace=True)
        tree_indices = np.concatenate((selected_target, selected_background)).astype(np.int64)
        tree_indices = tree_indices[rng.permutation(len(tree_indices))]
        tree_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int32).max)))
        trees.append(
            _build_tree(
                matrix,
                binary_labels,
                tree_indices,
                rng=tree_rng,
                max_depth=int(max_depth),
                min_samples_leaf=int(min_samples_leaf),
                max_features=maximum_features,
            )
        )
    source_records = [
        {
            "image_sha256": sample.image_sha256,
            "image_name": sample.image_name,
            "wafer_id": sample.wafer_id,
            "split": sample.split,
            "roi_xywh": list(sample.roi_xywh),
            "target_pixels": int(np.sum(sample.labels == TARGET_LABEL)),
            "background_pixels": int(np.sum(sample.labels == BACKGROUND_LABEL)),
            "ignore_pixels": int(np.sum(sample.labels == IGNORE_LABEL)),
        }
        for sample in validated
    ]
    model: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "software_version": __version__,
        "created_at_utc": utc_now_iso(),
        "feature_config": cfg,
        "feature_names": list(feature_names(cfg)),
        "class_mapping": {
            "target": TARGET_LABEL,
            "background_or_artifact": BACKGROUND_LABEL,
            "ignore_or_uncertain": IGNORE_LABEL,
        },
        "training_parameters": {
            "algorithm": "class_balanced_randomized_decision_forest",
            "n_trees": int(n_trees),
            "max_depth": int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            "maximum_pixels_per_class": int(maximum_pixels_per_class),
            "bootstrap_pixels_per_class": int(bootstrap_pixels_per_class),
            "random_seed": int(random_seed),
            "max_features_per_split": maximum_features,
            "absolute_coordinates_used": False,
        },
        "probability_threshold": float(probability_threshold),
        "minimum_object_area_px": int(minimum_object_area_px),
        "label_counts": total_counts,
        "training_sources": source_records,
        "trees": trees,
        "validation": {
            "status": "not_evaluated_yet",
            "real_sic_accuracy_claim_allowed": False,
        },
        "scientific_limit": (
            "The forest predicts an expert-labelled image class. It does not by itself "
            "confirm TSD, TED, BPD or another crystallographic defect identity."
        ),
    }
    model["training_duration_seconds"] = time.perf_counter() - training_started
    normalized = validate_pixel_model(model)
    holdout_metrics: dict[str, Any] = {}
    for split in ("validation", "locked_test"):
        split_samples = [sample for sample in validated if sample.split == split]
        if split_samples:
            holdout_metrics[split] = evaluate_samples(
                split_samples,
                normalized,
                probability_threshold=float(probability_threshold),
                minimum_object_area_px=int(minimum_object_area_px),
            )
    if holdout_metrics:
        model["validation"] = {
            "status": "held_out_metrics_available",
            "metrics": holdout_metrics,
            "real_sic_accuracy_claim_allowed": False,
            "note": "Physical interpretation still requires expert/independent validation.",
        }
    model["model_sha256"] = model_sha256(model)
    return validate_pixel_model(model)


def validate_pixel_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a pixel model and verify or assign its content digest."""

    normalized = json.loads(json.dumps(dict(model), ensure_ascii=False))
    if normalized.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise PixelClassifierError("不支持的像素模型 schema_version")
    if normalized.get("model_type") != MODEL_TYPE:
        raise PixelClassifierError("不支持的像素模型 model_type")
    cfg = _validated_feature_config(normalized.get("feature_config"))
    names = feature_names(cfg)
    if tuple(normalized.get("feature_names", ())) != names:
        raise PixelClassifierError("像素模型特征顺序与当前软件不一致")
    class_mapping = normalized.get("class_mapping")
    if class_mapping != {
        "target": TARGET_LABEL,
        "background_or_artifact": BACKGROUND_LABEL,
        "ignore_or_uncertain": IGNORE_LABEL,
    }:
        raise PixelClassifierError("像素模型类别映射不兼容")
    training_parameters = normalized.get("training_parameters")
    if not isinstance(training_parameters, Mapping):
        raise PixelClassifierError("像素模型缺少训练参数")
    if bool(training_parameters.get("absolute_coordinates_used", True)):
        raise PixelClassifierError("像素模型禁止使用绝对坐标特征")
    try:
        random_seed = int(training_parameters["random_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PixelClassifierError("像素模型缺少固定随机种子") from exc
    training_parameters["random_seed"] = random_seed
    sources = normalized.get("training_sources")
    if not isinstance(sources, list) or not sources:
        raise PixelClassifierError("像素模型缺少训练图像来源")
    source_samples: list[PixelTrainingSample] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise PixelClassifierError("像素模型训练来源格式无效")
        roi = tuple(int(value) for value in source.get("roi_xywh", ()))
        if len(roi) != 4 or roi[0] < 0 or roi[1] < 0 or roi[2] <= 0 or roi[3] <= 0:
            raise PixelClassifierError("像素模型训练来源 ROI 无效")
        digest = str(source.get("image_sha256", ""))
        split = str(source.get("split", ""))
        # A tiny placeholder only reuses the central provenance validator; no
        # source pixels or coordinates become model features.
        source_samples.append(
            PixelTrainingSample(
                image=np.zeros((1, 1), np.float32),
                labels=np.zeros((1, 1), np.uint8),
                image_sha256=digest,
                image_name=str(source.get("image_name", "")),
                wafer_id=str(source.get("wafer_id", "")),
                split=split,
                roi_xywh=(roi[0], roi[1], 1, 1),
            ).validated()
        )
    _assert_no_wafer_leakage(source_samples)
    counts = normalized.get("label_counts")
    if not isinstance(counts, Mapping) or int(counts.get("labelled_target_pixels", 0)) < 1 or int(
        counts.get("labelled_background_pixels", 0)
    ) < 1:
        raise PixelClassifierError("像素模型标签数量无效")
    if not isinstance(normalized.get("validation"), Mapping):
        raise PixelClassifierError("像素模型缺少验证状态")
    trees = normalized.get("trees")
    if not isinstance(trees, list) or not trees or len(trees) > 256:
        raise PixelClassifierError("像素模型没有决策树")
    for tree_index, tree in enumerate(trees):
        if not isinstance(tree, list) or not tree or len(tree) > 8191:
            raise PixelClassifierError(f"决策树 {tree_index} 为空")
        for node_index, node in enumerate(tree):
            if not isinstance(node, Mapping):
                raise PixelClassifierError("决策树节点必须为对象")
            feature = int(node.get("feature", -2))
            probability = float(node.get("probability", math.nan))
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise PixelClassifierError("决策树节点概率无效")
            if feature == -1:
                continue
            if not 0 <= feature < len(names):
                raise PixelClassifierError("决策树引用了未知特征")
            threshold = float(node.get("threshold", math.nan))
            left, right = int(node.get("left", -1)), int(node.get("right", -1))
            if not math.isfinite(threshold) or not 0 <= left < len(tree) or not 0 <= right < len(tree):
                raise PixelClassifierError("决策树分支索引或阈值无效")
            if left <= node_index or right <= node_index:
                raise PixelClassifierError("决策树必须为前向无环结构")
    threshold = float(normalized.get("probability_threshold", math.nan))
    minimum_area = int(normalized.get("minimum_object_area_px", 0))
    if not 0.0 < threshold < 1.0 or minimum_area < 1:
        raise PixelClassifierError("像素模型阈值或最小目标面积无效")
    expected = normalized.get("model_sha256")
    actual = model_sha256(normalized)
    if expected is not None and str(expected) != actual:
        raise PixelClassifierError("像素模型 SHA-256 与内容不一致")
    normalized["feature_config"] = cfg
    normalized["probability_threshold"] = threshold
    normalized["minimum_object_area_px"] = minimum_area
    normalized["model_sha256"] = actual
    return normalized


def load_pixel_model(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load a portable pixel model from a mapping or JSON file."""

    if isinstance(value, Mapping):
        return validate_pixel_model(value)
    path = Path(value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PixelClassifierError(f"无法读取像素模型：{path}") from exc
    if not isinstance(payload, Mapping):
        raise PixelClassifierError("像素模型 JSON 顶层必须为对象")
    return validate_pixel_model(payload)


def pixel_model_from_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Resolve the optional pixel model embedded in or referenced by config."""

    section = config.get("pixel_classifier", {})
    if not isinstance(section, Mapping) or not bool(section.get("enabled", False)):
        return None
    embedded = section.get("model")
    path = section.get("model_path")
    if embedded is not None and path:
        raise PixelClassifierError("pixel_classifier 不能同时提供 model 与 model_path")
    if embedded is not None:
        if not isinstance(embedded, Mapping):
            raise PixelClassifierError("pixel_classifier.model 必须为 JSON 对象")
        return validate_pixel_model(embedded)
    if path:
        return load_pixel_model(str(path))
    raise PixelClassifierError("已启用 pixel_classifier，但未提供 model 或 model_path")


def predict_pixel_probability(
    image: NDArray[np.generic], model: Mapping[str, Any]
) -> NDArray[np.float32]:
    """Predict target probability for every pixel with a validated model."""

    normalized = validate_pixel_model(model)
    features = extract_pixel_features(image, normalized["feature_config"])
    matrix = features.reshape(-1, features.shape[-1])
    total = np.zeros(len(matrix), dtype=np.float64)
    row_indices = np.arange(len(matrix), dtype=np.int64)
    for tree in normalized["trees"]:
        active = np.zeros(len(matrix), dtype=np.int32)
        output = np.empty(len(matrix), dtype=np.float32)
        pending = np.ones(len(matrix), dtype=bool)
        while pending.any():
            for node_index in np.unique(active[pending]):
                selector = pending & (active == node_index)
                node = tree[int(node_index)]
                feature = int(node["feature"])
                if feature < 0:
                    output[selector] = float(node["probability"])
                    pending[selector] = False
                    continue
                values = matrix[row_indices[selector], feature]
                local_rows = np.flatnonzero(selector)
                go_left = values <= float(node["threshold"])
                active[local_rows[go_left]] = int(node["left"])
                active[local_rows[~go_left]] = int(node["right"])
        total += output
    probability = (total / len(normalized["trees"])).reshape(features.shape[:2])
    return np.asarray(np.clip(probability, 0.0, 1.0), dtype=np.float32)


def required_feature_halo_px(model: Mapping[str, Any]) -> int:
    """Conservative halo that makes tiled features match full-frame interiors."""

    cfg = validate_pixel_model(model)["feature_config"]
    return int(
        math.ceil(
            max(
                max(cfg["gaussian_sigmas_px"]) * 4.0,
                max(cfg["local_radii_px"]),
                cfg["hessian_sigma_px"] * 4.0,
                cfg["structure_sigma_px"] * 4.0 + 1.0,
            )
        )
    )


def predict_pixel_probability_tiled(
    image: NDArray[np.generic],
    model: Mapping[str, Any],
    *,
    tile_size: int = 512,
    overlap: int | None = None,
) -> NDArray[np.float32]:
    """Predict a materialized image by overlap tiles without seam ownership gaps."""

    source = np.asarray(image)
    if source.ndim != 2:
        raise PixelClassifierError("分块预测输入必须是二维图像")
    normalized = validate_pixel_model(model)
    size = int(tile_size)
    halo = required_feature_halo_px(normalized) if overlap is None else int(overlap)
    if size < 32 or halo < required_feature_halo_px(normalized) or halo >= size:
        raise PixelClassifierError("tile_size/overlap 不足以覆盖像素特征邻域")
    output = np.zeros(source.shape, dtype=np.float32)
    height, width = source.shape
    for core_y in range(0, height, size):
        for core_x in range(0, width, size):
            core_y1 = min(height, core_y + size)
            core_x1 = min(width, core_x + size)
            y0, x0 = max(0, core_y - halo), max(0, core_x - halo)
            y1, x1 = min(height, core_y1 + halo), min(width, core_x1 + halo)
            tile_probability = predict_pixel_probability(source[y0:y1, x0:x1], normalized)
            output[core_y:core_y1, core_x:core_x1] = tile_probability[
                core_y - y0 : core_y1 - y0,
                core_x - x0 : core_x1 - x0,
            ]
    return output


def probability_to_mask(
    probability: NDArray[np.generic],
    *,
    threshold: float,
    minimum_object_area_px: int,
    valid_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Threshold probability and remove connected objects below the minimum area."""

    values = np.asarray(probability, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise PixelClassifierError("概率图必须为有限二维数组")
    if not 0.0 < float(threshold) < 1.0 or int(minimum_object_area_px) < 1:
        raise PixelClassifierError("概率阈值或最小目标面积无效")
    mask = values >= float(threshold)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != mask.shape:
            raise PixelClassifierError("valid_mask 与概率图尺寸不一致")
        mask &= valid
    labels = label(mask, connectivity=2)
    counts = np.bincount(labels.ravel())
    keep = counts >= int(minimum_object_area_px)
    if len(keep):
        keep[0] = False
    return np.asarray(keep[labels], dtype=bool)


def pixel_metrics(
    labels: NDArray[np.uint8],
    probability: NDArray[np.generic],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate only explicit target/background pixels; ignore is excluded."""

    truth = np.asarray(labels, dtype=np.uint8)
    values = np.asarray(probability, dtype=np.float32)
    if truth.shape != values.shape:
        raise PixelClassifierError("验证标签与概率图尺寸不一致")
    determinate = (truth == TARGET_LABEL) | (truth == BACKGROUND_LABEL)
    positive = truth == TARGET_LABEL
    prediction = values >= float(threshold)
    tp = int(np.sum(determinate & positive & prediction))
    fp = int(np.sum(determinate & ~positive & prediction))
    fn = int(np.sum(determinate & positive & ~prediction))
    tn = int(np.sum(determinate & ~positive & ~prediction))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    union = tp + fp + fn
    return {
        "evaluated_pixel_count": int(np.sum(determinate)),
        "ignored_pixel_count": int(np.sum(truth == IGNORE_LABEL)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": tp / union if union else None,
    }


def object_metrics(
    labels: NDArray[np.uint8],
    predicted_mask: NDArray[np.bool_],
    *,
    matching_tolerance_px: float = 5.0,
) -> dict[str, Any]:
    """Match predicted and labelled target objects once and report localization error."""

    truth_mask = np.asarray(labels, dtype=np.uint8) == TARGET_LABEL
    prediction = np.asarray(predicted_mask, dtype=bool)
    if truth_mask.shape != prediction.shape:
        raise PixelClassifierError("目标级验证掩膜尺寸不一致")
    truth_regions = regionprops(label(truth_mask, connectivity=2))
    predicted_regions = regionprops(label(prediction, connectivity=2))
    truth_centers = np.asarray([(r.centroid[1], r.centroid[0]) for r in truth_regions], float)
    predicted_centers = np.asarray([(r.centroid[1], r.centroid[0]) for r in predicted_regions], float)
    errors: list[float] = []
    matches = 0
    if len(truth_centers) and len(predicted_centers):
        distances = np.linalg.norm(
            truth_centers[:, None, :] - predicted_centers[None, :, :], axis=2
        )
        rows, columns = linear_sum_assignment(distances)
        for row, column in zip(rows, columns):
            distance = float(distances[row, column])
            if distance <= float(matching_tolerance_px):
                matches += 1
                errors.append(distance)
    fp = len(predicted_regions) - matches
    fn = len(truth_regions) - matches
    precision = matches / (matches + fp) if matches + fp else None
    recall = matches / (matches + fn) if matches + fn else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "labelled_object_count": len(truth_regions),
        "predicted_object_count": len(predicted_regions),
        "true_positive": matches,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_localization_error_px": float(np.mean(errors)) if errors else None,
        "max_localization_error_px": float(np.max(errors)) if errors else None,
        "matching_tolerance_px": float(matching_tolerance_px),
    }


def evaluate_samples(
    samples: Sequence[PixelTrainingSample],
    model: Mapping[str, Any],
    *,
    probability_threshold: float | None = None,
    minimum_object_area_px: int | None = None,
) -> dict[str, Any]:
    """Return auditable per-ROI pixel and object metrics for one split."""

    normalized = validate_pixel_model(model)
    threshold = (
        float(normalized["probability_threshold"])
        if probability_threshold is None
        else float(probability_threshold)
    )
    minimum_area = (
        int(normalized["minimum_object_area_px"])
        if minimum_object_area_px is None
        else int(minimum_object_area_px)
    )
    rows: list[dict[str, Any]] = []
    pixel_totals = {key: 0 for key in ("true_positive", "false_positive", "false_negative", "true_negative")}
    object_totals = {key: 0 for key in ("true_positive", "false_positive", "false_negative")}
    errors: list[float] = []
    for sample in samples:
        checked = sample.validated()
        probability = predict_pixel_probability(checked.image, normalized)
        predicted = probability_to_mask(
            probability,
            threshold=threshold,
            minimum_object_area_px=minimum_area,
        )
        pixels = pixel_metrics(checked.labels, probability, threshold=threshold)
        objects = object_metrics(checked.labels, predicted)
        for key in pixel_totals:
            pixel_totals[key] += int(pixels[key])
        for key in object_totals:
            object_totals[key] += int(objects[key])
        if objects["mean_localization_error_px"] is not None:
            errors.append(float(objects["mean_localization_error_px"]))
        rows.append(
            {
                "image_sha256": checked.image_sha256,
                "wafer_id": checked.wafer_id,
                "split": checked.split,
                "roi_xywh": list(checked.roi_xywh),
                "pixel": pixels,
                "object": objects,
            }
        )

    def summarize(counts: Mapping[str, int]) -> dict[str, Any]:
        tp, fp, fn = counts["true_positive"], counts["false_positive"], counts["false_negative"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        return {**counts, "precision": precision, "recall": recall, "f1": f1}

    pixel_summary = summarize(pixel_totals)
    pixel_union = pixel_totals["true_positive"] + pixel_totals["false_positive"] + pixel_totals["false_negative"]
    pixel_summary["iou"] = pixel_totals["true_positive"] / pixel_union if pixel_union else None
    object_summary = summarize(object_totals)
    object_summary["mean_roi_localization_error_px"] = float(np.mean(errors)) if errors else None
    return {
        "roi_count": len(rows),
        "pixel": pixel_summary,
        "object": object_summary,
        "per_roi": rows,
    }


def encode_label_mask_rle(labels: NDArray[np.uint8]) -> dict[str, Any]:
    """Losslessly encode non-zero labels as flat runs for JSON project files."""

    array = np.asarray(labels, dtype=np.uint8)
    if array.ndim != 2:
        raise PixelClassifierError("标签掩膜必须为二维数组")
    invalid = set(int(value) for value in np.unique(array)) - set(LABEL_NAMES)
    if invalid:
        raise PixelClassifierError(f"标签掩膜包含未知类别：{sorted(invalid)}")
    flat = array.ravel()
    runs: list[list[int]] = []
    index = 0
    while index < len(flat):
        value = int(flat[index])
        start = index
        index += 1
        while index < len(flat) and int(flat[index]) == value:
            index += 1
        if value != UNLABELED:
            runs.append([start, index - start, value])
    return {"encoding": "flat_nonzero_rle_v1", "shape": list(array.shape), "runs": runs}


def decode_label_mask_rle(payload: Mapping[str, Any]) -> NDArray[np.uint8]:
    """Decode and bounds-check a JSON label mask."""

    if payload.get("encoding") != "flat_nonzero_rle_v1":
        raise PixelClassifierError("不支持的标签 RLE 编码")
    shape = tuple(int(value) for value in payload.get("shape", ()))
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise PixelClassifierError("标签 RLE shape 无效")
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    occupied = np.zeros(len(flat), dtype=bool)
    for run in payload.get("runs", []):
        if not isinstance(run, Sequence) or len(run) != 3:
            raise PixelClassifierError("标签 RLE run 无效")
        start, length, value = (int(item) for item in run)
        end = start + length
        if start < 0 or length <= 0 or end > len(flat) or value not in LABEL_NAMES or value == 0:
            raise PixelClassifierError("标签 RLE run 越界或类别无效")
        if occupied[start:end].any():
            raise PixelClassifierError("标签 RLE run 发生重叠")
        flat[start:end] = value
        occupied[start:end] = True
    return flat.reshape(shape)


def build_training_project(
    *,
    image_sha256: str,
    image_name: str,
    source_shape: tuple[int, int],
    wafer_id: str,
    split: str,
    annotations: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any] | None = None,
    operation_history: Sequence[Mapping[str, Any]] = (),
    reviewer_id: str = "local_expert",
) -> dict[str, Any]:
    """Build a versioned, auditable project mapping for save/reopen."""

    digest = str(image_sha256).strip().lower()
    if len(digest) != 64:
        raise PixelClassifierError("训练项目图像 SHA-256 无效")
    split = str(split).strip().lower()
    if split not in ALLOWED_SPLITS:
        raise PixelClassifierError("训练项目 split 无效")
    height, width = (int(value) for value in source_shape)
    if height <= 0 or width <= 0:
        raise PixelClassifierError("训练项目源图尺寸无效")
    normalized_annotations: list[dict[str, Any]] = []
    for annotation in annotations:
        roi = tuple(int(value) for value in annotation.get("roi_xywh", ()))
        if len(roi) != 4:
            raise PixelClassifierError("训练项目 ROI 无效")
        labels = decode_label_mask_rle(annotation.get("labels", {}))
        if labels.shape != (roi[3], roi[2]) or roi[0] < 0 or roi[1] < 0:
            raise PixelClassifierError("训练项目 ROI 与标签尺寸不一致")
        if roi[0] + roi[2] > width or roi[1] + roi[3] > height:
            raise PixelClassifierError("训练项目 ROI 超出源图")
        normalized_annotations.append(
            {
                "annotation_id": str(annotation.get("annotation_id", f"roi_{len(normalized_annotations)+1:04d}")),
                "roi_xywh": list(roi),
                "labels": encode_label_mask_rle(labels),
                "updated_at_utc": str(annotation.get("updated_at_utc", utc_now_iso())),
            }
        )
    project: dict[str, Any] = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "project_type": "sic_xrt_pixel_training_project",
        "created_at_utc": utc_now_iso(),
        "source_image": {
            "file_name": str(image_name),
            "sha256": digest,
            "shape_yx": [height, width],
            "pixels_embedded": False,
        },
        "wafer_id": str(wafer_id),
        "reviewer_id": str(reviewer_id).strip() or "local_expert",
        "split": split,
        "class_mapping": {
            "target": TARGET_LABEL,
            "background_or_artifact": BACKGROUND_LABEL,
            "ignore_or_uncertain": IGNORE_LABEL,
        },
        "annotations": normalized_annotations,
        "operation_history": [dict(item) for item in operation_history],
        "model": validate_pixel_model(model) if model is not None else None,
        "scientific_limit": "Image target labels require expert physical validation before defect-class claims.",
    }
    project["project_sha256"] = training_project_sha256(project)
    return project


def validate_training_project(project: Mapping[str, Any]) -> dict[str, Any]:
    """Validate project integrity and all lossless label masks."""

    if project.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise PixelClassifierError("不支持的训练项目 schema_version")
    if project.get("project_type") != "sic_xrt_pixel_training_project":
        raise PixelClassifierError("不支持的训练项目类型")
    source = project.get("source_image")
    if not isinstance(source, Mapping):
        raise PixelClassifierError("训练项目缺少 source_image")
    normalized = build_training_project(
        image_sha256=str(source.get("sha256", "")),
        image_name=str(source.get("file_name", "")),
        source_shape=tuple(source.get("shape_yx", ())),
        wafer_id=str(project.get("wafer_id", "")),
        split=str(project.get("split", "")),
        annotations=list(project.get("annotations", [])),
        model=project.get("model"),
        operation_history=list(project.get("operation_history", [])),
        reviewer_id=str(project.get("reviewer_id", "local_expert")),
    )
    for key, value in source.items():
        if key not in normalized["source_image"]:
            normalized["source_image"][str(key)] = value
    # Preserve original timestamps so digest verification covers the exact file.
    normalized["created_at_utc"] = str(project.get("created_at_utc", ""))
    expected = project.get("project_sha256")
    actual = training_project_sha256(normalized)
    if expected is not None and str(expected) != actual:
        raise PixelClassifierError("训练项目 SHA-256 与内容不一致")
    normalized["project_sha256"] = actual
    return normalized


__all__ = [
    "ALLOWED_SPLITS",
    "BACKGROUND_LABEL",
    "IGNORE_LABEL",
    "LABEL_NAMES",
    "MODEL_SCHEMA_VERSION",
    "MODEL_TYPE",
    "PROJECT_SCHEMA_VERSION",
    "PixelClassifierError",
    "PixelTrainingSample",
    "TARGET_LABEL",
    "UNLABELED",
    "build_training_project",
    "decode_label_mask_rle",
    "default_feature_config",
    "encode_label_mask_rle",
    "evaluate_samples",
    "extract_pixel_features",
    "feature_names",
    "load_pixel_model",
    "model_sha256",
    "object_metrics",
    "pixel_metrics",
    "pixel_model_from_config",
    "predict_pixel_probability",
    "predict_pixel_probability_tiled",
    "probability_to_mask",
    "required_feature_halo_px",
    "train_pixel_classifier",
    "training_project_sha256",
    "validate_pixel_model",
    "validate_training_project",
]
