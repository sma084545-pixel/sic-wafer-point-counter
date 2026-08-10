"""Auditable supervised classification for already-segmented point candidates.

The detector remains responsible for proposing image regions.  This module
learns only a target-versus-artifact decision from expert-labelled candidate
features.  It deliberately uses a small NumPy logistic model so the exact same
JSON model can run in CPython and in the Pyodide browser worker.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .utils import utc_now_iso


MODEL_SCHEMA_VERSION = "1.0"
MODEL_TYPE = "standardized_logistic_candidate_classifier"
POSITIVE_LABEL = "target"
NEGATIVE_LABEL = "artifact"
UNCERTAIN_LABEL = "uncertain"
TRAINING_LABELS = frozenset({POSITIVE_LABEL, NEGATIVE_LABEL, UNCERTAIN_LABEL})

# Physical dimensions are used where possible so a model is not tied to one
# camera pixel pitch.  Position and wafer region are intentionally excluded to
# avoid learning a spatial sampling bias as a visual identity rule.
RAW_FEATURE_COLUMNS: tuple[str, ...] = (
    "area_mm2",
    "equivalent_diameter_um",
    "major_axis_length_um",
    "minor_axis_length_um",
    "aspect_ratio",
    "eccentricity",
    "circularity",
    "solidity",
    "contrast",
    "mean_dark_response",
)
MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "log1p_area_um2",
    "log1p_equivalent_diameter_um",
    "log1p_major_axis_um",
    "log1p_minor_axis_um",
    "aspect_ratio",
    "eccentricity",
    "circularity",
    "solidity",
    "contrast",
    "mean_dark_response",
)
HARD_REJECTION_REASONS = frozenset({"outside_valid_mask", "near_wafer_edge"})


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _model_digest(model: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in model.items() if key != "model_sha256"}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed model features and a finite-row mask."""

    area_um2 = np.clip(_numeric(frame, "area_mm2") * 1_000_000.0, 0.0, None)
    equivalent_um = np.clip(_numeric(frame, "equivalent_diameter_um"), 0.0, None)
    major_um = np.clip(_numeric(frame, "major_axis_length_um"), 0.0, None)
    minor_um = np.clip(_numeric(frame, "minor_axis_length_um"), 0.0, None)
    matrix = np.column_stack(
        (
            np.log1p(area_um2),
            np.log1p(equivalent_um),
            np.log1p(major_um),
            np.log1p(minor_um),
            _numeric(frame, "aspect_ratio"),
            _numeric(frame, "eccentricity"),
            _numeric(frame, "circularity"),
            _numeric(frame, "solidity"),
            _numeric(frame, "contrast"),
            _numeric(frame, "mean_dark_response"),
        )
    )
    valid = np.isfinite(matrix).all(axis=1)
    return matrix, valid


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    regularization: float,
    max_iterations: int = 100,
) -> tuple[np.ndarray, float, int]:
    """Fit a class-balanced L2 logistic model with backtracked Newton steps."""

    sample_count, feature_count = matrix.shape
    positives = int(labels.sum())
    negatives = int(sample_count - positives)
    sample_weights = np.where(
        labels > 0.5,
        sample_count / (2.0 * positives),
        sample_count / (2.0 * negatives),
    )
    design = np.column_stack((np.ones(sample_count), matrix))
    coefficients = np.zeros(feature_count + 1, dtype=np.float64)
    iterations = 0
    penalty = np.zeros(feature_count + 1, dtype=np.float64)
    penalty[1:] = float(regularization)

    def objective(candidate: np.ndarray) -> float:
        logits = design @ candidate
        signed_labels = np.where(labels > 0.5, 1.0, -1.0)
        losses = np.logaddexp(0.0, -signed_labels * logits)
        return float(np.sum(sample_weights * losses) + 0.5 * np.sum(penalty * candidate**2))

    for iterations in range(1, max_iterations + 1):
        probabilities = _sigmoid(design @ coefficients)
        residual = sample_weights * (probabilities - labels)
        gradient = design.T @ residual + penalty * coefficients
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        # ``einsum`` avoids platform BLAS kernels that can leave spurious
        # floating-point flags on very small curvature values.
        hessian = np.einsum("ni,nj,n->ij", design, design, curvature, optimize=True)
        hessian.flat[:: feature_count + 2] += penalty + 1e-9
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        if not np.isfinite(step).all():
            raise ValueError("Classifier optimization produced non-finite coefficients")
        previous_objective = objective(coefficients)
        step_scale = 1.0
        candidate = coefficients - step
        candidate_objective = objective(candidate)
        while step_scale >= 1.0 / 1024.0 and (
            not math.isfinite(candidate_objective)
            or candidate_objective > previous_objective
        ):
            step_scale *= 0.5
            candidate = coefficients - step_scale * step
            candidate_objective = objective(candidate)
        if step_scale < 1.0 / 1024.0:
            break
        update = candidate - coefficients
        coefficients = candidate
        if float(np.max(np.abs(update))) < 1e-8:
            break
    return coefficients[1:], float(coefficients[0]), iterations


def _decision_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    accept_threshold: float,
    reject_threshold: float,
) -> dict[str, Any]:
    target = probabilities >= accept_threshold
    artifact = probabilities <= reject_threshold
    covered = target | artifact
    predictions = target.astype(np.int8)
    positive = labels.astype(bool)
    tp = int(np.sum(covered & predictions.astype(bool) & positive))
    fp = int(np.sum(covered & predictions.astype(bool) & ~positive))
    fn = int(np.sum((~covered | ~predictions.astype(bool)) & positive))
    tn = int(np.sum(covered & ~predictions.astype(bool) & ~positive))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "sample_count": int(len(labels)),
        "target_count": int(positive.sum()),
        "artifact_count": int((~positive).sum()),
        "coverage_fraction": float(np.mean(covered)) if len(labels) else None,
        "uncertain_count": int(np.sum(~covered)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative_including_uncertain": fn,
        "true_negative": tn,
        "precision_at_accept_threshold": precision,
        "recall_counting_uncertain_as_missed": recall,
        "f1_counting_uncertain_as_missed": f1,
    }


def validate_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a candidate-classifier JSON mapping."""

    normalized = dict(model)
    if normalized.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("Unsupported candidate-classifier schema_version")
    if normalized.get("model_type") != MODEL_TYPE:
        raise ValueError("Unsupported candidate-classifier model_type")
    if tuple(normalized.get("feature_columns", ())) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Candidate-classifier feature schema does not match this software")
    expected = len(MODEL_FEATURE_COLUMNS)
    for key in ("means", "scales", "coefficients"):
        values = np.asarray(normalized.get(key), dtype=np.float64)
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError(f"Candidate classifier {key} must contain {expected} finite values")
        normalized[key] = values.tolist()
    intercept = float(normalized.get("intercept"))
    reject_threshold = float(normalized.get("reject_threshold"))
    accept_threshold = float(normalized.get("accept_threshold"))
    if not all(math.isfinite(value) for value in (intercept, reject_threshold, accept_threshold)):
        raise ValueError("Candidate-classifier thresholds and intercept must be finite")
    if not 0.0 < reject_threshold < accept_threshold < 1.0:
        raise ValueError("必须满足 0 < 拒绝概率上限 < 接受概率下限 < 1")
    scales = np.asarray(normalized["scales"], dtype=np.float64)
    if np.any(scales <= 0):
        raise ValueError("Candidate-classifier scales must be positive")
    normalized["intercept"] = intercept
    normalized["reject_threshold"] = reject_threshold
    normalized["accept_threshold"] = accept_threshold
    expected_digest = normalized.get("model_sha256")
    actual_digest = _model_digest(normalized)
    if expected_digest is not None and str(expected_digest) != actual_digest:
        raise ValueError("Candidate-classifier model_sha256 does not match its contents")
    normalized["model_sha256"] = actual_digest
    return normalized


def train_candidate_classifier(
    annotations: pd.DataFrame,
    *,
    accept_threshold: float = 0.75,
    reject_threshold: float = 0.25,
    regularization: float = 1.0,
    minimum_per_class: int = 5,
) -> dict[str, Any]:
    """Train from labelled candidate snapshots without using holdout rows."""

    if not 0.0 < reject_threshold < accept_threshold < 1.0:
        raise ValueError("必须满足 0 < 拒绝概率上限 < 接受概率下限 < 1")
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("正则化强度必须是正的有限数值")
    if minimum_per_class < 2:
        raise ValueError("每类最少标注数不能小于 2")
    frame = annotations.copy()
    if "label" not in frame:
        raise ValueError("训练标注缺少 label 列")
    if "split" not in frame:
        frame["split"] = "calibration"
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    frame["split"] = frame["split"].replace({"training": "calibration", "train": "calibration"})
    if "wafer_id" in frame:
        leakage = (
            frame.loc[frame["split"].isin({"calibration", "validation", "locked_test"})]
            .groupby("wafer_id", dropna=False)["split"]
            .nunique()
        )
        leaked_wafers = [str(value) for value in leakage[leakage > 1].index]
        if leaked_wafers:
            sample = ", ".join(leaked_wafers[:3])
            raise ValueError(
                "同一晶圆/图像组不能跨 calibration 与留出集合："
                + sample
            )
    binary = frame.loc[frame["label"].isin({POSITIVE_LABEL, NEGATIVE_LABEL})].copy()
    training = binary.loc[binary["split"] == "calibration"].copy()
    if training.empty:
        raise ValueError("尚无可用于训练的 calibration 候选标签")
    matrix, valid = feature_matrix(training)
    training = training.loc[valid].reset_index(drop=True)
    matrix = matrix[valid]
    labels = (training["label"].to_numpy() == POSITIVE_LABEL).astype(np.float64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives < minimum_per_class or negatives < minimum_per_class:
        raise ValueError(
            f"至少需要 {minimum_per_class} 个目标和 {minimum_per_class} 个伪影 calibration 标签；"
            f"当前目标={positives}，伪影={negatives}"
        )

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (matrix - means) / scales
    coefficients, intercept, iterations = _fit_logistic(
        standardized,
        labels,
        regularization=regularization,
    )
    train_probabilities = _sigmoid(standardized @ coefficients + intercept)

    annotation_columns = [
        column
        for column in ("run_id", "defect_id", "label", "split", *RAW_FEATURE_COLUMNS)
        if column in frame
    ]
    annotation_payload = frame.loc[:, annotation_columns].fillna("").to_dict(orient="records")
    annotation_digest = hashlib.sha256(
        json.dumps(
            annotation_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    model: dict[str, Any] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "raw_feature_columns": list(RAW_FEATURE_COLUMNS),
        "trained_at_utc": utc_now_iso(),
        "training_annotation_sha256": annotation_digest,
        "training_sample_count": int(len(training)),
        "training_target_count": positives,
        "training_artifact_count": negatives,
        "excluded_uncertain_count": int((frame["label"] == UNCERTAIN_LABEL).sum()),
        "excluded_nonfinite_training_rows": int(np.sum(~valid)),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "regularization": float(regularization),
        "optimizer_iterations": iterations,
        "accept_threshold": float(accept_threshold),
        "reject_threshold": float(reject_threshold),
        "hard_rejection_reasons": sorted(HARD_REJECTION_REASONS),
        "training_fit_metrics_not_validation": _decision_metrics(
            labels,
            train_probabilities,
            accept_threshold=accept_threshold,
            reject_threshold=reject_threshold,
        ),
        "validation": {
            "status": "not_available",
            "reason": "No held-out validation or locked-test labels with both classes were supplied.",
        },
        "scientific_limit": (
            "The model classifies image candidates from expert labels; it does not by itself "
            "prove that a target is a physical dislocation."
        ),
    }

    for split_name in ("validation", "locked_test"):
        holdout = binary.loc[binary["split"] == split_name].copy()
        if holdout.empty:
            continue
        holdout_matrix, holdout_valid = feature_matrix(holdout)
        holdout = holdout.loc[holdout_valid].reset_index(drop=True)
        holdout_matrix = holdout_matrix[holdout_valid]
        holdout_labels = (holdout["label"].to_numpy() == POSITIVE_LABEL).astype(np.float64)
        if not len(holdout_labels) or len(np.unique(holdout_labels)) < 2:
            continue
        probabilities = _sigmoid(
            ((holdout_matrix - means) / scales) @ coefficients + intercept
        )
        model["validation"] = {
            "status": f"held_out_{split_name}_metrics_available",
            "split": split_name,
            "metrics": _decision_metrics(
                holdout_labels,
                probabilities,
                accept_threshold=accept_threshold,
                reject_threshold=reject_threshold,
            ),
            "physical_identity_validated": False,
        }
        if split_name == "locked_test":
            break

    model["model_sha256"] = _model_digest(model)
    return validate_model(model)


def classifier_probabilities(frame: pd.DataFrame, model: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return target probabilities and a finite-feature mask."""

    normalized = validate_model(model)
    matrix, valid = feature_matrix(frame)
    probabilities = np.full(len(frame), np.nan, dtype=np.float64)
    if np.any(valid):
        means = np.asarray(normalized["means"], dtype=np.float64)
        scales = np.asarray(normalized["scales"], dtype=np.float64)
        coefficients = np.asarray(normalized["coefficients"], dtype=np.float64)
        standardized = (matrix[valid] - means) / scales
        probabilities[valid] = _sigmoid(
            standardized @ coefficients + float(normalized["intercept"])
        )
    return probabilities, valid


def apply_candidate_classifier(
    frame: pd.DataFrame,
    model: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Apply a trained model while preserving rule decisions and hard gates."""

    normalized = validate_model(model)
    result = frame.copy()
    if "accepted" not in result:
        result["accepted"] = True
    if "rejection_reason" not in result:
        result["rejection_reason"] = ""
    result["rule_accepted"] = result["accepted"].astype(bool)
    result["rule_rejection_reason"] = result["rejection_reason"].fillna("").astype(str)
    probabilities, valid = classifier_probabilities(result, normalized)
    result["classifier_applied"] = valid
    result["classifier_probability"] = probabilities
    result["classifier_decision"] = "feature_invalid"
    result["decision_basis"] = "trained_candidate_classifier"

    reasons = result["rule_rejection_reason"].map(
        lambda value: {item.strip() for item in str(value).split(";") if item.strip()}
    )
    hard_blocked = reasons.map(lambda values: bool(values & HARD_REJECTION_REASONS)).to_numpy()
    target = valid & (probabilities >= float(normalized["accept_threshold"])) & ~hard_blocked
    artifact = valid & (probabilities <= float(normalized["reject_threshold"])) & ~hard_blocked
    uncertain = valid & ~target & ~artifact & ~hard_blocked
    result.loc[target, "classifier_decision"] = POSITIVE_LABEL
    result.loc[artifact, "classifier_decision"] = NEGATIVE_LABEL
    result.loc[uncertain, "classifier_decision"] = UNCERTAIN_LABEL
    result.loc[hard_blocked, "classifier_decision"] = "blocked_by_geometry"
    result["accepted"] = target
    result["rejection_reason"] = ""
    result.loc[artifact, "rejection_reason"] = "classifier_artifact"
    result.loc[uncertain, "rejection_reason"] = "classifier_uncertain"
    result.loc[~valid & ~hard_blocked, "rejection_reason"] = "classifier_feature_invalid"
    result.loc[hard_blocked, "rejection_reason"] = result.loc[
        hard_blocked, "rule_rejection_reason"
    ]

    validation = normalized.get("validation", {})
    validation_status = str(validation.get("status", "not_available"))
    report = {
        "status": "applied",
        "model_sha256": normalized["model_sha256"],
        "model_type": MODEL_TYPE,
        "training_sample_count": normalized.get("training_sample_count"),
        "validation_status": validation_status,
        "accept_threshold": normalized["accept_threshold"],
        "reject_threshold": normalized["reject_threshold"],
        "target_count": int(np.sum(target)),
        "artifact_count": int(np.sum(artifact)),
        "uncertain_count": int(np.sum(uncertain)),
        "hard_blocked_count": int(np.sum(hard_blocked)),
        "feature_invalid_count": int(np.sum(~valid)),
        "rule_accept_count_before_classifier": int(result["rule_accepted"].sum()),
        "physical_identity_validated": False,
    }
    warnings: list[str] = []
    if not validation_status.startswith("held_out_"):
        warnings.append(
            "A trained candidate classifier was applied, but no held-out image-label "
            "validation with both classes is available; real-SiC accuracy remains unknown"
        )
    if int(np.sum(uncertain)):
        warnings.append(
            f"Candidate classifier left {int(np.sum(uncertain))} candidates uncertain; "
            "they were excluded from n and require expert review"
        )
    if int(np.sum(~valid)):
        warnings.append(
            f"Candidate classifier could not score {int(np.sum(~valid))} non-finite feature rows; "
            "they were excluded from n"
        )
    return result, report, warnings


def model_from_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Load an enabled inline or file-backed model from configuration."""

    section = config.get("classifier", {})
    if not isinstance(section, Mapping) or not bool(section.get("enabled", False)):
        return None
    inline = section.get("model")
    if isinstance(inline, Mapping):
        return validate_model(inline)
    path_value = section.get("model_path")
    if not path_value:
        raise ValueError("classifier.enabled=true requires classifier.model or classifier.model_path")
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Candidate-classifier model file does not exist: {path.name}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read candidate-classifier model: {path.name}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("Candidate-classifier JSON must contain an object")
    return validate_model(loaded)


__all__ = [
    "HARD_REJECTION_REASONS",
    "MODEL_FEATURE_COLUMNS",
    "MODEL_SCHEMA_VERSION",
    "MODEL_TYPE",
    "NEGATIVE_LABEL",
    "POSITIVE_LABEL",
    "RAW_FEATURE_COLUMNS",
    "TRAINING_LABELS",
    "UNCERTAIN_LABEL",
    "apply_candidate_classifier",
    "classifier_probabilities",
    "feature_matrix",
    "model_from_config",
    "train_candidate_classifier",
    "validate_model",
]
