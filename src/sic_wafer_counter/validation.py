"""Expert-annotation validation utilities for point-target measurement.

This module deliberately evaluates image-rule agreement, not physical defect
identity.  It keeps original reviewer annotations immutable, separates unclear
labels from explicit negatives, and enforces wafer-level data partitions.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .utils import atomic_write_json, atomic_write_text


ANNOTATION_SCHEMA_VERSION = "1.0"
ANNOTATION_COLUMNS: tuple[str, ...] = (
    "image_id",
    "wafer_id",
    "candidate_id",
    "x_px",
    "y_px",
    "x_mm",
    "y_mm",
    "label",
    "reviewer_id",
    "review_confidence",
    "notes",
    "source_image_sha256",
    "annotation_schema_version",
)
CONFIRMED_POINT = "confirmed_point"
EXPLICIT_NEGATIVE_LABELS = frozenset(
    {
        "line_fragment",
        "scratch",
        "particle_or_dust",
        "detector_artifact",
        "large_dark_region",
    }
)
INDETERMINATE_LABELS = frozenset({"possible_point", "uncertain"})
VALID_LABELS = frozenset(
    {CONFIRMED_POINT, *EXPLICIT_NEGATIVE_LABELS, *INDETERMINATE_LABELS}
)
SENSITIVITY_PARAMETERS: tuple[str, ...] = (
    "detection.threshold_offset",
    "filters.min_equivalent_diameter_um",
    "filters.max_equivalent_diameter_um",
    "filters.min_circularity",
    "filters.max_eccentricity",
    "filters.min_solidity",
    "filters.min_contrast",
    "detection.min_peak_distance_um",
)


@dataclass(frozen=True, slots=True)
class ValidationEvaluation:
    """Metrics and audit tables from one automatic-versus-reviewed comparison."""

    metrics: dict[str, Any]
    matches: pd.DataFrame
    per_wafer: pd.DataFrame
    bland_altman: pd.DataFrame


def source_image_sha256(path: str | Path) -> str:
    """Return the content hash stored beside annotations for traceability."""

    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_template(
    *, image_id: str = "", wafer_id: str = "", source_sha256: str = ""
) -> pd.DataFrame:
    """Return an empty, versioned CSV template without fabricated labels."""

    return pd.DataFrame(
        [
            {
                "image_id": image_id,
                "wafer_id": wafer_id,
                "candidate_id": "",
                "x_px": np.nan,
                "y_px": np.nan,
                "x_mm": np.nan,
                "y_mm": np.nan,
                "label": "uncertain",
                "reviewer_id": "",
                "review_confidence": "",
                "notes": "Replace this example row with an actual review decision.",
                "source_image_sha256": source_sha256,
                "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            }
        ],
        columns=ANNOTATION_COLUMNS,
    )


def write_annotation_template(
    path: str | Path, *, image_id: str = "", wafer_id: str = "", source_sha256: str = ""
) -> Path:
    """Write the versioned annotation template used by the validation script."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    annotation_template(
        image_id=image_id, wafer_id=wafer_id, source_sha256=source_sha256
    ).to_csv(destination, index=False)
    return destination


def load_annotations(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and validate raw annotations without changing reviewer decisions."""

    frames: list[pd.DataFrame] = []
    for item in paths:
        path = Path(item).expanduser()
        frame = pd.read_csv(path, dtype={"candidate_id": "string", "reviewer_id": "string"})
        missing = set(ANNOTATION_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing annotation columns: {sorted(missing)}")
        frame = frame.loc[:, ANNOTATION_COLUMNS].copy()
        frame["annotation_source_file"] = str(path.resolve())
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=(*ANNOTATION_COLUMNS, "annotation_source_file"))
    annotations = pd.concat(frames, ignore_index=True)
    annotations["annotation_schema_version"] = annotations[
        "annotation_schema_version"
    ].astype(str)
    versions = set(annotations["annotation_schema_version"])
    if versions != {ANNOTATION_SCHEMA_VERSION}:
        raise ValueError(
            "Unsupported annotation schema version(s): " + ", ".join(sorted(versions))
        )
    annotations["label"] = annotations["label"].astype(str).str.strip().str.lower()
    invalid_labels = sorted(set(annotations["label"]) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(f"Unknown annotation label(s): {invalid_labels}")
    for column in ("image_id", "wafer_id", "reviewer_id"):
        if annotations[column].isna().any() or (annotations[column].astype(str).str.strip() == "").any():
            raise ValueError(f"Annotation column {column} must be populated")
    for column in ("x_px", "y_px", "x_mm", "y_mm"):
        annotations[column] = pd.to_numeric(annotations[column], errors="coerce")
    has_pixel_pair = annotations[["x_px", "y_px"]].notna().all(axis=1)
    has_mm_pair = annotations[["x_mm", "y_mm"]].notna().all(axis=1)
    if not (has_pixel_pair | has_mm_pair).all():
        raise ValueError("Each annotation needs an x/y coordinate pair in pixels or millimetres")
    return annotations


def annotation_provenance_status(
    annotations: pd.DataFrame,
    source_hash_by_image: Mapping[str, str | None],
) -> pd.DataFrame:
    """Check declared annotation hashes against available source-image hashes.

    A missing source file is reported as unverified rather than fabricated as a
    pass.  A declared mismatch is a hard provenance error for downstream
    validation because coordinates may refer to another image revision.
    """

    if annotations.empty:
        return pd.DataFrame(columns=["image_id", "declared_hashes", "source_hash", "status"])
    rows: list[dict[str, Any]] = []
    for image_id, group in annotations.groupby(annotations["image_id"].astype(str), sort=True):
        declared = sorted(
            {
                value.strip().lower()
                for value in group["source_image_sha256"].fillna("").astype(str)
                if value.strip()
            }
        )
        actual = source_hash_by_image.get(str(image_id))
        actual_normalized = actual.strip().lower() if isinstance(actual, str) and actual.strip() else None
        if not declared:
            status = "missing_declared_hash"
        elif len(declared) > 1:
            status = "inconsistent_annotation_hashes"
        elif actual_normalized is None:
            status = "source_file_unavailable_for_verification"
        elif declared[0] == actual_normalized:
            status = "verified"
        else:
            status = "hash_mismatch"
        rows.append(
            {
                "image_id": str(image_id),
                "declared_hashes": ";".join(declared),
                "source_hash": actual_normalized,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _candidate_key(frame: pd.DataFrame) -> pd.Series:
    """Create a stable cross-review key while preserving source columns."""

    candidate = frame["candidate_id"].astype("string").fillna("").str.strip()
    fallback = (
        frame["image_id"].astype(str)
        + "|px:"
        + frame["x_px"].round(3).astype(str)
        + ","
        + frame["y_px"].round(3).astype(str)
    )
    return pd.Series(np.where(candidate != "", frame["image_id"].astype(str) + "|id:" + candidate, fallback), index=frame.index)


def _binary_label(label: str) -> int | None:
    if label == CONFIRMED_POINT:
        return 1
    if label in EXPLICIT_NEGATIVE_LABELS:
        return 0
    return None


def _cohen_kappa(first: Sequence[int], second: Sequence[int]) -> float | None:
    """Cohen's kappa without adding a scikit-learn dependency."""

    if len(first) != len(second) or len(first) < 2:
        return None
    a, b = np.asarray(first, dtype=int), np.asarray(second, dtype=int)
    observed = float(np.mean(a == b))
    expected = sum(float(np.mean(a == value) * np.mean(b == value)) for value in (0, 1))
    if math.isclose(1.0 - expected, 0.0):
        return None
    return (observed - expected) / (1.0 - expected)


def reviewer_agreement(annotations: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute raw/pairwise agreement and retain a list of disagreements."""

    if annotations.empty:
        return {
            "reviewer_count": 0,
            "raw_agreement": None,
            "cohen_kappa": None,
            "inter_reviewer_agreement_status": "inter_reviewer_agreement_not_available",
            "uncertain_fraction": None,
            "per_class_agreement": {},
        }, pd.DataFrame()
    raw = annotations.copy()
    raw["candidate_key"] = _candidate_key(raw)
    pair_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    for key, group in raw.groupby("candidate_key", sort=True):
        # Multiple decisions from the same reviewer are retained in raw data but
        # cannot form a unique agreement pair without an explicit adjudication.
        unique = group.drop_duplicates("reviewer_id", keep="last")
        labels = unique["label"].tolist()
        if len(set(labels)) > 1:
            disagreement_rows.append(
                {
                    "candidate_key": key,
                    "image_id": str(unique.iloc[0]["image_id"]),
                    "wafer_id": str(unique.iloc[0]["wafer_id"]),
                    "reviewer_labels": json.dumps(
                        dict(zip(unique["reviewer_id"].astype(str), labels)), ensure_ascii=False
                    ),
                    "raw_annotation_count": int(len(group)),
                }
            )
        for first, second in combinations(unique.itertuples(index=False), 2):
            pair_rows.append(
                {
                    "candidate_key": key,
                    "reviewer_a": str(first.reviewer_id),
                    "reviewer_b": str(second.reviewer_id),
                    "label_a": str(first.label),
                    "label_b": str(second.label),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    reviewers = sorted(raw["reviewer_id"].astype(str).unique())
    summary: dict[str, Any] = {
        "reviewer_count": len(reviewers),
        "reviewer_ids": reviewers,
        "uncertain_fraction": float(raw["label"].isin(INDETERMINATE_LABELS).mean()),
        "per_class_agreement": {},
    }
    if pairs.empty:
        summary.update(
            {
                "raw_agreement": None,
                "cohen_kappa": None,
                "inter_reviewer_agreement_status": "inter_reviewer_agreement_not_available",
            }
        )
    else:
        summary["raw_agreement"] = float((pairs["label_a"] == pairs["label_b"]).mean())
        binary = pairs.assign(
            binary_a=pairs["label_a"].map(_binary_label),
            binary_b=pairs["label_b"].map(_binary_label),
        ).dropna(subset=["binary_a", "binary_b"])
        summary["cohen_kappa"] = _cohen_kappa(
            binary["binary_a"].astype(int).tolist(), binary["binary_b"].astype(int).tolist()
        )
        summary["inter_reviewer_agreement_status"] = "available" if len(binary) >= 2 else "insufficient_clear_binary_pairs"
    for label in sorted(VALID_LABELS):
        if pairs.empty:
            summary["per_class_agreement"][label] = None
            continue
        either = (pairs["label_a"] == label) | (pairs["label_b"] == label)
        summary["per_class_agreement"][label] = (
            float(((pairs["label_a"] == label) & (pairs["label_b"] == label)).sum() / either.sum())
            if either.any()
            else None
        )
    return summary, pd.DataFrame(disagreement_rows)


def arbitrate_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    """Return separate adjudicated labels; raw labels are not overwritten."""

    if annotations.empty:
        return pd.DataFrame(
            columns=("candidate_key", "image_id", "wafer_id", "label", "arbitration_status", "reviewer_count")
        )
    raw = annotations.copy()
    raw["candidate_key"] = _candidate_key(raw)
    rows: list[dict[str, Any]] = []
    for key, group in raw.groupby("candidate_key", sort=True):
        counts = group["label"].value_counts()
        top_count = int(counts.max())
        top_labels = sorted(counts[counts == top_count].index.tolist())
        if len(top_labels) == 1:
            label = str(top_labels[0])
            status = "single_reviewer" if len(group) == 1 else ("unanimous" if len(counts) == 1 else "majority")
        else:
            label, status = "uncertain", "tie_to_uncertain"
        first = group.iloc[0]
        rows.append(
            {
                "candidate_key": key,
                "image_id": str(first["image_id"]),
                "wafer_id": str(first["wafer_id"]),
                "candidate_id": str(first["candidate_id"]),
                "x_px": float(first["x_px"]),
                "y_px": float(first["y_px"]),
                "x_mm": float(first["x_mm"]),
                "y_mm": float(first["y_mm"]),
                "label": label,
                "arbitration_status": status,
                "reviewer_count": int(group["reviewer_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def assign_wafer_splits(
    wafer_ids: Iterable[str],
    *,
    calibration_wafers: Iterable[str],
    validation_wafers: Iterable[str],
    locked_test_wafers: Iterable[str],
) -> pd.DataFrame:
    """Enforce mutually exclusive wafer-level calibration/validation/test sets."""

    known = {str(item) for item in wafer_ids}
    groups = {
        "calibration": {str(item) for item in calibration_wafers},
        "validation": {str(item) for item in validation_wafers},
        "locked_test": {str(item) for item in locked_test_wafers},
    }
    for first, second in combinations(groups, 2):
        overlap = groups[first] & groups[second]
        if overlap:
            raise ValueError(f"Wafer-level data leakage: {first}/{second} overlap {sorted(overlap)}")
    assigned = set().union(*groups.values())
    unknown = assigned - known
    missing = known - assigned
    if unknown:
        raise ValueError(f"Split names are not present in annotations/results: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Every wafer must have one split assignment; missing {sorted(missing)}")
    return pd.DataFrame(
        [
            {"wafer_id": wafer_id, "split": split}
            for split, group in groups.items()
            for wafer_id in sorted(group)
        ]
    )


def _distance_um(
    automatic: pd.DataFrame,
    reviewed: pd.DataFrame,
    *,
    mm_per_pixel_by_image: Mapping[str, float] | None,
) -> np.ndarray:
    """Pairwise coordinate distances in micrometres, preferring physical coordinates."""

    auto_mm = automatic[["x_mm", "y_mm"]].apply(pd.to_numeric, errors="coerce")
    review_mm = reviewed[["x_mm", "y_mm"]].apply(pd.to_numeric, errors="coerce")
    if auto_mm.notna().all().all() and review_mm.notna().all().all():
        dx = auto_mm["x_mm"].to_numpy()[:, None] - review_mm["x_mm"].to_numpy()[None, :]
        dy = auto_mm["y_mm"].to_numpy()[:, None] - review_mm["y_mm"].to_numpy()[None, :]
        return np.hypot(dx, dy) * 1000.0
    if mm_per_pixel_by_image is None:
        raise ValueError("Physical coordinates are required unless mm_per_pixel_by_image is provided")
    auto_px = automatic[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
    review_px = reviewed[["x_px", "y_px"]].to_numpy(dtype=float)
    distances = np.empty((len(automatic), len(reviewed)), dtype=float)
    for row, image_id in enumerate(automatic["image_id"].astype(str)):
        if image_id not in mm_per_pixel_by_image:
            raise ValueError(f"No pixel scale supplied for image {image_id}")
        distances[row] = np.hypot(
            auto_px[row, 0] - review_px[:, 0], auto_px[row, 1] - review_px[:, 1]
        ) * float(mm_per_pixel_by_image[image_id]) * 1000.0
    return distances


def _pairs_within_tolerance(
    automatic: pd.DataFrame,
    reviewed: pd.DataFrame,
    *,
    matching_tolerance_um: float,
    mm_per_pixel_by_image: Mapping[str, float] | None,
) -> list[tuple[int, int, float]]:
    """One-to-one optimal coordinate matches for one image/label category."""

    if automatic.empty or reviewed.empty:
        return []
    distances = _distance_um(
        automatic, reviewed, mm_per_pixel_by_image=mm_per_pixel_by_image)
    # Invalid pairs must not displace a valid match in the assignment.
    penalty = float(max(np.nanmax(distances, initial=0.0), matching_tolerance_um) + matching_tolerance_um + 1.0)
    costs = np.where(distances <= matching_tolerance_um, distances, penalty)
    rows, columns = linear_sum_assignment(costs)
    return [
        (int(row), int(column), float(distances[row, column]))
        for row, column in zip(rows, columns)
        if distances[row, column] <= matching_tolerance_um
    ]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else float(numerator / denominator)


def evaluate_automatic_detections(
    automatic_candidates: pd.DataFrame,
    reviewed_annotations: pd.DataFrame,
    *,
    matching_tolerance_um: float,
    annotation_coverage_complete: bool,
    area_cm2_by_wafer: Mapping[str, float] | None = None,
    mm_per_pixel_by_image: Mapping[str, float] | None = None,
) -> ValidationEvaluation:
    """Compare accepted automatic points to adjudicated expert annotations.

    ``possible_point`` and ``uncertain`` are never silently converted to
    negatives.  With incomplete annotation coverage, unmatched automatic points
    are reported as indeterminate and precision/F1 are intentionally withheld.
    """

    if not math.isfinite(float(matching_tolerance_um)) or matching_tolerance_um <= 0:
        raise ValueError("matching_tolerance_um must be finite and positive")
    auto = automatic_candidates.copy()
    if "accepted" in auto:
        auto = auto[auto["accepted"].astype(bool)].copy()
    required_auto = {"image_id", "wafer_id", "centroid_x_px", "centroid_y_px", "x_mm", "y_mm"}
    missing_auto = required_auto - set(auto.columns)
    if missing_auto:
        raise ValueError(f"Automatic candidate table is missing {sorted(missing_auto)}")
    reviewed = reviewed_annotations.copy()
    if not set(("image_id", "wafer_id", "label", "x_px", "y_px", "x_mm", "y_mm")) <= set(reviewed):
        raise ValueError("Reviewed annotations do not contain the validation schema fields")
    rows: list[dict[str, Any]] = []
    used_auto: set[int] = set()
    used_review: set[int] = set()
    all_image_ids = sorted(set(auto["image_id"].astype(str)) | set(reviewed["image_id"].astype(str)))
    for image_id in all_image_ids:
        auto_image = auto[auto["image_id"].astype(str) == image_id]
        review_image = reviewed[reviewed["image_id"].astype(str) == image_id]
        positive = review_image[review_image["label"] == CONFIRMED_POINT]
        for left, right, distance in _pairs_within_tolerance(
            auto_image, positive,
            matching_tolerance_um=matching_tolerance_um,
            mm_per_pixel_by_image=mm_per_pixel_by_image,
        ):
            auto_index, review_index = auto_image.index[left], positive.index[right]
            used_auto.add(int(auto_index)); used_review.add(int(review_index))
            record = auto.loc[auto_index].to_dict()
            record.update({"match_class": "TP", "review_index": int(review_index), "match_distance_um": distance})
            rows.append(record)
        remaining_auto = auto_image[~auto_image.index.isin(used_auto)]
        for label in (*EXPLICIT_NEGATIVE_LABELS, *INDETERMINATE_LABELS):
            labelled = review_image[(review_image["label"] == label) & ~review_image.index.isin(used_review)]
            for left, right, distance in _pairs_within_tolerance(
                remaining_auto, labelled,
                matching_tolerance_um=matching_tolerance_um,
                mm_per_pixel_by_image=mm_per_pixel_by_image,
            ):
                auto_index, review_index = remaining_auto.index[left], labelled.index[right]
                if int(auto_index) in used_auto:
                    continue
                used_auto.add(int(auto_index)); used_review.add(int(review_index))
                record = auto.loc[auto_index].to_dict()
                record.update(
                    {
                        "match_class": "FP" if label in EXPLICIT_NEGATIVE_LABELS else "indeterminate",
                        "review_index": int(review_index),
                        "match_distance_um": distance,
                        "review_label": label,
                    }
                )
                rows.append(record)
            remaining_auto = auto_image[~auto_image.index.isin(used_auto)]
    for index, record in auto.loc[~auto.index.isin(used_auto)].iterrows():
        row = record.to_dict()
        row.update(
            {
                "match_class": "FP" if annotation_coverage_complete else "indeterminate",
                "review_index": None,
                "match_distance_um": None,
            }
        )
        rows.append(row)
    for index, record in reviewed[(reviewed["label"] == CONFIRMED_POINT) & ~reviewed.index.isin(used_review)].iterrows():
        row = record.to_dict()
        row.update({"match_class": "FN", "review_index": int(index), "match_distance_um": None})
        rows.append(row)
    matches = pd.DataFrame(rows)
    if matches.empty:
        matches = pd.DataFrame(columns=["match_class", "wafer_id", "image_id"])
    counts = matches["match_class"].value_counts()
    tp, fp, fn = (int(counts.get(name, 0)) for name in ("TP", "FP", "FN"))
    indeterminate = int(counts.get("indeterminate", 0))
    precision = _safe_ratio(tp, tp + fp) if annotation_coverage_complete else None
    recall = _safe_ratio(tp, tp + fn)
    f1 = (
        _safe_ratio(2.0 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    total_area = float(sum((area_cm2_by_wafer or {}).values()))
    automatic_count = int((matches["match_class"] != "FN").sum())
    reviewed_count = tp + fn
    automatic_density = _safe_ratio(automatic_count, total_area)
    reviewed_density = _safe_ratio(reviewed_count, total_area)
    density_difference = (
        None
        if automatic_density is None or reviewed_density is None
        else automatic_density - reviewed_density
    )
    metrics = {
        "validation_status": "validated" if annotation_coverage_complete else "partial_annotation_coverage",
        "matching_tolerance_um": float(matching_tolerance_um),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "indeterminate_automatic_count": indeterminate,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "false_positives_per_cm2": _safe_ratio(fp, total_area),
        "false_negatives_per_cm2": _safe_ratio(fn, total_area),
        "automatic_count": automatic_count,
        "reviewed_confirmed_count": reviewed_count,
        "automatic_density_cm2": automatic_density,
        "reviewed_density_cm2": reviewed_density,
        "automatic_minus_reviewed_density_cm2": density_difference,
        "relative_density_bias": (
            None
            if density_difference is None or reviewed_density in (None, 0.0)
            else density_difference / reviewed_density
        ),
    }
    per_wafer_rows: list[dict[str, Any]] = []
    for wafer_id in sorted(set(auto["wafer_id"].astype(str)) | set(reviewed["wafer_id"].astype(str))):
        wafer_matches = matches[matches["wafer_id"].astype(str) == wafer_id]
        wafer_counts = wafer_matches["match_class"].value_counts()
        wafer_tp, wafer_fp, wafer_fn = (int(wafer_counts.get(name, 0)) for name in ("TP", "FP", "FN"))
        area = None if area_cm2_by_wafer is None else area_cm2_by_wafer.get(wafer_id)
        auto_count = int((wafer_matches["match_class"] != "FN").sum())
        reviewed_count = wafer_tp + wafer_fn
        per_wafer_rows.append(
            {
                "wafer_id": wafer_id,
                "TP": wafer_tp,
                "FP": wafer_fp,
                "FN": wafer_fn,
                "precision": _safe_ratio(wafer_tp, wafer_tp + wafer_fp) if annotation_coverage_complete else None,
                "recall": _safe_ratio(wafer_tp, wafer_tp + wafer_fn),
                "automatic_count": auto_count,
                "reviewed_confirmed_count": reviewed_count,
                "valid_area_cm2": area,
                "automatic_density_cm2": _safe_ratio(auto_count, float(area)) if area else None,
                "reviewed_density_cm2": _safe_ratio(reviewed_count, float(area)) if area else None,
            }
        )
    per_wafer = pd.DataFrame(per_wafer_rows)
    bland_altman = per_wafer.loc[:, ["wafer_id", "automatic_density_cm2", "reviewed_density_cm2"]].copy()
    bland_altman["mean_density_cm2"] = bland_altman[["automatic_density_cm2", "reviewed_density_cm2"]].mean(axis=1)
    bland_altman["difference_auto_minus_reviewed_cm2"] = (
        bland_altman["automatic_density_cm2"] - bland_altman["reviewed_density_cm2"]
    )
    return ValidationEvaluation(metrics, matches, per_wafer, bland_altman)


def regional_validation_metrics(
    matches: pd.DataFrame, *, wafer_radius_mm: float = 50.0
) -> pd.DataFrame:
    """Descriptive center/middle/edge metrics; CNR is withheld if absent."""

    if matches.empty or "match_class" not in matches:
        return pd.DataFrame(columns=["stratum_type", "stratum", "count", "TP", "FP", "FN", "precision", "recall"])
    data = matches.copy()
    if {"x_mm", "y_mm"} <= set(data):
        radial = np.hypot(pd.to_numeric(data["x_mm"], errors="coerce"), pd.to_numeric(data["y_mm"], errors="coerce"))
    else:
        radial = pd.Series(np.nan, index=data.index)
    normalized = radial / float(wafer_radius_mm)
    data["radial_region"] = pd.cut(
        normalized,
        bins=[-np.inf, 0.33, 0.67, 1.0 + np.finfo(float).eps],
        labels=["center", "middle", "edge"],
    )
    rows: list[dict[str, Any]] = []
    for region, group in data.groupby("radial_region", observed=False):
        counts = group["match_class"].value_counts()
        tp, fp, fn = (int(counts.get(name, 0)) for name in ("TP", "FP", "FN"))
        rows.append(
            {
                "stratum_type": "radial_region",
                "stratum": str(region),
                "count": int(len(group)),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": _safe_ratio(tp, tp + fp),
                "recall": _safe_ratio(tp, tp + fn),
                "note": "Descriptive only; no CNR stratum without local-noise measurements.",
            }
        )
    return pd.DataFrame(rows)


def bootstrap_classification_bias(
    per_wafer: pd.DataFrame, *, seed: int = 0, samples: int = 2000
) -> dict[str, Any]:
    """Bootstrap automatic-minus-reviewed density bias by wafer, never by patch."""

    required = {"automatic_count", "reviewed_confirmed_count", "valid_area_cm2"}
    if not required <= set(per_wafer) or len(per_wafer) < 2:
        return {"status": "not quantified", "reason": "at least two annotated wafers are required"}
    valid = per_wafer.dropna(subset=list(required)).copy()
    valid = valid[valid["valid_area_cm2"].astype(float) > 0]
    if len(valid) < 2:
        return {"status": "not quantified", "reason": "at least two wafers with known valid area are required"}
    rng = np.random.default_rng(seed)
    auto = valid["automatic_count"].to_numpy(dtype=float)
    reviewed = valid["reviewed_confirmed_count"].to_numpy(dtype=float)
    area = valid["valid_area_cm2"].to_numpy(dtype=float)
    draws = np.empty(int(samples), dtype=float)
    for index in range(len(draws)):
        choice = rng.integers(0, len(valid), size=len(valid))
        draws[index] = (auto[choice].sum() - reviewed[choice].sum()) / area[choice].sum()
    return {
        "status": "estimated_from_wafer_bootstrap",
        "seed": int(seed),
        "bootstrap_samples": int(samples),
        "bias_mean_cm2": float(np.mean(draws)),
        "bias_lower_95_cm2": float(np.percentile(draws, 2.5)),
        "bias_upper_95_cm2": float(np.percentile(draws, 97.5)),
    }


def local_parameter_values(value: float) -> list[float]:
    """Return the prescribed local -20/-10/current/+10/+20 percent sweep."""

    current = float(value)
    if not math.isfinite(current):
        raise ValueError("Sensitivity parameter must be finite")
    if current == 0.0:
        return [-0.02, -0.01, 0.0, 0.01, 0.02]
    return [current * factor for factor in (0.8, 0.9, 1.0, 1.1, 1.2)]


def run_local_parameter_sensitivity(
    parameters: Mapping[str, float],
    *,
    calibration_wafer_ids: Sequence[str],
    locked_test_wafer_ids: Sequence[str],
    evaluator: Callable[[str, float, Sequence[str]], Mapping[str, Any]],
) -> pd.DataFrame:
    """Run a small fixed sensitivity sweep, explicitly excluding locked tests."""

    overlap = set(calibration_wafer_ids) & set(locked_test_wafer_ids)
    if overlap:
        raise ValueError(f"Locked test leakage in sensitivity analysis: {sorted(overlap)}")
    rows: list[dict[str, Any]] = []
    for parameter, current in parameters.items():
        for value in local_parameter_values(float(current)):
            outcome = dict(evaluator(parameter, value, tuple(calibration_wafer_ids)))
            outcome.update({"parameter": parameter, "value": value, "split": "calibration"})
            rows.append(outcome)
    return pd.DataFrame(rows)


def spatial_heterogeneity_indicator(counts: Sequence[float], areas_cm2: Sequence[float]) -> dict[str, Any]:
    """Flag descriptive overdispersion relative to an equal-density Poisson model."""

    count = np.asarray(counts, dtype=float)
    area = np.asarray(areas_cm2, dtype=float)
    valid = np.isfinite(count) & np.isfinite(area) & (area > 0)
    if valid.sum() < 2 or count[valid].sum() <= 0:
        return {"status": "not quantified", "reason": "need at least two nonempty area blocks"}
    count, area = count[valid], area[valid]
    expected = count.sum() * area / area.sum()
    pearson = float(np.sum((count - expected) ** 2 / np.maximum(expected, np.finfo(float).eps)))
    ratio = pearson / max(1, len(count) - 1)
    return {
        "status": "descriptive",
        "block_count": int(len(count)),
        "pearson_chi_square": pearson,
        "dispersion_ratio": ratio,
        "overdispersion_indicated": bool(ratio > 1.5),
        "note": "A descriptive flag, not a replacement for experimental replication.",
    }


def write_unvalidated_bundle(output_dir: str | Path) -> dict[str, Path]:
    """Write only a transparent no-real-validation status and annotation template."""

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    summary = {
        "validation_status": "not validated on real SiC data",
        "performance_metrics_reported": False,
        "classification_uncertainty": "not quantified",
        "reason": "No real expert annotations were supplied.",
    }
    summary_path = atomic_write_json(folder / "validation_summary.json", summary)
    template_path = write_annotation_template(folder / "annotation_template.csv")
    instructions_path = atomic_write_text(
        folder / "ANNOTATION_README.txt",
        "Use annotation_template.csv with the documented schema. confirmed_point is the only "
        "clear positive; possible_point and uncertain are not negatives. Preserve raw reviewer "
        "files and split calibration/validation/locked-test sets by wafer_id.\n",
    )
    return {"validation_summary": summary_path, "annotation_template": template_path, "instructions": instructions_path}


def write_validation_outputs(
    output_dir: str | Path,
    *,
    summary: Mapping[str, Any],
    evaluation: ValidationEvaluation,
    regional_metrics: pd.DataFrame,
    disagreements: pd.DataFrame,
    parameter_sensitivity: pd.DataFrame,
    uncertainty_budget: pd.DataFrame,
) -> dict[str, Path]:
    """Persist auditable CSV/JSON/HTML validation outputs from actual labels."""

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "validation_summary": atomic_write_json(folder / "validation_summary.json", dict(summary)),
    }
    metric_frame = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in evaluation.metrics.items()]
    )
    tables = {
        "validation_metrics": metric_frame,
        "per_wafer_metrics": evaluation.per_wafer,
        "per_region_metrics": regional_metrics,
        "annotation_disagreements": disagreements,
        "parameter_sensitivity": parameter_sensitivity,
        "uncertainty_budget": uncertainty_budget,
        "bland_altman": evaluation.bland_altman,
        "match_audit": evaluation.matches,
    }
    for name, frame in tables.items():
        path = folder / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    headline = html.escape(json.dumps(dict(summary), ensure_ascii=False, indent=2, default=str))
    report = "<html><body><h1>SiC 点状目标真实标注验证</h1><pre>" + headline + "</pre>"
    report += "<p>性能指标只描述当前图像规则与专家标注的一致性，不证明每个黑点必为真实物理位错。</p>"
    report += "</body></html>\n"
    paths["validation_report"] = atomic_write_text(folder / "validation_report.html", report)
    return paths


__all__ = [
    "ANNOTATION_COLUMNS",
    "ANNOTATION_SCHEMA_VERSION",
    "CONFIRMED_POINT",
    "EXPLICIT_NEGATIVE_LABELS",
    "INDETERMINATE_LABELS",
    "SENSITIVITY_PARAMETERS",
    "ValidationEvaluation",
    "annotation_template",
    "annotation_provenance_status",
    "arbitrate_annotations",
    "assign_wafer_splits",
    "bootstrap_classification_bias",
    "evaluate_automatic_detections",
    "load_annotations",
    "local_parameter_values",
    "regional_validation_metrics",
    "reviewer_agreement",
    "run_local_parameter_sensitivity",
    "source_image_sha256",
    "spatial_heterogeneity_indicator",
    "write_annotation_template",
    "write_unvalidated_bundle",
    "write_validation_outputs",
]
