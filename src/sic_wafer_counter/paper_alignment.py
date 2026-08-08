"""Paper-aligned XRT reference overlays and quantitative agreement checks.

The Rigaku articles show automatically detected XRT targets as red rectangles
and independently observed DIC pits as yellow circles.  This module preserves
that distinction.  A yellow marker is accepted only from a registered,
hash-traceable external reference image; reference points never change the
automatic candidate count or density.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


REFERENCE_SCHEMA_VERSION = "1.0"
REFERENCE_COLUMNS: tuple[str, ...] = (
    "reference_id",
    "x_px",
    "y_px",
    "label",
    "reference_method",
    "registration_status",
    "registration_rmse_um",
    "source_image_sha256",
    "reference_image_sha256",
    "notes",
    "reference_schema_version",
)
CONFIRMED_REFERENCE_LABEL = "confirmed_point"
ALLOWED_REFERENCE_LABELS = frozenset(
    {CONFIRMED_REFERENCE_LABEL, "possible_point", "uncertain"}
)
ALLOWED_REFERENCE_METHODS = frozenset({"dic", "koh", "expert_annotation"})
ALLOWED_REGISTRATION_STATUSES = frozenset({"registered", "not_registered"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_ASSIGNMENT_COMPONENT_CELLS = 2_000_000


@dataclass(frozen=True, slots=True)
class RegisteredReferenceSet:
    """Validated external reference coordinates and provenance."""

    points: pd.DataFrame
    all_rows: pd.DataFrame
    summary: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    """Hash a source or independent-reference image for provenance."""

    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_template(
    *,
    source_image_sha256: str = "",
    reference_image_sha256: str = "",
    reference_method: str = "DIC",
) -> pd.DataFrame:
    """Return a one-row editable template without fabricating a confirmation."""

    return pd.DataFrame(
        [
            {
                "reference_id": "replace-me",
                "x_px": np.nan,
                "y_px": np.nan,
                "label": "uncertain",
                "reference_method": reference_method,
                "registration_status": "not_registered",
                "registration_rmse_um": np.nan,
                "source_image_sha256": source_image_sha256,
                "reference_image_sha256": reference_image_sha256,
                "notes": "Replace this row with registered independent observations.",
                "reference_schema_version": REFERENCE_SCHEMA_VERSION,
            }
        ],
        columns=REFERENCE_COLUMNS,
    )


def write_reference_template(
    output_path: str | Path,
    *,
    source_image_path: str | Path,
    reference_image_path: str | Path,
    reference_method: str = "DIC",
) -> Path:
    """Write a hash-bound registered-reference CSV template."""

    method = str(reference_method).strip().lower()
    if method not in ALLOWED_REFERENCE_METHODS:
        raise ValueError(
            "reference_method must be DIC, KOH, or expert_annotation"
        )
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    reference_template(
        source_image_sha256=file_sha256(source_image_path),
        reference_image_sha256=file_sha256(reference_image_path),
        reference_method=method.upper() if method in {"dic", "koh"} else method,
    ).to_csv(destination, index=False)
    return destination


def _normalise_hash(value: Any, *, field: str) -> str:
    result = str(value).strip().lower()
    if not SHA256_PATTERN.fullmatch(result):
        raise ValueError(f"{field} must contain a full 64-character SHA-256")
    return result


def _configuration_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0", "", "none"}:
        return False
    raise ValueError(f"{field} must be a boolean")


def load_registered_reference_points(
    csv_path: str | Path,
    *,
    source_image_path: str | Path,
    reference_image_path: str | Path,
    source_shape: tuple[int, int],
) -> RegisteredReferenceSet:
    """Load independently registered reference points with strict provenance.

    Only rows labelled ``confirmed_point`` and ``registered`` become yellow
    markers.  Both the XRT source image and independent reference image hashes
    must match the CSV.  Possible/uncertain rows remain in the audit table but
    cannot be rendered as confirmed physical evidence.
    """

    path = Path(csv_path).expanduser()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REFERENCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing reference columns: {sorted(missing)}")
    frame = frame.loc[:, REFERENCE_COLUMNS].copy()
    if frame.empty:
        raise ValueError("Registered reference CSV contains no rows")
    versions = set(frame["reference_schema_version"].str.strip())
    if versions != {REFERENCE_SCHEMA_VERSION}:
        raise ValueError(
            "Unsupported reference schema version(s): " + ", ".join(sorted(versions))
        )
    frame["label"] = frame["label"].str.strip().str.lower()
    invalid_labels = sorted(set(frame["label"]) - ALLOWED_REFERENCE_LABELS)
    if invalid_labels:
        raise ValueError(f"Unknown independent-reference label(s): {invalid_labels}")
    frame["reference_method"] = frame["reference_method"].str.strip().str.lower()
    invalid_methods = sorted(set(frame["reference_method"]) - ALLOWED_REFERENCE_METHODS)
    if invalid_methods:
        raise ValueError(f"Unknown independent-reference method(s): {invalid_methods}")
    if len(set(frame["reference_method"])) != 1:
        raise ValueError("One reference CSV must use exactly one independent method")
    frame["reference_id"] = frame["reference_id"].str.strip()
    if frame["reference_id"].eq("").any():
        raise ValueError("reference_id must be populated")
    if frame["reference_id"].duplicated().any():
        raise ValueError("reference_id values must be unique")

    source_hashes = {
        _normalise_hash(value, field="source_image_sha256")
        for value in frame["source_image_sha256"]
    }
    reference_hashes = {
        _normalise_hash(value, field="reference_image_sha256")
        for value in frame["reference_image_sha256"]
    }
    if len(source_hashes) != 1 or len(reference_hashes) != 1:
        raise ValueError("Reference CSV must declare one consistent source and reference hash")
    actual_source_hash = file_sha256(source_image_path)
    actual_reference_hash = file_sha256(reference_image_path)
    if actual_source_hash == actual_reference_hash:
        raise ValueError(
            "The independent reference image must not be the same file content as the XRT source"
        )
    if next(iter(source_hashes)) != actual_source_hash:
        raise ValueError("Registered-reference source_image_sha256 does not match the XRT input")
    if next(iter(reference_hashes)) != actual_reference_hash:
        raise ValueError(
            "Registered-reference reference_image_sha256 does not match the supplied independent image"
        )

    frame["registration_status"] = frame["registration_status"].str.strip().str.lower()
    invalid_statuses = sorted(
        set(frame["registration_status"]) - ALLOWED_REGISTRATION_STATUSES
    )
    if invalid_statuses:
        raise ValueError(f"Unknown registration_status value(s): {invalid_statuses}")
    frame["x_px"] = pd.to_numeric(frame["x_px"], errors="coerce")
    frame["y_px"] = pd.to_numeric(frame["y_px"], errors="coerce")
    rmse_text = frame["registration_rmse_um"].str.strip()
    frame["registration_rmse_um"] = pd.to_numeric(
        frame["registration_rmse_um"], errors="coerce"
    )
    if (rmse_text.ne("") & frame["registration_rmse_um"].isna()).any():
        raise ValueError("registration_rmse_um must be numeric or blank")
    confirmed = frame["label"].eq(CONFIRMED_REFERENCE_LABEL)
    registered = frame["registration_status"].eq("registered")
    eligible = confirmed & registered
    if not eligible.any():
        raise ValueError(
            "No confirmed_point row has registration_status=registered; no yellow markers can be drawn"
        )
    points = frame.loc[eligible].copy()
    if points[["x_px", "y_px"]].isna().any().any():
        raise ValueError("Every registered confirmed reference needs finite x_px and y_px")
    source_height, source_width = map(int, source_shape[:2])
    if source_height <= 0 or source_width <= 0:
        raise ValueError("source_shape must contain positive height and width")
    inside = (
        points["x_px"].between(0, source_width - 1)
        & points["y_px"].between(0, source_height - 1)
    )
    if not inside.all():
        raise ValueError("Registered confirmed reference coordinates fall outside the XRT image")
    coordinate_keys = points[["x_px", "y_px"]].round(6)
    if coordinate_keys.duplicated().any():
        raise ValueError("Registered confirmed reference coordinates contain duplicates")
    finite_rmse = points["registration_rmse_um"].dropna()
    if (finite_rmse < 0).any():
        raise ValueError("registration_rmse_um cannot be negative")
    method = str(points.iloc[0]["reference_method"])
    summary = {
        "status": "verified_registered_reference_supplied",
        "method": method.upper() if method in {"dic", "koh"} else method,
        "confirmed_registered_count": int(len(points)),
        "possible_or_uncertain_count": int((~confirmed).sum()),
        "unregistered_confirmed_count": int((confirmed & ~registered).sum()),
        "source_image_sha256": actual_source_hash,
        "reference_image_sha256": actual_reference_hash,
        "registration_rmse_um_max": (
            float(finite_rmse.max()) if not finite_rmse.empty else None
        ),
        "registration_uncertainty_status": (
            "quantified" if not finite_rmse.empty else "not quantified"
        ),
        "automatic_count_changed_by_reference": False,
    }
    return RegisteredReferenceSet(points=points, all_rows=frame, summary=summary)


def compare_automatic_to_registered_reference(
    automatic_candidates: pd.DataFrame,
    reference_points: pd.DataFrame,
    *,
    mm_per_pixel: float,
    matching_tolerance_um: float,
    reference_coverage_complete: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Match accepted automatic candidates to registered reference points."""

    if not np.isfinite(mm_per_pixel) or mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be finite and positive")
    if not np.isfinite(matching_tolerance_um) or matching_tolerance_um <= 0:
        raise ValueError("matching_tolerance_um must be finite and positive")
    automatic = automatic_candidates.copy()
    if "accepted" in automatic:
        accepted = automatic["accepted"].map(
            lambda value: str(value).strip().lower()
            in {"1", "true", "yes", "accepted", "accept"}
        )
        automatic = automatic.loc[accepted].copy()
    automatic = automatic.reset_index(drop=True)
    references = reference_points.reset_index(drop=True)
    pairs: list[tuple[int, int, float]] = []
    if not automatic.empty and not references.empty:
        automatic_xy = automatic[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
        reference_xy = references[["x_px", "y_px"]].to_numpy(dtype=float)
        if not np.isfinite(automatic_xy).all() or not np.isfinite(reference_xy).all():
            raise ValueError("Automatic and registered-reference coordinates must be finite")
        tolerance_px = float(matching_tolerance_um) / (float(mm_per_pixel) * 1000.0)
        tree = cKDTree(automatic_xy)
        reference_to_automatic = tree.query_ball_point(reference_xy, r=tolerance_px)
        automatic_to_reference: dict[int, list[int]] = {}
        for reference_index, automatic_indices in enumerate(reference_to_automatic):
            for automatic_index in automatic_indices:
                automatic_to_reference.setdefault(int(automatic_index), []).append(
                    int(reference_index)
                )

        visited_automatic: set[int] = set()
        visited_reference: set[int] = set()
        for seed_reference, neighbours in enumerate(reference_to_automatic):
            if not neighbours or seed_reference in visited_reference:
                continue
            component_automatic: set[int] = set()
            component_reference: set[int] = set()
            pending: list[tuple[str, int]] = [("reference", int(seed_reference))]
            while pending:
                kind, index = pending.pop()
                if kind == "reference":
                    if index in component_reference:
                        continue
                    component_reference.add(index)
                    visited_reference.add(index)
                    pending.extend(
                        ("automatic", int(item))
                        for item in reference_to_automatic[index]
                    )
                else:
                    if index in component_automatic:
                        continue
                    component_automatic.add(index)
                    visited_automatic.add(index)
                    pending.extend(
                        ("reference", int(item))
                        for item in automatic_to_reference.get(index, [])
                    )

            automatic_indices = np.asarray(sorted(component_automatic), dtype=int)
            reference_indices = np.asarray(sorted(component_reference), dtype=int)
            component_cells = int(automatic_indices.size * reference_indices.size)
            if component_cells > MAX_ASSIGNMENT_COMPONENT_CELLS:
                raise ValueError(
                    "Independent-reference matching has an excessively dense local "
                    f"component ({automatic_indices.size} x {reference_indices.size}); "
                    "reduce matching_tolerance_um or validate separately registered fields"
                )
            distances = np.hypot(
                automatic_xy[automatic_indices, None, 0]
                - reference_xy[None, reference_indices, 0],
                automatic_xy[automatic_indices, None, 1]
                - reference_xy[None, reference_indices, 1],
            ) * float(mm_per_pixel) * 1000.0
            penalty = float(matching_tolerance_um) * 2.0 + 1.0
            local_automatic, local_reference = linear_sum_assignment(
                np.where(distances <= matching_tolerance_um, distances, penalty)
            )
            pairs.extend(
                (
                    int(automatic_indices[automatic_index]),
                    int(reference_indices[reference_index]),
                    float(distances[automatic_index, reference_index]),
                )
                for automatic_index, reference_index in zip(
                    local_automatic, local_reference
                )
                if distances[automatic_index, reference_index]
                <= matching_tolerance_um
            )

    matched_auto = {row for row, _, _ in pairs}
    matched_reference = {column for _, column, _ in pairs}
    audit_rows: list[dict[str, Any]] = []
    for row, column, distance in pairs:
        audit_rows.append(
            {
                "match_status": "matched",
                "defect_id": automatic.iloc[row].get("defect_id"),
                "reference_id": references.iloc[column].get("reference_id"),
                "distance_um": distance,
            }
        )
    for row in sorted(set(range(len(automatic))) - matched_auto):
        audit_rows.append(
            {
                "match_status": (
                    "unmatched_automatic_false_positive"
                    if reference_coverage_complete
                    else "unmatched_automatic_indeterminate"
                ),
                "defect_id": automatic.iloc[row].get("defect_id"),
                "reference_id": None,
                "distance_um": None,
            }
        )
    for column in sorted(set(range(len(references))) - matched_reference):
        audit_rows.append(
            {
                "match_status": "unmatched_reference_false_negative",
                "defect_id": None,
                "reference_id": references.iloc[column].get("reference_id"),
                "distance_um": None,
            }
        )
    matched_count = len(pairs)
    reference_count = len(references)
    automatic_count = len(automatic)
    recall = matched_count / reference_count if reference_count else None
    precision = (
        matched_count / automatic_count
        if reference_coverage_complete and automatic_count
        else None
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    metrics = {
        "matching_tolerance_um": float(matching_tolerance_um),
        "reference_coverage_complete": bool(reference_coverage_complete),
        "automatic_accepted_count": int(automatic_count),
        "registered_reference_count": int(reference_count),
        "matched_count": int(matched_count),
        "unmatched_reference_count": int(reference_count - matched_count),
        "unmatched_automatic_count": int(automatic_count - matched_count),
        "recall_vs_registered_reference": recall,
        "precision_vs_registered_reference": precision,
        "f1_vs_registered_reference": f1,
        "precision_status": (
            "computed_from_complete_reference_coverage"
            if reference_coverage_complete
            else "not computed: reference coverage not declared complete"
        ),
    }
    return metrics, pd.DataFrame(
        audit_rows,
        columns=("match_status", "defect_id", "reference_id", "distance_um"),
    )


def references_from_config(
    config: Mapping[str, Any],
    *,
    source_image_path: str | Path | None,
    source_shape: tuple[int, int] | None,
    automatic_candidates: pd.DataFrame,
    mm_per_pixel: float | None,
) -> tuple[RegisteredReferenceSet | None, dict[str, Any], pd.DataFrame]:
    """Resolve optional registered-reference configuration for one run."""

    raw = config.get("independent_reference", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("independent_reference configuration must be a mapping")
    if not _configuration_bool(raw.get("enabled", False), field="independent_reference.enabled"):
        return None, {
            "status": "not provided",
            "automatic_count_changed_by_reference": False,
            "physical_validation_claim": False,
        }, pd.DataFrame()
    csv_path = raw.get("csv_path")
    reference_image_path = raw.get("reference_image_path")
    if not csv_path or not reference_image_path:
        raise ValueError(
            "Enabled independent_reference requires csv_path and reference_image_path"
        )
    if source_image_path is None or source_shape is None:
        raise ValueError("Independent-reference verification requires source image path and shape")
    if mm_per_pixel is None:
        raise ValueError("Independent-reference matching requires mm_per_pixel")
    registered = load_registered_reference_points(
        csv_path,
        source_image_path=source_image_path,
        reference_image_path=reference_image_path,
        source_shape=source_shape,
    )
    metrics, audit = compare_automatic_to_registered_reference(
        automatic_candidates,
        registered.points,
        mm_per_pixel=mm_per_pixel,
        matching_tolerance_um=float(raw.get("matching_tolerance_um", 75.0)),
        reference_coverage_complete=_configuration_bool(
            raw.get("coverage_complete", False),
            field="independent_reference.coverage_complete",
        ),
    )
    summary = dict(registered.summary)
    summary["agreement"] = metrics
    summary["physical_validation_claim"] = False
    summary["interpretation"] = (
        "Registered independent reference coordinates were overlaid and matched. "
        "This run alone does not establish universal TSD classification validity."
    )
    return registered, summary, audit


__all__ = [
    "ALLOWED_REFERENCE_LABELS",
    "ALLOWED_REFERENCE_METHODS",
    "ALLOWED_REGISTRATION_STATUSES",
    "CONFIRMED_REFERENCE_LABEL",
    "REFERENCE_COLUMNS",
    "REFERENCE_SCHEMA_VERSION",
    "RegisteredReferenceSet",
    "compare_automatic_to_registered_reference",
    "file_sha256",
    "load_registered_reference_points",
    "reference_template",
    "references_from_config",
    "write_reference_template",
]
