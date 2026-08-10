"""Persistent local annotations and candidate-classifier model management."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import threading
from typing import Any, Mapping

import pandas as pd

from .candidate_classifier import (
    RAW_FEATURE_COLUMNS,
    TRAINING_LABELS,
    UNCERTAIN_LABEL,
    train_candidate_classifier,
    validate_model,
)
from .run_repository import RunRepository, RunRepositoryError
from .utils import atomic_write_json, atomic_write_text, utc_now_iso


ANNOTATION_SCHEMA_VERSION = "1.0"
ANNOTATION_SPLITS = frozenset({"calibration", "validation", "locked_test"})
ANNOTATION_COLUMNS: tuple[str, ...] = (
    "annotation_schema_version",
    "run_id",
    "defect_id",
    "input_file_name",
    "wafer_id",
    "label",
    "split",
    "reviewer_id",
    "notes",
    "updated_at_utc",
    "defects_csv_sha256",
    "x_mm",
    "y_mm",
    "accepted_at_annotation_time",
    "rejection_reason_at_annotation_time",
    *RAW_FEATURE_COLUMNS,
)


class TrainingRepositoryError(ValueError):
    """Raised for unsafe, incomplete, or scientifically ambiguous training data."""


def _clean_short_text(value: Any, *, name: str, maximum: int) -> str:
    text = str(value or "").strip()
    if len(text) > maximum or any(ord(character) < 32 for character in text):
        raise TrainingRepositoryError(f"{name} is invalid or exceeds {maximum} characters")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_number(value: Any) -> float | int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return int(numeric) if numeric.is_integer() else numeric


class TrainingRepository:
    """Manage append-auditable labels and one active portable JSON model."""

    def __init__(self, workspace: str | Path, runs: RunRepository) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = self.workspace / "training"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or self.root.resolve().parent != self.workspace:
            raise TrainingRepositoryError("training directory must be a direct workspace child")
        self.annotations_path = self.root / "candidate_annotations.csv"
        self.model_path = self.root / "candidate_classifier.json"
        self.runs = runs
        self._lock = threading.RLock()
        self._ensure_annotations_file()

    def _ensure_annotations_file(self) -> None:
        if self.annotations_path.is_symlink():
            raise TrainingRepositoryError("annotation symlinks are not allowed")
        if self.annotations_path.is_file():
            return
        frame = pd.DataFrame(columns=ANNOTATION_COLUMNS)
        self._write_annotations(frame)

    def _read_annotations(self) -> pd.DataFrame:
        self._ensure_annotations_file()
        try:
            frame = pd.read_csv(
                self.annotations_path,
                dtype={
                    "run_id": str,
                    "defect_id": str,
                    "label": str,
                    "split": str,
                    "reviewer_id": str,
                },
                keep_default_na=False,
            )
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise TrainingRepositoryError("candidate_annotations.csv is unreadable") from exc
        missing = [column for column in ANNOTATION_COLUMNS if column not in frame]
        if missing:
            raise TrainingRepositoryError(
                "candidate_annotations.csv is missing columns: " + ", ".join(missing)
            )
        return frame.loc[:, ANNOTATION_COLUMNS]

    def _write_annotations(self, frame: pd.DataFrame) -> Path:
        buffer = io.StringIO(newline="")
        frame.loc[:, ANNOTATION_COLUMNS].to_csv(buffer, index=False, lineterminator="\n")
        return atomic_write_text(self.annotations_path, buffer.getvalue())

    def _candidate(self, run_id: str, defect_id: str) -> dict[str, str]:
        wanted = str(defect_id).strip()
        if not wanted or len(wanted) > 64:
            raise TrainingRepositoryError("defect_id is invalid")
        for row in self.runs._iter_candidates(run_id):
            if str(row.get("defect_id", "")).strip() == wanted:
                return row
        raise TrainingRepositoryError("candidate was not found in this run")

    def save_annotation(
        self,
        *,
        run_id: str,
        defect_id: str,
        label: str,
        split: str = "calibration",
        reviewer_id: str = "local_expert",
        notes: str = "",
    ) -> dict[str, Any]:
        """Upsert one reviewer's label while snapshotting the model features."""

        normalized_label = str(label).strip().lower()
        normalized_split = str(split).strip().lower()
        if normalized_label not in TRAINING_LABELS:
            raise TrainingRepositoryError("label must be target, artifact, or uncertain")
        if normalized_split not in ANNOTATION_SPLITS:
            raise TrainingRepositoryError(
                "split must be calibration, validation, or locked_test"
            )
        reviewer = _clean_short_text(reviewer_id, name="reviewer_id", maximum=80)
        if not reviewer:
            raise TrainingRepositoryError("reviewer_id cannot be empty")
        note = _clean_short_text(notes, name="notes", maximum=1000)
        validated_run_id = self.runs.validate_run_id(str(run_id))
        candidate = self._candidate(validated_run_id, str(defect_id))
        summary = self.runs.load_summary(validated_run_id)
        defects_path = self.runs.resolve_file(validated_run_id, "defects_all.csv")
        record: dict[str, Any] = {
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "run_id": validated_run_id,
            "defect_id": str(candidate.get("defect_id", defect_id)).strip(),
            "input_file_name": Path(
                str(summary.get("input_file_name") or validated_run_id)
            ).name,
            "wafer_id": str(summary.get("wafer_id") or validated_run_id),
            "label": normalized_label,
            "split": normalized_split,
            "reviewer_id": reviewer,
            "notes": note,
            "updated_at_utc": utc_now_iso(),
            "defects_csv_sha256": _file_sha256(defects_path),
            "x_mm": _json_number(candidate.get("x_mm")),
            "y_mm": _json_number(candidate.get("y_mm")),
            "accepted_at_annotation_time": str(candidate.get("accepted", "")).strip(),
            "rejection_reason_at_annotation_time": str(
                candidate.get("rejection_reason", "")
            ).strip(),
        }
        for column in RAW_FEATURE_COLUMNS:
            record[column] = _json_number(candidate.get(column))

        with self._lock:
            frame = self._read_annotations()
            key = (
                (frame["run_id"].astype(str) == validated_run_id)
                & (frame["defect_id"].astype(str) == record["defect_id"])
                & (frame["reviewer_id"].astype(str) == reviewer)
            )
            frame = frame.loc[~key].copy()
            addition = pd.DataFrame([record], columns=ANNOTATION_COLUMNS)
            frame = addition if frame.empty else pd.concat([frame, addition], ignore_index=True)
            frame = frame.sort_values(
                ["run_id", "defect_id", "reviewer_id"], kind="stable"
            ).reset_index(drop=True)
            self._write_annotations(frame)
        return {
            "run_id": validated_run_id,
            "defect_id": record["defect_id"],
            "label": normalized_label,
            "split": normalized_split,
            "reviewer_id": reviewer,
            "updated_at_utc": record["updated_at_utc"],
        }

    @staticmethod
    def _consensus(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        rows: list[pd.Series] = []
        conflicts = 0
        uncertain = 0
        split_conflicts = 0
        for _, group in frame.groupby(["run_id", "defect_id"], sort=False):
            labels = set(group["label"].astype(str).str.strip().str.lower())
            splits = set(group["split"].astype(str).str.strip().str.lower())
            if UNCERTAIN_LABEL in labels:
                uncertain += 1
                continue
            if len(labels) != 1:
                conflicts += 1
                continue
            if len(splits) != 1:
                split_conflicts += 1
                continue
            rows.append(group.iloc[-1])
        consensus = (
            pd.DataFrame(rows).reset_index(drop=True)
            if rows
            else pd.DataFrame(columns=frame.columns)
        )
        return consensus, {
            "consensus_candidate_count": int(len(consensus)),
            "conflicting_candidate_count": conflicts,
            "uncertain_candidate_count": uncertain,
            "split_conflict_count": split_conflicts,
        }

    def labels_for_run(self, run_id: str) -> dict[str, str]:
        """Return consensus labels, marking reviewer disagreement explicitly."""

        validated = self.runs.validate_run_id(str(run_id))
        with self._lock:
            frame = self._read_annotations()
        selected = frame.loc[frame["run_id"].astype(str) == validated]
        labels: dict[str, str] = {}
        for defect_id, group in selected.groupby("defect_id", sort=False):
            values = set(group["label"].astype(str).str.strip().str.lower())
            labels[str(defect_id)] = next(iter(values)) if len(values) == 1 else "conflict"
        return labels

    def train(
        self,
        *,
        accept_threshold: float = 0.75,
        reject_threshold: float = 0.25,
        regularization: float = 1.0,
        minimum_per_class: int = 5,
    ) -> dict[str, Any]:
        """Train and atomically activate a model from consensus labels."""

        with self._lock:
            raw = self._read_annotations()
            consensus, consensus_status = self._consensus(raw)
            try:
                model = train_candidate_classifier(
                    consensus,
                    accept_threshold=accept_threshold,
                    reject_threshold=reject_threshold,
                    regularization=regularization,
                    minimum_per_class=minimum_per_class,
                )
            except ValueError as exc:
                raise TrainingRepositoryError(str(exc)) from exc
            model["annotation_repository"] = consensus_status
            model["model_sha256"] = hashlib.sha256(
                json.dumps(
                    {key: value for key, value in model.items() if key != "model_sha256"},
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            model = validate_model(model)
            atomic_write_json(self.model_path, model, sort_keys=True)
        return model

    def active_model(self) -> dict[str, Any] | None:
        if self.model_path.is_symlink():
            raise TrainingRepositoryError("model symlinks are not allowed")
        if not self.model_path.is_file():
            return None
        try:
            loaded = json.loads(self.model_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainingRepositoryError("candidate_classifier.json is unreadable") from exc
        if not isinstance(loaded, Mapping):
            raise TrainingRepositoryError("candidate_classifier.json must contain an object")
        try:
            return validate_model(loaded)
        except ValueError as exc:
            raise TrainingRepositoryError(str(exc)) from exc

    def status(self) -> dict[str, Any]:
        with self._lock:
            raw = self._read_annotations()
            consensus, consensus_status = self._consensus(raw)
            try:
                model = self.active_model()
                model_error = None
            except TrainingRepositoryError as exc:
                model = None
                model_error = str(exc)
        label_counts = {
            label: int((raw["label"].astype(str).str.lower() == label).sum())
            for label in sorted(TRAINING_LABELS)
        }
        split_counts = {
            split: int((raw["split"].astype(str).str.lower() == split).sum())
            for split in sorted(ANNOTATION_SPLITS)
        }
        consensus_label_counts = {
            label: int((consensus["label"].astype(str).str.lower() == label).sum())
            for label in sorted(TRAINING_LABELS)
        }
        return {
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "annotation_count": int(len(raw)),
            "label_counts": label_counts,
            "split_counts": split_counts,
            "consensus_label_counts": consensus_label_counts,
            **consensus_status,
            "model_available": model is not None,
            "model_error": model_error,
            "model": (
                {
                    "model_sha256": model.get("model_sha256"),
                    "trained_at_utc": model.get("trained_at_utc"),
                    "training_sample_count": model.get("training_sample_count"),
                    "training_target_count": model.get("training_target_count"),
                    "training_artifact_count": model.get("training_artifact_count"),
                    "accept_threshold": model.get("accept_threshold"),
                    "reject_threshold": model.get("reject_threshold"),
                    "validation": model.get("validation"),
                }
                if model is not None
                else None
            ),
            "scientific_limit": (
                "Training labels improve image-candidate classification only; they do not "
                "establish physical dislocation identity without independent evidence."
            ),
        }


__all__ = [
    "ANNOTATION_COLUMNS",
    "ANNOTATION_SCHEMA_VERSION",
    "ANNOTATION_SPLITS",
    "TrainingRepository",
    "TrainingRepositoryError",
]
