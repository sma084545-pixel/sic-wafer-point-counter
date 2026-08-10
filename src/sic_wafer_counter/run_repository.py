"""Safe, restart-persistent access to completed analysis result folders.

The repository deliberately treats ``results/`` as an append-only collection.
It never follows result-directory symlinks, never opens the source image path
recorded in a summary, and streams candidate CSV rows for bounded pagination.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterator, Mapping


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MAX_PAGE_SIZE = 200
RESULT_FILE_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".log",
    ".png",
    ".tif",
    ".tiff",
    ".txt",
    ".xlsx",
    ".yaml",
    ".yml",
}
ARTIFACT_NAMES = (
    "report.html",
    "summary.json",
    "summary.csv",
    "defects_all.csv",
    "defects_accepted.csv",
    "defects_rejected.csv",
    "overlay_accepted.png",
    "overlay_all_candidates.png",
    "overlay_xrt_red_boxes.png",
    "xrt_detection_detail_montage.png",
    "paper_detection_field.png",
    "paper_aligned_result_figure.png",
    "defect_comparison_details.png",
    "wafer_mask.png",
    "valid_analysis_mask.png",
    "preprocessed_preview.png",
    "candidate_mask.png",
    "equivalent_diameter_histogram.png",
    "defect_size_histogram.png",
    "radial_density.png",
    "angular_density.png",
    "density_heatmap.png",
    "density_heatmap_grid.csv",
    "local_fields/00_global_overview.xlsx",
    "independent_reference_points.csv",
    "independent_reference_matches.csv",
    "radial_density.csv",
    "angular_density.csv",
    "regional_density.csv",
    "analysis_config.yaml",
    "resolved_physical_parameters.yaml",
    "run.log",
)


class RunRepositoryError(ValueError):
    """Raised when a requested result identifier or file is unsafe/invalid."""


@dataclass(frozen=True, slots=True)
class RunIndex:
    """A persistent run listing plus non-fatal discovery diagnostics."""

    runs: list[dict[str, Any]]
    skipped_invalid_summaries: list[str]


@dataclass(frozen=True, slots=True)
class CandidatePage:
    """One bounded page from a potentially very large candidate table."""

    rows: list[dict[str, Any]]
    page: int
    page_size: int
    total: int
    total_pages: int
    reason_counts: dict[str, int]


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted"}


def _basename(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return Path(value).name if Path(value).is_absolute() else value


def public_json(value: Any, *, key: str = "") -> Any:
    """Return strict JSON data while removing absolute filesystem disclosure."""

    if isinstance(value, Mapping):
        return {str(name): public_json(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [public_json(item, key=key) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and key.lower().endswith("path"):
        return Path(value).name
    if isinstance(value, str) and Path(value).is_absolute():
        return _basename(value)
    return value


def _dataset_kind(run_id: str, summary: Mapping[str, Any]) -> str:
    status = str(summary.get("status", "completed")).lower()
    if status == "failed":
        return "failed"
    name = str(summary.get("input_file_name", "")).lower()
    if "synthetic" in run_id.lower() or name.startswith("synthetic_"):
        return "synthetic"
    return "real_exploratory"


def _timestamp(summary: Mapping[str, Any], summary_path: Path) -> str:
    provided = summary.get("generated_at_utc")
    if isinstance(provided, str) and provided.strip():
        return provided
    return datetime.fromtimestamp(summary_path.stat().st_mtime, tz=timezone.utc).isoformat()


class RunRepository:
    """Read-only view of direct analysis result directories."""

    def __init__(self, result_root: str | Path) -> None:
        self.result_root = Path(result_root).expanduser().resolve()
        self.result_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_run_id(run_id: str) -> str:
        if not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
            raise RunRepositoryError("invalid run_id")
        return run_id

    def run_dir(self, run_id: str) -> Path:
        """Resolve a direct, non-symlink result directory."""

        self.validate_run_id(run_id)
        candidate = self.result_root / run_id
        if candidate.is_symlink() or not candidate.is_dir():
            raise RunRepositoryError("run not found")
        resolved = candidate.resolve()
        if resolved.parent != self.result_root:
            raise RunRepositoryError("run is outside the configured results directory")
        return resolved

    def resolve_file(self, run_id: str, relative_path: str) -> Path:
        """Resolve a permitted regular result file without following symlinks."""

        run_dir = self.run_dir(run_id)
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
            raise RunRepositoryError("invalid result file path")
        if pure.suffix.lower() not in RESULT_FILE_SUFFIXES:
            raise RunRepositoryError("result file type is not exposed")
        current = run_dir
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise RunRepositoryError("result file symlinks are not exposed")
        target = current.resolve()
        try:
            target.relative_to(run_dir)
        except ValueError as exc:
            raise RunRepositoryError("result file is outside its run directory") from exc
        if not target.is_file():
            raise RunRepositoryError("result file not found")
        return target

    def _load_summary_path(self, summary_path: Path) -> dict[str, Any]:
        if summary_path.is_symlink() or not summary_path.is_file():
            raise RunRepositoryError("summary not found")
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RunRepositoryError("invalid summary.json") from exc
        if not isinstance(loaded, dict):
            raise RunRepositoryError("summary.json must contain an object")
        return loaded

    def load_summary(self, run_id: str) -> dict[str, Any]:
        return self._load_summary_path(self.run_dir(run_id) / "summary.json")

    def artifacts(self, run_id: str) -> dict[str, str]:
        run_dir = self.run_dir(run_id)
        available: dict[str, str] = {}
        for name in ARTIFACT_NAMES:
            candidate = run_dir / name
            if candidate.is_file() and not candidate.is_symlink():
                available[name] = name
        return available

    def _entry(self, run_id: str, summary: Mapping[str, Any], summary_path: Path) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "status": str(summary.get("status", "completed")),
            "dataset_kind": _dataset_kind(run_id, summary),
            "input_file_name": Path(str(summary.get("input_file_name") or "—")).name,
            "generated_at_utc": _timestamp(summary, summary_path),
            "accepted_count": _safe_int(summary.get("accepted_count")),
            "valid_analysis_area_cm2": _safe_float(summary.get("valid_analysis_area_cm2")),
            "point_density_cm2": _safe_float(summary.get("point_density_cm2")),
            "runtime_seconds": _safe_float(summary.get("runtime_seconds")),
            "real_annotation_validation_status": str(
                summary.get("real_annotation_validation_status")
                or "not validated on real SiC data"
            ),
            "warning_count": len(summary.get("warnings", []))
            if isinstance(summary.get("warnings", []), list)
            else 1,
        }

    def list_runs(self) -> RunIndex:
        runs: list[dict[str, Any]] = []
        skipped: list[str] = []
        for candidate in self.result_root.iterdir():
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            run_id = candidate.name
            try:
                self.validate_run_id(run_id)
                summary_path = candidate / "summary.json"
                summary = self._load_summary_path(summary_path)
                runs.append(self._entry(run_id, summary, summary_path))
            except RunRepositoryError:
                if (candidate / "summary.json").exists():
                    skipped.append(run_id)
        runs.sort(key=lambda item: str(item["generated_at_utc"]), reverse=True)
        return RunIndex(runs=runs, skipped_invalid_summaries=sorted(skipped))

    def run_detail(self, run_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        summary_path = run_dir / "summary.json"
        summary = self._load_summary_path(summary_path)
        return {
            **self._entry(run_id, summary, summary_path),
            "summary": public_json(summary),
            "artifact_names": list(self.artifacts(run_id)),
        }

    def _iter_candidates(self, run_id: str) -> Iterator[dict[str, str]]:
        path = self.resolve_file(run_id, "defects_all.csv")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "defect_id" not in reader.fieldnames:
                raise RunRepositoryError("defects_all.csv has no defect_id column")
            yield from reader

    def _crop_relative(self, run_id: str, raw: Any) -> str | None:
        value = str(raw or "").strip().replace("\\", "/")
        if not value:
            return None
        try:
            self.resolve_file(run_id, value)
        except RunRepositoryError:
            return None
        return value

    def candidate_page(
        self,
        run_id: str,
        *,
        status: str = "all",
        reason: str = "",
        defect_id: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> CandidatePage:
        """Stream a filtered page without loading the full CSV into memory."""

        normalized_status = status.strip().lower() or "all"
        if normalized_status not in {"all", "accepted", "rejected"}:
            raise RunRepositoryError("status must be all, accepted, or rejected")
        if page < 1:
            raise RunRepositoryError("page must be at least 1")
        if page_size < 1 or page_size > MAX_PAGE_SIZE:
            raise RunRepositoryError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")
        wanted_reason = reason.strip()
        wanted_id = defect_id.strip()
        start = (page - 1) * page_size
        stop = start + page_size
        rows: list[dict[str, Any]] = []
        total = 0
        reason_counts: dict[str, int] = {}
        for raw in self._iter_candidates(run_id):
            accepted = _truthy(raw.get("accepted"))
            raw_reason = str(raw.get("rejection_reason") or "").strip()
            reasons = [item.strip() for item in raw_reason.split(";") if item.strip()]
            for item in reasons:
                reason_counts[item] = reason_counts.get(item, 0) + 1
            if normalized_status == "accepted" and not accepted:
                continue
            if normalized_status == "rejected" and accepted:
                continue
            if wanted_reason and wanted_reason not in reasons:
                continue
            if wanted_id and str(raw.get("defect_id", "")).strip() != wanted_id:
                continue
            if start <= total < stop:
                preview = self._crop_relative(run_id, raw.get("crop_preview_path"))
                crop = self._crop_relative(run_id, raw.get("crop_path"))
                row: dict[str, Any] = {
                    key: value
                    for key, value in raw.items()
                    if key not in {"crop_path", "crop_preview_path"}
                }
                row["accepted"] = accepted
                row["crop_preview_relative"] = preview
                row["crop_relative"] = crop
                rows.append(row)
            total += 1
        total_pages = max(1, math.ceil(total / page_size))
        return CandidatePage(
            rows=rows,
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
            reason_counts=dict(sorted(reason_counts.items())),
        )


__all__ = [
    "ARTIFACT_NAMES",
    "CandidatePage",
    "MAX_PAGE_SIZE",
    "RunIndex",
    "RunRepository",
    "RunRepositoryError",
    "public_json",
]
