"""Disk-bounded ZIP exports for one completed analysis run.

The exporter never reads all candidate images into memory.  It copies each
already-generated file into a ZIP64 archive with ``ZIP_STORED`` so TIFF/PNG
bytes are preserved without an additional lossy or scientific transformation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import threading
from typing import Iterable, Iterator
from uuid import uuid4
import zipfile

from .run_repository import ARTIFACT_NAMES, RunRepository, RunRepositoryError


EXPORT_SCHEMA_VERSION = 1
CROP_COLUMNS = ("crop_path", "crop_preview_path")
CROP_SUFFIXES = {".png", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class ExportBundle:
    """Public description of one on-demand archive."""

    kind: str
    filename: str
    label: str
    description: str


EXPORT_BUNDLES = {
    "figures": ExportBundle(
        kind="figures",
        filename="all_analysis_figures.zip",
        label="全部分析图 ZIP",
        description="本次运行生成的全部 PNG 分析图，保持原文件字节。",
    ),
    "data": ExportBundle(
        kind="data",
        filename="all_reports_and_tables.zip",
        label="全部报告与表格 ZIP",
        description="HTML、CSV、JSON、YAML 和运行日志，不包含输入原图。",
    ),
    "candidate-crops": ExportBundle(
        kind="candidate-crops",
        filename="all_candidate_crops.zip",
        label="全部局部分析包 ZIP",
        description="首项为全局 Excel；随后是每个视场的标记图、位置 Excel、原始 TIFF，并保留逐候选裁剪。",
    ),
}


class ResultExporter:
    """Create immutable, cached result archives with bounded memory use."""

    def __init__(self, repository: RunRepository) -> None:
        self.repository = repository
        self._lock = threading.Lock()

    def available(self, run_id: str) -> dict[str, ExportBundle]:
        """Return bundle kinds whose source files are present."""

        run_dir = self.repository.run_dir(run_id)
        artifacts = self.repository.artifacts(run_id)
        available: dict[str, ExportBundle] = {}
        if any(Path(name).suffix.lower() == ".png" for name in artifacts):
            available["figures"] = EXPORT_BUNDLES["figures"]
        if any(Path(name).suffix.lower() != ".png" for name in artifacts):
            available["data"] = EXPORT_BUNDLES["data"]
        crop_dir = run_dir / "candidate_crops"
        local_overview = run_dir / "local_fields" / "00_global_overview.xlsx"
        defect_table = run_dir / "defects_all.csv"
        if (
            (
                (crop_dir.is_dir() and not crop_dir.is_symlink())
                or (local_overview.is_file() and not local_overview.is_symlink())
            )
            and defect_table.is_file()
            and not defect_table.is_symlink()
        ):
            available["candidate-crops"] = EXPORT_BUNDLES["candidate-crops"]
        return available

    def archive(self, run_id: str, kind: str) -> Path:
        """Return a complete cached archive, building it atomically if needed."""

        bundle = self.available(run_id).get(kind)
        if bundle is None:
            raise RunRepositoryError("export bundle is not available")
        run_dir = self.repository.run_dir(run_id)
        export_dir = run_dir / "exports"
        if export_dir.is_symlink():
            raise RunRepositoryError("export directory symlinks are not allowed")
        export_dir.mkdir(exist_ok=True)
        if export_dir.resolve().parent != run_dir:
            raise RunRepositoryError("export directory is outside the run")
        destination = export_dir / f"v{EXPORT_SCHEMA_VERSION}_{bundle.filename}"
        if destination.is_symlink():
            raise RunRepositoryError("export archive symlinks are not allowed")

        with self._lock:
            if destination.is_file() and zipfile.is_zipfile(destination):
                return destination
            temporary = export_dir / f".{destination.name}.{uuid4().hex}.tmp"
            try:
                self._build_archive(run_id, kind, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination

    def _artifact_members(self, run_id: str, *, figures: bool) -> Iterator[tuple[str, Path]]:
        artifacts = self.repository.artifacts(run_id)
        for name in ARTIFACT_NAMES:
            if name not in artifacts:
                continue
            is_figure = Path(name).suffix.lower() == ".png"
            if is_figure != figures:
                continue
            folder = "figures" if figures else "reports_and_tables"
            yield f"{folder}/{name}", self.repository.resolve_file(run_id, name)

    def _crop_members(
        self,
        run_id: str,
    ) -> tuple[Iterator[tuple[str, Path]], dict[str, int]]:
        """Yield every CSV-referenced crop exactly once and track omissions."""

        statistics = {
            "candidate_rows": 0,
            "raw_crops_exported": 0,
            "preview_crops_exported": 0,
            "rows_without_raw_crop": 0,
            "rows_without_preview_crop": 0,
        }

        def iterator() -> Iterator[tuple[str, Path]]:
            table = self.repository.resolve_file(run_id, "defects_all.csv")
            seen: set[str] = set()
            with table.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                if not set(CROP_COLUMNS) <= fields:
                    raise RunRepositoryError("defects_all.csv has no candidate crop columns")
                for row in reader:
                    statistics["candidate_rows"] += 1
                    for column in CROP_COLUMNS:
                        value = str(row.get(column) or "").strip().replace("\\", "/")
                        if not value:
                            missing_key = (
                                "rows_without_raw_crop"
                                if column == "crop_path"
                                else "rows_without_preview_crop"
                            )
                            statistics[missing_key] += 1
                            continue
                        pure = PurePosixPath(value)
                        if (
                            pure.is_absolute()
                            or len(pure.parts) < 2
                            or pure.parts[0] != "candidate_crops"
                            or pure.suffix.lower() not in CROP_SUFFIXES
                        ):
                            raise RunRepositoryError("unsafe candidate crop path in defects_all.csv")
                        if value in seen:
                            continue
                        path = self.repository.resolve_file(run_id, value)
                        seen.add(value)
                        if column == "crop_path":
                            statistics["raw_crops_exported"] += 1
                        else:
                            statistics["preview_crops_exported"] += 1
                        yield value, path

        return iterator(), statistics

    def _local_field_members(self, run_id: str) -> Iterator[tuple[str, Path]]:
        """Yield every generated field file in deterministic human order."""

        run_dir = self.repository.run_dir(run_id)
        root = run_dir / "local_fields"
        if root.is_symlink() or not root.is_dir():
            return
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise RunRepositoryError("local field symlinks are not exported")
            if not path.is_file():
                continue
            relative = path.relative_to(run_dir).as_posix()
            if relative == "local_fields/00_global_overview.xlsx":
                continue
            yield relative, self.repository.resolve_file(run_id, relative)

    @staticmethod
    def _write_members(
        archive: zipfile.ZipFile,
        members: Iterable[tuple[str, Path]],
    ) -> int:
        count = 0
        for archive_name, source in members:
            archive.write(source, archive_name, compress_type=zipfile.ZIP_STORED)
            count += 1
        return count

    def _build_archive(self, run_id: str, kind: str, destination: Path) -> None:
        statistics: dict[str, int | str | bool] = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "run_id": run_id,
            "bundle_kind": kind,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "stored_without_recompression": True,
        }
        with zipfile.ZipFile(
            destination,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            if kind == "figures":
                statistics["file_count"] = self._write_members(
                    archive, self._artifact_members(run_id, figures=True)
                )
            elif kind == "data":
                statistics["file_count"] = self._write_members(
                    archive, self._artifact_members(run_id, figures=False)
                )
            elif kind == "candidate-crops":
                overview_path = (
                    self.repository.run_dir(run_id)
                    / "local_fields"
                    / "00_global_overview.xlsx"
                )
                if overview_path.is_file() and not overview_path.is_symlink():
                    local_overview = self.repository.resolve_file(
                        run_id, "local_fields/00_global_overview.xlsx"
                    )
                    archive.write(
                        local_overview,
                        "00_global_overview.xlsx",
                        compress_type=zipfile.ZIP_STORED,
                    )
                    statistics["global_overview_files"] = 1
                else:
                    statistics["global_overview_files"] = 0
                statistics["local_field_files"] = self._write_members(
                    archive, self._local_field_members(run_id)
                )
                members, crop_statistics = self._crop_members(run_id)
                statistics["file_count"] = self._write_members(archive, members)
                statistics.update(crop_statistics)
                defect_table = self.repository.resolve_file(run_id, "defects_all.csv")
                archive.write(
                    defect_table,
                    "index/defects_all.csv",
                    compress_type=zipfile.ZIP_STORED,
                )
                statistics["index_files"] = 1
            else:  # guarded by available(), retained for defensive callers
                raise RunRepositoryError("unknown export bundle kind")
            archive.writestr(
                "export_manifest.json",
                json.dumps(statistics, ensure_ascii=False, indent=2),
                compress_type=zipfile.ZIP_STORED,
            )
            archive.writestr(
                "README.txt",
                (
                    "SiC 晶圆点状目标分析导出包\n"
                    "文件来自单次运行目录；未重新缩放或改变科研图像内容。\n"
                    "candidate-crops 包含全部候选（接受与拒绝），判定见 index/defects_all.csv。\n"
                    "自动识别的是满足最终图像判定（透明规则或已记录训练分类器）的点状目标，不等同于已物理确认的位错。\n"
                ),
                compress_type=zipfile.ZIP_STORED,
            )


__all__ = ["EXPORT_BUNDLES", "ExportBundle", "ResultExporter"]
