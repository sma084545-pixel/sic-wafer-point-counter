"""Persistent result browsing, pagination, and filesystem safety tests."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import pytest

from sic_wafer_counter.run_repository import RunRepository, RunRepositoryError, public_json
from sic_wafer_counter.result_export import ResultExporter
from sic_wafer_counter.web import create_app


def _make_run(root: Path, run_id: str, *, status: str = "completed") -> Path:
    folder = root / "results" / run_id
    folder.mkdir(parents=True)
    summary = {
        "status": status,
        "input_file_name": "synthetic_clean.png",
        "input_path": "/private/research/secret/synthetic_clean.png",
        "generated_at_utc": "2026-01-02T03:04:05+00:00",
        "accepted_count": 2 if status == "completed" else None,
        "valid_analysis_area_cm2": 10.0 if status == "completed" else None,
        "point_density_cm2": 0.2 if status == "completed" else None,
        "real_annotation_validation_status": "not validated on real SiC data",
        "warnings": ["test warning"],
    }
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return folder


def _write_candidates(path: Path, count: int = 5) -> None:
    fields = [
        "defect_id", "x_mm", "y_mm", "equivalent_diameter_mm", "circularity",
        "contrast", "distance_to_valid_boundary_mm", "accepted", "rejection_reason",
        "crop_path", "crop_preview_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(count):
            writer.writerow({
                "defect_id": index + 1,
                "x_mm": index / 10,
                "y_mm": -index / 10,
                "equivalent_diameter_mm": 0.012,
                "circularity": 0.8,
                "contrast": 0.1,
                "distance_to_valid_boundary_mm": 1.0,
                "accepted": index % 2 == 0,
                "rejection_reason": "" if index % 2 == 0 else "too_elongated",
                "crop_path": "candidate_crops/candidate.tif",
                "crop_preview_path": "candidate_crops/candidate.png",
            })


def test_repository_discovers_direct_valid_runs_and_isolates_corruption(tmp_path: Path) -> None:
    valid = _make_run(tmp_path, "valid_run")
    corrupt = tmp_path / "results" / "corrupt_run"
    corrupt.mkdir()
    (corrupt / "summary.json").write_text("{broken", encoding="utf-8")
    nested = valid / "nested_run"
    nested.mkdir()
    (nested / "summary.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "results" / "linked_run").symlink_to(outside, target_is_directory=True)

    index = RunRepository(tmp_path / "results").list_runs()
    assert [run["run_id"] for run in index.runs] == ["valid_run"]
    assert index.skipped_invalid_summaries == ["corrupt_run"]


def test_summary_is_public_and_files_cannot_escape_or_follow_symlinks(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "safe_run")
    (run / "overlay_accepted.png").write_bytes(b"png")
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    (run / "leak.txt").symlink_to(secret)
    repository = RunRepository(tmp_path / "results")

    detail = repository.run_detail("safe_run")
    assert detail["summary"]["input_path"] == "synthetic_clean.png"
    assert public_json({"input_path": "private/research/secret/image.tif"})["input_path"] == "image.tif"
    assert repository.resolve_file("safe_run", "overlay_accepted.png").is_file()
    for unsafe in ("../secret.txt", "/etc/passwd", "leak.txt"):
        with pytest.raises(RunRepositoryError):
            repository.resolve_file("safe_run", unsafe)
    for unsafe_id in ("../safe_run", "safe/../run", ".."):
        with pytest.raises(RunRepositoryError):
            repository.run_dir(unsafe_id)


def test_candidate_pagination_filters_and_crop_paths_are_bounded(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "candidate_run")
    crop_dir = run / "candidate_crops"
    crop_dir.mkdir()
    (crop_dir / "candidate.tif").write_bytes(b"tif")
    (crop_dir / "candidate.png").write_bytes(b"png")
    _write_candidates(run / "defects_all.csv", count=7)
    repository = RunRepository(tmp_path / "results")

    page = repository.candidate_page("candidate_run", status="rejected", page=1, page_size=2)
    assert page.total == 3
    assert len(page.rows) == 2
    assert page.total_pages == 2
    assert page.reason_counts["too_elongated"] == 3
    assert page.rows[0]["crop_preview_relative"] == "candidate_crops/candidate.png"
    exact = repository.candidate_page("candidate_run", defect_id="7", page_size=10)
    assert exact.total == 1
    assert exact.rows[0]["defect_id"] == "7"
    with pytest.raises(RunRepositoryError):
        repository.candidate_page("candidate_run", page_size=201)


def test_result_exports_every_figure_table_and_candidate_crop_without_recompression(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path, "export_run")
    (run / "overlay_accepted.png").write_bytes(b"figure-bytes")
    (run / "report.html").write_text("<p>report</p>", encoding="utf-8")
    crop_dir = run / "candidate_crops"
    crop_dir.mkdir()
    local_dir = run / "local_fields"
    field_dir = local_dir / "field_0001_X_0.000_Y_0.000"
    field_dir.mkdir(parents=True)
    (local_dir / "00_global_overview.xlsx").write_bytes(b"global-xlsx")
    (field_dir / "01_marked.png").write_bytes(b"marked")
    (field_dir / "02_positions.xlsx").write_bytes(b"positions")
    (field_dir / "03_raw_original.tif").write_bytes(b"field-raw")
    second_field = local_dir / "field_0002_X_-6.000_Y_4.500"
    second_field.mkdir()
    (second_field / "01_marked.png").write_bytes(b"marked-2")
    (second_field / "02_positions.xlsx").write_bytes(b"positions-2")
    (second_field / "03_raw_original.tif").write_bytes(b"field-raw-2")
    fields = ["defect_id", "accepted", "rejection_reason", "crop_path", "crop_preview_path"]
    with (run / "defects_all.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for defect_id, accepted in ((1, True), (2, False)):
            raw_name = f"candidate_{defect_id:06d}.tif"
            preview_name = f"candidate_{defect_id:06d}_preview.png"
            (crop_dir / raw_name).write_bytes(f"raw-{defect_id}".encode())
            (crop_dir / preview_name).write_bytes(f"preview-{defect_id}".encode())
            writer.writerow({
                "defect_id": defect_id,
                "accepted": accepted,
                "rejection_reason": "" if accepted else "too_elongated",
                "crop_path": f"candidate_crops/{raw_name}",
                "crop_preview_path": f"candidate_crops/{preview_name}",
            })

    exporter = ResultExporter(RunRepository(tmp_path / "results"))
    assert set(exporter.available("export_run")) == {
        "figures", "data", "candidate-crops", "cu-style-fields"
    }

    figure_archive = exporter.archive("export_run", "figures")
    data_archive = exporter.archive("export_run", "data")
    crop_archive = exporter.archive("export_run", "candidate-crops")
    cu_style_archive = exporter.archive("export_run", "cu-style-fields")
    assert exporter.archive("export_run", "candidate-crops") == crop_archive

    with zipfile.ZipFile(figure_archive) as archive:
        assert archive.read("figures/overlay_accepted.png") == b"figure-bytes"
        assert archive.getinfo("figures/overlay_accepted.png").compress_type == zipfile.ZIP_STORED
    with zipfile.ZipFile(data_archive) as archive:
        assert archive.read("reports_and_tables/report.html") == b"<p>report</p>"
        assert archive.read("reports_and_tables/defects_all.csv")
    with zipfile.ZipFile(crop_archive) as archive:
        ordered_names = archive.namelist()
        names = set(ordered_names)
        assert ordered_names[0] == "00_global_overview.xlsx"
        assert {
            "local_fields/field_0001_X_0.000_Y_0.000/01_marked.png",
            "local_fields/field_0001_X_0.000_Y_0.000/02_positions.xlsx",
            "local_fields/field_0001_X_0.000_Y_0.000/03_raw_original.tif",
            "candidate_crops/candidate_000001.tif",
            "candidate_crops/candidate_000001_preview.png",
            "candidate_crops/candidate_000002.tif",
            "candidate_crops/candidate_000002_preview.png",
            "index/defects_all.csv",
            "export_manifest.json",
            "README.txt",
        } <= names
        assert archive.read("candidate_crops/candidate_000001.tif") == b"raw-1"
        assert archive.read("candidate_crops/candidate_000002_preview.png") == b"preview-2"
        manifest = json.loads(archive.read("export_manifest.json"))
        assert manifest["candidate_rows"] == 2
        assert manifest["raw_crops_exported"] == 2
        assert manifest["preview_crops_exported"] == 2
        assert manifest["stored_without_recompression"] is True
    with zipfile.ZipFile(cu_style_archive) as archive:
        assert archive.namelist() == [
            "00000_global_overview.xlsx",
            "synthetic_clean_00001_X_0_Y_0_01_marked.png",
            "synthetic_clean_00001_X_0_Y_0_02_positions.xlsx",
            "synthetic_clean_00001_X_0_Y_0_03_raw_original.tif",
            "synthetic_clean_00002_X_-6_Y_4.5_01_marked.png",
            "synthetic_clean_00002_X_-6_Y_4.5_02_positions.xlsx",
            "synthetic_clean_00002_X_-6_Y_4.5_03_raw_original.tif",
        ]
        assert archive.read("synthetic_clean_00001_X_0_Y_0_01_marked.png") == b"marked"
        assert archive.read("synthetic_clean_00002_X_-6_Y_4.5_02_positions.xlsx") == b"positions-2"
        assert all("/" not in name for name in archive.namelist())


def test_candidate_crop_export_fails_closed_for_csv_path_escape(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "unsafe_export")
    crop_dir = run / "candidate_crops"
    crop_dir.mkdir()
    (crop_dir / "preview.png").write_bytes(b"preview")
    with (run / "defects_all.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["defect_id", "accepted", "crop_path", "crop_preview_path"],
        )
        writer.writeheader()
        writer.writerow({
            "defect_id": 1,
            "accepted": True,
            "crop_path": "../secret.tif",
            "crop_preview_path": "candidate_crops/preview.png",
        })
    exporter = ResultExporter(RunRepository(tmp_path / "results"))
    with pytest.raises(RunRepositoryError, match="unsafe candidate crop path"):
        exporter.archive("unsafe_export", "candidate-crops")
    exports = run / "exports"
    assert not list(exports.glob("*.zip"))


def test_cu_style_export_is_hidden_when_any_field_triplet_is_incomplete(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path, "incomplete_fields")
    root = run / "local_fields"
    field = root / "field_0001_X_0.000_Y_0.000"
    field.mkdir(parents=True)
    (root / "00_global_overview.xlsx").write_bytes(b"global")
    (field / "01_marked.png").write_bytes(b"marked")
    (field / "03_raw_original.tif").write_bytes(b"raw")

    exporter = ResultExporter(RunRepository(tmp_path / "results"))
    assert "cu-style-fields" not in exporter.available("incomplete_fields")
    with pytest.raises(RunRepositoryError, match="not available"):
        exporter.archive("incomplete_fields", "cu-style-fields")


def test_api_streams_one_small_page_from_one_hundred_thousand_candidates(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "large_run")
    path = run / "defects_all.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["defect_id", "accepted", "rejection_reason"])
        for index in range(100_000):
            writer.writerow([index + 1, index % 2 == 0, "" if index % 2 == 0 else "too_small"])

    app = create_app(tmp_path, max_workers=1, max_upload_mb=1)
    app.config["TESTING"] = True
    manager = app.extensions["sic_wafer_job_manager"]
    try:
        with app.test_client() as client:
            response = client.get("/api/runs/large_run/defects?page=2&page_size=50")
            assert response.status_code == 200
            payload = response.get_json()
            assert payload["total"] == 100_000
            assert len(payload["rows"]) == 50
            assert len(response.data) < 100_000
            assert client.get("/api/runs/large_run/defects?page_size=201").status_code == 400
    finally:
        manager.shutdown()


def test_runs_survive_app_restart_and_security_headers_are_present(tmp_path: Path) -> None:
    _make_run(tmp_path, "persistent_run")
    first = create_app(tmp_path, max_workers=1, max_upload_mb=1)
    first.extensions["sic_wafer_job_manager"].shutdown()
    second = create_app(tmp_path, max_workers=1, max_upload_mb=1)
    second.config["TESTING"] = True
    manager = second.extensions["sic_wafer_job_manager"]
    try:
        with second.test_client() as client:
            response = client.get("/api/runs")
            assert response.status_code == 200
            assert response.get_json()["runs"][0]["run_id"] == "persistent_run"
            assert "input_path" not in response.get_data(as_text=True)
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert "default-src 'self'" in response.headers["Content-Security-Policy"]
            assert "'unsafe-inline'" not in response.headers["Content-Security-Policy"]
            assert response.headers["Cross-Origin-Opener-Policy"] == "same-origin"
            detail = client.get("/api/runs/persistent_run")
            assert detail.status_code == 200
            assert "/private/research" not in detail.get_data(as_text=True)
            assert client.get("/api/runs/../files/summary.json").status_code == 404
            assert client.get("/api/runs", headers={"Host": "attacker.example"}).status_code == 403
            cross_site = client.post(
                "/api/jobs",
                headers={"Origin": "https://attacker.example"},
                data={},
            )
            assert cross_site.status_code == 403
    finally:
        manager.shutdown()


def test_export_api_exposes_individual_downloads_and_run_scoped_zip_urls(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "download_run")
    (run / "overlay_accepted.png").write_bytes(b"png")
    (run / "report.html").write_text("report", encoding="utf-8")
    crop_dir = run / "candidate_crops"
    crop_dir.mkdir()
    (crop_dir / "candidate_000001.tif").write_bytes(b"raw")
    (crop_dir / "candidate_000001_preview.png").write_bytes(b"preview")
    local_dir = run / "local_fields"
    field_dir = local_dir / "field_0001_X_-10.000_Y_-46.000"
    field_dir.mkdir(parents=True)
    (local_dir / "00_global_overview.xlsx").write_bytes(b"global")
    (field_dir / "01_marked.png").write_bytes(b"marked")
    (field_dir / "02_positions.xlsx").write_bytes(b"positions")
    (field_dir / "03_raw_original.tif").write_bytes(b"field-raw")
    with (run / "defects_all.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["defect_id", "accepted", "crop_path", "crop_preview_path"],
        )
        writer.writeheader()
        writer.writerow({
            "defect_id": 1,
            "accepted": True,
            "crop_path": "candidate_crops/candidate_000001.tif",
            "crop_preview_path": "candidate_crops/candidate_000001_preview.png",
        })

    app = create_app(tmp_path, max_workers=1, max_upload_mb=1)
    app.config["TESTING"] = True
    manager = app.extensions["sic_wafer_job_manager"]
    try:
        with app.test_client() as client:
            detail = client.get("/api/runs/download_run")
            assert detail.status_code == 200
            payload = detail.get_json()
            assert set(payload["exports"]) == {
                "figures", "data", "candidate-crops", "cu-style-fields"
            }
            assert payload["artifacts"]["overlay_accepted.png"].endswith(
                "/api/runs/download_run/files/overlay_accepted.png"
            )
            individual = client.get(
                "/api/runs/download_run/files/overlay_accepted.png?download=1"
            )
            assert individual.status_code == 200
            assert "attachment" in individual.headers["Content-Disposition"]

            crop_bundle = client.get(payload["exports"]["candidate-crops"]["url"])
            assert crop_bundle.status_code == 200
            assert crop_bundle.mimetype == "application/zip"
            assert "all_candidate_crops.zip" in crop_bundle.headers["Content-Disposition"]
            with zipfile.ZipFile(io.BytesIO(crop_bundle.data)) as archive:
                assert archive.read("candidate_crops/candidate_000001.tif") == b"raw"
            cu_bundle = client.get(payload["exports"]["cu-style-fields"]["url"])
            assert cu_bundle.status_code == 200
            assert "Cu-0008-R_style_local_fields.zip" in cu_bundle.headers["Content-Disposition"]
            with zipfile.ZipFile(io.BytesIO(cu_bundle.data)) as archive:
                assert archive.namelist()[:4] == [
                    "00000_global_overview.xlsx",
                    "synthetic_clean_00001_X_-10_Y_-46_01_marked.png",
                    "synthetic_clean_00001_X_-10_Y_-46_02_positions.xlsx",
                    "synthetic_clean_00001_X_-10_Y_-46_03_raw_original.tif",
                ]
            latest_bundle = client.get("/api/runs/latest/exports/figures.zip")
            assert latest_bundle.status_code == 200
            assert latest_bundle.mimetype == "application/zip"
            assert client.get("/api/runs/download_run/exports/not-a-kind.zip").status_code == 404
    finally:
        manager.shutdown()
