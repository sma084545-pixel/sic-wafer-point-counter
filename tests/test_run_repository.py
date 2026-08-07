"""Persistent result browsing, pagination, and filesystem safety tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sic_wafer_counter.run_repository import RunRepository, RunRepositoryError, public_json
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
