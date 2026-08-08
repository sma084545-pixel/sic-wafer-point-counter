"""Local browser workbench integration tests against the real pipeline."""

from __future__ import annotations

import io
from pathlib import Path
import shutil
import time

from sic_wafer_counter.web import create_app


def _wait_for_job(client, job_id: str) -> dict:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.get_json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("browser analysis job did not finish within 30 seconds")


def test_browser_workbench_submits_real_analysis_and_serves_artifacts(
    generate_synthetic, tmp_path: Path
) -> None:
    generated = generate_synthetic("clean")
    app = create_app(tmp_path / "workspace", max_workers=1, max_upload_mb=32)
    app.config["TESTING"] = True
    manager = app.extensions["sic_wafer_job_manager"]
    try:
        with app.test_client() as client:
            page = client.get("/")
            assert page.status_code == 200
            assert "SiC 晶圆点状目标分析" in page.get_data(as_text=True)
            submit = client.post(
                "/api/jobs",
                data={
                    "image": (
                        io.BytesIO(Path(generated["image_path"]).read_bytes()),
                        "synthetic_clean.png",
                    ),
                    "wafer_diameter_mm": "100",
                    "exclude_edge_mm": "0",
                    "threshold_method": "otsu",
                    "use_watershed": "true",
                },
                content_type="multipart/form-data",
            )
            assert submit.status_code == 202
            job = _wait_for_job(client, submit.get_json()["job_id"])
            assert job["status"] == "completed", job.get("error")
            assert job["summary"]["accepted_count"] == 96
            assert job["summary"]["density_unit"] == "cm^-2"
            required_artifacts = {
                "overlay_accepted.png",
                "overlay_xrt_red_boxes.png",
                "xrt_detection_detail_montage.png",
                "paper_detection_field.png",
                "paper_aligned_result_figure.png",
                "defect_comparison_details.png",
                "defect_size_histogram.png",
                "density_heatmap.png",
                "density_heatmap_grid.csv",
            }
            assert required_artifacts <= set(job["artifacts"])
            for name in (
                "overlay_xrt_red_boxes.png",
                "xrt_detection_detail_montage.png",
                "paper_detection_field.png",
                "paper_aligned_result_figure.png",
                "defect_comparison_details.png",
                "density_heatmap.png",
            ):
                artifact = client.get(job["artifacts"][name])
                assert artifact.status_code == 200
                assert artifact.mimetype == "image/png"
            grid = client.get(job["artifacts"]["density_heatmap_grid.csv"])
            assert grid.status_code == 200
            assert grid.mimetype == "text/csv"
            assert client.get(f"/api/jobs/{job['job_id']}/files/../../README.md").status_code == 404
    finally:
        manager.shutdown()


def test_browser_workbench_rejects_unsafe_upload_and_partial_manual_geometry(tmp_path: Path) -> None:
    app = create_app(tmp_path / "workspace", max_workers=1, max_upload_mb=1)
    app.config["TESTING"] = True
    manager = app.extensions["sic_wafer_job_manager"]
    try:
        with app.test_client() as client:
            rejected = client.post(
                "/api/jobs",
                data={"image": (io.BytesIO(b"not an image"), "unsafe.txt")},
                content_type="multipart/form-data",
            )
            assert rejected.status_code == 400
            partial = client.post(
                "/api/jobs",
                data={
                    "image": (io.BytesIO(b"not an image"), "valid-name.tif"),
                    "center_x": "20",
                },
                content_type="multipart/form-data",
            )
            assert partial.status_code == 400
            assert "手工标定" in partial.get_json()["error"]
            invalid_number = client.post(
                "/api/jobs",
                data={
                    "image": (io.BytesIO(b"not an image"), "valid-name.tif"),
                    "wafer_diameter_mm": "NaN",
                },
                content_type="multipart/form-data",
            )
            assert invalid_number.status_code == 400
            assert "有限数值" in invalid_number.get_json()["error"]
            oversized = client.post(
                "/api/jobs",
                data={"image": (io.BytesIO(b"0" * (1024 * 1024 + 1)), "large.tif")},
                content_type="multipart/form-data",
            )
            assert oversized.status_code == 413
    finally:
        manager.shutdown()


def test_failed_analysis_is_persisted_for_restart_safe_review(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    app = create_app(workspace, max_workers=1, max_upload_mb=1)
    app.config["TESTING"] = True
    manager = app.extensions["sic_wafer_job_manager"]
    try:
        with app.test_client() as client:
            submitted = client.post(
                "/api/jobs",
                data={"image": (io.BytesIO(b"not really a tiff"), "broken.tif")},
                content_type="multipart/form-data",
            )
            assert submitted.status_code == 202
            failed = _wait_for_job(client, submitted.get_json()["job_id"])
            assert failed["status"] == "failed"
            assert failed["summary"]["point_density_cm2"] is None
            detail = client.get(f"/api/runs/{failed['run_id']}")
            assert detail.status_code == 200
            assert detail.get_json()["status"] == "failed"
            assert str(workspace) not in detail.get_data(as_text=True)
    finally:
        manager.shutdown()


def test_clean_demo_uses_real_pipeline_and_persists_result(
    generate_synthetic, tmp_path: Path
) -> None:
    generated = generate_synthetic("clean")
    workspace = tmp_path / "workspace"
    demo_dir = workspace / "sample_data" / "generated"
    demo_dir.mkdir(parents=True)
    shutil.copy2(generated["image_path"], demo_dir / "synthetic_clean.png")
    app = create_app(workspace, max_workers=1, max_upload_mb=32)
    app.config["TESTING"] = True
    manager = app.extensions["sic_wafer_job_manager"]
    try:
        with app.test_client() as client:
            submitted = client.post("/api/demo/clean")
            assert submitted.status_code == 202
            completed = _wait_for_job(client, submitted.get_json()["job_id"])
            assert completed["status"] == "completed"
            assert completed["summary"]["accepted_count"] == 96
            runs = client.get("/api/runs").get_json()["runs"]
            assert completed["run_id"] in {run["run_id"] for run in runs}
            detail = client.get(f"/api/runs/{completed['run_id']}").get_json()
            assert detail["dataset_kind"] == "synthetic"
            assert detail["summary"]["analysis_quantized_to_uint8"] is False
            assert detail["summary"]["real_annotation_validation_status"] == "not validated on real SiC data"
    finally:
        manager.shutdown()
