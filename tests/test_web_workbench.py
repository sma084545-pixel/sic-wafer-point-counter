"""Local browser workbench integration tests against the real pipeline."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import zipfile

import cv2
import numpy as np

from sic_wafer_counter.pixel_classifier import encode_label_mask_rle

from sic_wafer_counter.web import create_app
from sic_wafer_counter import __version__


def test_pixel_training_web_api_closes_source_label_model_preview_loop(tmp_path: Path) -> None:
    image = np.full((96, 112), 190, np.uint8)
    labels = np.zeros(image.shape, np.uint8)
    cv2.circle(image, (34, 38), 5, 45, -1)
    cv2.circle(image, (76, 61), 5, 52, -1)
    labels[32:45, 28:41] = 1
    labels[55:68, 70:83] = 1
    labels[4:18, 4:108] = 2
    labels[78:92, 4:108] = 2
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    app = create_app(tmp_path / "workspace", max_workers=1, max_upload_mb=4)
    app.config["TESTING"] = True
    manager = app.extensions["sic_wafer_job_manager"]
    try:
        with app.test_client() as client:
            page = client.get("/pixel-training")
            assert page.status_code == 200
            text = page.get_data(as_text=True)
            assert "源像素标注画布" in text and "目标概率图" in text
            created = client.post(
                "/api/pixel-training/projects",
                data={
                    "image": (io.BytesIO(encoded.tobytes()), "training.png"),
                    "wafer_id": "wafer-api-1",
                    "reviewer_id": "reviewer-A",
                    "split": "calibration",
                },
                content_type="multipart/form-data",
            )
            assert created.status_code == 201, created.get_json()
            project = created.get_json()["project"]
            project_id = project["project_id"]
            assert project["reviewer_id"] == "reviewer-A"
            assert client.get(project["preview_url"]).mimetype == "image/png"
            roi = client.get(
                f"/api/pixel-training/projects/{project_id}/roi?x=0&y=0&width=112&height=96"
            )
            assert roi.status_code == 200 and roi.mimetype == "image/png"
            saved = client.post(
                f"/api/pixel-training/projects/{project_id}/annotations",
                json={"roi_xywh": [0, 0, 112, 96], "labels": encode_label_mask_rle(labels)},
            )
            assert saved.status_code == 200, saved.get_json()
            annotation_id = saved.get_json()["annotation"]["annotation_id"]
            trained = client.post(
                "/api/pixel-training/train",
                json={"n_trees": 4, "random_seed": 22, "probability_threshold": 0.5},
            )
            assert trained.status_code == 200, trained.get_json()
            model = trained.get_json()["model"]
            assert len(model["model_sha256"]) == 64
            assert model["training_sources"][0]["wafer_id"] == "wafer-api-1"
            preview = client.post(
                f"/api/pixel-training/projects/{project_id}/predict/{annotation_id}",
                json={"probability_threshold": 0.5, "minimum_object_area_px": 5},
            )
            assert preview.status_code == 200, preview.get_json()
            report = preview.get_json()["prediction"]
            assert set(report["files"]) == {"probability", "segmentation", "overlay"}
            for url in report["files"].values():
                response = client.get(url)
                assert response.status_code == 200 and response.mimetype == "image/png"
            downloaded = client.get(f"/api/pixel-training/projects/{project_id}/download")
            payload = json.loads(downloaded.get_data(as_text=True))
            assert payload["model"]["model_sha256"] == model["model_sha256"]
            imported = client.post("/api/pixel-training/model", json=model)
            assert imported.status_code == 200
            assert imported.get_json()["model"]["model_sha256"] == model["model_sha256"]
    finally:
        manager.shutdown()


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
            assert "独立与批量导出" in page.get_data(as_text=True)
            assert "一次导出全部局部分析包 ZIP" in page.get_data(as_text=True)
            assert "一键导出 Cu-0008-R 顺序三联件 ZIP" in page.get_data(as_text=True)
            assert "专家标注与候选训练" in page.get_data(as_text=True)
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
            latest = client.get("/api/runs/latest")
            assert latest.status_code == 200
            latest_payload = latest.get_json()
            assert latest_payload["run_id"] == job["run_id"]
            assert set(latest_payload["exports"]) == {
                "figures", "data", "candidate-crops", "cu-style-fields"
            }
            latest_figure = client.get(
                "/api/runs/latest/files/paper_aligned_result_figure.png"
            )
            assert latest_figure.status_code == 200
            assert latest_figure.mimetype == "image/png"
            crop_bundle = client.get(latest_payload["exports"]["candidate-crops"]["url"])
            assert crop_bundle.status_code == 200
            with zipfile.ZipFile(io.BytesIO(crop_bundle.data)) as archive:
                names = archive.namelist()
                assert names[0] == "00_global_overview.xlsx"
                raw_crops = [
                    name for name in names
                    if name.startswith("candidate_crops/") and name.endswith(".tif")
                ]
                previews = [
                    name for name in names
                    if name.startswith("candidate_crops/") and name.endswith("_preview.png")
                ]
                local_raw = [
                    name for name in names
                    if name.startswith("local_fields/") and name.endswith("03_raw_original.tif")
                ]
                local_marked = [
                    name for name in names
                    if name.startswith("local_fields/") and name.endswith("01_marked.png")
                ]
                local_positions = [
                    name for name in names
                    if name.startswith("local_fields/") and name.endswith("02_positions.xlsx")
                ]
                candidate_total = (
                    job["summary"]["accepted_count"] + job["summary"]["rejected_count"]
                )
                assert len(raw_crops) == candidate_total
                assert len(previews) == candidate_total
                assert len(local_raw) == len(local_marked) == len(local_positions) > 0
                assert "index/defects_all.csv" in names
            cu_bundle = client.get(latest_payload["exports"]["cu-style-fields"]["url"])
            assert cu_bundle.status_code == 200
            with zipfile.ZipFile(io.BytesIO(cu_bundle.data)) as archive:
                names = archive.namelist()
                assert names[0] == "00000_global_overview.xlsx"
                assert all("/" not in name for name in names)
                assert (len(names) - 1) % 3 == 0
                for offset in range(1, len(names), 3):
                    unit = names[offset:offset + 3]
                    assert unit[0].endswith("_01_marked.png")
                    assert unit[1].endswith("_02_positions.xlsx")
                    assert unit[2].endswith("_03_raw_original.tif")
            assert client.get(f"/api/jobs/{job['job_id']}/files/../../README.md").status_code == 404
    finally:
        manager.shutdown()


def test_health_identifies_version_without_exposing_workspace_path(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()
    app = create_app(workspace, max_workers=1, max_upload_mb=1)
    app.config["TESTING"] = True
    manager = app.extensions["sic_wafer_job_manager"]
    try:
        with app.test_client() as client:
            response = client.get("/api/health")
            assert response.status_code == 200
            payload = response.get_json()
            assert payload == {
                "application": "sic-wafer-point-counter",
                "software_version": __version__,
                "git_revision": "未提供",
                "workspace_id": hashlib.sha256(
                    str(workspace).encode("utf-8")
                ).hexdigest(),
                "status": "ready",
            }
            assert str(workspace) not in response.get_data(as_text=True)
            assert response.headers["Cache-Control"] == "no-store"
    finally:
        manager.shutdown()


def test_latest_result_routes_return_404_without_persisted_runs(tmp_path: Path) -> None:
    app = create_app(tmp_path / "workspace", max_workers=1, max_upload_mb=1)
    app.config["TESTING"] = True
    manager = app.extensions["sic_wafer_job_manager"]
    try:
        with app.test_client() as client:
            assert client.get("/api/runs/latest").status_code == 404
            assert client.get(
                "/api/runs/latest/files/paper_aligned_result_figure.png"
            ).status_code == 404
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
