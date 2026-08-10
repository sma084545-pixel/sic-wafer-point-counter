"""Persistent local expert-label repository tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sic_wafer_counter.run_repository import RunRepository
from sic_wafer_counter.training_repository import TrainingRepository
from sic_wafer_counter.web import create_app


def _write_training_run(workspace: Path) -> None:
    run = workspace / "results" / "label-run"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"input_file_name": "label-wafer.tif", "wafer_id": "wafer-A"}),
        encoding="utf-8",
    )
    rows = []
    for index in range(1, 13):
        target = index <= 6
        rows.append({
            "defect_id": index,
            "x_mm": index / 10,
            "y_mm": -index / 10,
            "accepted": target,
            "rejection_reason": "" if target else "too_elongated",
            "area_mm2": 0.00002 if target else 0.00018,
            "equivalent_diameter_um": 5.0 if target else 18.0,
            "major_axis_length_um": 6.0 if target else 32.0,
            "minor_axis_length_um": 4.5 if target else 5.0,
            "aspect_ratio": 1.25 if target else 6.2,
            "eccentricity": 0.35 if target else 0.97,
            "circularity": 0.88 if target else 0.18,
            "solidity": 0.96 if target else 0.58,
            "contrast": 0.42 if target else 0.05,
            "mean_dark_response": 0.61 if target else 0.12,
        })
    pd.DataFrame(rows).to_csv(run / "defects_all.csv", index=False)


def test_training_repository_persists_labels_and_portable_model(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_training_run(workspace)
    repository = TrainingRepository(workspace, RunRepository(workspace / "results"))
    for defect_id in range(1, 13):
        label = "target" if defect_id <= 6 else "artifact"
        saved = repository.save_annotation(
            run_id="label-run",
            defect_id=str(defect_id),
            label=label,
            reviewer_id="expert-1",
        )
        assert saved["label"] == label

    status = repository.status()
    assert status["annotation_count"] == 12
    assert status["consensus_label_counts"] == {
        "artifact": 6,
        "target": 6,
        "uncertain": 0,
    }
    model = repository.train()
    assert repository.model_path.is_file()
    assert repository.annotations_path.is_file()
    assert repository.active_model()["model_sha256"] == model["model_sha256"]
    assert repository.status()["model_available"] is True

    labels = repository.labels_for_run("label-run")
    assert labels["1"] == "target"
    assert labels["12"] == "artifact"


def test_training_repository_marks_multi_reviewer_disagreement_as_conflict(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_training_run(workspace)
    repository = TrainingRepository(workspace, RunRepository(workspace / "results"))
    repository.save_annotation(
        run_id="label-run", defect_id="1", label="target", reviewer_id="expert-1"
    )
    repository.save_annotation(
        run_id="label-run", defect_id="1", label="artifact", reviewer_id="expert-2"
    )
    assert repository.labels_for_run("label-run")["1"] == "conflict"
    assert repository.status()["conflicting_candidate_count"] == 1


def test_training_web_api_labels_trains_downloads_and_returns_candidate_label(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_training_run(workspace)
    app = create_app(workspace, max_workers=1, max_upload_mb=1)
    app.config["TESTING"] = True
    manager = app.extensions["sic_wafer_job_manager"]
    try:
        with app.test_client() as client:
            for defect_id in (*range(1, 6), *range(7, 12)):
                response = client.post(
                    "/api/training/labels",
                    json={
                        "run_id": "label-run",
                        "defect_id": str(defect_id),
                        "label": "target" if defect_id <= 5 else "artifact",
                        "split": "calibration",
                        "reviewer_id": "expert-api",
                    },
                )
                assert response.status_code == 200

            trained = client.post(
                "/api/training/train",
                json={
                    "accept_threshold": 0.75,
                    "reject_threshold": 0.25,
                    "regularization": 1.0,
                },
            )
            assert trained.status_code == 200
            assert trained.get_json()["training"]["model_available"] is True
            model = client.get("/api/training/model")
            annotations = client.get("/api/training/annotations")
            assert model.status_code == 200 and model.mimetype == "application/json"
            assert annotations.status_code == 200 and annotations.mimetype == "text/csv"
            candidates = client.get("/api/runs/label-run/defects?page_size=20")
            assert candidates.status_code == 200
            assert candidates.get_json()["rows"][0]["training_label"] == "target"
    finally:
        manager.shutdown()
