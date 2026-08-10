from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sic_wafer_counter.pixel_classifier import encode_label_mask_rle
from sic_wafer_counter.pixel_training_repository import PixelTrainingRepository


def _source(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(6)
    image = np.clip(190 + rng.normal(0, 3, (128, 144)), 0, 255).astype(np.uint8)
    labels = np.zeros(image.shape, dtype=np.uint8)
    yy, xx = np.ogrid[:128, :144]
    for y, x in ((30, 35), (66, 75), (94, 112)):
        point = (yy - y) ** 2 + (xx - x) ** 2 <= 4**2
        image[point] = 55
        labels[point] = 1
    labels[5:14, 6:136] = 2
    labels[112:122, 8:136] = 2
    assert cv2.imwrite(str(path), image)
    return image, labels


def test_pixel_training_repository_persists_labels_trains_and_renders_feedback(
    tmp_path: Path,
    default_config: dict,
) -> None:
    source = tmp_path / "source.png"
    _, labels = _source(source)
    repository = PixelTrainingRepository(tmp_path, default_config)
    project = repository.create_project(
        source,
        original_name="source.png",
        wafer_id="wafer-001",
        split="calibration",
        reviewer_id="expert-A",
    )
    project_id = project["project_id"]
    assert repository.preview_path(project_id).is_file()
    assert cv2.imdecode(
        np.frombuffer(repository.roi_png(project_id, (0, 0, 144, 128)), np.uint8),
        cv2.IMREAD_GRAYSCALE,
    ).shape == labels.shape
    saved = repository.save_annotation(
        project_id,
        annotation_id=None,
        roi_xywh=(0, 0, 144, 128),
        labels_rle=encode_label_mask_rle(labels),
    )
    reopened = repository.public_project(project_id)
    assert reopened["annotations"][0]["annotation_id"] == saved["annotation_id"]
    model = repository.train(n_trees=6, random_seed=9)
    assert repository.active_model()["model_sha256"] == model["model_sha256"]
    persisted = json.loads(repository.project_file(project_id).read_text(encoding="utf-8"))
    assert persisted["reviewer_id"] == "expert-A"
    assert persisted["model"]["model_sha256"] == model["model_sha256"]
    assert persisted["operation_history"][-1]["action"] == "pixel_model_trained"
    feedback = repository.predict_annotation(project_id, saved["annotation_id"])
    assert feedback["model_sha256"] == model["model_sha256"]
    for name in ("probability.png", "segmentation.png", "overlay.png", "prediction.json"):
        assert repository.prediction_file(project_id, saved["annotation_id"], name).is_file()

    installed = repository.install_model(json.loads(json.dumps(model)))
    assert installed["model_sha256"] == model["model_sha256"]


def test_pixel_project_detects_modified_source(tmp_path: Path, default_config: dict) -> None:
    source = tmp_path / "source.png"
    _source(source)
    repository = PixelTrainingRepository(tmp_path, default_config)
    project = repository.create_project(
        source,
        original_name="source.png",
        wafer_id="wafer-001",
        split="calibration",
    )
    project_id = project["project_id"]
    metadata_path = tmp_path / "training" / "pixel_projects" / project_id / "project.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    stored = metadata_path.parent / payload["source_image"]["stored_file_name"]
    stored.write_bytes(stored.read_bytes() + b"tamper")
    try:
        repository.roi_png(project_id, (0, 0, 64, 64))
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("modified training source was not rejected")
