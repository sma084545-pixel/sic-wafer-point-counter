"""Persistent, auditable local projects for interactive pixel training."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import threading
from typing import Any, Mapping, Sequence
from uuid import uuid4

import cv2
import numpy as np

from .image_io import load_image
from .pixel_classifier import (
    BACKGROUND_LABEL,
    IGNORE_LABEL,
    PixelClassifierError,
    PixelTrainingSample,
    TARGET_LABEL,
    build_training_project,
    decode_label_mask_rle,
    encode_label_mask_rle,
    evaluate_samples,
    predict_pixel_probability,
    probability_to_mask,
    train_pixel_classifier,
    training_project_sha256,
    validate_pixel_model,
    validate_training_project,
)
from .utils import atomic_write_json, utc_now_iso


class PixelTrainingRepositoryError(ValueError):
    """Raised when a local pixel-training project is unsafe or incomplete."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text or len(text) > 80 or not all(char.isalnum() or char in "-_." for char in text):
        raise PixelTrainingRepositoryError(f"{name} 无效")
    return text


def _probability_png(probability: np.ndarray, destination: Path) -> Path:
    values = np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)
    coloured = cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS)
    if not cv2.imwrite(str(destination), coloured):
        raise OSError(f"无法保存概率预览：{destination}")
    return destination


def _mask_png(mask: np.ndarray, destination: Path) -> Path:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[np.asarray(mask, bool)] = (50, 210, 255)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"无法保存分割预览：{destination}")
    return destination


def _overlay_png(
    image: np.ndarray,
    labels: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    destination: Path,
) -> Path:
    gray = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    prediction_boundary = cv2.morphologyEx(
        np.asarray(prediction, np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    overlay[prediction_boundary] = (255, 70, 210)  # predicted: magenta
    overlay[labels == TARGET_LABEL] = (0, 0, 255)
    overlay[labels == BACKGROUND_LABEL] = (0, 190, 0)
    overlay[labels == IGNORE_LABEL] = (0, 210, 235)
    cv2.putText(
        overlay,
        "RED label target | GREEN label background | YELLOW ignore | MAGENTA prediction",
        (8, max(18, min(overlay.shape[0] - 4, 22))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(destination), overlay):
        raise OSError(f"无法保存训练叠加预览：{destination}")
    return destination


class PixelTrainingRepository:
    """Manage source-linked ROI labels and one active portable pixel model."""

    def __init__(self, workspace: str | Path, config: Mapping[str, Any]) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = self.workspace / "training" / "pixel_projects"
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_path = self.workspace / "training" / "pixel_classifier.json"
        self.config = json.loads(json.dumps(dict(config)))
        self._lock = threading.RLock()

    def _project_dir(self, project_id: str) -> Path:
        identifier = _safe_id(project_id, name="project_id")
        path = self.root / identifier
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != self.root:
            raise PixelTrainingRepositoryError("训练项目不存在")
        return path.resolve()

    def _project_path(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project.json"

    def _read(self, project_id: str) -> dict[str, Any]:
        path = self._project_path(project_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            project = validate_training_project(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, PixelClassifierError) as exc:
            raise PixelTrainingRepositoryError("训练项目文件损坏或不兼容") from exc
        project["project_id"] = _safe_id(project_id, name="project_id")
        return project

    def _source_path(self, project_id: str, project: Mapping[str, Any] | None = None) -> Path:
        project = self._read(project_id) if project is None else project
        file_name = Path(str(project["source_image"]["stored_file_name"])).name
        path = self._project_dir(project_id) / file_name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != self._project_dir(project_id):
            raise PixelTrainingRepositoryError("训练项目原图缺失")
        if _file_sha256(path) != str(project["source_image"]["sha256"]):
            raise PixelTrainingRepositoryError("训练项目原图 SHA-256 已改变")
        return path

    def create_project(
        self,
        source_path: str | Path,
        *,
        original_name: str,
        wafer_id: str,
        split: str,
        reviewer_id: str = "local_expert",
        move_source: bool = False,
    ) -> dict[str, Any]:
        """Create a project and a display preview without materializing a large TIFF."""

        source = Path(source_path).expanduser().resolve()
        if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
            raise PixelTrainingRepositoryError("训练原图不存在或为空")
        project_id = "px_" + uuid4().hex
        folder = self.root / project_id
        folder.mkdir(parents=False, exist_ok=False)
        suffix = source.suffix.lower() or ".tif"
        stored_name = "source" + suffix
        destination = folder / stored_name
        try:
            if move_source:
                shutil.move(str(source), str(destination))
            else:
                shutil.copy2(source, destination)
            digest = _file_sha256(destination)
            with load_image(destination, self.config) as image_data:
                preview = image_data.preview.copy()
                metadata = image_data.metadata.to_dict()
                source_shape = image_data.shape
            if not cv2.imwrite(str(folder / "preview.png"), preview):
                raise OSError("无法保存训练项目预览")
            base = build_training_project(
                image_sha256=digest,
                image_name=Path(original_name).name,
                source_shape=source_shape,
                wafer_id=str(wafer_id).strip() or digest[:16],
                split=split,
                reviewer_id=reviewer_id,
                annotations=[],
                operation_history=[{"at_utc": utc_now_iso(), "action": "project_created"}],
            )
            base["source_image"].update(
                {
                    "stored_file_name": stored_name,
                    "source_dtype": metadata.get("dtype"),
                    "analysis_dtype": metadata.get("analysis_dtype"),
                    "normalization_low_value": metadata.get("normalization_low_value"),
                    "normalization_high_value": metadata.get("normalization_high_value"),
                    "white_is_zero": metadata.get("white_is_zero"),
                    "preview_shape_yx": list(preview.shape),
                }
            )
            # The added source audit fields are part of the persisted digest.
            base["project_sha256"] = training_project_sha256(base)
            atomic_write_json(folder / "project.json", base)
            return self.public_project(project_id)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def _write_project(self, project_id: str, project: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(project)
        payload.pop("project_id", None)
        payload["project_sha256"] = training_project_sha256(payload)
        atomic_write_json(self._project_path(project_id), payload)
        return self._read(project_id)

    def public_project(self, project_id: str) -> dict[str, Any]:
        project = self._read(project_id)
        source = dict(project["source_image"])
        source.pop("stored_file_name", None)
        project["source_image"] = source
        project["preview_url"] = f"/api/pixel-training/projects/{project_id}/preview"
        project["model_available"] = self.model_path.is_file()
        return project

    def list_projects(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for folder in sorted(self.root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if folder.is_symlink() or not folder.is_dir():
                continue
            try:
                project = self.public_project(folder.name)
            except PixelTrainingRepositoryError:
                continue
            rows.append(
                {
                    "project_id": folder.name,
                    "image_name": project["source_image"]["file_name"],
                    "wafer_id": project["wafer_id"],
                    "split": project["split"],
                    "roi_count": len(project["annotations"]),
                    "model_available": project["model_available"],
                }
            )
        return rows

    def preview_path(self, project_id: str) -> Path:
        path = self._project_dir(project_id) / "preview.png"
        if not path.is_file() or path.is_symlink():
            raise PixelTrainingRepositoryError("训练项目预览缺失")
        return path

    def read_roi(
        self, project_id: str, roi_xywh: Sequence[int]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        project = self._read(project_id)
        if len(roi_xywh) != 4:
            raise PixelTrainingRepositoryError("ROI 必须包含 x、y、width、height")
        x, y, width, height = (int(value) for value in roi_xywh)
        source_height, source_width = (int(value) for value in project["source_image"]["shape_yx"])
        if (
            x < 0
            or y < 0
            or width < 32
            or height < 32
            or width > 2048
            or height > 2048
            or x + width > source_width
            or y + height > source_height
        ):
            raise PixelTrainingRepositoryError("ROI 必须位于原图内，且边长为 32–2048 px")
        source_path = self._source_path(project_id, project)
        with load_image(source_path, self.config) as image_data:
            roi = image_data.source.read_region(x, y, width, height, normalize=True)
            metadata = image_data.metadata.to_dict()
        return np.asarray(roi, dtype=np.float32), metadata

    def roi_png(self, project_id: str, roi_xywh: Sequence[int]) -> bytes:
        roi, _ = self.read_roi(project_id, roi_xywh)
        gray = np.rint(np.clip(roi, 0.0, 1.0) * 255.0).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", gray)
        if not ok:
            raise PixelTrainingRepositoryError("无法编码训练 ROI")
        return encoded.tobytes()

    def save_annotation(
        self,
        project_id: str,
        *,
        annotation_id: str | None,
        roi_xywh: Sequence[int],
        labels_rle: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist one source-coordinate ROI label mask without lossy conversion."""

        with self._lock:
            project = self._read(project_id)
            labels = decode_label_mask_rle(labels_rle)
            roi = tuple(int(value) for value in roi_xywh)
            if len(roi) != 4 or labels.shape != (roi[3], roi[2]):
                raise PixelTrainingRepositoryError("ROI 尺寸与标签掩膜不一致")
            # Bounds and maximum ROI size are validated by the same source reader.
            self.read_roi(project_id, roi)
            identifier = (
                _safe_id(annotation_id, name="annotation_id")
                if annotation_id
                else f"roi_{len(project['annotations']) + 1:04d}"
            )
            entry = {
                "annotation_id": identifier,
                "roi_xywh": list(roi),
                "labels": encode_label_mask_rle(labels),
                "updated_at_utc": utc_now_iso(),
            }
            annotations = [
                value for value in project["annotations"] if value["annotation_id"] != identifier
            ]
            annotations.append(entry)
            project["annotations"] = annotations
            project.setdefault("operation_history", []).append(
                {
                    "at_utc": utc_now_iso(),
                    "action": "labels_saved",
                    "annotation_id": identifier,
                    "target_pixels": int(np.sum(labels == TARGET_LABEL)),
                    "background_pixels": int(np.sum(labels == BACKGROUND_LABEL)),
                    "ignore_pixels": int(np.sum(labels == IGNORE_LABEL)),
                }
            )
            self._write_project(project_id, project)
            return entry

    def _sample(self, project_id: str, annotation: Mapping[str, Any]) -> PixelTrainingSample:
        project = self._read(project_id)
        roi = tuple(int(value) for value in annotation["roi_xywh"])
        image, _ = self.read_roi(project_id, roi)
        labels = decode_label_mask_rle(annotation["labels"])
        return PixelTrainingSample(
            image=image,
            labels=labels,
            image_sha256=str(project["source_image"]["sha256"]),
            image_name=str(project["source_image"]["file_name"]),
            wafer_id=str(project["wafer_id"]),
            split=str(project["split"]),
            roi_xywh=roi,
        )

    def all_samples(self) -> list[PixelTrainingSample]:
        samples: list[PixelTrainingSample] = []
        for row in self.list_projects():
            project = self._read(row["project_id"])
            for annotation in project["annotations"]:
                samples.append(self._sample(row["project_id"], annotation))
        return samples

    def train(
        self,
        *,
        probability_threshold: float = 0.5,
        minimum_object_area_px: int = 5,
        n_trees: int = 32,
        random_seed: int = 1729,
    ) -> dict[str, Any]:
        """Train across all calibration projects and score any held-out projects."""

        with self._lock:
            try:
                model = train_pixel_classifier(
                    self.all_samples(),
                    probability_threshold=probability_threshold,
                    minimum_object_area_px=minimum_object_area_px,
                    n_trees=n_trees,
                    random_seed=random_seed,
                )
            except PixelClassifierError as exc:
                raise PixelTrainingRepositoryError(str(exc)) from exc
            atomic_write_json(self.model_path, model)
            for row in self.list_projects():
                project = self._read(row["project_id"])
                project["model"] = model
                project.setdefault("operation_history", []).append(
                    {
                        "at_utc": utc_now_iso(),
                        "action": "pixel_model_trained",
                        "model_sha256": model["model_sha256"],
                        "validation_status": model.get("validation", {}).get("status"),
                    }
                )
                self._write_project(row["project_id"], project)
            return model

    def project_file(self, project_id: str) -> Path:
        """Return a validated project JSON for a safe download route."""

        self._read(project_id)
        return self._project_path(project_id)

    def active_model(self) -> dict[str, Any] | None:
        if not self.model_path.is_file():
            return None
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            return validate_pixel_model(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, PixelClassifierError) as exc:
            raise PixelTrainingRepositoryError("pixel_classifier.json 损坏或不兼容") from exc

    def install_model(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and atomically activate a portable model from another endpoint."""

        try:
            model = validate_pixel_model(value)
        except PixelClassifierError as exc:
            raise PixelTrainingRepositoryError(str(exc)) from exc
        atomic_write_json(self.model_path, model)
        return model

    def predict_annotation(
        self,
        project_id: str,
        annotation_id: str,
        *,
        threshold: float | None = None,
        minimum_object_area_px: int | None = None,
    ) -> dict[str, Any]:
        """Generate probability, segmentation and overlay feedback for one ROI."""

        project = self._read(project_id)
        wanted = _safe_id(annotation_id, name="annotation_id")
        annotation = next(
            (item for item in project["annotations"] if item["annotation_id"] == wanted), None
        )
        if annotation is None:
            raise PixelTrainingRepositoryError("训练 ROI 标注不存在")
        model = self.active_model()
        if model is None:
            raise PixelTrainingRepositoryError("尚未训练像素模型")
        sample = self._sample(project_id, annotation)
        probability = predict_pixel_probability(sample.image, model)
        used_threshold = float(model["probability_threshold"] if threshold is None else threshold)
        used_area = int(
            model["minimum_object_area_px"]
            if minimum_object_area_px is None
            else minimum_object_area_px
        )
        prediction = probability_to_mask(
            probability,
            threshold=used_threshold,
            minimum_object_area_px=used_area,
        )
        output = self._project_dir(project_id) / "predictions" / wanted
        output.mkdir(parents=True, exist_ok=True)
        probability_path = _probability_png(probability, output / "probability.png")
        mask_path = _mask_png(prediction, output / "segmentation.png")
        overlay_path = _overlay_png(
            sample.image,
            sample.labels,
            probability,
            prediction,
            output / "overlay.png",
        )
        metrics = evaluate_samples(
            [sample],
            model,
            probability_threshold=used_threshold,
            minimum_object_area_px=used_area,
        )
        audit = {
            "generated_at_utc": utc_now_iso(),
            "model_sha256": model["model_sha256"],
            "threshold": used_threshold,
            "minimum_object_area_px": used_area,
            "metrics": metrics,
            "files": {
                "probability": str(probability_path.relative_to(self._project_dir(project_id))),
                "segmentation": str(mask_path.relative_to(self._project_dir(project_id))),
                "overlay": str(overlay_path.relative_to(self._project_dir(project_id))),
            },
        }
        atomic_write_json(output / "prediction.json", audit)
        return audit

    def prediction_file(self, project_id: str, annotation_id: str, name: str) -> Path:
        if name not in {"probability.png", "segmentation.png", "overlay.png", "prediction.json"}:
            raise PixelTrainingRepositoryError("不允许的训练预览文件")
        path = self._project_dir(project_id) / "predictions" / _safe_id(
            annotation_id, name="annotation_id"
        ) / name
        if path.is_symlink() or not path.is_file():
            raise PixelTrainingRepositoryError("训练预览文件不存在")
        return path


__all__ = ["PixelTrainingRepository", "PixelTrainingRepositoryError"]
