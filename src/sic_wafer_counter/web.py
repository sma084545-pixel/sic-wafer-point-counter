"""Local browser workbench for the auditable SiC wafer analysis pipeline.

The server is intentionally local-first: it accepts only image uploads, saves
them under the selected workspace, runs the existing :func:`analyze_image`
pipeline in a bounded queue, and exposes only files produced for that job.
"""

from __future__ import annotations

import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import resources
import logging
import math
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from . import __version__
from .image_io import ImageReadError
from .pipeline import analyze_image
from .result_export import EXPORT_BUNDLES, ResultExporter
from .run_repository import RunRepository, RunRepositoryError, public_json
from .training_repository import TrainingRepository, TrainingRepositoryError
from .utils import ConfigurationError, deep_merge, load_config, setup_logging
from .wafer_detection import WaferDetectionError


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(resources.files("sic_wafer_counter").joinpath("resources/default.yaml"))
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".btf", ".bigtif", ".bigtiff"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _workspace_id(workspace: Path) -> str:
    """Return a stable, non-reversible identifier for one local checkout."""

    return hashlib.sha256(str(workspace.resolve()).encode("utf-8")).hexdigest()


def _git_revision(workspace: Path) -> str:
    """Return the current short revision without making Git a runtime dependency."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "未提供"
    revision = completed.stdout.strip()
    return revision if revision and all(character in "0123456789abcdef" for character in revision.lower()) else "未提供"


def _hostname(value: str) -> str | None:
    """Parse a Host header or absolute Origin without trusting raw separators."""

    candidate = value if "://" in value else f"//{value}"
    try:
        return urlsplit(candidate).hostname
    except ValueError:
        return None


@dataclass(slots=True)
class AnalysisJob:
    """One submitted local analysis job and the paths it is allowed to expose."""

    job_id: str
    upload_path: Path
    output_dir: Path
    config: dict[str, Any]
    manual_geometry: tuple[float | None, float | None, float | None]
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    error: str | None = None
    summary: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    future: Future[None] | None = field(default=None, repr=False)

    @property
    def run_id(self) -> str:
        """Persistent result-directory identifier for this transient job."""

        return self.output_dir.name


def _json_safe(value: Any) -> Any:
    """Convert analysis summary values to strict JSON response data."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_optional_float(form: Mapping[str, str], key: str, *, minimum: float | None = None) -> float | None:
    """Parse one optional finite web-form number with an actionable error."""

    raw = str(form.get(key, "")).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} 必须是数值") from exc
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        comparison = f"且不小于 {minimum}" if minimum is not None else ""
        raise ValueError(f"{key} 必须是有限数值{comparison}")
    return value


def _analysis_config_from_form(form: Mapping[str, str]) -> tuple[dict[str, Any], tuple[float | None, float | None, float | None]]:
    """Apply a deliberately small, auditable set of browser controls."""

    config = load_config(DEFAULT_CONFIG_PATH)
    diameter = _parse_optional_float(form, "wafer_diameter_mm", minimum=np.finfo(float).eps)
    edge = _parse_optional_float(form, "exclude_edge_mm", minimum=0.0)
    threshold = str(form.get("threshold_method", "")).strip().lower()
    if threshold and threshold not in {"otsu", "adaptive", "quantile"}:
        raise ValueError("threshold_method 必须是 otsu、adaptive 或 quantile")
    overrides: dict[str, Any] = {}
    if diameter is not None:
        overrides.setdefault("wafer", {})["diameter_mm"] = diameter
    if edge is not None:
        overrides.setdefault("wafer", {})["exclude_edge_mm"] = edge
    if threshold:
        overrides.setdefault("detection", {})["threshold_method"] = threshold
    overrides.setdefault("detection", {})["use_watershed"] = form.get("use_watershed") == "true"
    overrides.setdefault("output", {})["save_intermediates"] = True
    manual = tuple(
        _parse_optional_float(form, field, minimum=0.0)
        for field in ("center_x", "center_y", "radius_px")
    )
    if any(value is not None for value in manual) and not all(value is not None for value in manual):
        raise ValueError("手工标定时必须同时填写圆心 x、圆心 y 与半径")
    if manual[2] is not None and manual[2] <= 0:
        raise ValueError("手工半径必须大于零")
    return deep_merge(config, overrides), manual


class _JobManager:
    """Thread-safe bounded runner protecting large-image analyses from overlap."""

    def __init__(self, workspace: Path, *, max_workers: int) -> None:
        self.workspace = workspace
        self.upload_root = workspace / "web_uploads"
        self.result_root = workspace / "results"
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.result_root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sic-analysis")
        self._jobs: dict[str, AnalysisJob] = {}
        self._lock = threading.RLock()

    def submit(self, upload_path: Path, config: dict[str, Any], manual: tuple[float | None, float | None, float | None]) -> AnalysisJob:
        job_id = uuid4().hex
        job = AnalysisJob(
            job_id=job_id,
            upload_path=upload_path,
            output_dir=self.result_root / f"web_{job_id}",
            config=copy.deepcopy(config),
            manual_geometry=manual,
        )
        with self._lock:
            self._jobs[job_id] = job
            job.future = self._executor.submit(self._run, job_id)
        return job

    def get(self, job_id: str) -> AnalysisJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        with self._lock:
            job.status = "running"
        try:
            setup_logging(
                job.output_dir,
                level=str(job.config.get("logging", {}).get("level", "INFO")),
                logger_name=f"sic_wafer_counter.web.{job_id}",
            )
            result = analyze_image(
                job.upload_path,
                job.output_dir,
                job.config,
                center_x=job.manual_geometry[0],
                center_y=job.manual_geometry[1],
                radius_px=job.manual_geometry[2],
            )
            artifact_names = (
                "report.html", "summary.json", "defects_all.csv", "defects_accepted.csv",
                "defects_rejected.csv", "overlay_accepted.png", "overlay_all_candidates.png",
                "candidate_classifier.json",
                "overlay_xrt_red_boxes.png", "xrt_detection_detail_montage.png",
                "paper_detection_field.png", "paper_aligned_result_figure.png",
                "defect_comparison_details.png",
                "valid_analysis_mask.png", "preprocessed_preview.png", "defect_size_histogram.png",
                "radial_density.png", "angular_density.png", "density_heatmap.png",
                "density_heatmap_grid.csv", "radial_density.csv", "angular_density.csv",
                "regional_density.csv", "independent_reference_points.csv",
                "independent_reference_matches.csv",
            )
            artifacts = {name: name for name in artifact_names if (job.output_dir / name).is_file()}
            with self._lock:
                job.summary = public_json(_json_safe(result.summary))
                job.artifacts = artifacts
                job.status = "completed"
        except (ConfigurationError, ImageReadError, WaferDetectionError, ValueError, OSError) as exc:
            LOGGER.warning("Web job %s failed: %s", job_id, exc)
            public_error = self._public_error(job, exc)
            with self._lock:
                job.error = public_error
                job.summary = self._failure_summary(job, public_error)
                job.status = "failed"
        except Exception:
            LOGGER.exception("Unexpected web job failure: %s", job_id)
            public_error = "分析出现未预期错误；请查看该结果目录中的 run.log。"
            with self._lock:
                job.error = public_error
                job.summary = self._failure_summary(job, public_error)
                job.status = "failed"

    def _public_error(self, job: AnalysisJob, error: Exception) -> str:
        """Remove local absolute paths from an actionable job error."""

        message = str(error)
        for private in (str(job.upload_path), str(self.workspace), str(Path.home())):
            message = message.replace(private, "<本机路径>")
        return message

    @staticmethod
    def _failure_summary(job: AnalysisJob, public_error: str) -> dict[str, Any]:
        """Load a pipeline failure summary or persist a minimal restart-safe one."""

        path = job.output_dir / "summary.json"
        if path.is_file():
            try:
                import json

                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                loaded = None
            if isinstance(loaded, Mapping):
                return public_json(loaded)
        summary: dict[str, Any] = {
            "status": "failed",
            "input_file_name": job.upload_path.name,
            "input_path": job.upload_path.name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "accepted_count": None,
            "valid_analysis_area_cm2": None,
            "point_density_cm2": None,
            "counting_uncertainty_cm2": None,
            "software_version": __version__,
            "warnings": [public_error],
            "real_annotation_validation_status": "not validated on real SiC data",
        }
        job.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import json

            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            LOGGER.warning("Could not persist failure summary for web job %s", job.job_id)
        return summary


def _job_response(job: AnalysisJob) -> dict[str, Any]:
    """Return one browser-safe status document without exposing filesystem paths."""

    response: dict[str, Any] = {
        "job_id": job.job_id,
        "run_id": job.run_id,
        "status": job.status,
        "error": job.error,
        "created_at_unix": job.created_at,
    }
    if job.summary is not None:
        response["summary"] = public_json(job.summary)
    if job.artifacts:
        response["artifacts"] = {
            name: url_for("job_file", job_id=job.job_id, relative_path=relative)
            for name, relative in job.artifacts.items()
        }
    return response


def create_app(
    workspace: str | Path,
    *,
    max_workers: int = 1,
    max_upload_mb: int = 4096,
) -> Flask:
    """Create the local Flask workbench around the existing analysis pipeline."""

    if max_workers < 1:
        raise ValueError("max_workers must be at least one")
    if max_upload_mb < 1:
        raise ValueError("max_upload_mb must be at least one")
    root = Path(workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = int(max_upload_mb) * 1024 * 1024
    manager = _JobManager(root, max_workers=max_workers)
    repository = RunRepository(manager.result_root)
    exporter = ResultExporter(repository)
    training = TrainingRepository(root, repository)
    app.extensions["sic_wafer_job_manager"] = manager
    app.extensions["sic_wafer_run_repository"] = repository
    app.extensions["sic_wafer_result_exporter"] = exporter
    app.extensions["sic_wafer_training_repository"] = training

    demo_sources = {
        kind: root / "sample_data" / "generated" / f"synthetic_{kind}.png"
        for kind in ("clean", "noisy", "difficult")
    }

    @app.before_request
    def enforce_loopback_request():
        """Reject DNS-rebinding hosts and cross-origin state-changing requests."""

        if _hostname(request.host) not in LOOPBACK_HOSTS:
            return jsonify({"error": "本机工作台只接受回环地址请求"}), 403
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("Origin")
            if origin and _hostname(origin) not in LOOPBACK_HOSTS:
                return jsonify({"error": "拒绝跨站提交到本机工作台"}), 403
        return None

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; "
            "form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            max_upload_mb=max_upload_mb,
            software_version=__version__,
            git_revision=_git_revision(root),
            demo_available={name: path.is_file() for name, path in demo_sources.items()},
        )

    @app.get("/api/health")
    def health():
        """Identify the exact local checkout serving this loopback endpoint."""

        return jsonify({
            "application": "sic-wafer-point-counter",
            "software_version": __version__,
            "git_revision": _git_revision(root),
            "workspace_id": _workspace_id(root),
            "status": "ready",
        })

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_: RequestEntityTooLarge):
        return jsonify({"error": f"文件超过 {max_upload_mb} MB 上传限制"}), 413

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "请求的本机分析资源不存在"}), 404
        return error

    @app.post("/api/jobs")
    def create_job():
        if "image" not in request.files:
            return jsonify({"error": "请选择晶圆图像文件"}), 400
        image = request.files["image"]
        original_name = Path(image.filename or "").name
        suffix = Path(original_name).suffix.lower()
        if not original_name or suffix not in ALLOWED_SUFFIXES:
            return jsonify({"error": "仅支持 PNG、JPG、BMP、TIFF 或 BigTIFF 图像"}), 400
        filename = secure_filename(original_name) or f"uploaded_image{suffix}"
        try:
            config, manual = _analysis_config_from_form(request.form)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if request.form.get("use_trained_classifier") == "true":
            try:
                model = training.active_model()
            except TrainingRepositoryError as exc:
                return jsonify({"error": str(exc)}), 400
            if model is None:
                return jsonify({"error": "尚未训练可用的候选分类器"}), 400
            config = deep_merge(
                config,
                {"classifier": {"enabled": True, "model_path": None, "model": model}},
            )
        upload_id = uuid4().hex
        upload_dir = manager.upload_root / upload_id
        upload_dir.mkdir(parents=True, exist_ok=False)
        destination = upload_dir / filename
        image.save(destination)
        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            upload_dir.rmdir()
            return jsonify({"error": "上传文件为空"}), 400
        job = manager.submit(destination, config, manual)
        return jsonify(_job_response(job)), 202

    @app.post("/api/demo/<kind>")
    def create_demo_job(kind: str):
        """Run a checked-in/generated synthetic source through the real pipeline."""

        source = demo_sources.get(kind)
        if source is None:
            return jsonify({"error": "未知合成演示类型"}), 404
        try:
            source.resolve().relative_to(root)
        except ValueError:
            return jsonify({"error": "合成演示源不在工作区内"}), 404
        if source.is_symlink() or not source.is_file():
            return jsonify({
                "error": (
                    "合成演示图尚未生成。请先运行："
                    "python scripts/generate_synthetic_wafer.py --all --output-dir sample_data/generated"
                )
            }), 404
        config = load_config(DEFAULT_CONFIG_PATH)
        config = deep_merge(config, {"output": {"save_intermediates": True}})
        job = manager.submit(source, config, (None, None, None))
        return jsonify(_job_response(job)), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = manager.get(job_id)
        if job is None:
            abort(404)
        return jsonify(_job_response(job))

    @app.get("/api/jobs/<job_id>/files/<path:relative_path>")
    def job_file(job_id: str, relative_path: str):
        job = manager.get(job_id)
        if job is None:
            abort(404)
        try:
            target = repository.resolve_file(job.run_id, relative_path)
        except RunRepositoryError:
            abort(404)
        return send_file(target, as_attachment=request.args.get("download") == "1")

    @app.get("/api/runs")
    def list_runs():
        index = repository.list_runs()
        return jsonify({
            "runs": index.runs,
            "count": len(index.runs),
            "skipped_invalid_summaries": index.skipped_invalid_summaries,
        })

    def latest_persisted_run_id() -> str:
        """Return the newest completed run, or the newest failed run as fallback."""

        runs = repository.list_runs().runs
        if not runs:
            raise RunRepositoryError("no persisted runs")
        completed = next(
            (run for run in runs if str(run.get("status", "")).lower() == "completed"),
            None,
        )
        return str((completed or runs[0])["run_id"])

    def export_links(run_id: str, *, latest: bool = False) -> dict[str, dict[str, str]]:
        """Return only currently available, server-generated ZIP exports."""

        endpoint = "latest_run_export" if latest else "run_export"
        links: dict[str, dict[str, str]] = {}
        for kind, bundle in exporter.available(run_id).items():
            route_values = {"bundle_kind": kind}
            if not latest:
                route_values["run_id"] = run_id
            links[kind] = {
                "url": url_for(endpoint, **route_values),
                "filename": bundle.filename,
                "label": bundle.label,
                "description": bundle.description,
            }
        return links

    @app.get("/api/runs/latest")
    def latest_run_detail():
        """Expose a stable URL for the newest locally persisted result."""

        try:
            run_id = latest_persisted_run_id()
            detail = repository.run_detail(run_id)
        except RunRepositoryError:
            abort(404)
        detail["artifacts"] = {
            name: url_for("latest_run_file", relative_path=name)
            for name in detail.pop("artifact_names")
        }
        detail["exports"] = export_links(run_id, latest=True)
        return jsonify(detail)

    @app.get("/api/runs/latest/files/<path:relative_path>")
    def latest_run_file(relative_path: str):
        """Serve a whitelisted artifact through a stable latest-result URL."""

        try:
            target = repository.resolve_file(latest_persisted_run_id(), relative_path)
        except RunRepositoryError:
            abort(404)
        return send_file(target, as_attachment=request.args.get("download") == "1")

    @app.get("/api/runs/latest/exports/<bundle_kind>.zip")
    def latest_run_export(bundle_kind: str):
        """Build or serve a stable latest-result ZIP without loading it into memory."""

        try:
            run_id = latest_persisted_run_id()
            target = exporter.archive(run_id, bundle_kind)
            bundle = EXPORT_BUNDLES[bundle_kind]
        except (KeyError, RunRepositoryError):
            abort(404)
        return send_file(
            target,
            as_attachment=True,
            download_name=bundle.filename,
            mimetype="application/zip",
            conditional=True,
        )

    @app.get("/api/runs/<run_id>")
    def run_detail(run_id: str):
        try:
            detail = repository.run_detail(run_id)
        except RunRepositoryError:
            abort(404)
        detail["artifacts"] = {
            name: url_for("run_file", run_id=run_id, relative_path=name)
            for name in detail.pop("artifact_names")
        }
        detail["exports"] = export_links(run_id)
        return jsonify(detail)

    @app.get("/api/runs/<run_id>/files/<path:relative_path>")
    def run_file(run_id: str, relative_path: str):
        try:
            target = repository.resolve_file(run_id, relative_path)
        except RunRepositoryError:
            abort(404)
        return send_file(target, as_attachment=request.args.get("download") == "1")

    @app.get("/api/runs/<run_id>/exports/<bundle_kind>.zip")
    def run_export(run_id: str, bundle_kind: str):
        """Build or serve one run-scoped ZIP without buffering the archive in RAM."""

        try:
            target = exporter.archive(run_id, bundle_kind)
            bundle = EXPORT_BUNDLES[bundle_kind]
        except (KeyError, RunRepositoryError):
            abort(404)
        return send_file(
            target,
            as_attachment=True,
            download_name=bundle.filename,
            mimetype="application/zip",
            conditional=True,
        )

    @app.get("/api/runs/<run_id>/defects")
    def run_defects(run_id: str):
        try:
            page = int(request.args.get("page", "1"))
            page_size = int(request.args.get("page_size", "50"))
            candidates = repository.candidate_page(
                run_id,
                status=request.args.get("status", "all"),
                reason=request.args.get("reason", ""),
                defect_id=request.args.get("defect_id", ""),
                page=page,
                page_size=page_size,
            )
            training_labels = training.labels_for_run(run_id)
        except (RunRepositoryError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        rows: list[dict[str, Any]] = []
        for row in candidates.rows:
            item = dict(row)
            preview = item.pop("crop_preview_relative", None)
            crop = item.pop("crop_relative", None)
            item["crop_preview_url"] = (
                url_for("run_file", run_id=run_id, relative_path=preview) if preview else None
            )
            item["crop_url"] = (
                url_for("run_file", run_id=run_id, relative_path=crop) if crop else None
            )
            item["training_label"] = training_labels.get(str(item.get("defect_id", "")))
            rows.append(item)
        return jsonify({
            "run_id": run_id,
            "rows": rows,
            "page": candidates.page,
            "page_size": candidates.page_size,
            "total": candidates.total,
            "total_pages": candidates.total_pages,
            "reason_counts": candidates.reason_counts,
        })

    @app.get("/api/training")
    def training_status():
        """Summarize local labels and the active portable classifier."""

        try:
            return jsonify(public_json(training.status()))
        except TrainingRepositoryError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/training/labels")
    def save_training_label():
        """Persist one explicit expert candidate label."""

        payload = request.get_json(silent=True)
        if not isinstance(payload, Mapping):
            return jsonify({"error": "请求必须包含 JSON 标注对象"}), 400
        try:
            saved = training.save_annotation(
                run_id=str(payload.get("run_id", "")),
                defect_id=str(payload.get("defect_id", "")),
                label=str(payload.get("label", "")),
                split=str(payload.get("split", "calibration")),
                reviewer_id=str(payload.get("reviewer_id", "local_expert")),
                notes=str(payload.get("notes", "")),
            )
            return jsonify({"annotation": saved, "training": training.status()})
        except (TrainingRepositoryError, RunRepositoryError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/training/train")
    def train_candidate_model():
        """Train from consensus calibration labels and activate atomically."""

        payload = request.get_json(silent=True)
        values = payload if isinstance(payload, Mapping) else {}
        try:
            accept_threshold = float(values.get("accept_threshold", 0.75))
            reject_threshold = float(values.get("reject_threshold", 0.25))
            regularization = float(values.get("regularization", 1.0))
            model = training.train(
                accept_threshold=accept_threshold,
                reject_threshold=reject_threshold,
                regularization=regularization,
            )
            return jsonify({
                "training": training.status(),
                "model": public_json(model),
            })
        except (TypeError, ValueError, TrainingRepositoryError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/training/model")
    def download_candidate_model():
        try:
            if training.active_model() is None:
                abort(404)
        except TrainingRepositoryError:
            abort(404)
        return send_file(
            training.model_path,
            as_attachment=True,
            download_name="candidate_classifier.json",
            mimetype="application/json",
            conditional=True,
        )

    @app.get("/api/training/annotations")
    def download_candidate_annotations():
        return send_file(
            training.annotations_path,
            as_attachment=True,
            download_name="candidate_annotations.csv",
            mimetype="text/csv",
            conditional=True,
        )

    return app


def run_local_server(
    workspace: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    max_workers: int = 1,
    max_upload_mb: int = 4096,
) -> None:
    """Run the browser workbench on loopback by default."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("For data safety this local workbench only listens on loopback addresses")
    app = create_app(workspace, max_workers=max_workers, max_upload_mb=max_upload_mb)
    try:
        app.run(host=host, port=int(port), debug=False, use_reloader=False, threaded=True)
    finally:
        manager = app.extensions.get("sic_wafer_job_manager")
        if isinstance(manager, _JobManager):
            manager.shutdown()


__all__ = ["ALLOWED_SUFFIXES", "create_app", "run_local_server"]
