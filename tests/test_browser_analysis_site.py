from __future__ import annotations

import hashlib
import json
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sic_wafer_counter.reporting import save_defect_comparison_details
from sic_wafer_counter.visualization import create_xrt_detection_overlay


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class _AnalysisPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.h1_count = 0
        self.form_count = 0
        self.inputs: dict[str, dict[str, str | None]] = {}
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "h1":
            self.h1_count += 1
        elif tag == "form":
            self.form_count += 1
        elif tag in {"input", "select"} and attributes.get("id"):
            self.inputs[str(attributes["id"])] = attributes
        elif tag == "script" and attributes.get("src"):
            self.scripts.append(str(attributes["src"]))


def test_browser_page_has_real_upload_outputs_and_scientific_boundary() -> None:
    html = (DOCS / "analyze.html").read_text(encoding="utf-8")
    parser = _AnalysisPageParser()
    parser.feed(html)

    assert parser.h1_count == 1
    assert parser.form_count == 1
    assert {"image-file", "diameter", "edge-exclusion", "watershed"} <= set(parser.inputs)
    assert "原始 <code>File</code> 以只读方式挂载" in html
    assert "不在主线程整体读取，也不上传到网站服务器" in html
    assert "选择上限 100 MiB" in html
    assert "不会静默压缩、降采样或展示不可靠密度" in html
    assert 'id="analyze-button" type="submit" disabled' in html
    assert "大 CSV 标记为“仅 ZIP”" in html
    assert "自动 XRT 点状候选红框图" in html
    assert "论文风格局部视场" in html
    assert "论文式综合成果图" in html
    assert "不会伪造论文中的黄色 DIC 验证圈" in html
    assert "独立参考未提供" in html
    assert "非 DIC/KOH 验证" in html
    assert "整片密度热图" in html
    assert "未经材料专家标注或独立实验确认" in html
    assert "下载完整结果 ZIP" in html
    assert parser.scripts == ["assets/analyze.js?v=20260808c"]
    assert (DOCS / "assets" / "demo" / "synthetic_clean.png").is_file()


def test_worker_calls_packaged_pipeline_and_keeps_browser_limits_explicit() -> None:
    worker = (DOCS / "assets" / "analysis-worker.mjs").read_text(encoding="utf-8")
    assert "from sic_wafer_counter.pipeline import analyze_image" in worker
    assert "result = analyze_image(" in worker
    assert 'config["output"]["generate_defect_comparison"] = True' in worker
    assert 'config["output"]["generate_xrt_red_box_overlay"] = True' in worker
    assert 'config["output"]["generate_xrt_detection_detail_montage"] = True' in worker
    assert 'config["output"]["generate_heatmap"] = True' in worker
    assert 'config["output"]["generate_paper_aligned_figure"] = True' in worker
    assert 'config["output"]["save_candidate_crops"] = False' in worker
    assert "analysis_quantized_to_uint8" not in worker or '"candidate_crops_saved": False' in worker
    assert "jsdelivr.net/pyodide/v${version}/full/" in worker
    assert "fetchVerifiedWheel" in worker and 'crypto.subtle.digest("SHA-256"' in worker
    assert "fetch(" not in worker.replace("fetch(MANIFEST_URL", "").replace("fetch(url", "")


def test_browser_transfers_file_directly_and_worker_mounts_it_read_only() -> None:
    main = (DOCS / "assets" / "analyze.js").read_text(encoding="utf-8")
    worker = (DOCS / "assets" / "analysis-worker.mjs").read_text(encoding="utf-8")

    assert "worker.postMessage({ type: \"analyze\", file: runFile, options })" in main
    assert ".arrayBuffer(" not in main
    assert "fileBuffer" not in main
    assert "WORKERFS" in worker
    assert "FS.mount(workerFs, { files: [file] }, INPUT_MOUNT)" in worker
    assert 'input_transport: "WORKERFS_read_only_File"' in worker
    assert "message.fileBuffer" not in worker
    assert "new Uint8Array(message.fileBuffer)" not in worker
    assert "FS.writeFile(inputPath" not in worker


def test_worker_tiff_preflight_uses_axes_and_forces_bounded_tiling() -> None:
    worker = (DOCS / "assets" / "analysis-worker.mjs").read_text(encoding="utf-8")

    assert "series.axes" in worker
    assert 'shape[axes.index("X")]' in worker
    assert 'shape[axes.index("Y")]' in worker
    assert "shape[-2]" not in worker and "shape[-1]" not in worker
    assert 'config["io"]["tile_size"] = int(options["tier_limits"]["tile_size_px"])' in worker
    assert 'config["io"]["large_image_threshold_pixels"] = 1' in worker
    assert 'config["io"]["prefer_bounded_tiff_regions"] = True' in worker
    assert 'config["io"]["allow_tiff_memmap"] = False' in worker
    assert 'config["io"]["allow_tiff_full_decode"] = False' in worker
    assert "source_region_read_bounded" in worker
    assert "decoded_full_source_resident" in worker
    assert "TIFF_BOUNDED_BACKEND_REQUIRED" in worker


def test_worker_records_full_resolution_provenance_and_bounds_output_transfer() -> None:
    worker = (DOCS / "assets" / "analysis-worker.mjs").read_text(encoding="utf-8")
    main = (DOCS / "assets" / "analyze.js").read_text(encoding="utf-8")

    assert 'summary["input_transport"] = options["input_transport"]' in worker
    assert 'summary["source_resolution_px"] = list(options["source_resolution_px"])' in worker
    assert 'summary["analysis_downsample_factor"] = 1' in worker
    assert 'summary["scientific_downsampling_applied"] = False' in worker
    assert "max_inline_csv_bytes" in worker
    assert "max_bundle_bytes" in worker
    assert "bundleOnlyArtifacts.push(name)" in worker
    assert "OUTPUT_BUNDLE_LIMIT_EXCEEDED" in worker
    assert "bundleOnlyArtifacts" in main
    assert "仅 ZIP" in main
    assert 'results.hidden = true' in main
    assert 'payload.error || payload.message' in main


def test_browser_runtime_manifest_matches_published_wheels() -> None:
    runtime = DOCS / "assets" / "runtime"
    manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pyodide_version"] == "314.0.4"
    assert manifest["runtime_schema_version"] == 2
    assert manifest["limits"] == {
        "selection": {"max_file_bytes": 104857600},
        "raster_full_array": {
            "max_file_bytes": 25165824,
            "max_pixels": 6000000,
            "max_dimension_px": 6000,
        },
        "tiff_bounded": {
            "max_file_bytes": 104857600,
            "max_pixels": 120000000,
            "max_dimension_px": 16000,
            "tile_size_px": 1024,
            "tile_overlap_px": 128,
            "require_bounded_random_access": True,
        },
        "output_transfer": {
            "max_inline_artifact_bytes": 8388608,
            "max_inline_csv_bytes": 2097152,
            "max_bundle_bytes": 134217728,
        },
        "max_overlay_size_px": 2000,
    }
    assert "bounded random-access source backend" in manifest["scientific_runtime"]
    assert "memmap" not in manifest["scientific_runtime"].lower()
    for key in ("package_wheel", "tifffile_wheel"):
        entry = manifest[key]
        wheel = runtime / entry["file"]
        assert wheel.is_file()
        assert hashlib.sha256(wheel.read_bytes()).hexdigest() == entry["sha256"]

    project_wheel = runtime / manifest["package_wheel"]["file"]
    assert project_wheel.name == "sic_wafer_point_counter-0.2.3-py3-none-any.whl"
    with zipfile.ZipFile(project_wheel) as archive:
        packaged_image_io = archive.read("sic_wafer_counter/image_io.py").decode("utf-8")
        packaged_reporting = archive.read("sic_wafer_counter/reporting.py").decode("utf-8")
        packaged_paper_alignment = archive.read(
            "sic_wafer_counter/paper_alignment.py"
        ).decode("utf-8")
    assert "prefer_bounded_tiff_regions" in packaged_image_io
    assert "source_region_read_bounded" in packaged_image_io
    assert "decoded_full_source_resident" in packaged_image_io
    assert "allow_tiff_full_decode" in packaged_image_io
    assert "paper_aligned_result_figure.png" in packaged_reporting
    assert "source_image_sha256" in packaged_paper_alignment
    assert "reference_image_sha256" in packaged_paper_alignment


def test_detail_comparison_contains_accepted_and_rejected_markers(tmp_path: Path) -> None:
    image = np.full((120, 180), 0.8, dtype=np.float32)
    cv2.circle(image, (45, 55), 5, 0.1, -1)
    cv2.circle(image, (125, 65), 6, 0.15, -1)
    defects = pd.DataFrame(
        [
            {"defect_id": 1, "centroid_x_px": 45, "centroid_y_px": 55, "accepted": True,
             "equivalent_diameter_px": 10, "equivalent_diameter_mm": 0.2, "contrast": 0.7,
             "rejection_reason": ""},
            {"defect_id": 2, "centroid_x_px": 125, "centroid_y_px": 65, "accepted": False,
             "equivalent_diameter_px": 12, "equivalent_diameter_mm": 0.24, "contrast": 0.65,
             "rejection_reason": "too_elongated"},
        ]
    )
    output = save_defect_comparison_details(image, defects, tmp_path / "details.png", max_candidates=2)
    rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert rendered is not None and rendered.shape[0] > 200 and rendered.shape[1] > 700
    channels = rendered.astype(np.int16)
    assert np.count_nonzero(channels[:, :, 1] > channels[:, :, 2] + 60) > 20
    assert np.count_nonzero(channels[:, :, 2] > channels[:, :, 1] + 60) > 20



def test_xrt_overlay_uses_red_boxes_and_never_fabricates_yellow_references() -> None:
    image = np.full((160, 220), 0.75, dtype=np.float32)
    defects = pd.DataFrame(
        [
            {
                "defect_id": 7,
                "centroid_x_px": 80,
                "centroid_y_px": 70,
                "bounding_box": "[72,62,89,79]",
                "equivalent_diameter_px": 14,
                "accepted": True,
            },
            {
                "defect_id": 8,
                "centroid_x_px": 145,
                "centroid_y_px": 90,
                "bounding_box": "[136,82,155,99]",
                "equivalent_diameter_px": 15,
                "accepted": False,
            },
        ]
    )
    overlay = create_xrt_detection_overlay(
        image, defects, mm_per_pixel=0.2, scale_bar_mm=10.0
    )
    b, g, r = (overlay[:, :, index].astype(np.int16) for index in range(3))
    assert np.count_nonzero((r > 180) & (r > g + 80) & (r > b + 80)) > 20
    assert np.count_nonzero((r > 180) & (g > 180) & (b < 100)) == 0
    # The rejected candidate is not drawn in the accepted XRT red-box view.
    rejected_patch = overlay[80:102, 134:158]
    assert np.count_nonzero(
        (rejected_patch[:, :, 2] > rejected_patch[:, :, 1] + 80)
    ) == 0

    references = pd.DataFrame([{"x_px": 145, "y_px": 90}])
    with_reference = create_xrt_detection_overlay(
        image,
        defects,
        mm_per_pixel=0.2,
        scale_bar_mm=None,
        independent_reference_points=references,
    )
    b2, g2, r2 = (with_reference[:, :, index].astype(np.int16) for index in range(3))
    assert np.count_nonzero((r2 > 180) & (g2 > 180) & (b2 < 100)) > 10
