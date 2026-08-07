from __future__ import annotations

import hashlib
import json
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
    assert "文件只写入当前标签页的临时内存，不上传到网站服务器" in html
    assert "自动 XRT 点状候选红框图" in html
    assert "论文风格局部视场" in html
    assert "独立参考未提供" in html
    assert "非 DIC/KOH 验证" in html
    assert "整片密度热图" in html
    assert "未经材料专家标注或独立实验确认" in html
    assert "下载完整结果 ZIP" in html
    assert parser.scripts == ["assets/analyze.js?v=20260808a"]
    assert (DOCS / "assets" / "demo" / "synthetic_clean.png").is_file()


def test_worker_calls_packaged_pipeline_and_keeps_browser_limits_explicit() -> None:
    worker = (DOCS / "assets" / "analysis-worker.mjs").read_text(encoding="utf-8")
    assert "from sic_wafer_counter.pipeline import analyze_image" in worker
    assert "result = analyze_image(" in worker
    assert 'config["output"]["generate_defect_comparison"] = True' in worker
    assert 'config["output"]["generate_xrt_red_box_overlay"] = True' in worker
    assert 'config["output"]["generate_xrt_detection_detail_montage"] = True' in worker
    assert 'config["output"]["generate_heatmap"] = True' in worker
    assert 'config["output"]["save_candidate_crops"] = False' in worker
    assert "analysis_quantized_to_uint8" not in worker or '"candidate_crops_saved": False' in worker
    assert "jsdelivr.net/pyodide/v${version}/full/" in worker
    assert "fetchVerifiedWheel" in worker and 'crypto.subtle.digest("SHA-256"' in worker
    assert "fetch(" not in worker.replace("fetch(MANIFEST_URL", "").replace("fetch(url", "")


def test_browser_runtime_manifest_matches_published_wheels() -> None:
    runtime = DOCS / "assets" / "runtime"
    manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pyodide_version"] == "314.0.4"
    assert manifest["limits"] == {
        "max_file_bytes": 25165824,
        "max_pixels": 6000000,
        "max_dimension_px": 6000,
        "max_overlay_size_px": 2000,
    }
    for key in ("package_wheel", "tifffile_wheel"):
        entry = manifest[key]
        wheel = runtime / entry["file"]
        assert wheel.is_file()
        assert hashlib.sha256(wheel.read_bytes()).hexdigest() == entry["sha256"]


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
