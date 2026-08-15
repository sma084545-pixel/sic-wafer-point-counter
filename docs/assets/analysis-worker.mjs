const RUNTIME_BASE = new URL("./runtime/", self.location.href);
const MANIFEST_URL = new URL("manifest.json", RUNTIME_BASE);
const INPUT_MOUNT = "/browser-input";
const SCIENTIFIC_PACKAGES = [
  "numpy",
  "scipy",
  "opencv-python",
  "scikit-image",
  "matplotlib",
  "Pillow",
  "pyyaml",
  "pandas",
  "jinja2",
  "micropip",
];

let manifestPromise = null;
let runtimePromise = null;
let currentManifest = null;

class SafetyError extends Error {
  constructor(code, message, recovery, details = {}) {
    super(message);
    this.name = "SafetyError";
    this.code = code;
    this.recovery = recovery;
    this.details = details;
  }
}

function postStatus(message, phase, progress = null) {
  self.postMessage({ type: "status", message, phase, progress });
}

function safetyPayload(error) {
  if (error instanceof SafetyError) {
    return {
      code: error.code,
      message: error.message,
      recovery: error.recovery,
      details: error.details,
    };
  }
  return {
    code: "ANALYSIS_FAILED",
    message: error instanceof Error ? error.message : String(error),
    recovery: "请检查输入与参数后重试；若问题持续，请使用本机工作台并保留日志。",
    details: {},
  };
}

function humanMiB(bytes) {
  return `${(Number(bytes) / 1048576).toFixed(1)} MiB`;
}

function hexDigest(buffer) {
  return crypto.subtle.digest("SHA-256", buffer).then((digest) =>
    [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("")
  );
}

async function getManifest() {
  if (!manifestPromise) {
    manifestPromise = fetch(MANIFEST_URL, { cache: "no-cache", credentials: "same-origin" })
      .then((response) => {
        if (!response.ok) throw new Error(`无法读取运行时清单：HTTP ${response.status}`);
        return response.json();
      })
      .then((manifest) => {
        currentManifest = manifest;
        return manifest;
      })
      .catch((error) => {
        manifestPromise = null;
        throw error;
      });
  }
  return manifestPromise;
}

async function fetchVerifiedWheel(entry) {
  const url = new URL(entry.file, RUNTIME_BASE);
  const response = await fetch(url, { cache: "no-cache", credentials: "same-origin" });
  if (!response.ok) throw new Error(`无法下载科研代码包：HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  const actual = await hexDigest(buffer);
  if (actual !== entry.sha256) throw new Error(`科研代码包校验失败：${entry.file}`);
  return { name: entry.file, bytes: new Uint8Array(buffer) };
}

async function initializeRuntime() {
  const manifest = await getManifest();
  postStatus("正在载入浏览器 Python 运行时（首次约需下载数十 MB）…", "runtime");
  const version = manifest.pyodide_version;
  const indexURL = `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;
  const { loadPyodide } = await import(`${indexURL}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL });

  postStatus("正在载入 NumPy、SciPy、OpenCV 与 scikit-image…", "packages");
  await pyodide.loadPackage(SCIENTIFIC_PACKAGES);

  postStatus("正在验证并安装本项目的 Python 科研代码…", "package-verification");
  const [projectWheel, tifffileWheel] = await Promise.all([
    fetchVerifiedWheel(manifest.package_wheel),
    fetchVerifiedWheel(manifest.tifffile_wheel),
  ]);
  pyodide.FS.mkdirTree("/runtime");
  for (const wheel of [projectWheel, tifffileWheel]) {
    pyodide.FS.writeFile(`/runtime/${wheel.name}`, wheel.bytes);
  }
  pyodide.globals.set("project_wheel_name", projectWheel.name);
  pyodide.globals.set("tifffile_wheel_name", tifffileWheel.name);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(f"emfs:/runtime/{project_wheel_name}", deps=False)
await micropip.install(f"emfs:/runtime/{tifffile_wheel_name}", deps=False)
`);
  pyodide.globals.delete("project_wheel_name");
  pyodide.globals.delete("tifffile_wheel_name");
  postStatus("科研运行时已就绪。", "runtime-ready");
  return pyodide;
}

function getRuntime() {
  if (!runtimePromise) {
    runtimePromise = initializeRuntime().catch((error) => {
      runtimePromise = null;
      throw error;
    });
  }
  return runtimePromise;
}

function safeExtension(fileName) {
  const match = String(fileName).toLowerCase().match(/\.(png|jpe?g|bmp|tiff?|btf|bigtif|bigtiff)$/);
  if (!match) {
    throw new SafetyError(
      "UNSUPPORTED_FILE_TYPE",
      "仅支持 PNG、JPG、JPEG、BMP、TIF、TIFF 和 BigTIFF。当前文件未被读取。",
      "请转换为受支持格式，或使用本机工作台检查原始文件。"
    );
  }
  return match[0] === ".jpeg" ? ".jpg" : match[0];
}

function isTiffExtension(extension) {
  return [".tif", ".tiff", ".btf", ".bigtif", ".bigtiff"].includes(extension);
}

function assertFileLike(file) {
  if (
    !file || typeof file.name !== "string" || !Number.isFinite(Number(file.size)) ||
    typeof file.slice !== "function"
  ) {
    throw new SafetyError(
      "FILE_TRANSFER_UNAVAILABLE",
      "浏览器未能把 File 对象安全交给分析 Worker；未读取图像。",
      "请使用最新版 Chromium、Chrome、Edge、Firefox 或 Safari 后重试。"
    );
  }
  if (Number(file.size) <= 0) {
    throw new SafetyError("EMPTY_FILE", "输入文件为空；未生成密度。", "请选择非空图像文件。");
  }
}

function selectTier(extension, limits) {
  return isTiffExtension(extension)
    ? { id: "tiff_bounded", value: limits.tiff_bounded }
    : { id: "raster_full_array", value: limits.raster_full_array };
}

function enforceFileLimits(file, manifest, tier) {
  const selectionLimit = Number(manifest.limits.selection.max_file_bytes);
  const tierLimit = Number(tier.value.max_file_bytes);
  const effectiveLimit = Math.min(selectionLimit, tierLimit);
  if (Number(file.size) > effectiveLimit) {
    throw new SafetyError(
      "FILE_SIZE_LIMIT_EXCEEDED",
      `${file.name} 为 ${humanMiB(file.size)}，超过 ${tier.id} 的 ${humanMiB(effectiveLimit)} 上限。`,
      "图像未被压缩或降采样，也未生成密度；请改用本机工作台。",
      { file_size_bytes: Number(file.size), limit_bytes: effectiveLimit, tier: tier.id }
    );
  }
}

function mimeFor(name) {
  if (name.endsWith(".png")) return "image/png";
  if (name.endsWith(".html")) return "text/html;charset=utf-8";
  if (name.endsWith(".json")) return "application/json;charset=utf-8";
  if (name.endsWith(".csv")) return "text/csv;charset=utf-8";
  if (name.endsWith(".yaml") || name.endsWith(".yml")) return "text/yaml;charset=utf-8";
  if (name.endsWith(".log") || name.endsWith(".txt")) return "text/plain;charset=utf-8";
  return "application/octet-stream";
}

function resetInputMount(pyodide) {
  try { pyodide.FS.unmount(INPUT_MOUNT); } catch (_) { /* not mounted */ }
  try { pyodide.FS.rmdir(INPUT_MOUNT); } catch (_) { /* absent or non-empty */ }
}

function mountReadOnlyFile(pyodide, file) {
  const workerFs = pyodide.FS?.filesystems?.WORKERFS;
  if (!workerFs || typeof pyodide.FS.mount !== "function") {
    throw new SafetyError(
      "WORKERFS_UNAVAILABLE",
      "当前浏览器运行时不提供只读 File 挂载能力；为避免大图整份复制，分析已停止。",
      "请刷新页面升级运行时，或使用本机工作台。"
    );
  }
  resetInputMount(pyodide);
  pyodide.FS.mkdirTree(INPUT_MOUNT);
  pyodide.FS.mount(workerFs, { files: [file] }, INPUT_MOUNT);
  return `${INPUT_MOUNT}/${file.name}`;
}

async function readSourceMetadata(pyodide, inputPath, isTiff) {
  pyodide.globals.set("browser_preflight_input_path", inputPath);
  pyodide.globals.set("browser_preflight_is_tiff", isTiff);
  try {
    const raw = await pyodide.runPythonAsync(`
import json
from pathlib import Path

from PIL import Image
import tifffile

input_path = Path(browser_preflight_input_path)
is_tiff = bool(browser_preflight_is_tiff)

if is_tiff:
    with tifffile.TiffFile(input_path) as tif:
        if not tif.series:
            raise ValueError("TIFF 中没有可分析的图像序列。")
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        axes = str(series.axes or "")
        if len(shape) != len(axes) or axes.count("X") != 1 or axes.count("Y") != 1:
            raise ValueError(f"TIFF axes 无法唯一定位 X/Y 空间维：axes={axes!r}, shape={shape}")
        width = int(shape[axes.index("X")])
        height = int(shape[axes.index("Y")])
        unsupported_planes = [
            {"axis": axis, "size": int(size)}
            for axis, size in zip(axes, shape)
            if axis not in {"X", "Y", "S", "C"} and int(size) > 1
        ]
        channel_planes = [
            {"axis": axis, "size": int(size)}
            for axis, size in zip(axes, shape)
            if axis in {"S", "C"} and int(size) != 1
        ]
        metadata = {
            "width": width,
            "height": height,
            "shape": list(shape),
            "axes": axes,
            "dtype": str(series.dtype),
            "is_tiff": True,
            "is_bigtiff": bool(tif.is_bigtiff),
            "unsupported_planes": unsupported_planes + channel_planes,
        }
else:
    with Image.open(input_path) as image:
        width, height = (int(value) for value in image.size)
        metadata = {
            "width": width,
            "height": height,
            "shape": [height, width],
            "axes": "YX",
            "dtype": str(getattr(image, "mode", "unknown")),
            "is_tiff": False,
            "is_bigtiff": False,
            "unsupported_planes": [],
        }

json.dumps(metadata)
`);
    return JSON.parse(raw);
  } catch (error) {
    throw new SafetyError(
      "IMAGE_METADATA_INVALID",
      `无法安全读取图像尺寸或 TIFF axes：${error instanceof Error ? error.message : String(error)}`,
      "未运行点检测且未生成密度；请检查文件是否损坏、是否为多页/多时间点 TIFF。"
    );
  } finally {
    pyodide.globals.delete("browser_preflight_input_path");
    pyodide.globals.delete("browser_preflight_is_tiff");
  }
}

function enforceSourceLimits(metadata, tier) {
  const width = Number(metadata.width);
  const height = Number(metadata.height);
  const pixels = width * height;
  if (metadata.unsupported_planes?.length) {
    throw new SafetyError(
      "TIFF_MULTIPLANE_UNSUPPORTED",
      `TIFF axes=${metadata.axes}、shape=${JSON.stringify(metadata.shape)} 包含未支持的多平面维度。`,
      "浏览器不会静默选取单页；请在本机明确选择二维序列后再分析。",
      { axes: metadata.axes, shape: metadata.shape, unsupported_planes: metadata.unsupported_planes }
    );
  }
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0) {
    throw new SafetyError("INVALID_SOURCE_RESOLUTION", "图像分辨率无效；未生成密度。", "请检查图像文件。", metadata);
  }
  if (width > Number(tier.value.max_dimension_px) || height > Number(tier.value.max_dimension_px)) {
    throw new SafetyError(
      "SOURCE_DIMENSION_LIMIT_EXCEEDED",
      `源图为 ${width}×${height} px，超过 ${tier.id} 单边 ${Number(tier.value.max_dimension_px).toLocaleString()} px 上限。`,
      "图像未被缩放或降采样，也未生成密度；请使用本机工作台。",
      { width, height, pixel_count: pixels, tier: tier.id }
    );
  }
  if (!Number.isSafeInteger(pixels) || pixels > Number(tier.value.max_pixels)) {
    throw new SafetyError(
      "SOURCE_PIXEL_LIMIT_EXCEEDED",
      `源图为 ${pixels.toLocaleString()} 像素，超过 ${tier.id} 的 ${Number(tier.value.max_pixels).toLocaleString()} 像素上限。`,
      "图像未被缩放或降采样，也未生成密度；请使用本机工作台。",
      { width, height, pixel_count: pixels, tier: tier.id }
    );
  }
  return { width, height, pixelCount: pixels };
}

function assertBoundedTiffSummary(summary) {
  const metadata = summary?.image_metadata || {};
  const backend = String(metadata.backend || metadata.loader || "");
  if (
    metadata.source_region_read_bounded !== true ||
    metadata.decoded_full_source_resident === true ||
    /full fallback|asarray\(full/i.test(backend)
  ) {
    throw new SafetyError(
      "TIFF_BOUNDED_BACKEND_REQUIRED",
      `TIFF 未确认使用有界随机访问后端（backend=${backend || "unknown"}）；结果已拒绝。`,
      "未展示密度。请使用本机工作台，或改用支持分段读取的 TIFF 编码。",
      {
        backend,
        source_region_read_bounded: metadata.source_region_read_bounded,
        decoded_full_source_resident: metadata.decoded_full_source_resident,
      }
    );
  }
}

async function analyze(message) {
  const file = message.file;
  assertFileLike(file);
  const extension = safeExtension(file.name);
  const manifest = await getManifest();
  const limits = manifest.limits;
  const tier = selectTier(extension, limits);
  enforceFileLimits(file, manifest, tier);

  postStatus("文件大小已通过分层预检；正在启动隔离科研运行时…", "file-preflight");
  const pyodide = await getRuntime();
  const inputPath = mountReadOnlyFile(pyodide, file);

  await pyodide.runPythonAsync(`
import shutil
from pathlib import Path
work_root = Path("/work")
if work_root.exists():
    for previous_run in work_root.glob("run-*"):
        shutil.rmtree(previous_run, ignore_errors=True)
`);

  const runId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const runRoot = `/work/run-${runId}`;
  const outputPath = `${runRoot}/output`;
  pyodide.FS.mkdirTree(runRoot);

  postStatus("正在按 TIFF axes 核对源分辨率与浏览器安全上限…", "source-preflight");
  const sourceMetadata = await readSourceMetadata(pyodide, inputPath, isTiffExtension(extension));
  const sourceSize = enforceSourceLimits(sourceMetadata, tier);

  const browserOptions = {
    ...message.options,
    input_file_name: file.name,
    input_transport: "WORKERFS_read_only_File",
    source_metadata: sourceMetadata,
    source_resolution_px: [sourceSize.width, sourceSize.height],
    analysis_downsample_factor: 1,
    scientific_downsampling_applied: false,
    limit_tier: tier.id,
    tier_limits: tier.value,
    output_transfer_limits: limits.output_transfer,
    max_overlay_size_px: limits.max_overlay_size_px,
  };
  pyodide.globals.set("browser_options_json", JSON.stringify(browserOptions));
  pyodide.globals.set("browser_input_path", inputPath);
  pyodide.globals.set("browser_output_path", outputPath);

  postStatus("正在全分辨率运行晶圆识别、候选分割和面积统计…", "analysis");
  let resultJson;
  try {
    resultJson = await pyodide.runPythonAsync(`
import importlib.resources
import json
import zipfile
from pathlib import Path

import yaml

from sic_wafer_counter.pipeline import analyze_image
from sic_wafer_counter.calibration_profiles import apply_calibration_profile
from sic_wafer_counter.reporting import generate_html_report, write_summary_files

options = json.loads(browser_options_json)
input_path = Path(browser_input_path)
output_path = Path(browser_output_path)

config_text = importlib.resources.files("sic_wafer_counter").joinpath("resources/default.yaml").read_text(encoding="utf-8")
config = yaml.safe_load(config_text)
config["wafer"]["diameter_mm"] = float(options["wafer_diameter_mm"])
config["wafer"]["exclude_edge_mm"] = float(options["exclude_edge_mm"])
config["detection"]["threshold_method"] = str(options["threshold_method"])
config["detection"]["use_watershed"] = bool(options["use_watershed"])
config = apply_calibration_profile(config, options.get("analysis_profile"))
config["output"]["save_candidate_crops"] = False
config["output"]["save_intermediates"] = True
config["output"]["generate_html_report"] = True
config["output"]["generate_defect_comparison"] = True
config["output"]["generate_xrt_red_box_overlay"] = True
config["output"]["generate_xrt_detection_detail_montage"] = True
config["output"]["generate_heatmap"] = True
config["output"]["generate_paper_aligned_figure"] = True
config["io"]["max_overlay_size"] = int(options["max_overlay_size_px"])
classifier_model = options.get("classifier_model")
if classifier_model is not None:
    config["classifier"]["enabled"] = True
    config["classifier"]["model_path"] = None
    config["classifier"]["model"] = classifier_model
pixel_classifier_model = options.get("pixel_classifier_model")
if pixel_classifier_model is not None:
    config["pixel_classifier"]["enabled"] = True
    config["pixel_classifier"]["model_path"] = None
    config["pixel_classifier"]["model"] = pixel_classifier_model

is_tiff = bool(options["source_metadata"]["is_tiff"])
if is_tiff:
    config["io"]["tile_size"] = int(options["tier_limits"]["tile_size_px"])
    config["io"]["tile_overlap"] = int(options["tier_limits"]["tile_overlap_px"])
    config["io"]["large_image_threshold_pixels"] = 1
    config["io"]["prefer_bounded_tiff_regions"] = True
    config["io"]["allow_tiff_memmap"] = False
    config["io"]["allow_tiff_full_decode"] = False

manual = options.get("manual_geometry")
kwargs = {}
if manual:
    kwargs = {
        "center_x": float(manual["center_x"]),
        "center_y": float(manual["center_y"]),
        "radius_px": float(manual["radius_px"]),
    }

output_path.mkdir(parents=True, exist_ok=True)
result = analyze_image(input_path, output_path, config, **kwargs)
summary = dict(result.summary)
summary["input_file_name"] = options["input_file_name"]
summary["input_path"] = options["input_file_name"]
summary["input_transport"] = options["input_transport"]
summary["source_resolution_px"] = list(options["source_resolution_px"])
summary["analysis_resolution_px"] = list(options["source_resolution_px"])
summary["analysis_downsample_factor"] = 1
summary["scientific_downsampling_applied"] = False
summary["browser_runtime"] = {
    "execution": "Pyodide Web Worker",
    "input_transport": options["input_transport"],
    "uploaded_to_server": False,
    "candidate_crops_saved": False,
    "candidate_classifier_supplied": classifier_model is not None,
    "candidate_classifier_model_sha256": summary.get("candidate_classifier", {}).get("model_sha256"),
    "pixel_classifier_supplied": pixel_classifier_model is not None,
    "pixel_classifier_model_sha256": summary.get("pixel_classifier", {}).get("model_sha256"),
    "limit_tier": options["limit_tier"],
    "source_axes": options["source_metadata"]["axes"],
    "source_shape": options["source_metadata"]["shape"],
    "source_resolution_px": list(options["source_resolution_px"]),
    "analysis_downsample_factor": 1,
    "scientific_downsampling_applied": False,
    "bounded_tiff_required": is_tiff,
    "browser_limits": options["tier_limits"],
}
write_summary_files(summary, output_path)
generate_html_report(summary, output_path)

zip_path = output_path.parent / "sic_wafer_analysis_bundle.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(output_path.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(output_path))

json.dumps({
    "summary": summary,
    "width": int(options["source_resolution_px"][0]),
    "height": int(options["source_resolution_px"][1]),
    "output_files": [
        {"name": str(path.relative_to(output_path)), "size": int(path.stat().st_size)}
        for path in sorted(output_path.rglob("*")) if path.is_file()
    ],
    "zip_path": str(zip_path),
    "zip_size": int(zip_path.stat().st_size),
})
`);
  } finally {
    pyodide.globals.delete("browser_options_json");
    pyodide.globals.delete("browser_input_path");
    pyodide.globals.delete("browser_output_path");
  }

  const result = JSON.parse(resultJson);
  if (isTiffExtension(extension)) assertBoundedTiffSummary(result.summary);

  const transferLimits = limits.output_transfer;
  if (Number(result.zip_size) > Number(transferLimits.max_bundle_bytes)) {
    throw new SafetyError(
      "OUTPUT_BUNDLE_LIMIT_EXCEEDED",
      `完整结果 ZIP 为 ${humanMiB(result.zip_size)}，超过网页 ${humanMiB(transferLimits.max_bundle_bytes)} 传输上限。`,
      "为避免浏览器内存失控，本次不展示密度或部分结果；请使用本机工作台生成报告。",
      { bundle_size_bytes: Number(result.zip_size), limit_bytes: Number(transferLimits.max_bundle_bytes) }
    );
  }

  postStatus("正在整理报告预览；大 CSV 只保留在完整 ZIP 中…", "output-transfer");
  const artifacts = {};
  const bundleOnlyArtifacts = [];
  const transferable = [];
  for (const entry of result.output_files) {
    const name = entry.name;
    const size = Number(entry.size);
    const lower = name.toLowerCase();
    const previewable =
      !name.includes("/") &&
      (lower.endsWith(".png") || lower.endsWith(".html") || lower.endsWith(".json") ||
       lower.endsWith(".csv") || lower.endsWith(".yaml") || lower.endsWith(".log"));
    if (!previewable) continue;
    const isCsv = lower.endsWith(".csv");
    const inlineLimit = isCsv
      ? Number(transferLimits.max_inline_csv_bytes)
      : Number(transferLimits.max_inline_artifact_bytes);
    if (size > inlineLimit) {
      bundleOnlyArtifacts.push(name);
      continue;
    }
    const bytes = pyodide.FS.readFile(`${outputPath}/${name}`).slice();
    artifacts[name] = { buffer: bytes.buffer, mime: mimeFor(lower) };
    transferable.push(bytes.buffer);
  }
  const zipBytes = pyodide.FS.readFile(result.zip_path).slice();
  transferable.push(zipBytes.buffer);

  self.postMessage(
    {
      type: "complete",
      summary: result.summary,
      width: result.width,
      height: result.height,
      artifacts,
      bundleOnlyArtifacts,
      bundle: zipBytes.buffer,
      bundleName: `${String(file.name).replace(/\.[^.]+$/, "") || "wafer"}_analysis_bundle.zip`,
    },
    transferable
  );
  resetInputMount(pyodide);
}

self.addEventListener("message", async (event) => {
  if (event.data?.type !== "analyze") return;
  try {
    await analyze(event.data);
  } catch (error) {
    self.postMessage({ type: "error", error: safetyPayload(error) });
  }
});
