const RUNTIME_BASE = new URL("./runtime/", self.location.href);
const MANIFEST_URL = new URL("manifest.json", RUNTIME_BASE);
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

let runtimePromise = null;
let currentManifest = null;

function postStatus(message, progress = null) {
  self.postMessage({ type: "status", message, progress });
}

function hexDigest(buffer) {
  return crypto.subtle.digest("SHA-256", buffer).then((digest) =>
    [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("")
  );
}

async function fetchVerifiedWheel(entry) {
  const url = new URL(entry.file, RUNTIME_BASE);
  const response = await fetch(url, { cache: "no-cache", credentials: "same-origin" });
  if (!response.ok) {
    throw new Error(`无法下载科研代码包：HTTP ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  const actual = await hexDigest(buffer);
  if (actual !== entry.sha256) {
    throw new Error(`科研代码包校验失败：${entry.file}`);
  }
  return { name: entry.file, bytes: new Uint8Array(buffer) };
}

async function initializeRuntime() {
  postStatus("正在载入浏览器 Python 运行时（首次约需下载数十 MB）…", 0.04);
  const manifestResponse = await fetch(MANIFEST_URL, { cache: "no-cache" });
  if (!manifestResponse.ok) {
    throw new Error(`无法读取运行时清单：HTTP ${manifestResponse.status}`);
  }
  currentManifest = await manifestResponse.json();

  const version = currentManifest.pyodide_version;
  const indexURL = `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;
  const { loadPyodide } = await import(`${indexURL}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL });

  postStatus("正在载入 NumPy、SciPy、OpenCV 与 scikit-image…", 0.12);
  await pyodide.loadPackage(SCIENTIFIC_PACKAGES);

  postStatus("正在验证并安装本项目的 Python 科研代码…", 0.24);
  const [projectWheel, tifffileWheel] = await Promise.all([
    fetchVerifiedWheel(currentManifest.package_wheel),
    fetchVerifiedWheel(currentManifest.tifffile_wheel),
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
  postStatus("科研运行时已就绪。", 0.3);
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
  const match = String(fileName).toLowerCase().match(/\.(png|jpe?g|bmp|tiff?)$/);
  if (!match) {
    throw new Error("仅支持 PNG、JPG、JPEG、BMP、TIF 和 TIFF。当前文件不会被分析。");
  }
  return match[0] === ".jpeg" ? ".jpg" : match[0];
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

async function analyze(message) {
  const pyodide = await getRuntime();
  const extension = safeExtension(message.fileName);
  const limits = currentManifest.limits;
  if (message.fileBuffer.byteLength > limits.max_file_bytes) {
    throw new Error(
      `浏览器分析上限为 ${(limits.max_file_bytes / 1048576).toFixed(0)} MB；` +
      "该文件必须使用本机工作台的 TIFF/BigTIFF 分块路径。"
    );
  }

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
  const inputPath = `${runRoot}/input${extension}`;
  const outputPath = `${runRoot}/output`;
  pyodide.FS.mkdirTree(runRoot);
  pyodide.FS.writeFile(inputPath, new Uint8Array(message.fileBuffer));

  const browserOptions = {
    ...message.options,
    input_file_name: message.fileName,
    max_pixels: limits.max_pixels,
    max_dimension_px: limits.max_dimension_px,
    max_overlay_size_px: limits.max_overlay_size_px,
  };
  pyodide.globals.set("browser_options_json", JSON.stringify(browserOptions));
  pyodide.globals.set("browser_input_path", inputPath);
  pyodide.globals.set("browser_output_path", outputPath);

  postStatus("正在检查图像尺寸、格式和浏览器内存限制…", 0.34);
  let resultJson;
  try {
    resultJson = await pyodide.runPythonAsync(`
import importlib.resources
import json
import shutil
import zipfile
from pathlib import Path

import yaml
from PIL import Image
import tifffile

from sic_wafer_counter.pipeline import analyze_image
from sic_wafer_counter.reporting import generate_html_report, write_summary_files

options = json.loads(browser_options_json)
input_path = Path(browser_input_path)
output_path = Path(browser_output_path)

if input_path.suffix.lower() in {".tif", ".tiff"}:
    with tifffile.TiffFile(input_path) as tif:
        if not tif.series:
            raise ValueError("TIFF 中没有可分析的图像序列。")
        shape = tuple(int(value) for value in tif.series[0].shape)
        if len(shape) < 2:
            raise ValueError(f"TIFF 图像维度无效：{shape}")
        height, width = shape[-2], shape[-1]
else:
    with Image.open(input_path) as image:
        width, height = image.size

pixel_count = int(width) * int(height)
if width > int(options["max_dimension_px"]) or height > int(options["max_dimension_px"]):
    raise RuntimeError(
        f"浏览器模式单边最多 {options['max_dimension_px']} px；当前为 {width}×{height} px。"
        "请使用本机工作台的大图分块路径。"
    )
if pixel_count > int(options["max_pixels"]):
    raise RuntimeError(
        f"浏览器模式最多 {options['max_pixels']:,} 像素；当前为 {pixel_count:,} 像素。"
        "请使用本机工作台的大图分块路径。"
    )

config_text = importlib.resources.files("sic_wafer_counter").joinpath("resources/default.yaml").read_text(encoding="utf-8")
config = yaml.safe_load(config_text)
config["wafer"]["diameter_mm"] = float(options["wafer_diameter_mm"])
config["wafer"]["exclude_edge_mm"] = float(options["exclude_edge_mm"])
config["detection"]["threshold_method"] = str(options["threshold_method"])
config["detection"]["use_watershed"] = bool(options["use_watershed"])
config["output"]["save_candidate_crops"] = False
config["output"]["save_intermediates"] = True
config["output"]["generate_html_report"] = True
config["output"]["generate_defect_comparison"] = True
config["output"]["generate_heatmap"] = True
config["io"]["max_overlay_size"] = int(options["max_overlay_size_px"])

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
summary["browser_runtime"] = {
    "execution": "Pyodide Web Worker",
    "uploaded_to_server": False,
    "candidate_crops_saved": False,
    "browser_limits": {
        "max_pixels": options["max_pixels"],
        "max_dimension_px": options["max_dimension_px"],
    },
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
    "width": width,
    "height": height,
    "output_files": [str(path.relative_to(output_path)) for path in sorted(output_path.rglob("*")) if path.is_file()],
    "zip_path": str(zip_path),
})
`);
  } finally {
    pyodide.globals.delete("browser_options_json");
    pyodide.globals.delete("browser_input_path");
    pyodide.globals.delete("browser_output_path");
  }

  const result = JSON.parse(resultJson);
  postStatus("正在整理报告、图像和可下载审计包…", 0.92);

  const artifacts = {};
  const transferable = [];
  for (const name of result.output_files) {
    const lower = name.toLowerCase();
    const previewable =
      !name.includes("/") &&
      (lower.endsWith(".png") || lower.endsWith(".html") || lower.endsWith(".json") ||
       lower.endsWith(".csv") || lower.endsWith(".yaml") || lower.endsWith(".log"));
    if (!previewable) continue;
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
      bundle: zipBytes.buffer,
      bundleName: `${String(message.fileName).replace(/\.[^.]+$/, "") || "wafer"}_analysis_bundle.zip`,
    },
    transferable
  );
}

self.addEventListener("message", async (event) => {
  if (event.data?.type !== "analyze") return;
  try {
    await analyze(event.data);
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
