const form = document.querySelector("#analysis-form");
const fileInput = document.querySelector("#image-file");
const fileLine = document.querySelector("#file-line");
const dropZone = document.querySelector("#drop-zone");
const demoButton = document.querySelector("#demo-button");
const analyzeButton = document.querySelector("#analyze-button");
const cancelButton = document.querySelector("#cancel-button");
const statusBox = document.querySelector("#status-box");
const statusTitle = document.querySelector("#status-title");
const statusMessage = document.querySelector("#status-message");
const progress = document.querySelector("#analysis-progress");
const results = document.querySelector("#results");
const bundleDownload = document.querySelector("#bundle-download");

const imageLabels = {
  "xrt_detection_detail_montage.png": "论文风格局部视场（自动红框；独立参考未提供）",
  "overlay_xrt_red_boxes.png": "自动 XRT 点状候选红框图（无独立验证黄圈）",
  "defect_comparison_details.png": "原图与自动判定复核（非 DIC/KOH 验证）",
  "density_heatmap.png": "整片缺陷密度热图",
  "overlay_accepted.png": "接受目标编号叠加",
  "overlay_all_candidates.png": "全部候选叠加",
  "valid_analysis_mask.png": "最终有效区域",
  "candidate_mask.png": "候选二值掩膜",
  "preprocessed_preview.png": "暗目标响应",
  "radial_density.png": "径向面积归一化密度",
  "angular_density.png": "角向面积归一化密度",
  "wafer_position_scatter.png": "晶圆目标位置",
};

let selectedFile = null;
let worker = null;
let objectUrls = [];
let manifest = null;

function setStatus(kind, title, message, value = null) {
  statusBox.className = `status-box ${kind ? `is-${kind}` : ""}`;
  statusTitle.textContent = title;
  statusMessage.textContent = message;
  if (value === null) {
    progress.removeAttribute("value");
  } else {
    progress.value = Math.max(0, Math.min(1, Number(value)));
  }
}

function humanBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1048576) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / 1048576).toFixed(1)} MiB`;
}

function setSelectedFile(file) {
  selectedFile = file;
  fileLine.textContent = file ? `${file.name} · ${humanBytes(file.size)}` : "尚未选择文件";
  if (file) setStatus("", "图像已就绪", "确认物理标定与检测参数后开始。", 0);
}

function cleanupObjectUrls() {
  for (const url of objectUrls) URL.revokeObjectURL(url);
  objectUrls = [];
}

function makeUrl(buffer, mime) {
  const url = URL.createObjectURL(new Blob([buffer], { type: mime }));
  objectUrls.push(url);
  return url;
}

function validateFile(file) {
  if (!file) throw new Error("请先选择晶圆图像。 ");
  if (!/\.(png|jpe?g|bmp|tiff?)$/i.test(file.name)) {
    throw new Error("仅支持 PNG、JPG、JPEG、BMP、TIF 和 TIFF。 ");
  }
  if (manifest && file.size > manifest.limits.max_file_bytes) {
    throw new Error(`文件为 ${humanBytes(file.size)}，超过浏览器模式 ${humanBytes(manifest.limits.max_file_bytes)} 上限；请使用本机工作台。`);
  }
}

function readOptions() {
  const values = ["center-x", "center-y", "radius-px"].map((id) => document.querySelector(`#${id}`).value.trim());
  const supplied = values.filter(Boolean).length;
  if (supplied !== 0 && supplied !== 3) throw new Error("手动几何的圆心 X、圆心 Y 和半径必须同时提供。 ");
  const diameter = Number(document.querySelector("#diameter").value);
  const exclusion = Number(document.querySelector("#edge-exclusion").value);
  if (!Number.isFinite(diameter) || diameter <= 0) throw new Error("晶圆直径必须是正数。 ");
  if (!Number.isFinite(exclusion) || exclusion < 0 || exclusion * 2 >= diameter) {
    throw new Error("外缘排除必须非负，且小于晶圆半径。 ");
  }
  return {
    wafer_diameter_mm: diameter,
    exclude_edge_mm: exclusion,
    threshold_method: document.querySelector("#threshold-method").value,
    use_watershed: document.querySelector("#watershed").checked,
    manual_geometry: supplied === 3 ? { center_x: Number(values[0]), center_y: Number(values[1]), radius_px: Number(values[2]) } : null,
  };
}

function errorMessage(raw) {
  const message = String(raw || "未知错误");
  if (/memory|out of bounds|allocation/i.test(message)) return "浏览器内存不足，分析已安全停止且未输出密度。请缩小图像或使用本机大图工作台。";
  return message.split("\n").filter(Boolean).slice(-1)[0];
}

function formatNumber(value, digits = 6) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toLocaleString("zh-CN", { maximumSignificantDigits: digits }) : "—";
}

function patchReport(html, urls) {
  const documentValue = new DOMParser().parseFromString(html, "text/html");
  for (const node of documentValue.querySelectorAll("[src], [href]")) {
    const attribute = node.hasAttribute("src") ? "src" : "href";
    const target = (node.getAttribute(attribute) || "").replace(/^\.\//, "");
    if (urls[target]) node.setAttribute(attribute, urls[target]);
    else if (!/^(#|https?:|mailto:|data:|blob:)/i.test(target)) node.removeAttribute(attribute);
  }
  return `<!doctype html>\n${documentValue.documentElement.outerHTML}`;
}

function showArtifacts(payload) {
  cleanupObjectUrls();
  const urls = {};
  for (const [name, artifact] of Object.entries(payload.artifacts)) urls[name] = makeUrl(artifact.buffer, artifact.mime);

  document.querySelector("#metric-count").textContent = formatNumber(payload.summary.accepted_count, 10);
  document.querySelector("#metric-area").textContent = formatNumber(payload.summary.valid_analysis_area_cm2, 8);
  document.querySelector("#metric-density").textContent = formatNumber(payload.summary.point_density_cm2, 8);
  document.querySelector("#metric-uncertainty").textContent = `± ${formatNumber(payload.summary.counting_uncertainty_cm2, 7)}`;
  document.querySelector("#confidence-interval").textContent = `95% Poisson 计数区间：${formatNumber(payload.summary.poisson_95_ci_lower_cm2, 7)} – ${formatNumber(payload.summary.poisson_95_ci_upper_cm2, 7)} cm⁻²。该区间不包含漏检、误检和物理判定等系统误差。`;
  document.querySelector("#result-file").textContent = `${payload.summary.input_file_name || "当前图像"} · ${payload.width} × ${payload.height} px`;

  const warnings = Array.isArray(payload.summary.warnings) ? payload.summary.warnings : [];
  document.querySelector("#warning-list").replaceChildren(...warnings.map((warning) => {
    const item = document.createElement("p");
    item.textContent = warning;
    return item;
  }));

  const tabs = document.querySelector("#preview-tabs");
  const preview = document.querySelector("#artifact-preview");
  const caption = document.querySelector("#artifact-caption");
  tabs.replaceChildren();
  const available = Object.keys(imageLabels).filter((name) => urls[name]);
  available.forEach((name, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.role = "tab";
    button.textContent = imageLabels[name];
    button.setAttribute("aria-selected", index === 0 ? "true" : "false");
    button.addEventListener("click", () => {
      tabs.querySelectorAll("button").forEach((item) => item.setAttribute("aria-selected", String(item === button)));
      preview.src = urls[name];
      preview.alt = imageLabels[name];
      caption.textContent = `${imageLabels[name]} · 本次分析输出 ${name}`;
    });
    tabs.append(button);
  });
  if (available.length) tabs.firstElementChild.click();

  const report = payload.artifacts["report.html"];
  if (report) {
    const html = new TextDecoder().decode(report.buffer);
    document.querySelector("#report-preview").srcdoc = patchReport(html, urls);
  }

  const bundleUrl = makeUrl(payload.bundle, "application/zip");
  bundleDownload.href = bundleUrl;
  bundleDownload.download = payload.bundleName;

  const downloads = document.querySelector("#download-list");
  downloads.replaceChildren(...Object.keys(urls).sort().map((name) => {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = urls[name];
    link.download = name;
    link.textContent = name;
    item.append(link);
    return item;
  }));
  results.hidden = false;
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function stopWorker() {
  if (worker) worker.terminate();
  worker = null;
  analyzeButton.disabled = false;
  demoButton.disabled = false;
  cancelButton.hidden = true;
}

async function runAnalysis(event) {
  event.preventDefault();
  try {
    validateFile(selectedFile);
    const options = readOptions();
    results.hidden = true;
    cleanupObjectUrls();
    analyzeButton.disabled = true;
    demoButton.disabled = true;
    cancelButton.hidden = false;
    setStatus("running", "正在准备分析", "正在把图像写入当前标签页的临时科研运行环境…", 0.01);
    worker = new Worker("assets/analysis-worker.mjs", { type: "module" });
    worker.addEventListener("message", (messageEvent) => {
      const payload = messageEvent.data;
      if (payload.type === "status") {
        setStatus("running", "分析进行中", payload.message, payload.progress);
      } else if (payload.type === "complete") {
        showArtifacts(payload);
        setStatus("complete", "分析完成", "报告、对照细节图、整片密度热图和审计文件已生成。", 1);
        stopWorker();
      } else if (payload.type === "error") {
        setStatus("error", "分析已安全停止", `${errorMessage(payload.message)} 未生成或展示可能误导的密度结果。`, 0);
        stopWorker();
      }
    });
    worker.addEventListener("error", (error) => {
      setStatus("error", "运行环境启动失败", `${errorMessage(error.message)} 请检查网络后重试，或使用本机工作台。`, 0);
      stopWorker();
    });
    const fileBuffer = await selectedFile.arrayBuffer();
    worker.postMessage({ type: "analyze", fileName: selectedFile.name, fileBuffer, options }, [fileBuffer]);
  } catch (error) {
    setStatus("error", "无法开始分析", errorMessage(error.message), 0);
    stopWorker();
  }
}

fileInput.addEventListener("change", () => setSelectedFile(fileInput.files?.[0] || null));
for (const eventName of ["dragenter", "dragover"]) dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add("is-dragging"); });
for (const eventName of ["dragleave", "drop"]) dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove("is-dragging"); });
dropZone.addEventListener("drop", (event) => setSelectedFile(event.dataTransfer?.files?.[0] || null));
demoButton.addEventListener("click", async () => {
  try {
    setStatus("running", "正在载入合成样本", "该样本固定随机种子，预期接受 96 个点状目标。", null);
    const response = await fetch("assets/demo/synthetic_clean.png");
    if (!response.ok) throw new Error(`无法读取演示文件：HTTP ${response.status}`);
    const blob = await response.blob();
    setSelectedFile(new File([blob], "synthetic_clean.png", { type: "image/png" }));
  } catch (error) {
    setStatus("error", "演示样本载入失败", errorMessage(error.message), 0);
  }
});
cancelButton.addEventListener("click", () => {
  stopWorker();
  results.hidden = true;
  setStatus("", "本次运行已取消", "临时运行环境已释放；可以重新开始。", 0);
});
form.addEventListener("submit", runAnalysis);
window.addEventListener("pagehide", () => { stopWorker(); cleanupObjectUrls(); });

fetch("assets/runtime/manifest.json", { cache: "no-cache" })
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((value) => {
    manifest = value;
    const limits = value.limits;
    document.querySelector("#browser-limits").textContent = `浏览器安全上限：${humanBytes(limits.max_file_bytes)}、${limits.max_pixels.toLocaleString("zh-CN")} 像素、单边 ${limits.max_dimension_px.toLocaleString("zh-CN")} px。超出时明确拒绝并引导本机分块分析。`;
  })
  .catch(() => {
    document.querySelector("#browser-limits").textContent = "无法读取浏览器运行清单；开始分析前请刷新页面。";
    analyzeButton.disabled = true;
  });
