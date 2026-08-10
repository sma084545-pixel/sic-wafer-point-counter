const $ = (selector) => document.querySelector(selector);

const state = {
  project: null,
  overviewImage: null,
  overviewSelection: null,
  overviewDrag: null,
  roiImage: null,
  roi: null,
  annotationId: null,
  labels: null,
  tool: 'brush',
  label: 1,
  brushSize: 9,
  showLabels: true,
  zoom: 1,
  panX: 0,
  panY: 0,
  pointer: null,
  spaceDown: false,
  history: [],
  redo: [],
  stroke: null,
};

const canvas = $('#annotation-canvas');
const context = canvas.getContext('2d', {alpha: false});
const overviewCanvas = $('#overview-canvas');
const overviewContext = overviewCanvas.getContext('2d');
const labelCanvas = document.createElement('canvas');
const labelContext = labelCanvas.getContext('2d');
const stage = $('#annotation-stage');

class RequestError extends Error {}

async function jsonRequest(url, options = {}) {
  const response = await fetch(url, {cache: 'no-store', ...options});
  let payload;
  try { payload = await response.json(); } catch { throw new RequestError('本机服务返回了无法解析的响应'); }
  if (!response.ok) throw new RequestError(payload.error || `请求失败（HTTP ${response.status}）`);
  return payload;
}

function showError(message) {
  const target = $('#training-error');
  target.textContent = message;
  target.hidden = false;
  target.focus?.();
}

function clearError() { $('#training-error').hidden = true; $('#project-error').hidden = true; }

function sourceShape() {
  const [height, width] = state.project?.source_image?.shape_yx || [0, 0];
  return {width: Number(width), height: Number(height)};
}

function encodeRle(labels, width, height) {
  const runs = [];
  let index = 0;
  while (index < labels.length) {
    const value = labels[index];
    const start = index;
    index += 1;
    while (index < labels.length && labels[index] === value) index += 1;
    if (value !== 0) runs.push([start, index - start, value]);
  }
  return {encoding: 'flat_nonzero_rle_v1', shape: [height, width], runs};
}

function decodeRle(payload) {
  if (!payload || payload.encoding !== 'flat_nonzero_rle_v1') throw new Error('训练项目标签编码不兼容');
  const [height, width] = payload.shape.map(Number);
  const labels = new Uint8Array(width * height);
  for (const run of payload.runs || []) {
    const [start, length, value] = run.map(Number);
    labels.fill(value, start, start + length);
  }
  return {labels, width, height};
}

function labelColor(value, alpha = .58) {
  if (value === 1) return `rgba(235,47,57,${alpha})`;
  if (value === 2) return `rgba(31,181,91,${alpha})`;
  if (value === 3) return `rgba(244,200,36,${alpha})`;
  return 'rgba(0,0,0,0)';
}

function rebuildLabelCanvas() {
  if (!state.labels || !state.roi) return;
  const {width, height} = state.roi;
  labelCanvas.width = width;
  labelCanvas.height = height;
  const pixels = labelContext.createImageData(width, height);
  const colors = [[0,0,0,0],[235,47,57,148],[31,181,91,148],[244,200,36,160]];
  for (let index = 0; index < state.labels.length; index += 1) {
    const color = colors[state.labels[index]];
    const offset = index * 4;
    pixels.data[offset] = color[0]; pixels.data[offset + 1] = color[1];
    pixels.data[offset + 2] = color[2]; pixels.data[offset + 3] = color[3];
  }
  labelContext.putImageData(pixels, 0, 0);
}

function canvasCssSize() {
  const rect = stage.getBoundingClientRect();
  return {width: Math.max(1, rect.width), height: Math.max(1, rect.height)};
}

function resizeCanvas() {
  const {width, height} = canvasCssSize();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const nextWidth = Math.round(width * ratio), nextHeight = Math.round(height * ratio);
  if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
    canvas.width = nextWidth; canvas.height = nextHeight;
    canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  renderCanvas();
}

function fitView() {
  if (!state.roiImage || !state.roi) return;
  const {width, height} = canvasCssSize();
  state.zoom = Math.min(width / state.roi.width, height / state.roi.height) * .94;
  state.panX = (width - state.roi.width * state.zoom) / 2;
  state.panY = (height - state.roi.height * state.zoom) / 2;
  renderCanvas();
}

function renderCanvas() {
  const {width, height} = canvasCssSize();
  context.save();
  context.setTransform(context.getTransform().a, 0, 0, context.getTransform().d, 0, 0);
  context.fillStyle = '#11191b'; context.fillRect(0, 0, width, height);
  if (state.roiImage && state.roi) {
    context.imageSmoothingEnabled = state.zoom < 1;
    context.drawImage(state.roiImage, state.panX, state.panY, state.roi.width * state.zoom, state.roi.height * state.zoom);
    if (state.showLabels) context.drawImage(labelCanvas, state.panX, state.panY, state.roi.width * state.zoom, state.roi.height * state.zoom);
    context.strokeStyle = '#79d6d1'; context.lineWidth = 1;
    context.strokeRect(state.panX, state.panY, state.roi.width * state.zoom, state.roi.height * state.zoom);
  }
  context.restore();
  $('#viewport-status').textContent = state.roi
    ? `ROI (${state.roi.x}, ${state.roi.y}) ${state.roi.width}×${state.roi.height} px · 缩放 ${state.zoom.toFixed(2)}×`
    : '尚未载入 ROI';
}

function localPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {x: event.clientX - rect.left, y: event.clientY - rect.top};
}

function imagePoint(point) {
  return {x: (point.x - state.panX) / state.zoom, y: (point.y - state.panY) / state.zoom};
}

function paintAt(x, y, value) {
  if (!state.labels || !state.roi) return;
  const radius = state.brushSize / 2;
  const x0 = Math.max(0, Math.floor(x - radius)), x1 = Math.min(state.roi.width - 1, Math.ceil(x + radius));
  const y0 = Math.max(0, Math.floor(y - radius)), y1 = Math.min(state.roi.height - 1, Math.ceil(y + radius));
  for (let row = y0; row <= y1; row += 1) for (let column = x0; column <= x1; column += 1) {
    if ((column + .5 - x) ** 2 + (row + .5 - y) ** 2 > radius ** 2) continue;
    const index = row * state.roi.width + column;
    if (!state.stroke.old.has(index)) state.stroke.old.set(index, state.labels[index]);
    state.labels[index] = value;
  }
  labelContext.save();
  labelContext.beginPath(); labelContext.arc(x, y, radius, 0, Math.PI * 2);
  if (value === 0) { labelContext.globalCompositeOperation = 'destination-out'; labelContext.fillStyle = '#000'; }
  else { labelContext.globalCompositeOperation = 'source-over'; labelContext.fillStyle = labelColor(value); }
  labelContext.fill(); labelContext.restore();
  renderCanvas();
}

function finishStroke() {
  if (!state.stroke?.old?.size) { state.stroke = null; return; }
  const indices = [...state.stroke.old.keys()];
  state.history.push({indices, old: indices.map((i) => state.stroke.old.get(i)), next: indices.map((i) => state.labels[i])});
  if (state.history.length > 100) state.history.shift();
  state.redo = []; state.stroke = null; updateCounts();
}

function applyHistory(entry, direction) {
  if (!entry || !state.labels) return;
  const values = direction === 'undo' ? entry.old : entry.next;
  entry.indices.forEach((index, offset) => { state.labels[index] = values[offset]; });
  rebuildLabelCanvas(); renderCanvas(); updateCounts();
}

function updateCounts() {
  const counts = [0,0,0,0];
  if (state.labels) for (const value of state.labels) counts[value] += 1;
  $('#count-target').textContent = counts[1].toLocaleString('zh-CN');
  $('#count-background').textContent = counts[2].toLocaleString('zh-CN');
  $('#count-ignore').textContent = counts[3].toLocaleString('zh-CN');
}

function renderOverview() {
  if (!state.overviewImage) return;
  overviewCanvas.width = state.overviewImage.naturalWidth;
  overviewCanvas.height = state.overviewImage.naturalHeight;
  overviewContext.drawImage(state.overviewImage, 0, 0);
  if (state.overviewSelection) {
    const value = state.overviewSelection;
    overviewContext.fillStyle = 'rgba(8,127,131,.18)'; overviewContext.strokeStyle = '#00f0df'; overviewContext.lineWidth = 2;
    overviewContext.fillRect(value.x, value.y, value.width, value.height);
    overviewContext.strokeRect(value.x, value.y, value.width, value.height);
  }
}

function overviewPoint(event) {
  const rect = overviewCanvas.getBoundingClientRect();
  return {x: (event.clientX - rect.left) * overviewCanvas.width / rect.width, y: (event.clientY - rect.top) * overviewCanvas.height / rect.height};
}

function selectionToRoi(selection) {
  const shape = sourceShape();
  const scaleX = shape.width / overviewCanvas.width, scaleY = shape.height / overviewCanvas.height;
  let x = Math.max(0, Math.floor(selection.x * scaleX));
  let y = Math.max(0, Math.floor(selection.y * scaleY));
  let width = Math.max(32, Math.ceil(selection.width * scaleX));
  let height = Math.max(32, Math.ceil(selection.height * scaleY));
  width = Math.min(2048, width, shape.width - x); height = Math.min(2048, height, shape.height - y);
  $('#roi-x').value = x; $('#roi-y').value = y; $('#roi-width').value = width; $('#roi-height').value = height;
}

async function loadProjects() {
  const payload = await jsonRequest('/api/pixel-training/projects');
  const list = $('#project-list'); list.replaceChildren();
  if (!payload.projects.length) { list.textContent = '尚无像素训练项目。'; return; }
  for (const project of payload.projects) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'project-item';
    button.innerHTML = `<span><strong>${project.image_name}</strong><small>${project.wafer_id} · ${project.split} · ${project.roi_count} ROI</small></span><b>打开</b>`;
    button.addEventListener('click', () => openProject(project.project_id)); list.append(button);
  }
}

async function imageFromUrl(url) {
  const image = new Image(); image.decoding = 'async'; image.src = `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;
  await image.decode(); return image;
}

async function openProject(projectId) {
  clearError();
  const payload = await jsonRequest(`/api/pixel-training/projects/${encodeURIComponent(projectId)}`);
  state.project = payload.project;
  state.overviewImage = await imageFromUrl(state.project.preview_url);
  state.overviewSelection = null; renderOverview();
  $('#training-workspace').hidden = false;
  $('#current-project-title').textContent = state.project.source_image.file_name;
  $('#current-project-meta').textContent = `${state.project.wafer_id} · ${state.project.split} · 原图 ${sourceShape().width}×${sourceShape().height} px · SHA-256 ${state.project.source_image.sha256.slice(0,16)}…`;
  $('#download-project').href = `/api/pixel-training/projects/${encodeURIComponent(projectId)}/download`;
  $('#download-project').download = `${projectId}.sictrain.json`;
  $('#download-pixel-model').hidden = !state.project.model_available;
  const latest = state.project.annotations.at(-1);
  if (latest) await loadAnnotation(latest);
  else { const shape = sourceShape(); $('#roi-x').value = Math.max(0, Math.floor(shape.width/2-256)); $('#roi-y').value = Math.max(0, Math.floor(shape.height/2-256)); $('#roi-width').value = Math.min(512,shape.width); $('#roi-height').value = Math.min(512,shape.height); }
  $('#training-workspace').scrollIntoView({behavior:'smooth',block:'start'});
}

async function loadAnnotation(annotation) {
  const [x,y,width,height] = annotation.roi_xywh.map(Number);
  $('#roi-x').value=x; $('#roi-y').value=y; $('#roi-width').value=width; $('#roi-height').value=height;
  await loadRoi({x,y,width,height}, annotation);
}

async function loadRoi(roi, annotation = null) {
  if (!state.project) throw new Error('请先创建或打开训练项目');
  const params = new URLSearchParams(roi);
  state.roiImage = await imageFromUrl(`/api/pixel-training/projects/${encodeURIComponent(state.project.project_id)}/roi?${params}`);
  state.roi = {...roi}; state.annotationId = annotation?.annotation_id || null;
  if (annotation) state.labels = decodeRle(annotation.labels).labels;
  else state.labels = new Uint8Array(roi.width * roi.height);
  labelCanvas.width = roi.width; labelCanvas.height = roi.height; rebuildLabelCanvas();
  state.history=[]; state.redo=[]; $('#canvas-placeholder').hidden=true; updateCounts(); resizeCanvas(); fitView();
  $('#training-status').textContent = annotation ? `已恢复 ${annotation.annotation_id}，可继续纠错。` : '新 ROI 尚未保存标签。';
}

async function saveLabels() {
  if (!state.project || !state.roi || !state.labels) throw new Error('请先载入 ROI 并绘制标签');
  const payload = await jsonRequest(`/api/pixel-training/projects/${encodeURIComponent(state.project.project_id)}/annotations`, {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({
      annotation_id: state.annotationId, roi_xywh:[state.roi.x,state.roi.y,state.roi.width,state.roi.height],
      labels:encodeRle(state.labels,state.roi.width,state.roi.height),
    }),
  });
  state.annotationId = payload.annotation.annotation_id; state.project = payload.project;
  $('#training-status').textContent = `已保存 ${state.annotationId}；标签与原图 SHA-256 已绑定。`;
  return payload.annotation;
}

function metricCard(label, value) { return `<div class="metric"><span>${label}</span><strong>${value == null ? 'NA' : Number(value).toFixed(3)}</strong></div>`; }

async function trainAndPredict() {
  clearError(); await saveLabels();
  $('#training-status').textContent = '正在提取多尺度特征并训练像素随机森林…';
  const settings = {probability_threshold:Number($('#pixel-threshold').value),minimum_object_area_px:Number($('#pixel-min-area').value),n_trees:Number($('#pixel-trees').value),random_seed:Number($('#pixel-seed').value)};
  const trained = await jsonRequest('/api/pixel-training/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(settings)});
  $('#training-status').textContent = `模型已训练：${trained.model.model_sha256.slice(0,16)}…；正在生成当前 ROI 预测。`;
  $('#download-pixel-model').hidden=false;
  const result = await jsonRequest(`/api/pixel-training/projects/${encodeURIComponent(state.project.project_id)}/predict/${encodeURIComponent(state.annotationId)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(settings)});
  const prediction=result.prediction; const stamp=`?t=${Date.now()}`;
  $('#probability-preview').src=prediction.files.probability+stamp; $('#segmentation-preview').src=prediction.files.segmentation+stamp; $('#overlay-preview').src=prediction.files.overlay+stamp;
  $('#prediction-grid').hidden=false;
  const pixels=prediction.metrics.pixel, objects=prediction.metrics.object;
  $('#metrics').innerHTML=[metricCard('像素 Precision',pixels.precision),metricCard('像素 Recall',pixels.recall),metricCard('像素 F1',pixels.f1),metricCard('像素 IoU',pixels.iou),metricCard('目标 Precision',objects.precision),metricCard('目标 Recall',objects.recall),metricCard('目标 F1',objects.f1),metricCard('平均定位误差 px',objects.mean_roi_localization_error_px)].join('');
  $('#metrics').hidden=false;
  $('#training-status').textContent=`训练和预览完成。当前显示的是 ${state.project.split} ROI 的图像判定结果；真实物理准确率仍需独立验证。`;
}

$('#new-project-form').addEventListener('submit', async (event) => {
  event.preventDefault(); clearError();
  try { const payload=await jsonRequest('/api/pixel-training/projects',{method:'POST',body:new FormData(event.currentTarget)}); await loadProjects(); await openProject(payload.project.project_id); }
  catch(error){ const target=$('#project-error');target.textContent=error.message;target.hidden=false; }
});
$('#project-image').addEventListener('change',()=>{const file=$('#project-image').files?.[0];$('#project-file-line').textContent=file?`${file.name} · ${(file.size/1048576).toFixed(1)} MiB`:'支持 ImageJ TIFF / BigTIFF；原图仅保存在本机 training/';});
$('#import-pixel-model').addEventListener('change',async event=>{try{clearError();const file=event.target.files?.[0];if(!file) return;if(file.size<=0||file.size>16*1024*1024)throw new Error('像素模型 JSON 必须非空且不超过 16 MiB');const model=JSON.parse(await file.text());const payload=await jsonRequest('/api/pixel-training/model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(model)});$('#training-status').textContent=`像素模型已载入并核验：${payload.model.model_sha256.slice(0,16)}…`;$('#download-pixel-model').hidden=false;}catch(error){showError(error.message);}});
$('#refresh-projects').addEventListener('click',()=>loadProjects().catch(e=>showError(e.message)));

overviewCanvas.addEventListener('pointerdown',(event)=>{if(!state.overviewImage)return;overviewCanvas.setPointerCapture(event.pointerId);const point=overviewPoint(event);state.overviewDrag=point;state.overviewSelection={x:point.x,y:point.y,width:1,height:1};renderOverview();});
overviewCanvas.addEventListener('pointermove',(event)=>{if(!state.overviewDrag)return;const point=overviewPoint(event);state.overviewSelection={x:Math.min(point.x,state.overviewDrag.x),y:Math.min(point.y,state.overviewDrag.y),width:Math.abs(point.x-state.overviewDrag.x),height:Math.abs(point.y-state.overviewDrag.y)};renderOverview();});
overviewCanvas.addEventListener('pointerup',()=>{if(state.overviewSelection)selectionToRoi(state.overviewSelection);state.overviewDrag=null;});
$('#load-roi').addEventListener('click',async()=>{try{const roi={x:Number($('#roi-x').value),y:Number($('#roi-y').value),width:Number($('#roi-width').value),height:Number($('#roi-height').value)};await loadRoi(roi);}catch(e){showError(e.message);}});

canvas.addEventListener('wheel',(event)=>{if(!state.roi)return;event.preventDefault();const point=localPoint(event),before=imagePoint(point);const factor=Math.exp(-event.deltaY*.0015);state.zoom=Math.max(.08,Math.min(32,state.zoom*factor));state.panX=point.x-before.x*state.zoom;state.panY=point.y-before.y*state.zoom;renderCanvas();},{passive:false});
canvas.addEventListener('pointerdown',(event)=>{if(!state.roi)return;canvas.setPointerCapture(event.pointerId);const point=localPoint(event);const panning=state.tool==='pan'||state.spaceDown||event.button===1;state.pointer={id:event.pointerId,last:point,panning};if(!panning){state.stroke={old:new Map()};const image=imagePoint(point);paintAt(image.x,image.y,state.tool==='erase'?0:state.label);}});
canvas.addEventListener('pointermove',(event)=>{if(!state.pointer||event.pointerId!==state.pointer.id)return;const point=localPoint(event);if(state.pointer.panning){state.panX+=point.x-state.pointer.last.x;state.panY+=point.y-state.pointer.last.y;renderCanvas();}else{const image=imagePoint(point);paintAt(image.x,image.y,state.tool==='erase'?0:state.label);}state.pointer.last=point;});
function pointerEnd(event){if(!state.pointer||event.pointerId!==state.pointer.id)return;finishStroke();state.pointer=null;}
canvas.addEventListener('pointerup',pointerEnd);canvas.addEventListener('pointercancel',pointerEnd);

document.querySelectorAll('[data-tool]').forEach(button=>button.addEventListener('click',()=>{state.tool=button.dataset.tool;document.querySelectorAll('[data-tool]').forEach(item=>item.classList.toggle('active',item===button));canvas.style.cursor=state.tool==='pan'?'grab':state.tool==='erase'?'cell':'crosshair';}));
document.querySelectorAll('[data-label]').forEach(button=>button.addEventListener('click',()=>{state.label=Number(button.dataset.label);state.tool='brush';document.querySelectorAll('[data-label]').forEach(item=>item.classList.toggle('active',item===button));document.querySelectorAll('[data-tool]').forEach(item=>item.classList.toggle('active',item.dataset.tool==='brush'));}));
$('#brush-size').addEventListener('input',(event)=>{state.brushSize=Number(event.target.value);$('#brush-size-value').textContent=`${state.brushSize} px`;});
$('#undo').addEventListener('click',()=>{const entry=state.history.pop();if(entry){applyHistory(entry,'undo');state.redo.push(entry);}});
$('#redo').addEventListener('click',()=>{const entry=state.redo.pop();if(entry){applyHistory(entry,'redo');state.history.push(entry);}});
$('#fit-view').addEventListener('click',fitView);
$('#toggle-labels').addEventListener('click',(event)=>{state.showLabels=!state.showLabels;event.currentTarget.setAttribute('aria-pressed',String(state.showLabels));event.currentTarget.textContent=state.showLabels?'隐藏标签':'显示标签';renderCanvas();});
$('#clear-labels').addEventListener('click',()=>{if(!state.labels)return;const selected=$('#clear-class').value;const indices=[];const old=[];for(let i=0;i<state.labels.length;i+=1){if(selected==='all'||state.labels[i]===Number(selected)){if(state.labels[i]){indices.push(i);old.push(state.labels[i]);state.labels[i]=0;}}}if(indices.length){state.history.push({indices,old,next:indices.map(()=>0)});state.redo=[];rebuildLabelCanvas();renderCanvas();updateCounts();}});
$('#save-labels').addEventListener('click',()=>{clearError();saveLabels().catch(e=>showError(e.message));});
$('#train-pixel-model').addEventListener('click',()=>trainAndPredict().catch(e=>{showError(e.message);$('#training-status').textContent='训练已安全停止，未覆盖当前有效模型。';}));
window.addEventListener('keydown',(event)=>{if(event.code==='Space'&&!/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)){state.spaceDown=true;event.preventDefault();}if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='z'){event.preventDefault();(event.shiftKey?$('#redo'):$('#undo')).click();}});
window.addEventListener('keyup',(event)=>{if(event.code==='Space')state.spaceDown=false;});
new ResizeObserver(resizeCanvas).observe(stage);
loadProjects().catch(error=>{$('#project-list').textContent=`读取失败：${error.message}`;});
