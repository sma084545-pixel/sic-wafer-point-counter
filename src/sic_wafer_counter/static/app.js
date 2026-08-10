import {api} from './api.js';
import {
  renderCandidateError, renderCandidatePage, renderLatestPlaceholder,
  renderLatestRun, renderRunDetail, renderRunIndex, renderRunIndexError,
  updateResultImage,
} from './results.js';
import {escapeHtml} from './format.js';

const state = {
  runs: [],
  currentRun: null,
  currentRunToken: 0,
  candidates: {page: 1, totalPages: 1},
  jobActive: false,
  training: null,
};

const elements = {
  form: document.querySelector('#analysis-form'),
  fileInput: document.querySelector('#image'),
  fileName: document.querySelector('#file-name'),
  dropZone: document.querySelector('.drop-zone'),
  submit: document.querySelector('#submit-analysis'),
  progress: document.querySelector('#job-progress'),
  progressText: document.querySelector('#progress-text'),
  live: document.querySelector('#live-region'),
  alert: document.querySelector('#alert-region'),
  formError: document.querySelector('#form-error'),
  sidebar: document.querySelector('#sidebar'),
  menuButton: document.querySelector('#mobile-menu-button'),
  candidateFilters: document.querySelector('#candidate-filters'),
  trainingForm: document.querySelector('#training-form'),
  trainingStatus: document.querySelector('#training-status'),
  trainingError: document.querySelector('#training-error'),
  trainingModelDownload: document.querySelector('#training-model-download'),
  useTrainedClassifier: document.querySelector('#use-trained-classifier'),
  analysisClassifierStatus: document.querySelector('#analysis-classifier-status'),
};

function announce(message) {
  elements.live.textContent = '';
  window.setTimeout(() => { elements.live.textContent = message; }, 40);
}

function showAlert(message) {
  elements.alert.textContent = message;
  elements.alert.hidden = false;
  window.clearTimeout(showAlert.timer);
  showAlert.timer = window.setTimeout(() => { elements.alert.hidden = true; }, 8000);
}

function clearFormError() {
  elements.formError.hidden = true;
  elements.formError.textContent = '';
  elements.form.querySelectorAll('[aria-invalid="true"]').forEach((control) => {
    control.removeAttribute('aria-invalid');
    control.removeAttribute('aria-describedby');
  });
}

function showFormError(message, control = null) {
  clearFormError();
  elements.formError.textContent = message;
  elements.formError.hidden = false;
  if (control) {
    control.setAttribute('aria-invalid', 'true');
    control.setAttribute('aria-describedby', 'form-error');
    control.focus();
  } else {
    elements.formError.setAttribute('tabindex', '-1');
    elements.formError.focus();
  }
}

function routeParts() {
  const raw = window.location.hash.replace(/^#/, '') || 'overview';
  const [view, encodedId] = raw.split('/', 2);
  return {view, runId: encodedId ? decodeURIComponent(encodedId) : null};
}

function setActiveNavigation(active) {
  const navView = active === 'result' ? 'runs' : active;
  document.querySelectorAll('[data-nav]').forEach((link) => {
    if (link.dataset.nav === navView) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
}

function showView(name) {
  const allowed = ['overview', 'analyze', 'runs', 'training', 'result', 'methods'];
  const target = allowed.includes(name) ? name : 'overview';
  document.querySelectorAll('[data-view]').forEach((view) => {
    view.hidden = view.dataset.view !== target;
  });
  setActiveNavigation(target);
  const labels = {overview: '总览', analyze: '新建分析', runs: '结果记录', training: '标注训练', result: '分析结果', methods: '方法与验证'};
  document.title = `${labels[target]} | SiC 晶圆点状目标分析平台`;
  elements.sidebar.classList.remove('is-open');
  elements.menuButton.setAttribute('aria-expanded', 'false');
  window.scrollTo({top: 0, behavior: 'auto'});
  return target;
}

function renderTrainingStatus(payload) {
  state.training = payload;
  const counts = payload.consensus_label_counts || {};
  const model = payload.model;
  const validation = model?.validation || {};
  const validationStatus = validation.status || 'not_available';
  elements.trainingStatus.className = '';
  elements.trainingStatus.removeAttribute('aria-busy');
  elements.trainingStatus.innerHTML = `<div class="training-status-grid">
    <div><span>一致目标标签</span><strong>${Number(counts.target || 0)}</strong><small>进入相应 split</small></div>
    <div><span>一致伪影标签</span><strong>${Number(counts.artifact || 0)}</strong><small>进入相应 split</small></div>
    <div><span>不确定 / 冲突</span><strong>${Number(payload.uncertain_candidate_count || 0) + Number(payload.conflicting_candidate_count || 0)}</strong><small>不进入训练</small></div>
  </div><p class="training-model-summary">${model
    ? `当前模型：${escapeHtml(String(model.model_sha256 || '').slice(0, 16))}… · 训练 ${Number(model.training_sample_count || 0)} 条 · 留出评估 ${escapeHtml(validationStatus)}`
    : `当前没有可用模型${payload.model_error ? `：${escapeHtml(payload.model_error)}` : '。完成足量校准标注后训练。'}`}</p>`;
  elements.trainingModelDownload.hidden = !payload.model_available;
  elements.useTrainedClassifier.disabled = !payload.model_available;
  if (!payload.model_available) elements.useTrainedClassifier.checked = false;
  elements.analysisClassifierStatus.textContent = payload.model_available
    ? `可用模型 ${String(model?.model_sha256 || '').slice(0, 12)}…；${validationStatus}`
    : '尚无可用模型；请先在结果页标注并训练';
}

async function refreshTraining({announceResult = false} = {}) {
  try {
    const payload = await api.getTraining();
    renderTrainingStatus(payload);
    if (announceResult) announce(`已读取 ${payload.annotation_count || 0} 条本机专家标注`);
  } catch (error) {
    elements.trainingStatus.className = 'empty-state';
    elements.trainingStatus.removeAttribute('aria-busy');
    elements.trainingStatus.textContent = `训练状态读取失败：${error.message}`;
    elements.useTrainedClassifier.disabled = true;
    elements.useTrainedClassifier.checked = false;
    elements.analysisClassifierStatus.textContent = '训练状态不可用';
    if (announceResult) showAlert(error.message);
  }
}

async function refreshRuns({announceResult = false} = {}) {
  try {
    const payload = await api.listRuns();
    state.runs = renderRunIndex(payload);
    const latest = state.runs.find((run) => run.status === 'completed') || state.runs[0];
    if (!latest) {
      renderLatestPlaceholder('尚无可用结果。可从“新建分析”运行 clean 合成演示。');
    } else {
      try {
        renderLatestRun(await api.getRun(latest.run_id));
      } catch (error) {
        renderLatestPlaceholder(`最近结果读取失败：${error.message}`);
      }
    }
    if (announceResult) announce(`已读取 ${state.runs.length} 条本机分析记录`);
  } catch (error) {
    renderRunIndexError(error.message);
    renderLatestPlaceholder(`无法读取结果：${error.message}`);
    if (announceResult) showAlert(error.message);
  }
}

async function loadCandidates(page = 1) {
  if (!state.currentRun) return;
  const form = new FormData(elements.candidateFilters);
  const filters = {
    status: form.get('status') || 'all',
    reason: form.get('reason') || '',
    defect_id: form.get('defect_id') || '',
    page,
    page_size: form.get('page_size') || 50,
  };
  document.querySelector('#candidate-body').innerHTML = '<tr><td colspan="11" class="muted-cell">正在读取当前页…</td></tr>';
  try {
    const payload = await api.getDefects(state.currentRun.run_id, filters);
    state.candidates.page = payload.page;
    state.candidates.totalPages = payload.total_pages;
    renderCandidatePage(payload);
    announce(`候选第 ${payload.page} 页，共 ${payload.total} 条筛选结果`);
  } catch (error) {
    renderCandidateError(error.message);
  }
}

async function loadRun(runId) {
  const token = ++state.currentRunToken;
  state.currentRun = null;
  document.querySelector('#result-title').textContent = '正在读取分析结果…';
  document.querySelector('#result-subtitle').textContent = runId;
  document.querySelector('#result-metrics').innerHTML = '';
  document.querySelector('#result-main').hidden = true;
  document.querySelector('#result-error').hidden = true;
  try {
    const detail = await api.getRun(runId);
    if (token !== state.currentRunToken) return;
    state.currentRun = detail;
    renderRunDetail(detail);
    if (!document.querySelector('#candidate-browser').hidden) await loadCandidates(1);
    announce(`已打开 ${detail.input_file_name || detail.run_id} 的分析结果`);
  } catch (error) {
    if (token !== state.currentRunToken) return;
    const box = document.querySelector('#result-error');
    box.hidden = false;
    box.textContent = `结果读取失败：${error.message}`;
    showAlert(error.message);
  }
}

async function handleRoute() {
  const route = routeParts();
  if (route.view === 'run' && route.runId) {
    showView('result');
    await loadRun(route.runId);
    return;
  }
  if (!['overview', 'analyze', 'runs', 'training', 'methods'].includes(route.view)) {
    window.location.hash = '#overview';
    return;
  }
  showView(route.view);
}

function setJobState(active, message = '') {
  state.jobActive = active;
  elements.progress.hidden = !active;
  elements.submit.disabled = active;
  document.querySelectorAll('.demo-button').forEach((button) => {
    const available = document.body.dataset[`demo${button.dataset.demoKind[0].toUpperCase()}${button.dataset.demoKind.slice(1)}`] === 'true';
    button.disabled = active || !available;
  });
  if (message) elements.progressText.textContent = message;
}

async function pollJob(jobId) {
  let delay = 700;
  while (state.jobActive) {
    const job = await api.getJob(jobId);
    if (job.status === 'queued') elements.progressText.textContent = '任务正在本机单任务队列中等待…';
    if (job.status === 'running') elements.progressText.textContent = '正在读取图像、标定晶圆并检测点状候选…';
    if (job.status === 'completed' || job.status === 'failed') {
      setJobState(false);
      await refreshRuns();
      announce(job.status === 'completed' ? '分析完成，正在打开结果' : '分析已停止，正在打开失败记录');
      window.location.hash = `#run/${encodeURIComponent(job.run_id)}`;
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, delay));
    delay = Math.min(1600, delay + 100);
  }
}

async function submitJob(submitter, {onError = showAlert} = {}) {
  if (state.jobActive) return;
  setJobState(true, '正在提交到本机分析队列…');
  announce('已提交本机分析');
  try {
    const job = await submitter();
    await pollJob(job.job_id);
  } catch (error) {
    setJobState(false);
    onError(error.message || '无法提交分析');
  }
}

function updateFileLabel() {
  const file = elements.fileInput.files?.[0];
  elements.fileName.textContent = file ? `已选择：${file.name}` : '点击选择，或将文件拖到此处';
  clearFormError();
}

elements.fileInput.addEventListener('change', updateFileLabel);
elements.dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    elements.fileInput.click();
  }
});
elements.dropZone.addEventListener('dragover', (event) => {
  event.preventDefault();
  elements.dropZone.classList.add('is-dragging');
});
elements.dropZone.addEventListener('dragleave', () => elements.dropZone.classList.remove('is-dragging'));
elements.dropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove('is-dragging');
  if (event.dataTransfer.files.length) {
    elements.fileInput.files = event.dataTransfer.files;
    updateFileLabel();
  }
});

elements.form.addEventListener('submit', (event) => {
  event.preventDefault();
  clearFormError();
  if (!elements.fileInput.files?.length) {
    showFormError('请选择一张晶圆图像。', elements.dropZone);
    return;
  }
  const file = elements.fileInput.files[0];
  const maxBytes = Number(document.body.dataset.maxUploadMb) * 1024 * 1024;
  if (Number.isFinite(maxBytes) && file.size > maxBytes) {
    showFormError(`文件超过 ${document.body.dataset.maxUploadMb} MB 上传限制。`, elements.dropZone);
    return;
  }
  const manualControls = ['center-x', 'center-y', 'radius-px'].map((id) => document.querySelector(`#${id}`));
  const supplied = manualControls.filter((control) => control.value.trim() !== '');
  if (supplied.length > 0 && supplied.length < manualControls.length) {
    const missing = manualControls.find((control) => control.value.trim() === '');
    showFormError('手工标定时必须同时填写圆心 x、圆心 y 与半径。', missing);
    return;
  }
  submitJob(() => api.submitAnalysis(new FormData(elements.form)), {onError: (message) => showFormError(message)});
});

document.querySelectorAll('.demo-button').forEach((button) => {
  button.addEventListener('click', () => submitJob(() => api.submitDemo(button.dataset.demoKind)));
});

elements.candidateFilters.addEventListener('submit', (event) => {
  event.preventDefault();
  loadCandidates(1);
});
document.querySelector('#candidate-browser').addEventListener('click', async (event) => {
  const button = event.target.closest('.candidate-label-button');
  if (!button || !state.currentRun) return;
  const peers = [...document.querySelectorAll(`.candidate-label-button[data-defect-id="${CSS.escape(button.dataset.defectId)}"]`)];
  peers.forEach((item) => { item.disabled = true; });
  try {
    const payload = await api.saveTrainingLabel({
      run_id: state.currentRun.run_id,
      defect_id: button.dataset.defectId,
      label: button.dataset.label,
      split: document.querySelector('#annotation-split').value,
      reviewer_id: document.querySelector('#annotation-reviewer').value.trim() || 'local_expert',
    });
    peers.forEach((item) => item.setAttribute('aria-pressed', String(item.dataset.label === button.dataset.label)));
    renderTrainingStatus(payload.training);
    announce(`候选 ${button.dataset.defectId} 已标注为${button.textContent}`);
  } catch (error) {
    showAlert(error.message);
  } finally {
    peers.forEach((item) => { item.disabled = false; });
  }
});

elements.trainingForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  elements.trainingError.hidden = true;
  elements.trainingError.textContent = '';
  const form = new FormData(elements.trainingForm);
  const acceptThreshold = Number(form.get('accept_threshold'));
  const rejectThreshold = Number(form.get('reject_threshold'));
  const regularization = Number(form.get('regularization'));
  if (!Number.isFinite(acceptThreshold) || !Number.isFinite(rejectThreshold) || rejectThreshold >= acceptThreshold) {
    elements.trainingError.textContent = '拒绝概率上限必须小于接受概率下限。';
    elements.trainingError.hidden = false;
    return;
  }
  const submit = document.querySelector('#train-classifier');
  submit.disabled = true;
  submit.textContent = '正在训练…';
  try {
    const payload = await api.trainClassifier({
      accept_threshold: acceptThreshold,
      reject_threshold: rejectThreshold,
      regularization: regularization,
    });
    renderTrainingStatus(payload.training);
    announce('候选分类器训练完成并已启用');
  } catch (error) {
    elements.trainingError.textContent = error.message;
    elements.trainingError.hidden = false;
    elements.trainingError.setAttribute('tabindex', '-1');
    elements.trainingError.focus();
  } finally {
    submit.disabled = false;
    submit.textContent = '训练并启用候选分类器';
  }
});
document.querySelector('#candidate-first').addEventListener('click', () => loadCandidates(1));
document.querySelector('#candidate-prev').addEventListener('click', () => loadCandidates(Math.max(1, state.candidates.page - 1)));
document.querySelector('#candidate-next').addEventListener('click', () => loadCandidates(Math.min(state.candidates.totalPages, state.candidates.page + 1)));
document.querySelector('#candidate-last').addEventListener('click', () => loadCandidates(state.candidates.totalPages));
document.querySelector('#image-artifact').addEventListener('change', (event) => updateResultImage(state.currentRun, event.target.value));

document.querySelector('#refresh-runs').addEventListener('click', () => refreshRuns({announceResult: true}));
document.querySelector('#refresh-history').addEventListener('click', () => refreshRuns({announceResult: true}));
document.querySelector('#refresh-training').addEventListener('click', () => refreshTraining({announceResult: true}));
elements.menuButton.addEventListener('click', () => {
  const open = !elements.sidebar.classList.contains('is-open');
  elements.sidebar.classList.toggle('is-open', open);
  elements.menuButton.setAttribute('aria-expanded', String(open));
});
document.addEventListener('click', (event) => {
  if (window.innerWidth > 960 || !elements.sidebar.classList.contains('is-open')) return;
  if (!elements.sidebar.contains(event.target) && !elements.menuButton.contains(event.target)) {
    elements.sidebar.classList.remove('is-open');
    elements.menuButton.setAttribute('aria-expanded', 'false');
  }
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && elements.sidebar.classList.contains('is-open')) {
    elements.sidebar.classList.remove('is-open');
    elements.menuButton.setAttribute('aria-expanded', 'false');
    elements.menuButton.focus();
  }
});

document.querySelector('.skip-link').addEventListener('click', () => {
  window.setTimeout(() => document.querySelector('#main-content').focus(), 0);
});

window.addEventListener('hashchange', handleRoute);
refreshRuns();
refreshTraining();
handleRoute();
