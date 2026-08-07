import {api} from './api.js';
import {
  renderCandidateError, renderCandidatePage, renderLatestPlaceholder,
  renderLatestRun, renderRunDetail, renderRunIndex, renderRunIndexError,
  updateResultImage,
} from './results.js';

const state = {
  runs: [],
  currentRun: null,
  currentRunToken: 0,
  candidates: {page: 1, totalPages: 1},
  jobActive: false,
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
  const allowed = ['overview', 'analyze', 'runs', 'result', 'methods'];
  const target = allowed.includes(name) ? name : 'overview';
  document.querySelectorAll('[data-view]').forEach((view) => {
    view.hidden = view.dataset.view !== target;
  });
  setActiveNavigation(target);
  const labels = {overview: '总览', analyze: '新建分析', runs: '结果记录', result: '分析结果', methods: '方法与验证'};
  document.title = `${labels[target]} | SiC 晶圆点状目标分析平台`;
  elements.sidebar.classList.remove('is-open');
  elements.menuButton.setAttribute('aria-expanded', 'false');
  window.scrollTo({top: 0, behavior: 'auto'});
  return target;
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
  document.querySelector('#candidate-body').innerHTML = '<tr><td colspan="10" class="muted-cell">正在读取当前页…</td></tr>';
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
  if (!['overview', 'analyze', 'runs', 'methods'].includes(route.view)) {
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
document.querySelector('#candidate-first').addEventListener('click', () => loadCandidates(1));
document.querySelector('#candidate-prev').addEventListener('click', () => loadCandidates(Math.max(1, state.candidates.page - 1)));
document.querySelector('#candidate-next').addEventListener('click', () => loadCandidates(Math.min(state.candidates.totalPages, state.candidates.page + 1)));
document.querySelector('#candidate-last').addEventListener('click', () => loadCandidates(state.candidates.totalPages));
document.querySelector('#image-artifact').addEventListener('change', (event) => updateResultImage(state.currentRun, event.target.value));

document.querySelector('#refresh-runs').addEventListener('click', () => refreshRuns({announceResult: true}));
document.querySelector('#refresh-history').addEventListener('click', () => refreshRuns({announceResult: true}));
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
handleRoute();
