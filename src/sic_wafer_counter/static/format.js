/** Small display helpers. No scientific values are invented for missing data. */

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
  })[character]);
}

export function finiteNumber(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function formatNumber(value, options = {}) {
  const number = finiteNumber(value);
  if (number === null) return '未提供';
  return new Intl.NumberFormat('zh-CN', options).format(number);
}

export function formatScientific(value, digits = 4) {
  const number = finiteNumber(value);
  return number === null ? '未提供' : number.toExponential(digits);
}

export function formatDate(value) {
  if (!value) return '未提供';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).format(date);
}

export function datasetLabel(kind) {
  return ({synthetic: '合成演示', real_exploratory: '真实样品探索', failed: '失败运行'})[kind] || '未分类';
}

export function statusMarkup(status) {
  const normalized = String(status || '').toLowerCase();
  if (normalized === 'completed') return '<span class="status-badge success">已完成</span>';
  if (normalized === 'failed') return '<span class="status-badge error">已拒绝输出</span>';
  if (normalized === 'running') return '<span class="status-badge neutral">分析中</span>';
  return `<span class="status-badge neutral">${escapeHtml(status || '未知')}</span>`;
}

export function unvalidated(value) {
  const status = String(value || '').toLowerCase();
  return !status || status.includes('not validated') || status.includes('未验证');
}
