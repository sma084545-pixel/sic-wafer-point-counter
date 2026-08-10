/** Browser API client for the local-only scientific workbench. */

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {cache: 'no-store', ...options});
  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError('本机服务返回了无法解析的响应', response.status);
  }
  if (!response.ok) {
    throw new ApiError(data.error || `请求失败（HTTP ${response.status}）`, response.status);
  }
  return data;
}

export const api = {
  listRuns: () => requestJson('/api/runs'),
  getRun: (runId) => requestJson(`/api/runs/${encodeURIComponent(runId)}`),
  getDefects(runId, filters = {}) {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && String(value) !== '') params.set(key, String(value));
    });
    return requestJson(`/api/runs/${encodeURIComponent(runId)}/defects?${params}`);
  },
  getTraining: () => requestJson('/api/training'),
  getPixelTraining: () => requestJson('/api/pixel-training/projects'),
  saveTrainingLabel: (payload) => requestJson('/api/training/labels', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  }),
  trainClassifier: (payload) => requestJson('/api/training/train', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  }),
  submitAnalysis: (formData) => requestJson('/api/jobs', {method: 'POST', body: formData}),
  submitDemo: (kind) => requestJson(`/api/demo/${encodeURIComponent(kind)}`, {method: 'POST'}),
  getJob: (jobId) => requestJson(`/api/jobs/${encodeURIComponent(jobId)}`),
};
