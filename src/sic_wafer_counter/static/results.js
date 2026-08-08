import {
  datasetLabel, escapeHtml, finiteNumber, formatDate, formatNumber,
  formatScientific, statusMarkup, unvalidated,
} from './format.js';

const ARTIFACT_LABELS = {
  'report.html': '完整 HTML 报告',
  'summary.json': '摘要 JSON',
  'summary.csv': '摘要 CSV',
  'defects_all.csv': '全部候选 CSV',
  'defects_accepted.csv': '接受目标 CSV',
  'defects_rejected.csv': '拒绝目标 CSV',
  'analysis_config.yaml': '本次配置 YAML',
  'resolved_physical_parameters.yaml': '物理参数换算 YAML',
  'radial_density.csv': '径向密度 CSV',
  'angular_density.csv': '方位角密度 CSV',
  'regional_density.csv': '区域密度 CSV',
  'density_heatmap_grid.csv': '二维密度逐格审计 CSV',
  'independent_reference_points.csv': '独立参考登记审计 CSV',
  'independent_reference_matches.csv': '自动候选与独立参考匹配 CSV',
  'run.log': '运行日志',
};

const IMAGE_LABELS = {
  'overlay_xrt_red_boxes.png': '论文语义对齐图（自动红框；黄圈仅在独立参考已核验时出现）',
  'xrt_detection_detail_montage.png': '论文风格局部视场（自动红框；独立参考状态见图内说明）',
  'paper_detection_field.png': '单视场论文语义对照图',
  'paper_aligned_result_figure.png': '论文式点状目标与整片密度综合成果',
  'defect_comparison_details.png': '原图与自动判定复核（非 DIC/KOH 验证）',
  'overlay_accepted.png': '接受目标编号叠加图',
  'overlay_all_candidates.png': '全部候选叠加图',
  'density_heatmap.png': '实际有效面积归一化密度热图',
  'defect_size_histogram.png': '目标尺寸分布',
  'valid_analysis_mask.png': '最终有效分析掩膜',
  'wafer_mask.png': '完整晶圆掩膜',
  'preprocessed_preview.png': '暗目标响应预览',
  'candidate_mask.png': '候选二值掩膜',
};

function pick(object, ...paths) {
  for (const path of paths) {
    let current = object;
    let found = true;
    for (const part of path.split('.')) {
      if (!current || typeof current !== 'object' || !(part in current)) {
        found = false;
        break;
      }
      current = current[part];
    }
    if (found && current !== null && current !== undefined) return current;
  }
  return null;
}

function numeric(value, suffix = '', options = {maximumFractionDigits: 6}) {
  const parsed = finiteNumber(value);
  return parsed === null ? '未提供' : `${formatNumber(parsed, options)}${suffix}`;
}

function validationBadge(value) {
  if (unvalidated(value)) return '<span class="status-badge warning">未在真实 SiC 专家标注上验证</span>';
  return `<span class="status-badge success">${escapeHtml(value)}</span>`;
}

function runRows(runs, limit = null) {
  const selected = limit ? runs.slice(0, limit) : runs;
  if (!selected.length) return `<tr><td colspan="${limit ? 7 : 8}" class="muted-cell">尚无可用结果。可运行合成演示或上传一张完整晶圆图。</td></tr>`;
  return selected.map((run) => `<tr>
    <td><a class="run-link" href="#run/${encodeURIComponent(run.run_id)}">${escapeHtml(run.input_file_name || run.run_id)}</a><span class="subtext">${escapeHtml(run.run_id)}${limit ? '' : ` · ${escapeHtml(formatDate(run.generated_at_utc))}`}</span></td>
    <td>${escapeHtml(datasetLabel(run.dataset_kind))}</td>
    <td>${statusMarkup(run.status)}</td>
    <td>${formatNumber(run.accepted_count, {maximumFractionDigits: 0})}</td>
    <td>${numeric(run.valid_analysis_area_cm2, '', {minimumFractionDigits: 3, maximumFractionDigits: 6})}</td>
    <td>${formatScientific(run.point_density_cm2, 3)}</td>
    ${limit ? `<td>${formatDate(run.generated_at_utc)}</td>` : `<td>${unvalidated(run.real_annotation_validation_status) ? '<span class="status-badge warning">未验证</span>' : validationBadge(run.real_annotation_validation_status)}</td><td>${numeric(run.runtime_seconds, ' s', {maximumFractionDigits: 1})}</td>`}
  </tr>`).join('');
}

export function renderRunIndex(payload) {
  const runs = Array.isArray(payload.runs) ? payload.runs : [];
  document.querySelector('#recent-runs-body').innerHTML = runRows(runs, 6);
  document.querySelector('#history-body').innerHTML = runRows(runs);
  const completed = runs.filter((run) => run.status === 'completed').length;
  const failed = runs.filter((run) => run.status === 'failed').length;
  const synthetic = runs.filter((run) => run.dataset_kind === 'synthetic').length;
  document.querySelector('#history-summary').textContent = `共 ${runs.length} 条合法记录 · 完成 ${completed} · 失败 ${failed} · 合成演示 ${synthetic}`;
  const diagnostics = document.querySelector('#history-diagnostics');
  const skipped = Array.isArray(payload.skipped_invalid_summaries) ? payload.skipped_invalid_summaries : [];
  diagnostics.hidden = skipped.length === 0;
  diagnostics.textContent = skipped.length ? `已隔离 ${skipped.length} 个损坏或不兼容的 summary.json；其他记录未受影响。` : '';
  return runs;
}

export function renderRunIndexError(message) {
  const row = `<tr><td colspan="8" class="muted-cell">读取失败：${escapeHtml(message)}</td></tr>`;
  document.querySelector('#recent-runs-body').innerHTML = row;
  document.querySelector('#history-body').innerHTML = row;
  document.querySelector('#history-summary').textContent = '无法读取本机结果记录';
}

export function renderLatestPlaceholder(message) {
  const container = document.querySelector('#latest-run');
  container.className = 'empty-state';
  container.removeAttribute('aria-busy');
  container.textContent = message;
}

export function renderLatestRun(detail) {
  const container = document.querySelector('#latest-run');
  const run = detail;
  const s = detail.summary || {};
  const overlay = detail.artifacts?.['overlay_xrt_red_boxes.png']
    || detail.artifacts?.['overlay_accepted.png'];
  const preview = overlay
    ? `<a class="latest-preview" href="#run/${encodeURIComponent(run.run_id)}"><img src="${escapeHtml(overlay)}" alt="${escapeHtml(run.input_file_name)} 的接受目标叠加图" width="640" height="480" decoding="async" fetchpriority="high"></a>`
    : '<div class="latest-preview"><span class="status-badge neutral">没有叠加图</span></div>';
  container.className = 'latest-content';
  container.removeAttribute('aria-busy');
  container.innerHTML = `${preview}<div class="latest-info">
    <div>${statusMarkup(run.status)} <span class="status-badge neutral">${escapeHtml(datasetLabel(run.dataset_kind))}</span></div>
    <h3><a class="run-link" href="#run/${encodeURIComponent(run.run_id)}">${escapeHtml(run.input_file_name || run.run_id)}</a></h3>
    <span class="subtext">${escapeHtml(run.run_id)} · ${escapeHtml(formatDate(run.generated_at_utc))}</span>
    <div class="metric-mini-grid">
      <div class="metric-mini"><small>接受 n</small><strong>${formatNumber(s.accepted_count, {maximumFractionDigits: 0})}</strong></div>
      <div class="metric-mini"><small>有效 S (cm²)</small><strong>${numeric(s.valid_analysis_area_cm2, '', {maximumFractionDigits: 3})}</strong></div>
      <div class="metric-mini"><small>ρ (cm⁻²)</small><strong>${formatScientific(s.point_density_cm2, 2)}</strong></div>
    </div>
    ${validationBadge(s.real_annotation_validation_status)}
  </div>`;
}

function metric(label, value, note = '') {
  return `<div class="metric-card"><span>${escapeHtml(label)}</span><strong>${value}</strong><small>${escapeHtml(note)}</small></div>`;
}

function metadataRows(rows) {
  return rows.map(([label, value]) => `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`).join('');
}

function metadataSection(title, rows) {
  return `<section class="metadata-section"><h3>${escapeHtml(title)}</h3><dl class="metadata-list">${metadataRows(rows)}</dl></section>`;
}

function boolLabel(value) {
  if (value === true) return '是';
  if (value === false) return '否';
  return '未提供';
}

function arrayPair(value, suffix = '') {
  if (!Array.isArray(value) || value.length < 2) return '未提供';
  return `${numeric(value[0], '', {maximumFractionDigits: 3})}, ${numeric(value[1], '', {maximumFractionDigits: 3})}${suffix}`;
}

function renderMetadata(s) {
  const imageSize = Array.isArray(s.image_size) ? `${s.image_size[0]} × ${s.image_size[1]} px` : '未提供';
  const center = Array.isArray(s.wafer_center_px)
    ? arrayPair(s.wafer_center_px, ' px')
    : `${numeric(s.center_x_px, '', {maximumFractionDigits: 3})}, ${numeric(s.center_y_px, '', {maximumFractionDigits: 3})} px`;
  const sections = [
    ['灰度与图像', [
      ['源数据类型', String(s.source_dtype ?? s.image_dtype ?? '未提供')],
      ['科研分析类型', String(s.analysis_dtype ?? '未提供')],
      ['图像尺寸', imageSize],
      ['归一化低值', numeric(s.normalization_low_value, '', {maximumFractionDigits: 6})],
      ['归一化高值', numeric(s.normalization_high_value, '', {maximumFractionDigits: 6})],
      ['低端裁剪比例', numeric(s.low_clipped_fraction, '', {maximumFractionDigits: 6})],
      ['高端裁剪比例', numeric(s.high_clipped_fraction, '', {maximumFractionDigits: 6})],
      ['WhiteIsZero', boolLabel(s.white_is_zero)],
      ['检测前量化为 uint8', boolLabel(s.analysis_quantized_to_uint8)],
    ]],
    ['几何与标定', [
      ['实际直径', numeric(s.wafer_diameter_mm, ' mm')],
      ['拟合圆心', center],
      ['拟合半径', numeric(s.wafer_radius_px, ' px', {maximumFractionDigits: 3})],
      ['像素尺度', numeric(s.mm_per_pixel, ' mm/px', {maximumFractionDigits: 9})],
      ['微米尺度', numeric(s.um_per_pixel, ' µm/px', {maximumFractionDigits: 5})],
      ['几何可信度', numeric(pick(s, 'wafer_detection.confidence'), '', {maximumFractionDigits: 3})],
      ['晶圆轮廓圆度', numeric(pick(s, 'wafer_detection.circularity'), '', {maximumFractionDigits: 3})],
      ['拟合残差 / 半径', numeric(pick(s, 'wafer_detection.fit_residual_fraction_radius'), '', {maximumFractionDigits: 6})],
    ]],
    ['面积与候选', [
      ['理论完整圆面积', numeric(s.theoretical_area_cm2, ' cm²')],
      ['拟合圆面积', numeric(s.fitted_wafer_area_cm2, ' cm²')],
      ['像素掩膜完整面积', numeric(s.pixel_mask_full_wafer_area_cm2, ' cm²')],
      ['最终有效面积', numeric(s.valid_analysis_area_cm2, ' cm²')],
      ['边缘排除面积', numeric(s.edge_excluded_area_cm2, ' cm²')],
      ['其他无效面积', numeric(s.other_invalid_area_cm2, ' cm²')],
      ['原始候选数', numeric(s.raw_candidate_count, '', {maximumFractionDigits: 0})],
      ['分水岭后候选数', numeric(s.post_watershed_candidate_count, '', {maximumFractionDigits: 0})],
      ['拒绝数', numeric(s.rejected_count, '', {maximumFractionDigits: 0})],
    ]],
    ['运行', [
      ['处理模式', String(s.processing_mode ?? '未提供')],
      ['检测阈值', numeric(s.detection_threshold_value, '', {maximumFractionDigits: 6})],
      ['处理 tile 数', numeric(s.processed_tile_count, '', {maximumFractionDigits: 0})],
      ['运行耗时', numeric(s.runtime_seconds, ' s', {maximumFractionDigits: 2})],
      ['软件版本', String(s.software_version ?? '未提供')],
      ['真实标注验证', String(s.real_annotation_validation_status ?? 'not validated on real SiC data')],
    ]],
  ];
  return sections.map(([title, rows]) => metadataSection(title, rows)).join('');
}

export function renderRunDetail(detail) {
  const s = detail.summary || {};
  const title = detail.input_file_name && detail.input_file_name !== '—' ? detail.input_file_name : detail.run_id;
  document.querySelector('#result-title').textContent = `${title} · ${datasetLabel(detail.dataset_kind)}`;
  document.querySelector('#result-subtitle').textContent = `${detail.run_id} · ${formatDate(detail.generated_at_utc)} · ${detail.status}`;
  const validation = s.real_annotation_validation_status || detail.real_annotation_validation_status;
  const badge = document.querySelector('#result-validation-badge');
  badge.className = `status-badge ${unvalidated(validation) ? 'warning' : 'success'}`;
  badge.textContent = unvalidated(validation) ? '未在真实 SiC 专家标注上验证' : String(validation);

  const ciLow = finiteNumber(s.poisson_95_ci_lower_cm2);
  const ciHigh = finiteNumber(s.poisson_95_ci_upper_cm2);
  const ci = ciLow === null || ciHigh === null ? '未提供' : `[${formatScientific(ciLow, 3)}, ${formatScientific(ciHigh, 3)}]`;
  document.querySelector('#result-metrics').innerHTML = [
    metric('接受点状目标 n', formatNumber(s.accepted_count, {maximumFractionDigits: 0}), '当前图像规则'),
    metric('有效分析面积 S', `${numeric(s.valid_analysis_area_cm2, '', {maximumFractionDigits: 6})} cm²`, '来自最终有效掩膜'),
    metric('点状目标密度 ρ', `${formatScientific(s.point_density_cm2, 4)} cm⁻²`, 'ρ = n / S'),
    metric('计数不确定度 1σ', `± ${formatScientific(s.counting_uncertainty_cm2, 4)} cm⁻²`, '只含有限计数波动'),
    metric('Garwood 95% 区间', `${ci} cm⁻²`, '不含分类与标定系统误差'),
  ].join('');

  const failed = String(detail.status).toLowerCase() === 'failed';
  const error = document.querySelector('#result-error');
  const main = document.querySelector('#result-main');
  error.hidden = !failed;
  main.hidden = failed;
  if (failed) {
    const warnings = Array.isArray(s.warnings) ? s.warnings : [];
    error.innerHTML = `<strong>该运行拒绝输出密度。</strong><p>${escapeHtml(warnings[0] || '晶圆几何或输入数据未通过可靠性检查。')}</p>`;
  }

  document.querySelector('#result-metadata').innerHTML = renderMetadata(s);
  const warnings = Array.isArray(s.warnings) ? s.warnings : [];
  const warningBox = document.querySelector('#result-warnings');
  warningBox.hidden = warnings.length === 0;
  warningBox.innerHTML = warnings.length ? `<strong>运行警告（${warnings.length}）</strong><ul>${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join('')}</ul>` : '';

  const artifacts = detail.artifacts || {};
  const select = document.querySelector('#image-artifact');
  const imageOptions = Object.entries(IMAGE_LABELS).filter(([name]) => artifacts[name]);
  select.innerHTML = imageOptions.map(([name, label]) => `<option value="${escapeHtml(name)}">${escapeHtml(label)}</option>`).join('');
  const preferred = imageOptions.find(([name]) => name === 'paper_aligned_result_figure.png')
    || imageOptions.find(([name]) => name === 'xrt_detection_detail_montage.png')
    || imageOptions.find(([name]) => name === 'overlay_xrt_red_boxes.png')
    || imageOptions.find(([name]) => name === 'overlay_accepted.png')
    || imageOptions[0];
  select.disabled = imageOptions.length < 2;
  if (preferred) select.value = preferred[0];
  updateResultImage(detail, select.value);

  const linkEntries = Object.entries(ARTIFACT_LABELS).filter(([name]) => artifacts[name]);
  const artifactPanel = document.querySelector('#artifact-panel');
  artifactPanel.hidden = linkEntries.length === 0;
  document.querySelector('#artifact-links').innerHTML = linkEntries.map(([name, label]) => {
    const download = name === 'report.html' ? '' : '?download=1';
    return `<a href="${escapeHtml(artifacts[name])}${download}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`;
  }).join('');

  const candidateBrowser = document.querySelector('#candidate-browser');
  candidateBrowser.hidden = !artifacts['defects_all.csv'] || failed;
  const figures = [['radial_density.png', '径向密度'], ['angular_density.png', '方位角密度'], ['density_heatmap.png', '二维密度热图']]
    .filter(([name]) => artifacts[name]);
  const spatialPanel = document.querySelector('#spatial-panel');
  spatialPanel.hidden = figures.length === 0 || failed;
  document.querySelector('#spatial-figures').innerHTML = figures.map(([name, label]) => `<figure><a href="${escapeHtml(artifacts[name])}" target="_blank" rel="noopener"><img loading="lazy" decoding="async" src="${escapeHtml(artifacts[name])}" alt="${escapeHtml(label)}" width="800" height="560"></a><figcaption>${escapeHtml(label)}</figcaption></figure>`).join('');
}

export function updateResultImage(detail, name) {
  const image = document.querySelector('#result-image');
  const missing = document.querySelector('#image-missing');
  const caption = document.querySelector('#image-caption');
  const url = detail?.artifacts?.[name];
  image.hidden = !url;
  missing.hidden = Boolean(url);
  if (url) {
    image.src = url;
    image.alt = `${detail.input_file_name || detail.run_id} 的${IMAGE_LABELS[name] || '分析图像'}`;
    caption.textContent = IMAGE_LABELS[name] || name;
  } else {
    image.removeAttribute('src');
    image.alt = '';
    caption.textContent = '图像未保存';
  }
}

function candidateStatus(row) {
  return row.accepted
    ? '<span class="accepted-text">已接受</span>'
    : '<span class="rejected-text">已拒绝</span>';
}

function candidateValues(row) {
  const diameter = finiteNumber(row.equivalent_diameter_mm);
  const boundary = finiteNumber(row.distance_to_valid_boundary_mm);
  return {
    id: row.defect_id ?? '—',
    x: numeric(row.x_mm, '', {maximumFractionDigits: 4}),
    y: numeric(row.y_mm, '', {maximumFractionDigits: 4}),
    diameter: diameter === null ? '未提供' : `${formatNumber(diameter * 1000, {maximumFractionDigits: 2})} µm`,
    circularity: numeric(row.circularity, '', {maximumFractionDigits: 3}),
    contrast: numeric(row.contrast, '', {maximumFractionDigits: 4}),
    boundary: boundary === null ? '未提供' : `${formatNumber(boundary * 1000, {maximumFractionDigits: 1})} µm`,
    reason: row.rejection_reason || '—',
  };
}

export function renderCandidatePage(payload) {
  const rows = Array.isArray(payload.rows) ? payload.rows : [];
  const body = document.querySelector('#candidate-body');
  const cards = document.querySelector('#candidate-cards');
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="10" class="muted-cell">当前筛选条件下没有候选。</td></tr>';
    cards.innerHTML = '<p class="empty-state">当前筛选条件下没有候选。</p>';
  } else {
    body.innerHTML = rows.map((row) => {
      const value = candidateValues(row);
      const thumbnail = row.crop_preview_url
        ? `<a href="${escapeHtml(row.crop_url || row.crop_preview_url)}" target="_blank" rel="noopener"><img loading="lazy" decoding="async" src="${escapeHtml(row.crop_preview_url)}" alt="候选 ${escapeHtml(value.id)} 局部裁剪" width="40" height="40"></a>`
        : '<span aria-label="无裁剪图">—</span>';
      return `<tr><td>${escapeHtml(value.id)}</td><td>${thumbnail}</td><td>${value.x}</td><td>${value.y}</td><td>${value.diameter}</td><td>${value.circularity}</td><td>${value.contrast}</td><td>${value.boundary}</td><td>${candidateStatus(row)}</td><td>${escapeHtml(value.reason)}</td></tr>`;
    }).join('');
    cards.innerHTML = rows.map((row) => {
      const value = candidateValues(row);
      const thumbnail = row.crop_preview_url
        ? `<a href="${escapeHtml(row.crop_url || row.crop_preview_url)}" target="_blank" rel="noopener"><img loading="lazy" decoding="async" src="${escapeHtml(row.crop_preview_url)}" alt="候选 ${escapeHtml(value.id)} 局部裁剪" width="52" height="52"></a>`
        : '<div aria-label="无裁剪图"></div>';
      return `<article class="candidate-card">${thumbnail}<div><strong>#${escapeHtml(value.id)} · ${candidateStatus(row)}</strong><dl><div><dt>x / y</dt><dd>${value.x} / ${value.y} mm</dd></div><div><dt>直径</dt><dd>${value.diameter}</dd></div><div><dt>圆度</dt><dd>${value.circularity}</dd></div><div><dt>距边界</dt><dd>${value.boundary}</dd></div></dl><span class="subtext">${escapeHtml(value.reason)}</span></div></article>`;
    }).join('');
  }
  document.querySelector('#candidate-total').textContent = `筛选后 ${payload.total} 条`;
  document.querySelector('#candidate-page-status').textContent = `第 ${payload.page} / ${payload.total_pages} 页`;
  document.querySelector('#candidate-first').disabled = payload.page <= 1;
  document.querySelector('#candidate-prev').disabled = payload.page <= 1;
  document.querySelector('#candidate-next').disabled = payload.page >= payload.total_pages;
  document.querySelector('#candidate-last').disabled = payload.page >= payload.total_pages;
  const reasonSelect = document.querySelector('#candidate-reason');
  const selected = reasonSelect.value;
  const reasons = Object.keys(payload.reason_counts || {});
  reasonSelect.innerHTML = '<option value="">全部原因</option>' + reasons.map((reason) => `<option value="${escapeHtml(reason)}">${escapeHtml(reason)} (${payload.reason_counts[reason]})</option>`).join('');
  if (reasons.includes(selected)) reasonSelect.value = selected;
}

export function renderCandidateError(message) {
  document.querySelector('#candidate-body').innerHTML = `<tr><td colspan="10" class="muted-cell">候选读取失败：${escapeHtml(message)}</td></tr>`;
  document.querySelector('#candidate-cards').innerHTML = `<p class="empty-state">候选读取失败：${escapeHtml(message)}</p>`;
}
