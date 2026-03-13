const state = {
  selectedDate: null,
};

const ROUTE_LABELS = {
  'base_rule': '基础规则',
  'generic_makeup': '调休日路由',
  'holiday_family:Spring Festival': '春节路由',
  'ordinary_similar:weekend': '周末相似日',
  'ordinary_similar:workday': '工作日相似日',
};

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  return Number(value).toLocaleString('zh-CN', { maximumFractionDigits: digits });
}

function formatPercent(value) {
  return value === null || value === undefined ? '-' : `${formatNumber(value)}%`;
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value ?? '-';
}

function setRouteText(id, routeName) {
  const node = document.getElementById(id);
  if (!node) return;
  const label = formatRouteName(routeName);
  node.textContent = label;
  node.title = label;
}

function formatCopyValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
  return Number(value).toFixed(4);
}

function formatRouteName(routeName) {
  if (!routeName) return '-';
  return ROUTE_LABELS[routeName] || routeName;
}

function buildHorizontalCopyText(series) {
  return (series || []).map((item) => formatCopyValue(item?.value)).join('\t');
}

async function copyText(text) {
  if (!text) throw new Error('当前没有可复制的模型电量');

  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const helper = document.createElement('textarea');
  helper.value = text;
  helper.setAttribute('readonly', 'readonly');
  helper.style.position = 'fixed';
  helper.style.opacity = '0';
  document.body.appendChild(helper);
  helper.select();
  document.execCommand('copy');
  document.body.removeChild(helper);
}

let toastTimer = null;

function showToast(message, type = 'success') {
  let toast = document.getElementById('appToast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'appToast';
    toast.className = 'app-toast';
    document.body.appendChild(toast);
  }

  toast.textContent = message;
  toast.className = `app-toast show ${type === 'error' ? 'error' : 'success'}`;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toast.classList.remove('show');
  }, 1800);
}

function buildPath(points, width, height, padding, minValue, maxValue) {
  if (!points.length) return '';
  const range = maxValue - minValue || 1;
  return points.map((item, index) => {
    const x = padding + (index / Math.max(points.length - 1, 1)) * (width - padding * 2);
    const y = height - padding - ((item.value - minValue) / range) * (height - padding * 2);
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');
}

function renderChart(record) {
  const svg = document.getElementById('profileChart');
  const labels = document.getElementById('chartAxisLabels');
  if (!svg || !labels) return;

  const actualSeries = record.actual_series || [];
  const modelSeries = record.model_series || [];
  const fixedSeries = record.fixed_series || [];
  const allPoints = [...actualSeries, ...modelSeries, ...fixedSeries];
  const width = 960;
  const height = 360;
  const padding = 28;

  if (!allPoints.length) {
    svg.innerHTML = '';
    labels.innerHTML = '';
    return;
  }

  const maxValue = Math.max(...allPoints.map((item) => item.value));
  const minValue = Math.min(...allPoints.map((item) => item.value));
  const grid = [0, 1, 2, 3, 4].map((index) => {
    const y = padding + index * ((height - padding * 2) / 4);
    return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" class="grid-line" />`;
  }).join('');

  svg.innerHTML = `
    <g>${grid}</g>
    ${actualSeries.length ? `<path d="${buildPath(actualSeries, width, height, padding, minValue, maxValue)}" class="chart-path actual-path"></path>` : ''}
    ${modelSeries.length ? `<path d="${buildPath(modelSeries, width, height, padding, minValue, maxValue)}" class="chart-path model-path"></path>` : ''}
    ${fixedSeries.length ? `<path d="${buildPath(fixedSeries, width, height, padding, minValue, maxValue)}" class="chart-path fixed-path"></path>` : ''}
  `;
  labels.innerHTML = ['00:00', '06:00', '12:00', '18:00', '23:00'].map((label) => `<span>${label}</span>`).join('');
}

function renderMonthlyCards(monthly) {
  const container = document.getElementById('monthlyCards');
  if (!container) return;
  container.innerHTML = monthly.map((row) => `
    <button type="button" class="month-card-item table-link-block" data-target-date="${row.latest_target_date || ''}">
      <div class="month-card-head">
        <strong>${row.month}</strong>
        <span>${row.matched_days_count || 0}/${row.days_count} 天</span>
      </div>
      <div class="month-metric-row">
        <span>模型日总量准确率</span>
        <strong>${formatPercent(row.model_metrics?.day_total_accuracy)}</strong>
      </div>
      <div class="month-metric-row">
        <span>固定系数日总量准确率</span>
        <strong>${formatPercent(row.fixed_metrics?.day_total_accuracy)}</strong>
      </div>
      <div class="month-metric-row">
        <span>模型分时准确率</span>
        <strong>${formatPercent(row.model_metrics?.hourly_accuracy)}</strong>
      </div>
      <div class="month-metric-row">
        <span>固定系数分时准确率</span>
        <strong>${formatPercent(row.fixed_metrics?.hourly_accuracy)}</strong>
      </div>
    </button>
  `).join('');
}

function renderHistoryTable(rows) {
  const body = document.getElementById('recentTableBody');
  if (!body) return;
  body.innerHTML = rows.map((row) => `
    <tr class="${row.target_date === state.selectedDate ? 'active-row' : ''}">
      <td><button type="button" class="table-link" data-target-date="${row.target_date}">${row.target_date}</button></td>
      <td>${formatNumber(row.actual_total)}</td>
      <td>${formatNumber(row.model_total)}</td>
      <td>${formatNumber(row.fixed_total)}</td>
      <td>${formatPercent(row.model_metrics?.day_total_accuracy)}</td>
      <td>${formatPercent(row.fixed_metrics?.day_total_accuracy)}</td>
      <td class="truncate-cell" title="${formatRouteName(row.route_name)}">${formatRouteName(row.route_name)}</td>
    </tr>
  `).join('');
}

function renderHourlyTable(record) {
  const body = document.getElementById('hourlyTableBody');
  if (!body) return;

  const actualSeries = record.actual_series || [];
  const modelSeries = record.model_series || [];
  const fixedSeries = record.fixed_series || [];

  body.innerHTML = Array.from({ length: 24 }, (_, hour) => {
    const actual = actualSeries[hour]?.value;
    const model = modelSeries[hour]?.value;
    const fixed = fixedSeries[hour]?.value;
    return `
      <tr>
        <td>${String(hour).padStart(2, '0')}:00</td>
        <td>${formatNumber(actual, 4)}</td>
        <td>${formatNumber(model, 4)}</td>
        <td>${formatNumber(fixed, 4)}</td>
      </tr>
    `;
  }).join('');
}

function renderDateNavigation(navigation) {
  const input = document.getElementById('selectedDateInput');
  const prevButton = document.getElementById('prevDateButton');
  const nextButton = document.getElementById('nextDateButton');
  if (!input || !prevButton || !nextButton) return;

  input.value = state.selectedDate || '';
  input.min = navigation.min_date || '';
  input.max = navigation.max_date || '';
  prevButton.disabled = !navigation.prev_date;
  nextButton.disabled = !navigation.next_date;
  prevButton.dataset.targetDate = navigation.prev_date || '';
  nextButton.dataset.targetDate = navigation.next_date || '';
}

function renderCopyPanel(record) {
  const output = document.getElementById('modelCopyOutput');
  const copyButton = document.getElementById('copyModelRowButton');
  if (!output || !copyButton) return;

  const copyTextValue = buildHorizontalCopyText(record.model_series || []);
  output.value = copyTextValue;
  copyButton.disabled = !copyTextValue;
  setText('copyModelStatus', copyTextValue ? '已按 Excel 横向粘贴格式生成 24 个点' : '当前日期没有可复制的模型电量');
}

function renderDashboard(payload) {
  const record = payload.selected_record;
  const accuracy = payload.accuracy;
  state.selectedDate = payload.selected_date;

  setText('siteName', payload.site_name);
  setText('asOfActualDate', payload.max_actual_date);
  setText('currentTargetDate', payload.current_target_date);
  setText('nextExpectedDate', payload.next_upload_date);
  setText('forecastMode', record.forecast_mode || '-');
  setRouteText('routeName', record.route_name);
  setText('issueDate', record.issue_date || '-');
  setText('matchedDays', accuracy.matched_days);
  setText('heroTargetDate', record.target_date);
  setText('heroSummaryNote', `${record.has_actual ? '已匹配实际，可查看准确率和分时对比。' : '未来目标日，仅展示模型与固定系数预测。'} 当前路由：${formatRouteName(record.route_name)}`);
  setText('heroModelDayAccuracy', formatPercent(record.model_metrics?.day_total_accuracy));
  setText('heroFixedTotalHint', `固定系数日总量准确率：${formatPercent(record.fixed_metrics?.day_total_accuracy)}`);
  setText('modelDayAccuracy', formatPercent(record.model_metrics?.day_total_accuracy));
  setText('fixedDayAccuracy', formatPercent(record.fixed_metrics?.day_total_accuracy));
  setText('modelTotal', formatNumber(record.model_total));
  setText('fixedTotal', formatNumber(record.fixed_total));
  setText('historyStartText', `从 ${accuracy.history_start || '2026-01-01'} 开始`);
  setText('fixedFormulaText', payload.formula_label || '0.5*D-7 + 0.3*D-14 + 0.2*D-21');
  setText('selectedMonthHint', `当前月份 ${accuracy.selected_month?.month || '-'}，已核验 ${accuracy.selected_month?.matched_days_count || 0} 天 / 展示 ${accuracy.selected_month?.days_count || 0} 天`);
  setText('selectedDateHint', `当前查看 ${payload.selected_date}`);

  const actualDateInput = document.getElementById('actualDateInput');
  if (actualDateInput) actualDateInput.value = payload.next_upload_date;

  renderChart(record);
  renderCopyPanel(record);
  renderHourlyTable(record);
  renderMonthlyCards(accuracy.monthly || []);
  renderHistoryTable(payload.history || []);
  renderDateNavigation(payload.date_navigation || {});
}

async function loadDashboard(targetDate = null) {
  const url = new URL('/api/dashboard', window.location.origin);
  if (targetDate) {
    url.searchParams.set('target_date', targetDate);
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error('dashboard load failed');
  renderDashboard(await response.json());
}

async function submitUpload(event) {
  event.preventDefault();
  const status = document.getElementById('uploadStatus');
  const form = event.currentTarget;
  const formData = new FormData(form);
  if (status) status.textContent = '正在更新历史并生成下一次预测...';

  const response = await fetch('/api/actual-upload', {
    method: 'POST',
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || 'upload failed');
  }

  renderDashboard(payload.dashboard);
  form.reset();
  const actualDateInput = document.getElementById('actualDateInput');
  if (actualDateInput) actualDateInput.value = payload.dashboard.next_upload_date;
  setText('uploadStatus', `已更新 ${payload.actual_date}，下一目标日 ${payload.next_target_date}`);
}

function bindNavigation() {
  document.addEventListener('click', async (event) => {
    const target = event.target.closest('[data-target-date]');
    if (!target || !target.dataset.targetDate) return;
    try {
      await loadDashboard(target.dataset.targetDate);
    } catch (error) {
      setText('uploadStatus', error.message);
    }
  });

  const selectedDateInput = document.getElementById('selectedDateInput');
  if (selectedDateInput) {
    selectedDateInput.addEventListener('change', async (event) => {
      if (!event.target.value) return;
      try {
        await loadDashboard(event.target.value);
      } catch (error) {
        setText('uploadStatus', error.message);
      }
    });
  }

  const copyButton = document.getElementById('copyModelRowButton');
  if (copyButton) {
    copyButton.addEventListener('click', async () => {
      try {
        await copyText(document.getElementById('modelCopyOutput')?.value || '');
        setText('copyModelStatus', '已复制模型电量，可直接横向粘贴到 Excel');
        showToast('复制成功');
      } catch (error) {
        setText('copyModelStatus', error.message || '复制失败');
        showToast(error.message || '复制失败', 'error');
      }
    });
  }

  const copyOutput = document.getElementById('modelCopyOutput');
  if (copyOutput) {
    copyOutput.addEventListener('focus', () => {
      copyOutput.select();
    });
    copyOutput.addEventListener('click', () => {
      copyOutput.select();
    });
  }
}

async function boot() {
  try {
    await loadDashboard();
  } catch (error) {
    setText('uploadStatus', error.message);
  }

  bindNavigation();

  const form = document.getElementById('actualUploadForm');
  if (form) {
    form.addEventListener('submit', async (event) => {
      try {
        await submitUpload(event);
      } catch (error) {
        setText('uploadStatus', error.message);
      }
    });
  }
}

boot();
