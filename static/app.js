const state = {
  selectedDate: null,
  loadChartRecord: null,
  accuracyTrend: [],
  monthlyRows: [],
};

const SOURCE_LABELS = {
  manual: "手工录入",
  "manual-web": "网页录入",
  "manual-ems": "EMS回填",
  api: "接口上传",
  "api-demo": "接口样例",
  "ems-demo": "EMS样例",
  "dispatch-center": "调度中心",
  ems: "EMS系统",
};

function buildSampleValues(base, swing, workBoost) {
  return Array.from({ length: 24 }, (_, hour) =>
    Math.round(base + Math.sin(hour / 24 * Math.PI * 2) * swing + (hour >= 8 && hour <= 19 ? workBoost : 0)),
  );
}

function formatNumber(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${Number(value).toLocaleString("zh-CN", {
    maximumFractionDigits: 2,
  })}${suffix}`;
}

function formatPercent(value) {
  return value === null || value === undefined ? "-" : `${formatNumber(value)}%`;
}

function getAccuracyRate(metrics) {
  if (!metrics || metrics.mape === undefined || metrics.mape === null) {
    return null;
  }
  return Math.max(0, roundToTwo(100 - Number(metrics.mape)));
}

function roundToTwo(value) {
  return Math.round(value * 100) / 100;
}

function formatDate(value) {
  if (!value) return "-";
  return new Date(value).toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatLocalDate(value) {
  const year = value.getFullYear();
  const month = String(value.getMonth() + 1).padStart(2, "0");
  const day = String(value.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function formatSourceLabel(source) {
  if (!source) return "-";
  return SOURCE_LABELS[source] || source;
}

function formatStatusLabel(status) {
  if (status === "complete") return "预测/实际齐全";
  if (status === "forecast_only") return "仅有负荷预测";
  if (status === "actual_only") return "仅有负荷实际";
  return "暂无数据";
}

function buildStatusMeta(record, selectedDate) {
  if (record.data_status === "complete") {
    return `${selectedDate} 已接收对应负荷预测与负荷实际`;
  }
  if (record.data_status === "forecast_only") {
    return `${selectedDate} 已接收负荷预测，等待负荷实际回填`;
  }
  if (record.data_status === "actual_only") {
    return `${selectedDate} 仅有负荷实际数据`;
  }
  return `${selectedDate} 暂无负荷预测或负荷实际数据`;
}

function buildHistorySource(item) {
  const sources = [item.forecast_source, item.actual_source].filter(Boolean);
  if (!sources.length) return "-";
  return Array.from(new Set(sources.map((source) => formatSourceLabel(source)))).join(" / ");
}

function buildHistoryCaption({ month = null, fallback = false } = {}) {
  if (fallback) {
    return "最近记录";
  }

  if (month) {
    return month;
  }

  return "全部记录";
}

function buildHeroMeta(record, monthSummary) {
  const parts = [];

  if (monthSummary.month) {
    parts.push(`统计月份 ${monthSummary.month}`);
  }

  if (monthSummary.days_count) {
    parts.push(`样本 ${monthSummary.days_count} 天`);
  }

  if (record.data_status) {
    parts.push(formatStatusLabel(record.data_status));
  }

  return parts.length ? parts.join(" · ") : "按目标日查看负荷预测、实际与准确率";
}

function buildCurlExample(endpoint, payload) {
  const origin = window.location.origin || "";
  const body = JSON.stringify(payload || {}, null, 2);

  return [
    `curl -X POST '${origin}${endpoint}' \\`,
    "  -H 'Content-Type: application/json' \\",
    "  # 如启用鉴权，取消下一行注释",
    "  # -H 'X-API-Key: your-token' \\",
    "  --data @- <<'JSON'",
    body,
    "JSON",
  ].join("\n");
}

function normalizePasteText(raw) {
  return raw
    .replace(/\r/g, "")
    .replace(/[，、；]/g, ",")
    .replace(/[“”]/g, '"')
    .trim();
}

function isTimeOnlyCell(value) {
  return /^([01]?\d|2[0-3])(?::\d{2})?(?:\s*[-~至到]\s*(?:[01]?\d|2[0-3])(?::\d{2})?)?$/.test(value)
    || /^([01]?\d|2[0-3])时$/.test(value);
}

function isDateLikeCell(value) {
  return /^\d{4}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?$/.test(value);
}

function parseNumberFromCell(cell) {
  const value = cell.trim().replace(/\s+/g, " ");
  if (!value || isTimeOnlyCell(value) || isDateLikeCell(value)) {
    return null;
  }

  const numbers = value.match(/-?\d+(?:\.\d+)?/g);
  if (!numbers) {
    return null;
  }

  if (numbers.length === 1) {
    return Number(numbers[0]);
  }

  const firstNumber = Number(numbers[0]);
  if (
    /:/.test(value)
    || /时|点|hour|load|负荷|预测|实绩|实际/i.test(value)
    || (Number.isInteger(firstNumber) && firstNumber >= 0 && firstNumber <= 23)
  ) {
    return Number(numbers[numbers.length - 1]);
  }

  return null;
}

function pairSequenceValues(numbers) {
  if (numbers.length < 48) {
    return null;
  }

  for (let start = 0; start <= numbers.length - 48; start += 1) {
    const windowValues = numbers.slice(start, start + 48);
    const hours = windowValues.filter((_, index) => index % 2 === 0);
    const looksLikeHours = hours.every((value) => Number.isInteger(value) && value >= 0 && value <= 23);
    if (!looksLikeHours) {
      continue;
    }
    return windowValues.filter((_, index) => index % 2 === 1).slice(0, 24);
  }

  return null;
}

function recognizeHourlyValues(raw) {
  const normalized = normalizePasteText(raw);
  if (!normalized) {
    return {
      values: [],
      complete: false,
      message: "请粘贴 24 个小时值",
    };
  }

  const cells = normalized.split(/[\t\n,]+/).map((item) => item.trim()).filter(Boolean);
  const cellValues = cells.map(parseNumberFromCell).filter((value) => value !== null);
  if (cellValues.length === 24) {
    return {
      values: cellValues,
      complete: true,
      message: "已识别 24 个值，可直接提交",
    };
  }

  const lines = normalized.split("\n").map((item) => item.trim()).filter(Boolean);
  const lineValues = lines.map(parseNumberFromCell).filter((value) => value !== null);
  if (lineValues.length === 24) {
    return {
      values: lineValues,
      complete: true,
      message: "已按逐行文本识别 24 个值，可直接提交",
    };
  }

  const allNumbers = (normalized.match(/-?\d+(?:\.\d+)?/g) || []).map(Number);
  const pairedValues = pairSequenceValues(allNumbers);
  if (pairedValues) {
    return {
      values: pairedValues,
      complete: true,
      message: "已从时段和值序列中识别 24 个值，可直接提交",
    };
  }

  if (allNumbers.length === 24) {
    return {
      values: allNumbers,
      complete: true,
      message: "已识别 24 个值，可直接提交",
    };
  }

  return {
    values: cellValues.length ? cellValues : lineValues.length ? lineValues : allNumbers,
    complete: false,
    message: `当前识别到 ${cellValues.length || lineValues.length || allNumbers.length} 个数字，需要正好 24 个`,
  };
}

function renderPreview(containerId, values) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  for (let hour = 0; hour < 24; hour += 1) {
    const item = document.createElement("div");
    item.className = `hour-item ${values[hour] === undefined ? "pending" : ""}`;
    item.innerHTML = `
      <span>${String(hour).padStart(2, "0")}:00</span>
      <strong>${values[hour] === undefined ? "--" : formatNumber(values[hour])}</strong>
    `;
    container.appendChild(item);
  }
}

function updatePastePreview({ rawInputId, previewId, parseStatusId }) {
  const rawInput = document.getElementById(rawInputId);
  const parseStatus = document.getElementById(parseStatusId);
  const result = recognizeHourlyValues(rawInput.value);

  renderPreview(previewId, result.values);
  parseStatus.textContent = result.message;
  parseStatus.className = `parse-status ${result.complete ? "positive" : result.values.length ? "neutral" : ""}`;

  return result;
}

function drawLineChart(svgId, seriesList, options = {}) {
  const svg = document.getElementById(svgId);
  const rect = svg.getBoundingClientRect();
  const width = Math.max(Math.round(rect.width || options.width || 860), 320);
  const height = Math.max(Math.round(rect.height || options.height || 320), 180);
  const validSeries = seriesList.filter((item) => item.values && item.values.length);
  const padding = { top: 22, right: 18, bottom: 38, left: 52 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const allValues = validSeries.flatMap((item) => item.values);

  if (!allValues.length) {
    svg.innerHTML = "";
    return;
  }

  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const rawRange = maxValue - minValue || 1;
  const yPadding = Math.max(rawRange * 0.08, maxValue * 0.02, 1);
  const chartMin = Math.max(minValue - yPadding, 0);
  const chartMax = maxValue + yPadding;
  const range = chartMax - chartMin || 1;
  const stepX = innerWidth / Math.max((validSeries[0].values.length || 1) - 1, 1);

  const scaleY = (value) => padding.top + ((chartMax - value) / range) * innerHeight;
  const scaleX = (index) => padding.left + stepX * index;

  const gridLines = Array.from({ length: 5 }, (_, index) => {
    const y = padding.top + (innerHeight / 4) * index;
    const labelValue = chartMax - (range / 4) * index;
    return `
      <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="rgba(17,17,17,0.08)" stroke-width="1" />
      <text x="${padding.left - 12}" y="${y + 4}" text-anchor="end" fill="#6b7280" font-size="12">${labelValue.toFixed(0)}</text>
    `;
  }).join("");

  const labelSeries = validSeries[0].labels || [];
  const xLabels = labelSeries.map((label, index) => {
    if (index % 3 !== 0 && index !== labelSeries.length - 1) {
      return "";
    }
    return `<text x="${scaleX(index)}" y="${height - 12}" text-anchor="middle" fill="#6b7280" font-size="12">${label}</text>`;
  }).join("");

  const paths = validSeries
    .map((series) => {
      const points = series.values.map((value, index) => `${scaleX(index)},${scaleY(value)}`).join(" ");
      const areaPoints = [
        `${padding.left},${height - padding.bottom}`,
        points,
        `${scaleX(series.values.length - 1)},${height - padding.bottom}`,
      ].join(" ");
      return `
        ${series.fill ? `<polygon fill="${series.fill}" points="${areaPoints}" />` : ""}
        <polyline fill="none" stroke="${series.color}" stroke-width="3" points="${points}" stroke-linejoin="round" stroke-linecap="round" />
      `;
    })
    .join("");

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="transparent" />
    ${gridLines}
    ${paths}
    ${xLabels}
  `;
}

function drawDailyAccuracyChart(trend) {
  state.accuracyTrend = trend;
  if (!trend.length) {
    document.getElementById("accuracyChart").innerHTML = "";
    document.getElementById("dailyTrendHint").textContent = "当前月份暂无日准确率数据";
    return;
  }

  const labels = trend.map((item) => item.target_date.slice(8));
  const values = trend.map((item) => getAccuracyRate(item.metrics));
  drawLineChart(
    "accuracyChart",
    [
      {
        labels,
        values,
        color: "#0f766e",
        fill: "rgba(15, 118, 110, 0.14)",
      },
    ],
    { height: 220 },
  );
  document.getElementById("dailyTrendHint").textContent = `展示 ${trend[0].target_date.slice(0, 7)} 的逐日准确率`;
}

function drawLoadChart(record) {
  state.loadChartRecord = record;
  const seriesList = [];

  if (record.forecast_series && record.forecast_series.length) {
    seriesList.push({
      labels: record.forecast_series.map((item) => item.hour.slice(0, 2)),
      values: record.forecast_series.map((item) => item.value),
      color: "#2563eb",
      fill: "rgba(37, 99, 235, 0.14)",
    });
  }

  if (record.actual_series && record.actual_series.length) {
    seriesList.push({
      labels: record.actual_series.map((item) => item.hour.slice(0, 2)),
      values: record.actual_series.map((item) => item.value),
      color: "#f97316",
    });
  }

  drawLineChart("loadChart", seriesList);

  if (!seriesList.length) {
    document.getElementById("chartHint").textContent = "选定日期没有负荷预测或负荷实际数据";
    return;
  }

  if (record.has_forecast && record.has_actual) {
    document.getElementById("chartHint").textContent = `负荷预测生成时间：${formatDate(record.generated_at)}`;
    return;
  }

  if (record.has_forecast) {
    document.getElementById("chartHint").textContent = "该日期仅有负荷预测数据";
    return;
  }

  document.getElementById("chartHint").textContent = "该日期仅有负荷实际数据";
}

function populateHistory(items, options = {}) {
  const { month = null, fallback = false } = options;
  const body = document.getElementById("historyTableBody");
  const caption = document.getElementById("historyCaption");
  body.innerHTML = "";

  caption.textContent = buildHistoryCaption({ month, fallback });

  if (!items.length) {
    body.innerHTML = `
      <tr>
        <td colspan="9" class="empty-state">${fallback ? "暂无记录" : "当前月份没有记录"}</td>
      </tr>
    `;
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("tr");
    const metrics = item.metrics || {};
    const accuracyRate = getAccuracyRate(metrics);
    const isActive = item.target_date === state.selectedDate;
    row.className = isActive ? "active-row" : "";
    row.innerHTML = `
      <td class="history-col-date"><button type="button" class="table-link" data-target-date="${item.target_date}">${item.target_date}</button></td>
      <td class="history-col-status"><span class="table-status ${item.data_status || "empty"}">${formatStatusLabel(item.data_status)}</span></td>
      <td class="history-col-source">${buildHistorySource(item)}</td>
      <td class="history-col-total">${item.forecast_total === null ? "-" : formatNumber(item.forecast_total)}</td>
      <td class="history-col-total">${item.actual_total === null ? "-" : formatNumber(item.actual_total)}</td>
      <td class="history-col-rate history-group-start">${formatPercent(accuracyRate)}</td>
      <td class="history-col-metric">${metrics.mae === undefined ? "-" : formatNumber(metrics.mae)}</td>
      <td class="history-col-metric">${metrics.rmse === undefined ? "-" : formatNumber(metrics.rmse)}</td>
      <td class="history-col-rate">${metrics.hit_rate_5 === undefined ? "-" : formatPercent(metrics.hit_rate_5)}</td>
    `;
    body.appendChild(row);
  });
}

function renderMonthlyList(items, currentMonth) {
  const container = document.getElementById("monthlyList");
  container.innerHTML = "";

  if (!items.length) {
    container.innerHTML = '<div class="empty-state">暂无月统计数据</div>';
    return;
  }

  items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `month-item ${item.month === currentMonth ? "active" : ""}`;
    if (item.latest_target_date) {
      button.dataset.targetDate = item.latest_target_date;
    }
    button.innerHTML = `
      <div class="month-item-head">
        <span>${item.month}</span>
        <span>${item.days_count} 天</span>
      </div>
      <strong>${item.metrics ? formatPercent(getAccuracyRate(item.metrics)) : "-"}</strong>
      <small>5% 命中 ${item.metrics ? formatPercent(item.metrics.hit_rate_5) : "-"} / MAE ${item.metrics ? formatNumber(item.metrics.mae) : "-"}</small>
    `;
    container.appendChild(button);
  });
}

function renderHeroMonthSwitch(items, currentMonth) {
  const container = document.getElementById("heroMonthSwitch");
  const currentLabel = document.getElementById("heroMonthSwitchCurrent");
  const hint = document.getElementById("heroMonthSwitchHint");

  state.monthlyRows = items;
  currentLabel.textContent = currentMonth || "-";
  container.innerHTML = "";

  if (!items.length) {
    hint.textContent = "暂无可切换月份";
    container.innerHTML = '<div class="empty-state">暂无月统计数据</div>';
    return;
  }

  hint.textContent = "点击月份直接切换当月统计、曲线与按日记录";

  items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `hero-month-pill ${item.month === currentMonth ? "active" : ""}`;
    button.dataset.month = item.month;
    if (item.latest_target_date) {
      button.dataset.targetDate = item.latest_target_date;
    }
    button.innerHTML = `
      <span class="hero-month-pill-label">${item.month}</span>
      <span class="hero-month-pill-meta">${item.days_count} 天${item.metrics ? ` · ${formatPercent(getAccuracyRate(item.metrics))}` : ""}</span>
    `;
    container.appendChild(button);
  });

  const activeButton = container.querySelector(".hero-month-pill.active");
  if (activeButton) {
    const targetLeft = activeButton.offsetLeft - (container.clientWidth - activeButton.offsetWidth) / 2;
    const maxScrollLeft = Math.max(container.scrollWidth - container.clientWidth, 0);
    container.scrollLeft = Math.max(0, Math.min(targetLeft, maxScrollLeft));
  }
}

function applyDateNavigation(data) {
  const input = document.getElementById("selectedDateInput");
  const prevButton = document.getElementById("prevDateButton");
  const nextButton = document.getElementById("nextDateButton");
  const navigation = data.date_navigation || {};

  input.value = data.selected_date || "";
  input.removeAttribute("min");
  input.removeAttribute("max");

  prevButton.dataset.targetDate = navigation.prev_date || "";
  nextButton.dataset.targetDate = navigation.next_date || "";
  prevButton.disabled = !navigation.prev_date;
  nextButton.disabled = !navigation.next_date;
}

function updateOverview(data) {
  const record = data.selected_record || {};
  const accuracy = data.accuracy || {};
  const selectedDay = accuracy.selected_day || {};
  const dayMetrics = selectedDay.metrics || record.daily_metrics || {};
  const monthSummary = accuracy.selected_month || {};
  const monthMetrics = monthSummary.metrics || {};
  const rolling = accuracy.rolling_7d || {};
  const selectedDate = data.selected_date || "-";
  const statusChip = document.getElementById("dataStatusChip");
  const dayAccuracyRate = getAccuracyRate(dayMetrics);
  const rollingAccuracyRate = getAccuracyRate(rolling);
  const monthAccuracyRate = getAccuracyRate(monthMetrics);

  document.getElementById("heroSelectedDate").textContent = selectedDate;
  document.getElementById("heroMonthMeta").textContent = buildHeroMeta(record, monthSummary);
  document.getElementById("siteName").textContent = data.site_name || "-";
  document.getElementById("selectedMonthLabel").textContent = monthSummary.month || (data.selected_date || "-").slice(0, 7);
  document.getElementById("rollingAccuracy").textContent = formatPercent(rollingAccuracyRate);
  document.getElementById("matchedDaysTop").textContent = formatNumber(accuracy.matched_days);
  document.getElementById("forecastSource").textContent = formatSourceLabel(record.forecast_source);
  document.getElementById("actualSource").textContent = formatSourceLabel(record.actual_source);
  document.getElementById("forecastGeneratedAt").textContent = formatDate(record.generated_at);
  document.getElementById("actualUpdatedAt").textContent = formatDate(record.actual_updated_at);
  statusChip.textContent = formatStatusLabel(record.data_status);
  statusChip.className = `status-chip ${record.data_status || "empty"}`;
  document.getElementById("targetStatusMeta").textContent = buildStatusMeta(record, selectedDate);

  document.getElementById("forecastTotal").textContent = formatNumber(record.forecast_total);
  document.getElementById("actualTotal").textContent = formatNumber(record.actual_total);
  document.getElementById("avgLoad").textContent = formatNumber(record.avg);
  document.getElementById("peakInfo").textContent = record.peak ? `${record.peak.hour} / ${formatNumber(record.peak.value)}` : "-";
  document.getElementById("valleyInfo").textContent = record.valley ? `${record.valley.hour} / ${formatNumber(record.valley.value)}` : "-";
  document.getElementById("dayAccuracy").textContent = formatPercent(dayAccuracyRate);
  document.getElementById("dayAccuracyDetail").textContent = formatPercent(dayAccuracyRate);
  document.getElementById("rollingAccuracyMain").textContent = formatPercent(rollingAccuracyRate);
  document.getElementById("monthAccuracyMain").textContent = formatPercent(monthAccuracyRate);
  document.getElementById("dayHitRateMain").textContent = formatPercent(dayMetrics.hit_rate_5);
  document.getElementById("dayMaxAbsErrorMain").textContent = formatNumber(dayMetrics.max_abs_error);
  document.getElementById("dayAccuracyHint").textContent = dayAccuracyRate === null
    ? "无负荷实际时不计算"
    : `结合 5% 命中与绝对偏差综合查看`;

  document.getElementById("metricMae").textContent = formatNumber(dayMetrics.mae);
  document.getElementById("metricRmse").textContent = formatNumber(dayMetrics.rmse);
  document.getElementById("dayHitRate").textContent = formatPercent(dayMetrics.hit_rate_5);
  document.getElementById("dayMaxAbsError").textContent = formatNumber(dayMetrics.max_abs_error);

  document.getElementById("monthAccuracy").textContent = formatPercent(monthAccuracyRate);
  document.getElementById("monthMae").textContent = formatNumber(monthMetrics.mae);
  document.getElementById("monthRmse").textContent = formatNumber(monthMetrics.rmse);
  document.getElementById("monthDays").textContent = formatNumber(monthSummary.days_count);
  document.getElementById("monthHitRate").textContent = formatPercent(monthMetrics.hit_rate_5);
  document.getElementById("monthHint").textContent = monthSummary.days_count
    ? `${monthSummary.month} 共 ${monthSummary.days_count} 天，月准确率 ${formatPercent(monthAccuracyRate)}`
    : `${monthSummary.month || (data.selected_date || "-").slice(0, 7)} 暂无月统计样本`;

  document.querySelector('#forecastForm input[name="site_name"]').value = data.site_name || "辉华";
}

async function loadDashboard(targetDate = state.selectedDate) {
  const search = targetDate ? `?target_date=${encodeURIComponent(targetDate)}` : "";
  const response = await fetch(`/api/dashboard${search}`);
  const data = await response.json();
  const record = data.selected_record || {};
  const accuracy = data.accuracy || {};
  const selectedMonth = (accuracy.selected_month || {}).month || (data.selected_date || "").slice(0, 7);

  state.selectedDate = data.selected_date || null;

  applyDateNavigation(data);
  updateOverview(data);
  drawLoadChart(record);
  drawDailyAccuracyChart(accuracy.selected_month_trend || []);
  renderMonthlyList(accuracy.monthly || [], selectedMonth);
  renderHeroMonthSwitch(accuracy.monthly || [], selectedMonth);

  let historyItems = Array.isArray(data.history) ? data.history : [];
  let historyFallback = false;
  if (!historyItems.length) {
    try {
      const historyResponse = await fetch("/api/history?limit=30");
      if (historyResponse.ok) {
        const historyPayload = await historyResponse.json();
        historyItems = Array.isArray(historyPayload.items) ? historyPayload.items : [];
        historyFallback = true;
      }
    } catch (error) {
      historyItems = [];
      historyFallback = false;
    }
  }
  populateHistory(historyItems, { month: selectedMonth, fallback: historyFallback });

  const apiExample = data.api_example || {};
  document.getElementById("loadRecordCurlExample").textContent = buildCurlExample("/api/load-records", apiExample.load_record || {});
  document.getElementById("loadRecordExample").textContent = JSON.stringify(apiExample.load_record || {}, null, 2);
  document.getElementById("forecastCurlExample").textContent = buildCurlExample("/api/forecasts", apiExample.forecast || {});
  document.getElementById("forecastExample").textContent = JSON.stringify(apiExample.forecast || {}, null, 2);
  document.getElementById("actualCurlExample").textContent = buildCurlExample("/api/actuals", apiExample.actual || {});
  document.getElementById("actualExample").textContent = JSON.stringify(apiExample.actual || {}, null, 2);
}

async function submitJsonForm(formId, endpoint, rawInputId, parseStatusId, previewId, statusId) {
  const form = document.getElementById(formId);
  const status = document.getElementById(statusId);
  const result = updatePastePreview({ rawInputId, previewId, parseStatusId });

  if (!result.complete) {
    status.textContent = "无法提交：请先让系统识别出正好 24 个值";
    status.className = "form-status";
    return;
  }

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());
  delete payload.raw_values;
  payload.values = result.values.map((value) => Number(value));

  status.textContent = "提交中...";
  status.className = "form-status neutral";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const resultPayload = await response.json();
    if (!response.ok) {
      throw new Error(resultPayload.detail || "提交失败");
    }
    status.textContent = resultPayload.message || "提交成功";
    status.className = "form-status positive";
    await loadDashboard(payload.target_date);
  } catch (error) {
    status.textContent = error.message || "提交失败";
    status.className = "form-status";
  }
}

function attachPasteForm(config) {
  const rawInput = document.getElementById(config.rawInputId);
  const parseButton = document.querySelector(`[data-parse-target="${config.parseTarget}"]`);

  rawInput.value = config.sampleValues.join("\t");
  updatePastePreview(config);

  rawInput.addEventListener("input", () => {
    updatePastePreview(config);
  });

  rawInput.addEventListener("paste", () => {
    window.setTimeout(() => updatePastePreview(config), 0);
  });

  parseButton.addEventListener("click", () => {
    updatePastePreview(config);
  });
}

function initializeForms() {
  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(today.getDate() + 1);
  document.querySelector('#forecastForm input[name="target_date"]').value = formatLocalDate(tomorrow);
  document.querySelector('#actualForm input[name="target_date"]').value = formatLocalDate(today);

  attachPasteForm({
    rawInputId: "forecastRawInput",
    previewId: "forecastPreview",
    parseStatusId: "forecastParseStatus",
    parseTarget: "forecast",
    sampleValues: buildSampleValues(720, 80, 120),
  });

  attachPasteForm({
    rawInputId: "actualRawInput",
    previewId: "actualPreview",
    parseStatusId: "actualParseStatus",
    parseTarget: "actual",
    sampleValues: buildSampleValues(700, 75, 105),
  });

  document.getElementById("forecastForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitJsonForm("forecastForm", "/api/forecasts", "forecastRawInput", "forecastParseStatus", "forecastPreview", "forecastFormStatus");
  });

  document.getElementById("actualForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitJsonForm("actualForm", "/api/actuals", "actualRawInput", "actualParseStatus", "actualPreview", "actualFormStatus");
  });
}

function initializeDateControls() {
  document.getElementById("selectedDateInput").addEventListener("change", async (event) => {
    const value = event.target.value;
    if (!value) {
      return;
    }
    await loadDashboard(value);
  });

  document.getElementById("prevDateButton").addEventListener("click", async (event) => {
    const targetDate = event.currentTarget.dataset.targetDate;
    if (targetDate) {
      await loadDashboard(targetDate);
    }
  });

  document.getElementById("nextDateButton").addEventListener("click", async (event) => {
    const targetDate = event.currentTarget.dataset.targetDate;
    if (targetDate) {
      await loadDashboard(targetDate);
    }
  });

  document.getElementById("monthlyList").addEventListener("click", async (event) => {
    const button = event.target.closest(".month-item");
    if (!button || !button.dataset.targetDate) {
      return;
    }
    await loadDashboard(button.dataset.targetDate);
  });

  document.getElementById("heroMonthSwitch").addEventListener("click", async (event) => {
    const button = event.target.closest(".hero-month-pill");
    if (!button || !button.dataset.targetDate) {
      return;
    }
    await loadDashboard(button.dataset.targetDate);
  });

  document.getElementById("historyTableBody").addEventListener("click", async (event) => {
    const button = event.target.closest(".table-link");
    if (!button || !button.dataset.targetDate) {
      return;
    }
    await loadDashboard(button.dataset.targetDate);
  });
}

function initializeChartResizeHandler() {
  let resizeTimer = null;

  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      if (state.loadChartRecord) {
        drawLoadChart(state.loadChartRecord);
      }
      if (state.accuracyTrend) {
        drawDailyAccuracyChart(state.accuracyTrend);
      }
    }, 120);
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  initializeForms();
  initializeDateControls();
  initializeChartResizeHandler();
  await loadDashboard();
});
