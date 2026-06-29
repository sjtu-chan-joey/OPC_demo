const colors = {
  llm: "#d71934",
  oracle: "#7a0012",
  nearest: "#ff6b2c",
  conservative: "#b5333f",
  fixed: "#7b5b60",
};

const app = {
  sessionId: null,
  data: null,
  timer: null,
  playing: false,
  busy: false,
};

const el = (id) => document.getElementById(id);

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function bindControls() {
  const bindings = [
    ["initial_soc", (v) => Number(v).toFixed(2)],
    ["initial_soh", (v) => Number(v).toFixed(3)],
    ["ambient_c", (v) => `${Number(v).toFixed(0)} C`],
    ["resistance_mohm", (v) => `${Number(v).toFixed(0)} mΩ`],
    ["speed", (v) => `${Number(v).toFixed(0)}x`],
  ];
  bindings.forEach(([id, formatter]) => {
    const input = el(id);
    const output = el(`${id}_value`);
    const update = () => {
      output.textContent = formatter(input.value);
    };
    input.addEventListener("input", update);
    update();
  });
  el("startBtn").addEventListener("click", start);
  el("pauseBtn").addEventListener("click", pause);
  el("resetBtn").addEventListener("click", reset);
}

function queryString() {
  const keys = ["initial_soc", "initial_soh", "ambient_c", "resistance_mohm", "target_cycles", "decision_cycle_interval"];
  const params = new URLSearchParams();
  keys.forEach((key) => params.set(key, el(key).value));
  return params.toString();
}

async function start() {
  pause();
  app.sessionId = null;
  app.data = null;
  app.busy = true;
  el("runStatus").textContent = "初始化";
  el("decisionCard").innerHTML = "<strong>初始化闭环控制</strong><p>正在创建仿真会话，随后按分钟推进模型并在 Cycle 边界触发策略决策。</p>";
  try {
    const response = await fetch(`/api/start?${queryString()}`);
    const payload = await response.json();
    if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
    app.sessionId = payload.session_id;
    app.data = payload;
    app.playing = true;
    app.busy = false;
    renderLegend();
    renderFrame();
    el("runStatus").textContent = "实时运行";
    scheduleTick(0);
  } catch (error) {
    app.playing = false;
    app.busy = false;
    el("runStatus").textContent = "运行失败";
    el("decisionCard").innerHTML = `<strong>初始化失败</strong><p>${error.message}</p>`;
  }
}

function pause() {
  app.playing = false;
  if (app.timer) clearTimeout(app.timer);
  app.timer = null;
  if (app.data) el("runStatus").textContent = "已暂停";
}

function reset() {
  pause();
  app.sessionId = null;
  app.data = null;
  app.busy = false;
  el("runStatus").textContent = "待启动";
  el("metricGrid").innerHTML = "";
  el("capacityLegend").innerHTML = "";
  el("stateList").innerHTML = "";
  el("policyTable").innerHTML = "";
  el("decisionStep").textContent = "--";
  el("batterySoc").textContent = "--%";
  el("batteryMode").textContent = "rest";
  el("decisionCard").innerHTML = "";
  drawEmptyCharts();
}

function scheduleTick(delay = null) {
  if (!app.playing) return;
  const speed = Number(el("speed").value);
  const ms = delay === null ? Math.max(35, 1000 / speed) : delay;
  app.timer = setTimeout(tick, ms);
}

async function tick() {
  if (!app.playing || app.busy || !app.sessionId) return;
  app.busy = true;
  el("runStatus").textContent = "计算中";
  try {
    const response = await fetch(`/api/step?session_id=${encodeURIComponent(app.sessionId)}`);
    const payload = await response.json();
    if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
    mergeStep(payload);
    renderFrame();
    app.busy = false;
    if (payload.done) {
      app.playing = false;
      el("runStatus").textContent = "Cycle 结束";
      return;
    }
    el("runStatus").textContent = "实时运行";
    scheduleTick();
  } catch (error) {
    app.playing = false;
    app.busy = false;
    el("runStatus").textContent = "运行失败";
    el("decisionCard").innerHTML = `<strong>步进失败</strong><p>${error.message}</p>`;
  }
}

function mergeStep(payload) {
  app.data.done = payload.done;
  app.data.llm_status = payload.llm_status;
  app.data.has_real_llm = payload.has_real_llm;
  const summaryByKey = new Map(payload.policies.map((policy) => [policy.key, policy.summary]));
  const updatesByKey = new Map(payload.updates.map((update) => [update.key, update]));
  app.data.policies.forEach((policy) => {
    const update = updatesByKey.get(policy.key);
    if (update) {
      policy.rows.push(...update.new_rows);
      if (update.new_decision) policy.decisions.push(update.new_decision);
    }
    if (summaryByKey.has(policy.key)) policy.summary = summaryByKey.get(policy.key);
  });
}

function latestRow(policy) {
  return policy.rows[policy.rows.length - 1];
}

function llmPolicy() {
  return app.data?.policies.find((policy) => policy.key === "llm");
}

function renderLegend() {
  const legend = el("capacityLegend");
  legend.innerHTML = "";
  app.data.policies.forEach((policy) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<i class="legend-dot" style="background:${colors[policy.key] || "#333"}"></i>${policy.label}`;
    legend.appendChild(item);
  });
}

function renderFrame() {
  if (!app.data) {
    drawEmptyCharts();
    return;
  }
  renderMetrics();
  renderBattery();
  renderDecision();
  renderPolicyTable();
  drawCharts();
}

function renderMetrics() {
  const grid = el("metricGrid");
  const llm = llmPolicy();
  const llmRow = latestRow(llm);
  const fixed = app.data.policies.find((policy) => policy.key === "fixed");
  const fixedRow = latestRow(fixed);
  const currentRows = app.data.policies.map((policy) => ({ policy, row: latestRow(policy) }));
  const best = [...currentRows].sort((a, b) => b.row.soh - a.row.soh)[0];
  const cyclePct = Math.min(100, (llmRow.equivalent_cycles / app.data.target_cycles) * 100);
  const sohDelta = llmRow.soh - fixedRow.soh;
  const metrics = [
    ["当前 Cycle", `${fmt(llmRow.equivalent_cycles, 2)} / ${fmt(app.data.target_cycles, 1)}`, `${fmt(cyclePct, 1)}%`],
    ["实时决策", `第 ${llmRow.decision_step} 轮`, `${fmt(app.data.decision_cycle_interval, 2)} Cycle 间隔`],
    ["LLM SOC", `${fmt(llmRow.soc * 100, 1)}%`, `${llmRow.mode} ${fmt(llmRow.c_rate, 2)}C`],
    ["LLM SOH", fmt(llmRow.soh, 4), `较固定基线 ${fmt(sohDelta * 100, 3)}%`],
    ["容量", `${fmt(llmRow.capacity_ah, 3)} Ah`, `能量吞吐 ${fmt(llmRow.cumulative_energy_wh, 1)} Wh`],
    ["当前最优", best.policy.label, `SOH ${fmt(best.row.soh, 4)}`],
  ];
  grid.innerHTML = metrics
    .map(([name, value, sub]) => `<div class="metric-card"><span>${name}</span><strong>${value}</strong><small>${sub}</small></div>`)
    .join("");
}

function renderBattery() {
  const row = latestRow(llmPolicy());
  el("batteryFill").style.height = `${Math.max(2, Math.min(99, row.soc * 100))}%`;
  el("batterySoc").textContent = `${fmt(row.soc * 100, 0)}%`;
  el("batteryMode").textContent = row.mode;
  el("pulse").classList.toggle("rest", row.mode === "rest");
  const items = [
    ["时间", `${fmt(row.time_min, 0)} min`],
    ["Cycle", fmt(row.equivalent_cycles, 3)],
    ["SOH", fmt(row.soh, 4)],
    ["容量", `${fmt(row.capacity_ah, 3)} Ah`],
    ["温度", `${fmt(row.temperature_c, 1)} C`],
    ["电压", `${fmt(row.voltage_v, 3)} V`],
    ["电流", `${fmt(row.current_a, 2)} A`],
    ["内阻", `${fmt(row.resistance_mohm, 2)} mΩ`],
  ];
  el("stateList").innerHTML = items
    .map(([label, value]) => `<div class="state-item"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

function renderDecision() {
  const policy = llmPolicy();
  const decision = policy.decisions[policy.decisions.length - 1];
  if (!decision) {
    el("decisionStep").textContent = "--";
    el("decisionCard").innerHTML = "<strong>等待首次决策</strong><p>下一次步进将读取当前状态并生成控制动作。</p>";
    return;
  }
  const action = decision.action;
  el("decisionStep").textContent = `第 ${decision.decision_step} 轮`;
  el("decisionCard").innerHTML = `
    <strong>${action.mode} · ${fmt(action.c_rate, 2)}C · 实时执行</strong>
    <div>决策区间 ${fmt(decision.decision_start_cycle, 2)} → ${fmt(decision.decision_target_cycle, 2)} Cycle</div>
    <div>目标 SOC ${action.target_soc === null ? "--" : fmt(action.target_soc, 2)} · 电压限制 ${action.voltage_limit_v === null ? "--" : `${fmt(action.voltage_limit_v, 2)} V`}</div>
    <p>${decision.reason}</p>
    <small>${app.data.llm_status}</small>
  `;
}

function renderPolicyTable() {
  const rows = app.data.policies.map((policy) => {
    const row = latestRow(policy);
    return `
      <div class="policy-row">
        <b>${policy.label}</b>
        <span>SOC ${fmt(row.soc, 3)}</span>
        <span>SOH ${fmt(row.soh, 4)}</span>
        <span>Cycle ${fmt(row.equivalent_cycles, 2)}</span>
        <span>${row.mode} ${fmt(row.c_rate, 2)}C</span>
        <span>${fmt(row.temperature_c, 1)} C</span>
      </div>`;
  });
  el("policyTable").innerHTML = rows.join("");
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * dpr));
  canvas.height = Math.max(220, Math.floor(Number(canvas.getAttribute("height")) * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function drawEmptyCharts() {
  ["capacityChart", "socChart", "thermalChart"].forEach((id) => {
    const canvas = el(id);
    const ctx = setupCanvas(canvas);
    ctx.clearRect(0, 0, canvas.clientWidth, Number(canvas.getAttribute("height")));
    ctx.fillStyle = "#7b5b60";
    ctx.font = "15px Segoe UI";
    ctx.fillText("设置参数后点击启动", 24, 42);
  });
}

function drawCharts() {
  drawLineChart("capacityChart", [
    { field: "capacity_ah", policies: app.data.policies },
    { field: "soh", policies: [llmPolicy()], alt: true },
  ]);
  drawLineChart("socChart", [
    { field: "soc", policies: app.data.policies },
  ], { min: 0, max: 1 });
  drawLineChart("thermalChart", [
    { field: "temperature_c", policies: app.data.policies },
    { field: "voltage_v", policies: [llmPolicy()], alt: true },
  ]);
}

function drawLineChart(canvasId, seriesGroups, fixedScale = null) {
  const canvas = el(canvasId);
  const ctx = setupCanvas(canvas);
  const width = canvas.clientWidth;
  const height = Number(canvas.getAttribute("height"));
  const pad = { left: 48, right: 18, top: 18, bottom: 34 };
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);

  const seriesList = [];
  seriesGroups.forEach((group) => group.policies.forEach((policy) => seriesList.push({ ...group, policy })));
  const values = [];
  seriesList.forEach((series) => {
    series.policy.rows.forEach((row) => values.push(row[series.field]));
  });
  let min = fixedScale?.min ?? Math.min(...values);
  let max = fixedScale?.max ?? Math.max(...values);
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    min = 0;
    max = 1;
  }
  const span = max - min;
  min -= fixedScale ? 0 : span * 0.12;
  max += fixedScale ? 0 : span * 0.12;

  drawGrid(ctx, width, height, pad, min, max);
  const maxLen = Math.max(...app.data.policies.map((policy) => policy.rows.length));
  seriesList.forEach((series) => {
    ctx.beginPath();
    series.policy.rows.forEach((row, index) => {
      const x = pad.left + (index / Math.max(1, maxLen - 1)) * (width - pad.left - pad.right);
      const y = pad.top + (1 - (row[series.field] - min) / (max - min)) * (height - pad.top - pad.bottom);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineWidth = series.alt ? 2 : 2.6;
    ctx.setLineDash(series.alt ? [5, 5] : []);
    ctx.strokeStyle = series.alt ? "#111" : colors[series.policy.key] || "#333";
    ctx.stroke();
    ctx.setLineDash([]);
  });
}

function drawGrid(ctx, width, height, pad, min, max) {
  ctx.strokeStyle = "#f2c5cb";
  ctx.lineWidth = 1;
  ctx.font = "12px Segoe UI";
  ctx.fillStyle = "#7b5b60";
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + (i / 4) * (height - pad.top - pad.bottom);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    const value = max - (i / 4) * (max - min);
    ctx.fillText(fmt(value, 2), 8, y + 4);
  }
  ctx.fillText("time", width - pad.right - 28, height - 10);
}

window.addEventListener("resize", renderFrame);
bindControls();
drawEmptyCharts();
