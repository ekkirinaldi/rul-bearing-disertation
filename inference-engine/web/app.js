"use strict";

const $ = (id) => document.getElementById(id);
const fmtNum = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "\u2014" : Number(v).toFixed(d));

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "\u2014";
  let s = Math.max(0, Math.round(seconds));
  const d = Math.floor(s / 86400); s -= d * 86400;
  const h = Math.floor(s / 3600); s -= h * 3600;
  const m = Math.floor(s / 60); s -= m * 60;
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

const state = {
  ws: null,
  datasets: {},
  featureNames: [],
  hasGtRul: true,
  streamMode: "benchmark",
};

const charts = {};

function makeLineChart(canvasId, datasets, yTitle) {
  const ctx = $(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: { labels: [], datasets },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ticks: { color: "#8b949e", maxTicksLimit: 8 }, grid: { color: "#222831" } },
        y: { title: { display: !!yTitle, text: yTitle, color: "#8b949e" }, ticks: { color: "#8b949e" }, grid: { color: "#222831" } },
      },
      plugins: { legend: { labels: { color: "#e6edf3", boxWidth: 12 } } },
    },
  });
}

function initCharts() {
  charts.rul = makeLineChart("rul-chart", [
    { label: "Predicted RUL", data: [], borderColor: "#58a6ff", backgroundColor: "rgba(88,166,255,0.1)", borderWidth: 2, pointRadius: 0, tension: 0.2 },
    { label: "Ground-truth RUL", data: [], borderColor: "#3fb950", borderDash: [6, 4], borderWidth: 2, pointRadius: 0, tension: 0.2 },
  ], "RUL (norm.)");

  charts.hi = makeLineChart("hi-chart", [
    { label: "RMS-H / accel", data: [], borderColor: "#58a6ff", borderWidth: 1.5, pointRadius: 0 },
    { label: "RMS-V / velocity", data: [], borderColor: "#f0883e", borderWidth: 1.5, pointRadius: 0 },
    { label: "Kurtosis-H / envelope", data: [], borderColor: "#d29922", borderWidth: 1.5, pointRadius: 0 },
  ], "amplitude");

  charts.wave = makeLineChart("wave-chart", [
    { label: "Horizontal", data: [], borderColor: "#58a6ff", borderWidth: 1, pointRadius: 0 },
    { label: "Vertical", data: [], borderColor: "#f0883e", borderWidth: 1, pointRadius: 0 },
  ], "g");

  const igctx = $("ig-chart").getContext("2d");
  charts.ig = new Chart(igctx, {
    type: "bar",
    data: { labels: [], datasets: [{ label: "Integrated Gradients attribution", data: [], backgroundColor: "#f0883e" }] },
    options: {
      indexAxis: "y",
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#8b949e" }, grid: { color: "#222831" } },
        y: { ticks: { color: "#8b949e" }, grid: { display: false } },
      },
      plugins: { legend: { labels: { color: "#e6edf3" } } },
    },
  });
}

function resetCharts() {
  for (const key of ["rul", "hi", "wave"]) {
    const c = charts[key];
    c.data.labels = [];
    c.data.datasets.forEach((d) => (d.data = []));
    c.update();
  }
}

function pushPoint(chart, label, values, maxPoints = 400) {
  chart.data.labels.push(label);
  values.forEach((v, i) => chart.data.datasets[i].data.push(v));
  if (chart.data.labels.length > maxPoints) {
    chart.data.labels.shift();
    chart.data.datasets.forEach((d) => d.data.shift());
  }
  chart.update();
}

async function loadDatasets() {
  const resp = await fetch("/api/datasets");
  const data = await resp.json();
  const sel = $("dataset-select");
  sel.innerHTML = "";
  data.datasets.forEach((ds) => {
    state.datasets[ds.key] = ds;
    const opt = document.createElement("option");
    opt.value = ds.key;
    opt.textContent = ds.label;
    sel.appendChild(opt);
  });
  if (data.datasets.length) onDatasetChange();
}

function onDatasetChange() {
  const ds = state.datasets[$("dataset-select").value];
  if (!ds) return;
  state.featureNames = ds.feature_names || [];
  state.hasGtRul = ds.has_gt_rul;
  state.streamMode = ds.stream_mode;
  const bsel = $("bearing-select");
  bsel.innerHTML = "";
  (ds.test_bearings || []).forEach((b) => {
    const opt = document.createElement("option");
    opt.value = b;
    opt.textContent = (ds.bearing_labels && ds.bearing_labels[b]) || b;
    bsel.appendChild(opt);
  });
  $("meta-model").textContent = ds.model || "\u2014";
  $("meta-checkpoint").textContent = ds.checkpoint || "\u2014";
  $("meta-window").textContent = ds.window_length != null ? `${ds.window_length} acq` : "\u2014";
  $("meta-interval").textContent = ds.acquisition_interval_s != null ? `${ds.acquisition_interval_s}s` : "\u2014";
  const note = $("transfer-note");
  if (ds.transfer_note) { note.textContent = ds.transfer_note; note.classList.remove("hidden"); }
  else note.classList.add("hidden");
  // Relabel HI series for SKF trending mode.
  const skf = ds.stream_mode === "skf_trending";
  charts.hi.data.datasets[0].label = skf ? "Accel (g RMS)" : "RMS-H";
  charts.hi.data.datasets[1].label = skf ? "Velocity (mm/s)" : "RMS-V";
  charts.hi.data.datasets[2].label = skf ? "Envelope (gE)" : "Kurtosis-H";
  charts.hi.update();
}

function connect() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) return state.ws;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.onopen = () => setConn(true);
  ws.onclose = () => { setConn(false); state.ws = null; };
  ws.onerror = () => setConn(false);
  ws.onmessage = (ev) => handleMessage(JSON.parse(ev.data));
  state.ws = ws;
  return ws;
}

function setConn(online) {
  const el = $("conn-status");
  el.textContent = online ? "online" : "offline";
  el.className = `conn ${online ? "online" : "offline"}`;
}

function send(obj) {
  const ws = connect();
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  else ws.addEventListener("open", () => ws.send(JSON.stringify(obj)), { once: true });
}

function handleMessage(msg) {
  switch (msg.type) {
    case "started":
      state.featureNames = msg.feature_names || state.featureNames;
      state.hasGtRul = msg.has_gt_rul;
      $("meta-device").textContent = msg.device || "\u2014";
      $("meta-model").textContent = msg.model || "\u2014";
      $("meta-checkpoint").textContent = msg.checkpoint || "\u2014";
      $("meta-window").textContent = `${msg.window_length} acq`;
      $("meta-interval").textContent = `${fmtNum(msg.interval_s, 1)}s`;
      resetCharts();
      break;
    case "frame":
      renderFrame(msg);
      break;
    case "seek":
      if (msg.frame) renderFrame(msg.frame);
      break;
    case "done":
      $("progress-label").textContent = `done @ acq ${msg.t}`;
      break;
    case "paused": $("progress-label").textContent += " (paused)"; break;
    case "reset": resetCharts(); $("progress-bar").style.width = "0%"; $("progress-label").textContent = "reset"; break;
    case "explanation": renderExplanation(msg); break;
    case "error": $("ig-status").textContent = `Error: ${msg.message}`; break;
    default: break;
  }
}

function renderFrame(f) {
  const total = f.n_total || 1;
  const pct = (f.t / total) * 100;
  $("progress-bar").style.width = `${Math.min(100, pct)}%`;
  $("progress-label").textContent = `acq ${f.t} / ${total}`;
  $("metric-acq").textContent = `${f.t} / ${total}`;

  if (f.warmup) {
    $("warmup-banner").classList.remove("hidden");
    $("warmup-banner").textContent = `Warming up\u2026 ${f.warmup_remaining} acquisitions to first prediction`;
  } else {
    $("warmup-banner").classList.add("hidden");
  }

  $("metric-pred-rul").textContent = fmtNum(f.pred_rul, 3);
  $("metric-gt-rul").textContent = state.hasGtRul ? fmtNum(f.gt_rul, 3) : "n/a";
  $("metric-remaining").textContent = fmtDuration(f.pred_remaining_s) + (f.ttf_capped ? " (capped)" : "");
  $("metric-eol").textContent = f.pred_eol_iso ? new Date(f.pred_eol_iso).toLocaleString() : "\u2014";
  $("metric-elapsed").textContent = fmtDuration(f.elapsed_s);

  if (f.branch_gate) {
    const x = Math.round((f.branch_gate.xlstm || 0) * 100);
    $("gate-xlstm").style.width = `${x}%`;
    $("gate-mamba").style.width = `${100 - x}%`;
    $("gate-xlstm").textContent = `xLSTM ${x}%`;
    $("gate-mamba").textContent = `Mamba ${100 - x}%`;
  }

  if (!f.warmup && f.pred_rul !== null) {
    pushPoint(charts.rul, f.t, [f.pred_rul, state.hasGtRul ? f.gt_rul : null]);
  }
  if (f.hi) pushPoint(charts.hi, f.t, [f.hi.rms_h, f.hi.rms_v, f.hi.kurtosis_h]);

  if (f.waveform && (f.waveform.horizontal || f.waveform.vertical)) {
    const wc = charts.wave;
    const h = f.waveform.horizontal || [];
    const v = f.waveform.vertical || [];
    wc.data.labels = h.map((_, i) => i);
    wc.data.datasets[0].data = h;
    wc.data.datasets[1].data = v;
    wc.update();
  }

  renderDrivers(f.top_drivers);
}

function renderDrivers(drivers) {
  const ul = $("drivers-list");
  if (!drivers || !drivers.length) { ul.innerHTML = "<li class='muted'>no saliency this frame</li>"; return; }
  const max = Math.max(...drivers.map((d) => Math.abs(d.weight))) || 1;
  ul.innerHTML = "";
  drivers.forEach((d) => {
    const li = document.createElement("li");
    const w = (Math.abs(d.weight) / max) * 100;
    li.innerHTML = `<span class="dname">${d.name}</span>
      <span class="dbar-wrap"><span class="dbar ${d.dir}" style="width:${w}%"></span></span>
      <span class="dval">${fmtNum(d.weight, 3)}</span>`;
    ul.appendChild(li);
  });
}

function renderExplanation(msg) {
  if (!msg.ok) { $("ig-status").textContent = `Explanation unavailable: ${msg.reason}`; return; }
  $("ig-status").textContent = `Integrated Gradients @ acq ${msg.t} (${msg.n_steps} steps)`;
  const feats = (msg.features || []).slice(0, 12);
  charts.ig.data.labels = feats.map((x) => x.name);
  charts.ig.data.datasets[0].data = feats.map((x) => x.value);
  charts.ig.data.datasets[0].backgroundColor = feats.map((x) => (x.value >= 0 ? "#f85149" : "#3fb950"));
  charts.ig.update();
}

function wireControls() {
  $("dataset-select").addEventListener("change", onDatasetChange);
  $("speed-range").addEventListener("input", (e) => {
    $("speed-value").textContent = e.target.value;
    send({ action: "set_speed", speed_ms: parseInt(e.target.value, 10) });
  });
  $("btn-start").addEventListener("click", () => {
    send({
      action: "start",
      dataset: $("dataset-select").value,
      bearing: $("bearing-select").value,
      speed_ms: parseInt($("speed-range").value, 10),
    });
  });
  $("btn-pause").addEventListener("click", () => send({ action: "pause" }));
  $("btn-resume").addEventListener("click", () => send({ action: "resume" }));
  $("btn-step").addEventListener("click", () => send({ action: "step" }));
  $("btn-reset").addEventListener("click", () => send({ action: "reset" }));
  $("btn-stop").addEventListener("click", () => send({ action: "stop" }));
  $("btn-explain").addEventListener("click", () => {
    $("ig-status").textContent = "Computing Integrated Gradients\u2026";
    send({ action: "explain" });
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  initCharts();
  wireControls();
  await loadDatasets();
});
