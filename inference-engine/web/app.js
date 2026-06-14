/* RUL streaming dashboard — canvas charts + WebSocket client */
"use strict";

const els = {
  status: document.getElementById("status"),
  statusText: document.getElementById("statusText"),
  dataset: document.getElementById("dataset"),
  bearing: document.getElementById("bearing"),
  speed: document.getElementById("speed"),
  speedVal: document.getElementById("speedVal"),
  warn: document.getElementById("warn"),
  crit: document.getElementById("crit"),
  btnStart: document.getElementById("btnStart"),
  btnPause: document.getElementById("btnPause"),
  btnStep: document.getElementById("btnStep"),
  btnReset: document.getElementById("btnReset"),
  btnExplain: document.getElementById("btnExplain"),
  btnCsv: document.getElementById("btnCsv"),
  btnClear: document.getElementById("btnClear"),
  btnLogExport: document.getElementById("btnLogExport"),
  transferNote: document.getElementById("transfer-note"),
  seek: document.getElementById("seek"),
  seekIdx: document.getElementById("seekIdx"),
  seekMax: document.getElementById("seekMax"),
  mUseful: document.getElementById("mUseful"),
  mUsefulCell: document.getElementById("mUsefulCell"),
  mEol: document.getElementById("mEol"),
  mTruth: document.getElementById("mTruth"),
  mAcq: document.getElementById("mAcq"),
  mElapsed: document.getElementById("mElapsed"),
  fusionBar: document.getElementById("fusionBar"),
  segX: document.getElementById("segX"),
  segM: document.getElementById("segM"),
  pcX: document.getElementById("pcX"),
  pcM: document.getElementById("pcM"),
  drivers: document.getElementById("drivers"),
  heatgrid: document.getElementById("heatgrid"),
  infoModel: document.getElementById("info-model"),
  infoCkpt: document.getElementById("info-ckpt"),
  infoDevice: document.getElementById("info-device"),
  infoWindow: document.getElementById("info-window"),
  infoNtotal: document.getElementById("info-ntotal"),
  infoInterval: document.getElementById("info-interval"),
  stage: document.getElementById("stage"),
  featBody: document.getElementById("featBody"),
  log: document.getElementById("log"),
  footClock: document.getElementById("footClock"),
  gtLegend: document.querySelector(".gt-legend"),
};

const css = getComputedStyle(document.documentElement);
const C = {
  blue: css.getPropertyValue("--blue").trim(),
  blueB: css.getPropertyValue("--blue-bright").trim(),
  amber: css.getPropertyValue("--amber").trim(),
  green: css.getPropertyValue("--green").trim(),
  red: css.getPropertyValue("--red").trim(),
  rust: css.getPropertyValue("--rust").trim(),
  line: css.getPropertyValue("--line").trim(),
  ink3: css.getPropertyValue("--ink-3").trim(),
  ink4: css.getPropertyValue("--ink-4").trim(),
};

const VIEW = 400;
const MONO = '"IBM Plex Mono", ui-monospace, monospace';

let ws = null;
let datasets = [];
let currentSpec = null;
let streaming = false;
let paused = false;
let featureNames = [];
let runFrames = [];
let lastAlertState = "ok";
let logEntries = [];
let currentIdx = 0;
let latestWaveform = null;
let igFeatures = [];
let showGt = true;

const series = { pred: [], gt: [], rmsH: [], rmsV: [], kurt: [] };

function setStatus(state, text) {
  els.status.dataset.state = state;
  els.statusText.textContent = text;
}

function pct(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtTime(s) {
  if (s == null || Number.isNaN(s)) return "—";
  if (s < 60) return `${Math.floor(s)}s`;
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${sec}s`;
}

function fmtEol(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  } catch (e) {
    return iso.slice(0, 16);
  }
}

function stageFor(rul) {
  if (rul == null) return { text: "—", cls: "ok" };
  if (rul <= 0.2) return { text: "Critical", cls: "crit" };
  if (rul <= 0.4) return { text: "Degrading", cls: "warn" };
  return { text: "Healthy", cls: "ok" };
}

function log(msg, level = "info") {
  const ts = new Date().toLocaleTimeString();
  const line = `[${ts}] ${msg}`;
  logEntries.push(line);
  const ln = document.createElement("span");
  ln.className = `ln ${level === "info" ? "info" : level}`;
  ln.innerHTML = `<span class="ts">[${ts}]</span> ${msg}`;
  els.log.appendChild(ln);
  els.log.parentElement.scrollTop = els.log.parentElement.scrollHeight;
  if (els.log.children.length > 500) els.log.removeChild(els.log.firstChild);
}

function setup(canvas, h) {
  if (!canvas) return { ctx: null, w: 0, h: 0 };
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = canvas.clientWidth || canvas.parentElement?.clientWidth || 600;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function gridLines(ctx, x0, y0, x1, y1, rows) {
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  for (let r = 0; r <= rows; r++) {
    const y = y0 + ((y1 - y0) * r) / rows;
    ctx.beginPath();
    ctx.moveTo(x0, y + 0.5);
    ctx.lineTo(x1, y + 0.5);
    ctx.stroke();
  }
}

function drawRul() {
  const cv = document.getElementById("cRul");
  const { ctx, w, h } = setup(cv, 240);
  if (!ctx) return;
  ctx.clearRect(0, 0, w, h);
  const warn = Number(els.warn.value) || 40;
  const crit = Number(els.crit.value) || 20;
  const padL = 42, padR = 12, padT = 12, padB = 26;
  const x0 = padL, x1 = w - padR, y0 = padT, y1 = h - padB;
  const idx = currentIdx;
  const start = Math.max(0, idx - VIEW);
  const end = idx;
  const xAt = (i) => x0 + ((x1 - x0) * (i - start)) / Math.max(1, end - start);
  const yAt = (v) => y1 - ((y1 - y0) * v) / 100;

  ctx.fillStyle = css.getPropertyValue("--amber-wash").trim();
  ctx.fillRect(x0, yAt(warn), x1 - x0, yAt(crit) - yAt(warn));
  ctx.fillStyle = css.getPropertyValue("--red-wash").trim();
  ctx.fillRect(x0, yAt(crit), x1 - x0, y1 - yAt(crit));

  gridLines(ctx, x0, y0, x1, y1, 10);
  ctx.fillStyle = C.ink4;
  ctx.textAlign = "right";
  ctx.font = `10px ${MONO}`;
  ctx.textBaseline = "middle";
  for (let p = 0; p <= 100; p += 10) ctx.fillText(`${p}%`, x0 - 6, yAt(p));
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  for (let k = 0; k <= 5; k++) {
    const i = Math.round(start + ((end - start) * k) / 5);
    ctx.fillText(String(i), x0 + ((x1 - x0) * k) / 5, y1 + 14);
  }

  if (showGt) {
    ctx.strokeStyle = C.red;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    for (let i = start; i <= end; i++) {
      const gt = series.gt[i];
      if (gt == null) continue;
      const X = xAt(i), Y = yAt(gt);
      if (i === start || series.gt[i - 1] == null) ctx.moveTo(X, Y);
      else ctx.lineTo(X, Y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  let lastPred = null;
  ctx.beginPath();
  for (let i = start; i <= end; i++) {
    const p = series.pred[i];
    if (p == null) continue;
    const X = xAt(i), Y = yAt(p);
    if (lastPred == null) ctx.moveTo(X, Y);
    else ctx.lineTo(X, Y);
    lastPred = p;
  }
  if (lastPred != null) {
    ctx.lineTo(xAt(end), y1);
    ctx.lineTo(xAt(start), y1);
    ctx.closePath();
    ctx.fillStyle = "rgba(31,96,145,.10)";
    ctx.fill();
    ctx.beginPath();
    for (let i = start; i <= end; i++) {
      const p = series.pred[i];
      if (p == null) continue;
      const X = xAt(i), Y = yAt(p);
      if (i === start || series.pred[i - 1] == null) ctx.moveTo(X, Y);
      else ctx.lineTo(X, Y);
    }
    ctx.strokeStyle = C.blue;
    ctx.lineWidth = 1.8;
    ctx.stroke();
    const head = series.pred[end];
    if (head != null) {
      ctx.fillStyle = C.blue;
      ctx.beginPath();
      ctx.arc(xAt(end), yAt(head), 3.2, 0, 7);
      ctx.fill();
    }
  }
}

function drawWave() {
  const cv = document.getElementById("cWave");
  const { ctx, w, h } = setup(cv, 240);
  if (!ctx) return;
  ctx.clearRect(0, 0, w, h);
  const padL = 42, padR = 12, padT = 12, padB = 26;
  const x0 = padL, x1 = w - padR, y0 = padT, y1 = h - padB;
  const H = latestWaveform?.horizontal || [];
  const V = latestWaveform?.vertical || [];
  const n = Math.max(H.length, V.length, 1);
  if (!n || (!H.length && !V.length)) {
    ctx.fillStyle = C.ink4;
    ctx.font = `12px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("No waveform", w / 2, h / 2);
    return;
  }
  let maxAbs = 1e-6;
  for (let i = 0; i < n; i++) {
    maxAbs = Math.max(maxAbs, Math.abs(H[i] || 0), Math.abs(V[i] || 0));
  }
  const max = maxAbs * 1.1;
  const xAt = (i) => x0 + ((x1 - x0) * i) / Math.max(1, n - 1);
  const yAt = (v) => (y0 + y1) / 2 - ((y1 - y0) / 2) * (v / max);
  gridLines(ctx, x0, y0, x1, y1, 10);
  ctx.fillStyle = C.ink4;
  ctx.textAlign = "right";
  ctx.font = `10px ${MONO}`;
  ctx.textBaseline = "middle";
  const tickStep = max > 10 ? Math.pow(10, Math.floor(Math.log10(max))) : 2;
  for (let v = -max; v <= max + 1e-9; v += tickStep) {
    ctx.fillText(v.toFixed(max > 10 ? 0 : 1), x0 - 6, yAt(v));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  for (let k = 0; k <= 7; k++) {
    const s = Math.round((n * k) / 7);
    ctx.fillText(String(s), x0 + ((x1 - x0) * k) / 7, y1 + 14);
  }
  ctx.strokeStyle = C.line;
  ctx.beginPath();
  ctx.moveTo(x0, yAt(0));
  ctx.lineTo(x1, yAt(0));
  ctx.stroke();
  const line = (arr, col, lw) => {
    ctx.strokeStyle = col;
    ctx.lineWidth = lw;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const X = xAt(i), Y = yAt(arr[i] || 0);
      if (i === 0) ctx.moveTo(X, Y);
      else ctx.lineTo(X, Y);
    }
    ctx.stroke();
  };
  line(V, C.rust, 1);
  line(H, C.blue, 1.1);
}

function drawAttr() {
  const cv = document.getElementById("cAttr");
  const { ctx, w, h } = setup(cv, 200);
  if (!ctx) return;
  ctx.clearRect(0, 0, w, h);
  const padL = 34, padR = 10, padT = 10, padB = 58;
  const x0 = padL, x1 = w - padR, y0 = padT, y1 = h - padB;
  if (!igFeatures.length) {
    ctx.fillStyle = C.ink4;
    ctx.font = `12px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("Click Explain to compute attribution", w / 2, h / 2);
    return;
  }
  const names = igFeatures.map((f) => f.name);
  const vals = igFeatures.map((f) => f.value);
  const maxVal = Math.max(...vals, 1e-9);
  const norm = vals.map((v) => v / maxVal);
  gridLines(ctx, x0, y0, x1, y1, 10);
  ctx.fillStyle = C.ink4;
  ctx.textAlign = "right";
  ctx.font = `10px ${MONO}`;
  ctx.textBaseline = "middle";
  for (let p = 0; p <= 10; p++) ctx.fillText((p / 10).toFixed(1), x0 - 5, y1 - ((y1 - y0) * p) / 10);
  const n = norm.length, gap = 10, bw = (x1 - x0 - gap * (n - 1)) / n;
  for (let i = 0; i < n; i++) {
    const bx = x0 + i * (bw + gap), bh = (y1 - y0) * norm[i];
    const grad = ctx.createLinearGradient(0, y1 - bh, 0, y1);
    grad.addColorStop(0, C.blueB);
    grad.addColorStop(1, C.blue);
    ctx.fillStyle = grad;
    ctx.fillRect(bx, y1 - bh, bw, bh);
    ctx.strokeStyle = "rgba(15,65,99,.5)";
    ctx.strokeRect(bx + 0.5, y1 - bh + 0.5, bw - 1, bh - 1);
    ctx.save();
    ctx.translate(bx + bw / 2, y1 + 6);
    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = C.ink3;
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(names[i], 0, 0);
    ctx.restore();
  }
}

function drawHealth() {
  const cv = document.getElementById("cHealth");
  const { ctx, w, h } = setup(cv, 240);
  if (!ctx) return;
  ctx.clearRect(0, 0, w, h);
  const padL = 46, padR = 40, padT = 12, padB = 26;
  const x0 = padL, x1 = w - padR, y0 = padT, y1 = h - padB;
  const idx = currentIdx;
  const start = Math.max(0, idx - VIEW);
  const end = idx;
  const xAt = (i) => x0 + ((x1 - x0) * (i - start)) / Math.max(1, end - start);

  let rmsMax = 100;
  let kurMax = 10;
  for (let i = start; i <= end; i++) {
    rmsMax = Math.max(rmsMax, series.rmsH[i] || 0, series.rmsV[i] || 0);
    kurMax = Math.max(kurMax, series.kurt[i] || 0);
  }
  rmsMax = Math.ceil(rmsMax / 200) * 200 || 200;
  kurMax = Math.ceil(kurMax / 10) * 10 || 10;

  const yRms = (v) => y1 - ((y1 - y0) * (v || 0)) / rmsMax;
  const yKur = (v) => y1 - ((y1 - y0) * Math.min(v || 0, kurMax)) / kurMax;
  gridLines(ctx, x0, y0, x1, y1, 9);
  ctx.fillStyle = C.ink4;
  ctx.textAlign = "right";
  ctx.font = `10px ${MONO}`;
  ctx.textBaseline = "middle";
  const rmsStep = rmsMax / 9;
  for (let v = 0; v <= rmsMax + 1e-9; v += rmsStep) ctx.fillText(Math.round(v), x0 - 6, yRms(v));
  ctx.textAlign = "left";
  const kurStep = kurMax / 8;
  for (let v = 0; v <= kurMax + 1e-9; v += kurStep) ctx.fillText(Math.round(v), x1 + 6, yKur(v));
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  for (let k = 0; k <= 5; k++) {
    const i = Math.round(start + ((end - start) * k) / 5);
    ctx.fillText(String(i), x0 + ((x1 - x0) * k) / 5, y1 + 14);
  }
  const lineR = (arr, col) => {
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    for (let i = start; i <= end; i++) {
      const X = xAt(i), Y = yRms(arr[i]);
      if (i === start || arr[i - 1] == null) ctx.moveTo(X, Y);
      else ctx.lineTo(X, Y);
    }
    ctx.stroke();
  };
  lineR(series.rmsV, C.rust);
  lineR(series.rmsH, C.blue);
  ctx.strokeStyle = C.green;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  for (let i = start; i <= end; i++) {
    const X = xAt(i), Y = yKur(series.kurt[i]);
    if (i === start || series.kurt[i - 1] == null) ctx.moveTo(X, Y);
    else ctx.lineTo(X, Y);
  }
  ctx.stroke();
}

function heatColor(v) {
  const a = Math.min(1, Math.abs(v));
  if (v >= 0) return `rgba(165,43,30,${a})`;
  return `rgba(31,96,145,${a})`;
}

function drawHeatmap(hm) {
  els.heatgrid.innerHTML = "";
  if (!hm || !hm.values || !hm.values.length) return;
  const flat = hm.values.flat();
  const absMax = Math.max(...flat.map((v) => Math.abs(v)), 1e-9);
  hm.values.forEach((row, rI) => {
    const r = document.createElement("div");
    r.className = "heatrow";
    const lbl = document.createElement("div");
    lbl.className = "rl";
    lbl.textContent = hm.feature_names[rI];
    const cells = document.createElement("div");
    cells.className = "cells";
    row.forEach((v) => {
      const c = document.createElement("i");
      c.style.background = heatColor(v / absMax);
      c.title = v.toExponential(2);
      cells.appendChild(c);
    });
    r.appendChild(lbl);
    r.appendChild(cells);
    els.heatgrid.appendChild(r);
  });
}

function renderAll() {
  drawRul();
  drawWave();
  drawAttr();
  drawHealth();
}

function clearSeries() {
  series.pred = [];
  series.gt = [];
  series.rmsH = [];
  series.rmsV = [];
  series.kurt = [];
  currentIdx = 0;
  latestWaveform = null;
}

function storeFrame(frame) {
  while (series.pred.length <= frame.t) {
    series.pred.push(null);
    series.gt.push(null);
    series.rmsH.push(null);
    series.rmsV.push(null);
    series.kurt.push(null);
  }
  series.pred[frame.t] = frame.warmup ? null : (frame.pred_rul != null ? frame.pred_rul * 100 : null);
  series.gt[frame.t] = frame.gt_rul != null ? frame.gt_rul * 100 : null;
  if (frame.hi) {
    series.rmsH[frame.t] = frame.hi.rms_h;
    series.rmsV[frame.t] = frame.hi.rms_v;
    series.kurt[frame.t] = frame.hi.kurtosis_h;
  }
  currentIdx = frame.t;
  if (frame.waveform) latestWaveform = frame.waveform;
}

function updateBranchGate(g) {
  if (!g) {
    els.fusionBar.classList.add("hidden");
    return;
  }
  els.fusionBar.classList.remove("hidden");
  const xl = Math.round(g.xlstm * 100);
  els.segX.style.flexBasis = `${xl}%`;
  els.segM.style.flexBasis = `${100 - xl}%`;
  els.pcX.textContent = `${xl}%`;
  els.pcM.textContent = `${100 - xl}%`;
}

function updateDrivers(drivers) {
  els.drivers.innerHTML = "";
  if (!drivers || !drivers.length) {
    els.drivers.innerHTML = '<div class="driver muted">Awaiting predictions…</div>';
    return;
  }
  const max = Math.max(...drivers.map((d) => d.weight), 1e-9);
  drivers.forEach((d) => {
    const el = document.createElement("div");
    el.className = "driver";
    const arrow = d.dir === "up" ? "▲" : "▼";
    const w = Math.round((d.weight / max) * 100);
    el.innerHTML =
      `<span class="nm">${d.name} <span class="arr ${d.dir === "up" ? "up" : "dn"}">${arrow}</span></span>` +
      `<span class="track"><i style="width:${w}%"></i></span>` +
      `<span class="pc">${(d.weight * 100).toFixed(1)}%</span>`;
    els.drivers.appendChild(el);
  });
}

function updateHiTable(raw, scaled) {
  if (!raw || !featureNames.length) {
    els.featBody.innerHTML = '<tr><td colspan="3" class="muted">—</td></tr>';
    return;
  }
  els.featBody.innerHTML = featureNames
    .map((name, i) => {
      const r = raw[i] != null ? raw[i].toExponential(2) : "—";
      const sc = scaled && scaled[i] != null ? scaled[i] : null;
      const s = sc != null ? sc.toFixed(3) : "—";
      const bar = sc != null ? `<span class="scaled-mini"><i style="width:${Math.round(sc * 100)}%"></i></span>` : "";
      return `<tr><td class="feat">${name}</td><td class="num">${r}</td><td class="num">${s}${bar}</td></tr>`;
    })
    .join("");
}

function zoneClass(rulNorm) {
  const warn = Number(els.warn.value) / 100;
  const crit = Number(els.crit.value) / 100;
  if (rulNorm == null) return "";
  if (rulNorm <= crit) return "is-crit";
  if (rulNorm <= warn) return "is-warn";
  return "";
}

function updateHero(frame) {
  els.mUsefulCell.classList.remove("is-warn", "is-crit");
  if (frame.warmup) {
    els.mUseful.textContent = `Buffering (${frame.warmup_remaining} left)…`;
    return;
  }
  const p = frame.pred_rul;
  if (p == null) {
    els.mUseful.textContent = "—";
    return;
  }
  const zc = zoneClass(p);
  if (zc) els.mUsefulCell.classList.add(zc);
  els.mUseful.textContent = frame.ttf_capped ? "> long (early life)" : fmtTime(frame.pred_remaining_s);
}

function applyAlert(rul) {
  if (rul == null) return;
  const warn = Number(els.warn.value) / 100;
  const crit = Number(els.crit.value) / 100;
  let state = "ok";
  if (rul <= crit) state = "crit";
  else if (rul <= warn) state = "warn";
  if (state !== lastAlertState) {
    if (state === "crit") log(`CRITICAL: RUL ${pct(rul)} ≤ critical threshold`, "crit");
    else if (state === "warn") log(`WARNING: RUL ${pct(rul)} ≤ warning threshold`, "warn");
    else log(`RUL recovered above warning threshold (${pct(rul)})`, "ok");
    lastAlertState = state;
  }
}

function appendFrame(frame, opts = {}) {
  if (!frame) return;
  storeFrame(frame);
  updateBranchGate(frame.branch_gate);
  updateDrivers(frame.top_drivers);
  updateHiTable(frame.hi_raw, frame.hi_scaled);
  updateHero(frame);

  els.mAcq.textContent = `${frame.t + 1} / ${frame.n_total}`;
  els.mElapsed.textContent = fmtTime(frame.elapsed_s);
  els.mEol.textContent = frame.pred_eol_iso ? fmtEol(frame.pred_eol_iso) : "—";
  els.mTruth.textContent = frame.gt_remaining_s == null ? "N/A (plant)" : fmtTime(frame.gt_remaining_s);

  const st = stageFor(frame.warmup ? null : frame.pred_rul);
  els.stage.textContent = st.text;
  els.stage.className = `v stage stage-${st.cls}`;

  if (!frame.warmup) applyAlert(frame.pred_rul);
  if (frame.warmup && frame.warmup_remaining === 0) log("Warm-up complete — predictions live", "ok");

  els.seek.value = String(frame.t);
  els.seekIdx.textContent = String(frame.t + 1);
  els.seekMax.textContent = String(frame.n_total);

  renderAll();

  if (!opts.noStore) {
    runFrames.push(frame);
    if (!frame.warmup) els.btnExplain.disabled = false;
  }
}

function renderIg(msg) {
  if (!msg.ok) {
    log(`Explain failed: ${msg.reason || "unknown"}`, "warn");
    return;
  }
  igFeatures = msg.features || [];
  drawAttr();
  drawHeatmap(msg.heatmap);
  log(`Integrated Gradients computed (${msg.n_steps} steps) at acq ${msg.t + 1}`, "ok");
}

function resetUi() {
  clearSeries();
  igFeatures = [];
  els.heatgrid.innerHTML = "";
  els.mUsefulCell.classList.remove("is-warn", "is-crit");
  els.mUseful.textContent = "—";
  els.mEol.textContent = "—";
  els.mTruth.textContent = "—";
  els.mAcq.textContent = "— / —";
  els.mElapsed.textContent = "—";
  els.stage.textContent = "—";
  els.stage.className = "v stage";
  updateDrivers(null);
  updateHiTable(null, null);
  renderAll();
}

async function loadDatasets() {
  const res = await fetch("/api/datasets");
  const data = await res.json();
  datasets = data.datasets || [];
  els.dataset.innerHTML = datasets
    .map((d) => `<option value="${d.key}">${d.label}</option>`)
    .join("");
  els.dataset.dispatchEvent(new Event("change"));
}

function populateBearings() {
  const key = els.dataset.value;
  currentSpec = datasets.find((d) => d.key === key) || null;
  if (!currentSpec) return;
  const labels = currentSpec.bearing_labels || {};
  els.bearing.innerHTML = currentSpec.test_bearings
    .map((b) => `<option value="${b}">${labels[b] || b}</option>`)
    .join("");
  featureNames = currentSpec.feature_names || [];
  els.infoModel.textContent = currentSpec.model || "—";
  els.infoCkpt.textContent = currentSpec.checkpoint || "—";
  els.infoWindow.textContent = currentSpec.window_length ?? "—";
  els.infoInterval.textContent = currentSpec.acquisition_interval_s
    ? fmtTime(currentSpec.acquisition_interval_s)
    : "—";
  showGt = currentSpec.has_gt_rul !== false;
  if (els.gtLegend) els.gtLegend.classList.toggle("hidden", !showGt);
  if (currentSpec.transfer_note) {
    els.transferNote.hidden = false;
    els.transferNote.textContent = currentSpec.transfer_note;
  } else {
    els.transferNote.hidden = true;
    els.transferNote.textContent = "";
  }
}

function connectWs() {
  if (ws) {
    ws.close();
    ws = null;
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.onopen = () => setStatus("streaming", "Connected");
  ws.onclose = () => {
    setStatus("idle", "Disconnected");
    streaming = false;
    paused = false;
    els.btnStart.disabled = false;
    els.btnPause.disabled = true;
    els.btnReset.disabled = true;
  };
  ws.onerror = () => setStatus("idle", "Error");
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "frame") appendFrame(msg);
    if (msg.type === "started") {
      setStatus("streaming", "Streaming");
      els.btnPause.disabled = false;
      els.btnReset.disabled = false;
      els.btnStep.disabled = false;
      els.btnCsv.disabled = false;
      els.infoDevice.textContent = msg.device || "—";
      els.infoNtotal.textContent = msg.n_total ?? "—";
      els.infoWindow.textContent = msg.window_length ?? els.infoWindow.textContent;
      els.infoCkpt.textContent = msg.checkpoint || els.infoCkpt.textContent;
      if (msg.feature_names && msg.feature_names.length) featureNames = msg.feature_names;
      els.seek.max = String(Math.max(0, (msg.n_total ?? 1) - 1));
      els.seek.disabled = false;
      els.seekMax.textContent = String(msg.n_total ?? "—");
      log(`Stream started — ${msg.n_total} acquisitions on ${msg.device}`, "ok");
    }
    if (msg.type === "done") {
      setStatus("idle", "Complete");
      streaming = false;
      els.btnStart.disabled = false;
      els.btnPause.disabled = true;
      log("Stream complete (end of run)", "ok");
    }
    if (msg.type === "paused") {
      setStatus("paused", "Paused");
      paused = true;
      log("Paused", "warn");
    }
    if (msg.type === "resumed") {
      setStatus("streaming", "Streaming");
      paused = false;
      sendStream();
    }
    if (msg.type === "reset") {
      resetUi();
      log("Reset", "ok");
    }
    if (msg.type === "seek") {
      clearSeries();
      runFrames = [];
      if (msg.frame) appendFrame(msg.frame, { noStore: true });
      log(`Sought to acquisition ${msg.t + 1}`, "ok");
    }
    if (msg.type === "explanation") renderIg(msg);
    if (msg.type === "error") {
      alert(msg.message);
      setStatus("idle", "Error");
      log(`Error: ${msg.message}`, "crit");
    }
  };
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function sendStream() {
  if (!streaming || paused) return;
  send({
    action: "stream",
    dataset: els.dataset.value,
    bearing: els.bearing.value,
    speed_ms: Number(els.speed.value),
  });
}

els.dataset.addEventListener("change", populateBearings);

els.speed.addEventListener("input", () => {
  els.speedVal.textContent = els.speed.value;
  if (streaming && !paused) send({ action: "set_speed", speed_ms: Number(els.speed.value) });
});

[els.warn, els.crit].forEach((el) => {
  el.addEventListener("input", () => renderAll());
});

els.btnStart.addEventListener("click", () => {
  resetUi();
  runFrames = [];
  lastAlertState = "ok";
  els.btnExplain.disabled = true;
  connectWs();
  ws.onopen = () => {
    setStatus("streaming", "Streaming");
    streaming = true;
    paused = false;
    els.btnStart.disabled = true;
    els.btnPause.disabled = false;
    els.btnReset.disabled = false;
    els.btnPause.textContent = "❚❚ Pause";
    sendStream();
  };
});

els.btnPause.addEventListener("click", () => {
  if (paused) {
    send({ action: "resume" });
    paused = false;
    els.btnPause.textContent = "❚❚ Pause";
  } else {
    send({ action: "pause" });
    paused = true;
    els.btnPause.textContent = "▶ Resume";
    setStatus("paused", "Paused");
  }
});

els.btnStep.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    connectWs();
    ws.onopen = () => send({ action: "step", dataset: els.dataset.value, bearing: els.bearing.value });
    return;
  }
  send({ action: "step", dataset: els.dataset.value, bearing: els.bearing.value });
});

els.btnReset.addEventListener("click", () => {
  send({ action: "reset" });
  resetUi();
  runFrames = [];
  lastAlertState = "ok";
  els.btnExplain.disabled = true;
});

els.btnExplain.addEventListener("click", () => {
  log("Requesting Integrated Gradients…", "info");
  send({ action: "explain" });
});

els.seek.addEventListener("change", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    connectWs();
    ws.onopen = () =>
      send({ action: "seek", t: Number(els.seek.value), dataset: els.dataset.value, bearing: els.bearing.value });
    return;
  }
  send({ action: "seek", t: Number(els.seek.value) });
});

els.btnClear.addEventListener("click", () => {
  els.log.innerHTML = "";
  logEntries = [];
});

function download(filename, text, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

els.btnLogExport.addEventListener("click", () => {
  download(`rul-log-${Date.now()}.txt`, logEntries.join("\n"));
});

els.btnCsv.addEventListener("click", () => {
  if (!runFrames.length) return;
  const cols = [
    "t", "elapsed_s", "pred_rul", "gt_rul",
    "pred_remaining_s", "pred_total_life_s", "ttf_capped", "pred_eol_iso", "gt_remaining_s",
  ];
  const header = cols.join(",");
  const lines = runFrames.map((f) =>
    cols.map((c) => (f[c] == null ? "" : String(f[c]).replace(/,/g, ";"))).join(",")
  );
  download(`rul-run-${els.bearing.value}-${Date.now()}.csv`, [header, ...lines].join("\n"), "text/csv");
  log("Run CSV exported", "ok");
});

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderAll, 150);
});

setInterval(() => {
  els.footClock.textContent = new Date().toLocaleTimeString([], { hour12: false });
}, 1000);

setStatus("idle", "Idle");
loadDatasets().catch((e) => console.error(e));
