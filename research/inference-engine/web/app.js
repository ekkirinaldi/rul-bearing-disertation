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
  insightPanel: document.getElementById("insightPanel"),
  insightState: document.getElementById("insightState"),
  insightHeadline: document.getElementById("insightHeadline"),
  insightSees: document.getElementById("insightSees"),
  insightAction: document.getElementById("insightAction"),
  accRmse: document.getElementById("accRmse"),
  accMae: document.getElementById("accMae"),
  accR2: document.getElementById("accR2"),
  accPhm: document.getElementById("accPhm"),
  accBackbone: document.getElementById("accBackbone"),
  accParams: document.getElementById("accParams"),
  accEpoch: document.getElementById("accEpoch"),
  accCap: document.getElementById("accCap"),
  bpfxVerdict: document.getElementById("bpfxVerdict"),
  bpfxList: document.getElementById("bpfxList"),
  bpfxFoot: document.getElementById("bpfxFoot"),
  bpfxCaveat: document.getElementById("bpfxCaveat"),
  bpfxCap: document.getElementById("bpfxCap"),
  hiThirdLabel: document.getElementById("hiThirdLabel"),
  igCaption: document.getElementById("igCaption"),
  accNote: document.getElementById("accNote"),
  runSelect: document.getElementById("runSelect"),
  failZone: document.getElementById("failZone"),
  dropCount: document.getElementById("dropCount"),
  dropList: document.getElementById("dropList"),
  dropHeadline: document.getElementById("dropHeadline"),
  hiDelta: document.getElementById("hiDelta"),
};

/* Plain-language names for the HI feature codes, used to explain the
   "top drivers" to non-technical users. Keyed by the suffix after the
   channel token (td_c0_<suffix> / fd_c1_<suffix>). */
const FEATURE_PLAIN = {
  rms: "overall vibration energy",
  peak: "sharp peak impacts",
  kurtosis: "spikiness of the vibration",
  skewness: "lopsided vibration pattern",
  crest: "impact sharpness",
  shape: "waveform shape",
  impulse: "impulsiveness",
  margin: "impact margin",
  variance: "spread of the vibration",
  centroid: "pitch of the vibration",
  entropy: "irregularity across frequencies",
  mean_freq: "average frequency",
  rms_freq: "frequency energy",
  band0: "low-frequency energy",
  band1: "low-mid-frequency energy",
  band2: "mid-frequency energy",
  band3: "high-mid-frequency energy",
  band4: "high-frequency energy",
};

function driverPlain(name) {
  // name like "td_c0_skewness" or "fd_c1_band0"
  const m = /^(td|fd)_c(\d+)_(.+)$/.exec(name || "");
  if (!m) return name || "a vibration feature";
  const axis = m[2] === "0" ? "horizontal" : "vertical";
  const phrase = FEATURE_PLAIN[m[3]] || m[3].replace(/_/g, " ");
  return `${phrase} (${axis})`;
}

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

// Replay state — a loaded saved run animated/scrubbed entirely client-side.
let replayMode = false;
let replayRun = null;
let replayDrops = [];
let replayTimer = null;
let selectedDrop = null;
// HI feature overlaid on the Health Indicator Trend to explain a drop:
// {idx, name, label} or null.
let hiOverlay = null;
// run_id → {dataset, bearing, label} for the saved-run picker.
let runsIndex = {};

// When true, the live view tracks the newest acquisition. Dragging SEEK back to
// an earlier acquisition detaches it (DVR-style review of buffered frames);
// scrubbing to the live edge re-attaches.
let followLive = true;

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

  if (replayMode && replayDrops.length) {
    replayDrops.forEach((d) => {
      if (d.to_t < start || d.to_t > end) return;
      const sel = selectedDrop && selectedDrop.to_t === d.to_t;
      const X = xAt(d.to_t);
      const fromP = d.from_rul != null ? d.from_rul * 100 : 100;
      const toP = d.to_rul != null ? d.to_rul * 100 : 0;
      ctx.strokeStyle = sel ? C.red : "rgba(167,43,30,.5)";
      ctx.lineWidth = sel ? 2 : 1;
      ctx.setLineDash(sel ? [] : [3, 3]);
      ctx.beginPath();
      ctx.moveTo(X + 0.5, y0);
      ctx.lineTo(X + 0.5, y1);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = sel ? C.red : C.rust;
      ctx.beginPath();
      ctx.moveTo(X, yAt(fromP) - 5);
      ctx.lineTo(X - 4, yAt(fromP) - 12);
      ctx.lineTo(X + 4, yAt(fromP) - 12);
      ctx.closePath();
      ctx.fill();
      if (sel) {
        ctx.fillStyle = "rgba(167,43,30,.12)";
        ctx.fillRect(xAt(d.from_t), y0, X - xAt(d.from_t), y1 - y0);
      }
    });
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
    ctx.fillText("Select a drop in Drop Events", w / 2, h / 2 - 8);
    ctx.fillText("to see its input attribution", w / 2, h / 2 + 8);
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

// Vertical drop markers (+ shaded span for the selected drop) on a time chart.
function drawDropMarkers(ctx, xAt, y0, y1, start, end) {
  if (!replayMode || !replayDrops.length) return;
  replayDrops.forEach((d) => {
    if (d.to_t < start || d.to_t > end) return;
    const sel = selectedDrop && selectedDrop.to_t === d.to_t;
    const X = xAt(d.to_t);
    if (sel) {
      ctx.fillStyle = "rgba(167,43,30,.10)";
      ctx.fillRect(xAt(d.from_t), y0, X - xAt(d.from_t), y1 - y0);
    }
    ctx.strokeStyle = sel ? C.red : "rgba(167,43,30,.45)";
    ctx.lineWidth = sel ? 2 : 1;
    ctx.setLineDash(sel ? [] : [3, 3]);
    ctx.beginPath();
    ctx.moveTo(X + 0.5, y0);
    ctx.lineTo(X + 0.5, y1);
    ctx.stroke();
    ctx.setLineDash([]);
  });
}

// Trajectory of one raw HI feature (from the recorded frames), normalised to the
// visible window — used to plot the "why it dropped" feature on the HI chart.
function drawHiOverlay(ctx, xAt, x0, y0, y1, start, end) {
  if (!hiOverlay || hiOverlay.idx < 0) return;
  const valAt = (i) => {
    const f = runFrames[i];
    const v = f && f.hi_raw ? f.hi_raw[hiOverlay.idx] : null;
    return v == null || Number.isNaN(v) ? null : v;
  };
  let vmin = Infinity, vmax = -Infinity;
  for (let i = start; i <= end; i++) {
    const v = valAt(i);
    if (v == null) continue;
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
  }
  if (!(vmax > vmin)) return;
  const yOv = (v) => y1 - ((y1 - y0) * (v - vmin)) / (vmax - vmin);
  ctx.strokeStyle = C.amber;
  ctx.lineWidth = 1.8;
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  let started = false;
  for (let i = start; i <= end; i++) {
    const v = valAt(i);
    if (v == null) { started = false; continue; }
    const X = xAt(i), Y = yOv(v);
    if (!started) { ctx.moveTo(X, Y); started = true; }
    else ctx.lineTo(X, Y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = C.amber;
  ctx.font = `10px ${MONO}`;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(`◆ ${hiOverlay.label}`, x0 + 5, y0 + 11);
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

  // Link to "Why It Dropped": mark the drop(s) and overlay the changed feature.
  drawDropMarkers(ctx, xAt, y0, y1, start, end);
  drawHiOverlay(ctx, xAt, x0, y0, y1, start, end);
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

function setInsight(state, stateText, headlineHtml, sees, action) {
  if (els.insightPanel) els.insightPanel.dataset.state = state;
  if (els.insightState) els.insightState.textContent = stateText;
  if (els.insightHeadline) els.insightHeadline.innerHTML = headlineHtml;
  if (els.insightSees) els.insightSees.textContent = sees;
  if (els.insightAction) els.insightAction.textContent = action;
}

function describeDrivers(drivers) {
  if (!drivers || !drivers.length) return "Gathering enough readings to explain the assessment…";
  const top = drivers.slice(0, 3).map((d) => driverPlain(d.name));
  const uniq = [...new Set(top)];
  return `The model is mainly reacting to ${uniq.join(", ")}.`;
}

function updateInsight(frame) {
  const isPlant = currentSpec && currentSpec.has_gt_rul === false;
  const basis = isPlant ? " (based on the vibration health index)" : "";

  if (frame.warmup) {
    const left = frame.warmup_remaining;
    setInsight(
      "idle",
      "Warming up",
      `Collecting initial sensor readings — <b>${left}</b> more needed before the first health assessment.`,
      "Not enough history yet to explain a verdict.",
      "Please wait — the first assessment appears once enough data is buffered.",
    );
    return;
  }

  const rul = frame.pred_rul;
  if (rul == null) {
    setInsight("idle", "Standing by", "Waiting for the next prediction…", "—", "—");
    return;
  }

  const timeLeft = frame.ttf_capped
    ? "plenty (still early in life)"
    : fmtTime(frame.pred_remaining_s);
  const sees = describeDrivers(frame.top_drivers);
  const pctTxt = pct(rul);

  if (rul <= 0.05) {
    setInsight(
      "crit",
      "End of life",
      `This bearing has reached the <b>end of its useful life</b> in the replay${basis}.`,
      sees,
      "Replace the bearing now and inspect it for the root cause of failure.",
    );
  } else if (rul <= 0.2) {
    setInsight(
      "crit",
      "Critical",
      `<b>Critical wear.</b> Only about <b>${timeLeft}</b> of useful life remains (${pctTxt} health)${basis}.`,
      sees,
      "Prioritize replacement and inspect for bearing damage at the earliest opportunity.",
    );
  } else if (rul <= 0.4) {
    setInsight(
      "warn",
      "Early wear",
      `<b>Early signs of wear.</b> Roughly <b>${timeLeft}</b> of useful life left (${pctTxt} health)${basis}.`,
      sees,
      "Schedule an inspection at the next convenient maintenance window and keep monitoring.",
    );
  } else {
    setInsight(
      "ok",
      "Healthy",
      `This bearing looks <b>healthy</b> — about <b>${timeLeft}</b> of useful life remaining (${pctTxt} health)${basis}.`,
      sees,
      "No action needed. Continue routine monitoring.",
    );
  }
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

function renderFrameUi(frame) {
  updateBranchGate(frame.branch_gate);
  updateDrivers(frame.top_drivers);
  updateHiTable(frame.hi_raw, frame.hi_scaled);
  updateHero(frame);
  updateInsight(frame);

  els.mAcq.textContent = `${frame.t + 1} / ${frame.n_total}`;
  els.mElapsed.textContent = fmtTime(frame.elapsed_s);
  els.mEol.textContent = frame.pred_eol_iso ? fmtEol(frame.pred_eol_iso) : "—";
  els.mTruth.textContent = frame.gt_remaining_s == null ? "N/A (plant)" : fmtTime(frame.gt_remaining_s);

  const st = stageFor(frame.warmup ? null : frame.pred_rul);
  els.stage.textContent = st.text;
  els.stage.className = `v stage stage-${st.cls}`;

  els.seek.value = String(frame.t);
  els.seekIdx.textContent = String(frame.t + 1);
  els.seekMax.textContent = String(frame.n_total);

  renderAll();
}

function appendFrame(frame, opts = {}) {
  if (!frame) return;
  storeFrame(frame);

  // Buffer every streamed frame by acquisition index so SEEK can replay it
  // instantly without a server round-trip.
  if (!opts.noStore) {
    runFrames[frame.t] = frame;
    if (!frame.warmup) {
      applyAlert(frame.pred_rul);
    } else if (frame.warmup_remaining === 0) {
      log("Warm-up complete — predictions live", "ok");
    }
  }

  // Only move the view when following the live edge (or for an explicit
  // single-frame display such as a server seek result).
  if (followLive || opts.noStore) {
    currentIdx = frame.t;
    if (frame.waveform) latestWaveform = frame.waveform;
    renderFrameUi(frame);
  }
}

function showBufferedFrame(t) {
  const f = runFrames[t];
  if (!f) return false;
  currentIdx = f.t;
  if (f.waveform) latestWaveform = f.waveform;
  renderFrameUi(f);
  return true;
}

// Scrub the buffered live run to acquisition ``t`` — instant, no server call.
// Lands on the latest buffered frame if ``t`` is past the streamed head.
function liveSeek(t) {
  if (!runFrames.length) return;
  let head = runFrames.length - 1;
  while (head > 0 && !runFrames[head]) head--;
  const idx = Math.max(0, Math.min(t, head));
  followLive = idx >= head;
  showBufferedFrame(idx);
  if (streaming && !paused) {
    if (followLive) setStatus("streaming", "Streaming");
    else setStatus("paused", `Reviewing acq ${idx + 1} / ${runFrames[idx]?.n_total ?? "—"}`);
  }
}

function renderIg(msg) {
  if (!msg.ok) {
    // Streamed IG failures are silent (throttled, best-effort); only log manual ones.
    if (!msg.streaming) log(`Explain failed: ${msg.reason || "unknown"}`, "warn");
    return;
  }
  igFeatures = msg.features || [];
  drawAttr();
  drawHeatmap(msg.heatmap);
  if (els.igCaption) {
    els.igCaption.textContent = `IG @ acq ${msg.t + 1} · ${msg.n_steps} steps · ${msg.streaming ? "auto" : "manual"}`;
  }
  // The auto stream fires ~every 12 acq — don't spam the event log with it.
  if (!msg.streaming) log(`Integrated Gradients computed (${msg.n_steps} steps) at acq ${msg.t + 1}`, "ok");
}

function resetUi() {
  clearSeries();
  followLive = true;
  igFeatures = [];
  els.heatgrid.innerHTML = "";
  if (els.igCaption) els.igCaption.textContent = "No drop selected";
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
  setInsightIdle();
  renderAll();
}

function setInsightIdle() {
  const isPlant = currentSpec && currentSpec.has_gt_rul === false;
  const what = isPlant
    ? "live plant data from PT SKF (no labelled failure point — the system estimates health from vibration alone)"
    : "a run-to-failure test bearing";
  setInsight(
    "idle",
    "Ready",
    `Press <b>Start</b> to run and replay ${what}, with its health and every insight aligned from the first acquisition.`,
    "Waiting for the first readings…",
    "No action yet — press Start to begin.",
  );
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
  if (els.hiThirdLabel) {
    const skf = currentSpec.stream_mode === "skf_trending";
    els.hiThirdLabel.textContent = skf ? "gE Envelope" : "Kurt H";
    els.hiThirdLabel.title = skf
      ? "Acceleration-envelope (gE) health index from the SKF Observer export — the industry-standard early bearing-fault indicator. Replaces kurtosis for plant streams."
      : "Spikiness of the vibration — high values flag sharp, repeated impacts.";
  }
  if (currentSpec.transfer_note) {
    els.transferNote.hidden = false;
    els.transferNote.textContent = currentSpec.transfer_note;
  } else {
    els.transferNote.hidden = true;
    els.transferNote.textContent = "";
  }
  setInsightIdle();
  loadDissertation(key);
}

function fmt3(v) {
  return v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(3).replace(".", ",");
}

function fmtParams(n) {
  if (n == null) return "—";
  return `${Math.round(n / 1000).toLocaleString()}k`;
}

async function loadDissertation(key) {
  try {
    const res = await fetch(`/api/dissertation/${encodeURIComponent(key)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderDissertation(await res.json());
  } catch (e) {
    renderDissertation(null);
  }
}

function renderAccuracy(d) {
  const m = d && d.metrics;
  els.accRmse.textContent = m ? fmt3(m.rmse) : "—";
  els.accMae.textContent = m ? fmt3(m.mae) : "—";
  els.accR2.textContent = m ? fmt3(m.r2) : "—";
  els.accPhm.textContent = m ? fmt3(m.phm_score) : "—";
  els.accBackbone.textContent = d && d.backbone ? d.backbone.name : "—";
  els.accParams.textContent = m ? fmtParams(m.n_params) : "—";
  els.accEpoch.textContent = m && m.best_epoch != null ? `${m.best_epoch} / 75` : "—";
  els.accCap.textContent =
    d && d.is_transfer
      ? `Held-out test accuracy of the transfer source model (${(m && m.source_dataset) || "benchmark"}). Plant data has no labelled RUL to score against.`
      : "Test-set accuracy of the trained model (Bab V). Lower RMSE/MAE is better; higher R² and PHM Score are better.";
  if (els.accNote) {
    const note = d && d.backbone && d.backbone.note;
    els.accNote.hidden = !note;
    els.accNote.textContent = note || "";
  }
}

function renderBpfx(d) {
  const bpfx = d && d.bpfx;
  const sae = d && d.sae;
  els.bpfxList.innerHTML = "";
  if (!bpfx) {
    els.bpfxVerdict.hidden = true;
    els.bpfxFoot.hidden = true;
    if (els.bpfxCaveat) els.bpfxCaveat.hidden = true;
    const msg = (d && d.bpfx_note) || "SAE→BPFx mapping is available on the benchmark datasets (PHM2012, XJTU-SY).";
    els.bpfxList.innerHTML = `<div class="bpfx-na">${msg}</div>`;
    return;
  }

  const maxHit = Math.max(...bpfx.bands.map((b) => b.hit_rate || 0), 1e-6);
  els.bpfxList.innerHTML = bpfx.bands
    .map((b) => {
      const w = Math.max(2, ((b.hit_rate || 0) / maxHit) * 100);
      const freq = b.freq_hz != null ? `${b.freq_hz.toFixed(0)} Hz` : "";
      return `<div class="bpfx-row ${b.dominant ? "is-dominant" : ""}" title="${b.full} — ${b.fault}">
        <span class="nm">${b.bpfx} <span class="freq">${freq}</span></span>
        <span class="track"><span class="fill" style="width:${w}%"></span></span>
        <span class="pc">${((b.hit_rate || 0) * 100).toFixed(2)}%</span>
      </div>`;
    })
    .join("");

  // Grounded verdict/caveat come from the server (Bab V §V.4–§V.8) — faithful,
  // no overclaim. Rendered as plain text.
  if (bpfx.verdict) {
    els.bpfxVerdict.hidden = false;
    els.bpfxVerdict.textContent = bpfx.verdict;
  } else {
    els.bpfxVerdict.hidden = true;
  }

  const bf = bpfx.best_feature;
  const saeTxt = sae
    ? `${sae.d_latent}-dim Top-k SAE, k=${sae.k} (~${sae.sparsity_pct}% sparsity)`
    : `${bpfx.d_latent}-dim dictionary`;
  if (bf) {
    els.bpfxFoot.hidden = false;
    els.bpfxFoot.textContent = `Strongest single latent: SAE feature #${bf.feature_idx} vs ${bf.bpfx}, Pearson r = ${fmt3(bf.r)} · ${saeTxt} over ${bpfx.n_recordings} recordings (|r| ≥ ${bpfx.corr_threshold} counts as a hit).`;
  } else {
    els.bpfxFoot.hidden = true;
  }

  if (els.bpfxCaveat) {
    els.bpfxCaveat.hidden = !bpfx.caveat;
    els.bpfxCaveat.textContent = bpfx.caveat || "";
  }
}

function renderDissertation(d) {
  renderAccuracy(d);
  renderBpfx(d);
}

/* ============================ RUN LIBRARY / REPLAY ============================ */

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

async function refreshRuns(selectId) {
  try {
    const res = await fetch("/api/runs");
    const data = await res.json();
    const runs = data.runs || [];
    runsIndex = {};
    const groups = {};
    runs.forEach((r) => {
      const m = r.meta || {};
      runsIndex[r.run_id] = { dataset: m.dataset, bearing: m.bearing, label: m.bearing_label || m.bearing };
      const key = m.dataset_label || m.dataset || "runs";
      (groups[key] = groups[key] || []).push({ r, m, s: r.summary || {} });
    });
    let html = `<option value="">— ${runs.length ? "select a recorded run" : "record runs first"} —</option>`;
    Object.keys(groups).forEach((g) => {
      html += `<optgroup label="${escapeHtml(g)}">`;
      groups[g].forEach(({ r, m, s }) => {
        const drops = s.n_drops != null ? ` · ${s.n_drops} drops` : "";
        html += `<option value="${escapeHtml(r.run_id)}">${escapeHtml(m.bearing_label || m.bearing)}${drops}</option>`;
      });
      html += "</optgroup>";
    });
    els.runSelect.innerHTML = html;
    if (selectId) els.runSelect.value = selectId;
    else if (replayMode && replayRun) els.runSelect.value = replayRun.run_id;
  } catch (e) {
    console.error(e);
  }
}

async function loadReplay(runId, opts = {}) {
  if (!runId) return;
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    const run = await res.json();
    if (run.error) {
      log(`Load failed: ${run.error}`, "warn");
      return;
    }
    enterReplay(run, opts);
  } catch (e) {
    log(`Load failed: ${e}`, "warn");
  }
}

function enterReplay(run, opts = {}) {
  replayPause();
  // Switch the dashboard to match the recorded run's dataset (drives GT visibility,
  // HI labels, dissertation context) without sending anything to the server.
  if (run.meta && run.meta.dataset && els.dataset.value !== run.meta.dataset) {
    els.dataset.value = run.meta.dataset;
    populateBearings();
  }
  if (run.meta && run.meta.bearing) els.bearing.value = run.meta.bearing;
  if (run.meta && run.meta.feature_names && run.meta.feature_names.length) {
    featureNames = run.meta.feature_names;
  }

  replayMode = true;
  replayRun = run;
  replayDrops = run.drops || [];
  selectedDrop = null;
  hiOverlay = null;
  streaming = false;
  paused = false;

  clearSeries();
  runFrames = run.frames || [];
  runFrames.forEach(storeFrame);

  const n = runFrames.length;
  els.seek.max = String(Math.max(0, n - 1));
  els.seek.disabled = false;
  els.seekMax.textContent = String(n);
  els.infoNtotal.textContent = run.meta?.n_total ?? n;
  els.infoDevice.textContent = run.meta?.device || els.infoDevice.textContent;
  els.infoCkpt.textContent = run.meta?.checkpoint || els.infoCkpt.textContent;

  // The acquisition controls now drive replay playback (Pause/Step/Reset/Seek).
  els.btnCsv.disabled = false;
  els.btnStart.disabled = false;
  els.btnPause.disabled = false;
  els.btnStep.disabled = false;
  els.btnReset.disabled = false;

  // IG is drop-anchored: start empty until a drop is selected.
  igFeatures = [];
  els.heatgrid.innerHTML = "";
  if (els.igCaption) els.igCaption.textContent = "No drop selected";

  if (runsIndex[run.run_id]) els.runSelect.value = run.run_id;
  els.failZone.hidden = false;

  renderDropList();
  if (opts.autoplay) {
    // Start-driven playback: begin at acquisition 0 with everything aligned.
    replaySeek(0);
    replayPlay();
  } else if (replayDrops.length) {
    // Manual load: land on the first significant drop for analysis.
    setStatus("paused", "Replay loaded");
    showDrop(replayDrops[0]);
  } else {
    setStatus("paused", "Replay loaded");
    replaySeek(n - 1);
  }
  log(`Replay loaded — ${run.run_id} (${n} frames, ${replayDrops.length} drops)`, "ok");
}

function exitReplay() {
  replayPause();
  replayMode = false;
  replayRun = null;
  replayDrops = [];
  selectedDrop = null;
  hiOverlay = null;
  els.failZone.hidden = true;
  els.dropList.innerHTML = '<div class="muted">Pick a recorded run to locate significant drops.</div>';
  resetUi();
  runFrames = [];
  // Restore the idle control state.
  els.btnStart.disabled = false;
  els.btnPause.disabled = true;
  els.btnStep.disabled = true;
  els.btnReset.disabled = true;
  els.btnCsv.disabled = true;
  els.seek.disabled = true;
  setStatus("idle", "Idle");
}

// Reflect playback state on the Pause button.
function setPlayingUi(playing) {
  els.btnPause.textContent = playing ? "❚❚ Pause" : "▶ Resume";
  els.btnPause.classList.toggle("warn", playing);
}

function replaySeek(t) {
  if (!replayMode || !runFrames.length) return;
  const idx = Math.max(0, Math.min(t, runFrames.length - 1));
  const frame = runFrames[idx];
  currentIdx = frame.t;
  if (frame.waveform) latestWaveform = frame.waveform;
  renderFrameUi(frame);
}

function replayPlay() {
  if (!replayMode || replayTimer) return;
  // Restart from the beginning if parked at the end.
  if (currentIdx >= runFrames.length - 1) replaySeek(0);
  const speed = Math.max(10, Number(els.speed.value) || 50);
  setPlayingUi(true);
  setStatus("streaming", "Replaying");
  replayTimer = setInterval(() => {
    const next = currentIdx + 1;
    if (next >= runFrames.length) {
      replayPause();
      setStatus("idle", "Replay complete");
      return;
    }
    replaySeek(next);
  }, speed);
}

function replayPause() {
  if (replayTimer) {
    clearInterval(replayTimer);
    replayTimer = null;
  }
  setPlayingUi(false);
  if (replayMode && replayTimer === null) setStatus("paused", "Replay paused");
}

function replayToggle() {
  if (replayTimer) replayPause();
  else replayPlay();
}

// Apply a new playback speed without interrupting an active play session.
function replayRetime() {
  if (!replayTimer) return;
  clearInterval(replayTimer);
  replayTimer = null;
  replayPlay();
}

function renderDropList() {
  els.dropCount.textContent = String(replayDrops.length);
  if (!replayDrops.length) {
    els.dropList.innerHTML = '<div class="muted">No significant drops detected in this run.</div>';
    return;
  }
  els.dropList.innerHTML = "";
  replayDrops.forEach((d, i) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "drop-row";
    const sel = selectedDrop && selectedDrop.to_t === d.to_t;
    if (sel) row.classList.add("is-sel");
    const dropPct = (d.magnitude * 100).toFixed(1);
    const fromPct = (d.from_rul * 100).toFixed(0);
    const toPct = (d.to_rul * 100).toFixed(0);
    const when = d.timestamp ? fmtEol(d.timestamp) : fmtTime(d.elapsed_s);
    row.innerHTML =
      `<span class="dn">#${i + 1}</span>` +
      `<span class="dmid"><span class="dmag">−${dropPct}%</span>` +
      `<span class="drange">${fromPct}% → ${toPct}% · acq ${d.to_t + 1}</span></span>` +
      `<span class="dwhen">${when}</span>`;
    row.addEventListener("click", () => showDrop(d));
    els.dropList.appendChild(row);
  });
}

function showDrop(drop) {
  selectedDrop = drop;
  replayPause();
  replaySeek(drop.to_t);
  renderDropList();

  // Override the live-attribution widgets with this drop's saved explanation.
  if (drop.branch_gate) updateBranchGate(drop.branch_gate);
  if (drop.drivers) updateDrivers(drop.drivers);
  if (drop.ig) {
    igFeatures = drop.ig.features || [];
    drawAttr();
    drawHeatmap(drop.ig.heatmap);
    if (els.igCaption) els.igCaption.textContent = `IG @ drop · acq ${drop.to_t + 1} · ${drop.ig.n_steps} steps`;
  }
  // Auto-overlay the most-changed HI feature onto the Health Indicator Trend so
  // the "why" is visible on the chart; clicking another row switches it.
  const top = (drop.hi_delta || [])[0];
  const topIdx = top ? featureNames.indexOf(top.name) : -1;
  hiOverlay = top && topIdx >= 0 ? { idx: topIdx, name: top.name, label: driverPlain(top.name) } : null;

  renderHiDelta(drop);
  renderAll();
}

function renderHiDelta(drop) {
  const dropPct = (drop.magnitude * 100).toFixed(1);
  const top = (drop.hi_delta || []).slice(0, 3).map((h) => driverPlain(h.name));
  const uniq = [...new Set(top)];
  const lead = uniq.length
    ? `Health fell <b>${dropPct}%</b> (acq ${drop.to_t + 1}). The biggest physical change was in ${uniq.join(", ")}.`
    : `Health fell <b>${dropPct}%</b> at acquisition ${drop.to_t + 1}.`;
  els.dropHeadline.innerHTML = lead;

  const hd = drop.hi_delta || [];
  if (!hd.length) {
    els.hiDelta.innerHTML = '<div class="muted">No raw HI deltas captured.</div>';
    return;
  }
  const maxRel = Math.max(...hd.map((h) => h.rel_change), 1e-9);
  els.hiDelta.innerHTML = hd
    .map((h) => {
      const w = Math.round((h.rel_change / maxRel) * 100);
      const arrow = h.dir === "up" ? "▲" : "▼";
      const cls = h.dir === "up" ? "up" : "dn";
      const fidx = featureNames.indexOf(h.name);
      const active = hiOverlay && hiOverlay.idx === fidx;
      const fmtNum = (v) => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toFixed(2));
      return (
        `<div class="hid-row${active ? " is-active" : ""}" data-idx="${fidx}" data-name="${h.name}"` +
        ` title="Click to plot '${h.name}' on the Health Indicator Trend chart">` +
        `<span class="nm"><span class="t">${driverPlain(h.name)}</span> <span class="arr ${cls}">${arrow}</span></span>` +
        `<span class="track"><i class="${cls}" style="width:${w}%"></i></span>` +
        `<span class="pc">${fmtNum(h.before)} → ${fmtNum(h.after)}</span>` +
        `</div>`
      );
    })
    .join("");
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
    streaming = false;
    paused = false;
    // A dropped socket must not disable the replay player (which runs offline).
    if (!replayMode) {
      setStatus("idle", "Disconnected");
      els.btnStart.disabled = false;
      els.btnPause.disabled = true;
      els.btnReset.disabled = true;
    }
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

// The dataset/bearing selects are hidden and driven programmatically by the
// loaded run; this keeps dataset-derived UI state (GT, HI labels, dissertation)
// in sync when a replay loads.
els.dataset.addEventListener("change", populateBearings);

els.speed.addEventListener("input", () => {
  els.speedVal.textContent = els.speed.value;
  if (replayMode) replayRetime();
});

[els.warn, els.crit].forEach((el) => {
  el.addEventListener("input", () => renderAll());
});

// Play: (re)start the loaded replay from the beginning, or load the picked run.
els.btnStart.addEventListener("click", () => {
  if (replayMode && replayRun) {
    replayPause();
    replaySeek(0);
    replayPlay();
  } else if (els.runSelect.value) {
    loadReplay(els.runSelect.value, { autoplay: true });
  } else {
    log("No recorded runs yet — use Run Library → Record All", "warn");
  }
});

// Pause/Resume the replay playback.
els.btnPause.addEventListener("click", () => {
  if (replayMode) replayToggle();
});

// Step one acquisition forward through the replay.
els.btnStep.addEventListener("click", () => {
  if (!replayMode) return;
  replayPause();
  replaySeek(Math.min(currentIdx + 1, runFrames.length - 1));
});

// Reset the replay back to acquisition 0 (paused).
els.btnReset.addEventListener("click", () => {
  if (!replayMode) return;
  replayPause();
  selectedDrop = null;
  hiOverlay = null;
  igFeatures = [];
  els.heatgrid.innerHTML = "";
  if (els.igCaption) els.igCaption.textContent = "No drop selected";
  renderDropList();
  replaySeek(0);
});

// "Why It Dropped" → Health Indicator Trend: click a feature row to plot that
// feature's trajectory on the HI chart (toggles off if already shown).
els.hiDelta.addEventListener("click", (e) => {
  const row = e.target.closest(".hid-row");
  if (!row) return;
  const idx = Number(row.dataset.idx);
  const name = row.dataset.name;
  if (!Number.isInteger(idx) || idx < 0) return;
  hiOverlay = hiOverlay && hiOverlay.idx === idx ? null : { idx, name, label: driverPlain(name) };
  if (selectedDrop) renderHiDelta(selectedDrop);
  drawHealth();
});

function onSeekInput() {
  const t = Number(els.seek.value);
  if (replayMode) {
    replayPause();
    replaySeek(t);
  } else {
    liveSeek(t);
  }
}

els.seek.addEventListener("input", onSeekInput);
els.seek.addEventListener("change", onSeekInput);

// Picking a run from the library loads and plays it from the start.
els.runSelect.addEventListener("change", () => {
  if (els.runSelect.value) loadReplay(els.runSelect.value, { autoplay: true });
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
refreshRuns().catch((e) => console.error(e));
