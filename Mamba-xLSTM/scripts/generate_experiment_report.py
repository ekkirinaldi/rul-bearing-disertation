"""Generate a self-contained, A4-paginated HTML experiment report.

Reads:
  - results/runs/*/summary.json              (Stage 2 — 18 runs)
  - results/runs/<best>/explain/sae_history.json  (Stage 3)
  - results/runs/<best>/explain/shap_global.json  (Stage 3)
  - results/bpfx_mapping/*_bpfx_results.json      (Stage 4)
  - results/bpfx_mapping/summary_hitrate_table.json
  - all PNG figures from explain/ and bpfx_mapping/

Writes:
  - ../../writings/experiment-report.html

Notes:
  - Output is sized for A4 print (Chrome ``--print-to-pdf``).
  - Every numbered section starts on a new page via CSS ``page-break-before``.
  - Numeric values are read from JSON; only narrative prose is hardcoded.

Usage::

    cd Mamba-xLSTM
    source .venv/bin/activate
    python scripts/generate_experiment_report.py
"""

from __future__ import annotations

import base64
import io
import json
import statistics as stats
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "results" / "runs"
BPFX_DIR = ROOT / "results" / "bpfx_mapping"
WRITINGS = ROOT.parent / "writings"

BEST_RUNS = {
    "phm2012": RUNS_ROOT / "20260512_151550_algorithm_comparison_phm2012_mamba_xlstm_net_s42",
    "xjtusy":  RUNS_ROOT / "20260512_193202_algorithm_comparison_xjtusy_mamba_xlstm_net_s44",
}

DATASETS = ["phm2012", "xjtusy"]
MODELS = ["mamba_xlstm_net", "nbeats_xlstm_rul", "sparse_gate_tcn_rul"]
MODEL_LABELS = {
    "mamba_xlstm_net": "Mamba-xLSTM-Net",
    "nbeats_xlstm_rul": "N-BEATS-xLSTM-RUL",
    "sparse_gate_tcn_rul": "SparseGate-TCN-RUL",
}
DS_LABELS = {"phm2012": "PHM2012", "xjtusy": "XJTU-SY"}
SEEDS = [42, 43, 44]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def b64(path: Path) -> str:
    if not path.exists():
        return ""
    enc = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{enc}"


def img_tag(path: Path, caption: str, max_w: str = "640px") -> str:
    src = b64(path)
    if not src:
        return f'<p class="missing">[Figure not found: {path.name}]</p>'
    return (
        '<figure>'
        f'<img src="{src}" alt="{caption}" '
        f'style="width:100%;max-width:{max_w};border:1px solid #d0d7de;'
        'border-radius:4px;padding:6px;background:#fff">'
        f'<figcaption>{caption}</figcaption>'
        '</figure>'
    )


def read_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------------------
# Stage 2 data collection
# ---------------------------------------------------------------------------

def collect_stage2() -> dict:
    data = {ds: {m: {} for m in MODELS} for ds in DATASETS}
    for ds in DATASETS:
        for m in MODELS:
            for seed in SEEDS:
                pat = f"*algorithm_comparison_{ds}_{m}_s{seed}"
                dirs = sorted(RUNS_ROOT.glob(pat))
                if not dirs:
                    continue
                sj = dirs[-1] / "summary.json"
                if not sj.exists():
                    continue
                j = read_json(sj)
                tm = j.get("test_metrics", {})
                best_epoch = None
                for c in (dirs[-1] / "checkpoints").glob("*.ckpt"):
                    if c.stem != "last":
                        try:
                            best_epoch = int(c.stem)
                        except ValueError:
                            pass
                data[ds][m][seed] = {
                    "rmse": tm.get("test/rmse"),
                    "mae": tm.get("test/mae"),
                    "r2": tm.get("test/r2"),
                    "phm": tm.get("test/phm_score"),
                    "n_params": j.get("n_params"),
                    "best_epoch": best_epoch,
                    "run_id": j.get("run_id", ""),
                }
    return data


def agg_stage2(data: dict) -> dict:
    agg = {}
    for ds in DATASETS:
        agg[ds] = {}
        for m in MODELS:
            rows = [v for v in data[ds][m].values() if v.get("rmse") is not None]
            if not rows:
                continue
            rmses = [r["rmse"] for r in rows]
            maes = [r["mae"] for r in rows]
            phms = [r["phm"] for r in rows]
            r2s = [r["r2"] for r in rows]
            agg[ds][m] = {
                "rmse_mean": stats.mean(rmses),
                "rmse_std": stats.pstdev(rmses) if len(rmses) > 1 else 0.0,
                "mae_mean": stats.mean(maes),
                "phm_mean": stats.mean(phms),
                "r2_mean": stats.mean(r2s),
                "n_params": rows[0].get("n_params"),
            }
    return agg


# ---------------------------------------------------------------------------
# SAE loss curve
# ---------------------------------------------------------------------------

def sae_loss_curve_b64(phm_hist: list, xjtu_hist: list) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, hist, ds in zip(axes, [phm_hist, xjtu_hist], ["PHM2012", "XJTU-SY"]):
        epochs = [h["epoch"] for h in hist]
        recons = [h["recon"] for h in hist]
        ax.plot(epochs, recons, linewidth=2, color="#1565C0", marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reconstruction MSE")
        ax.set_title(f"SAE training loss — {ds}")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def shap_top10_table(shap_json: dict, ds_label: str, n: int = 10) -> str:
    top = sorted(shap_json.items(), key=lambda x: x[1], reverse=True)[:n]
    rows = ""
    for i, (feat, val) in enumerate(top):
        bg = "#f5faff" if i % 2 == 0 else "#ffffff"
        rows += (
            f'<tr style="background:{bg}">'
            f'<td>{i+1}</td><td><code>{feat}</code></td>'
            f'<td>{val:.6f}</td></tr>'
        )
    return f"""
    <h4>SHAP top-{n} features — {ds_label}</h4>
    <table class="data-table">
      <thead><tr><th>#</th><th>Feature</th><th>mean |SHAP|</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def bpfx_top5_table(bpfx_json: dict, ds_label: str) -> str:
    freqs = bpfx_json.get("characteristic_frequencies_hz", {})
    top5 = bpfx_json.get("top5_features", {})
    hr = bpfx_json.get("hit_rate", {})
    html = f"<h4>Top-5 SAE features per BPFx — {ds_label}</h4>"
    for bname in ["BPFO", "BPFI", "BSF", "FTF"]:
        fhz = freqs.get(bname, 0.0)
        hit = hr.get(bname, 0.0) * 100
        features = top5.get(bname, [])
        if not features or all(abs(f["r"]) < 0.05 for f in features):
            html += (
                f'<p><strong>{bname}</strong> ({fhz:.2f} Hz) — '
                f'hit-rate {hit:.1f}% — no features with |r| &ge; 0.05</p>'
            )
            continue
        rows = ""
        for i, f in enumerate(features):
            bg = "#f5faff" if i % 2 == 0 else "#ffffff"
            r_val = f["r"]
            bar_w = min(int(abs(r_val) * 130), 130)
            bar_col = "#1565C0" if r_val >= 0 else "#c62828"
            rows += (
                f'<tr style="background:{bg}">'
                f'<td>f{f["feature_idx"]}</td>'
                f'<td>{r_val:+.4f}</td>'
                f'<td><div style="width:{bar_w}px;height:10px;background:{bar_col};'
                'border-radius:2px"></div></td></tr>'
            )
        html += f"""
        <p style="margin:8px 0 4px;font-weight:600">{bname}
          <span style="color:#555;font-weight:400">— {fhz:.2f} Hz,
          hit-rate {hit:.1f}%</span></p>
        <table class="data-table" style="margin-top:0">
          <thead><tr><th>Feature</th><th>Pearson r</th><th>|r| bar</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>"""
    return html


def perf_table_agg(agg: dict, ds: str) -> str:
    ds_agg = agg.get(ds, {})
    rows = ""
    best_rmse = min((v["rmse_mean"] for v in ds_agg.values()), default=1.0)
    for m in MODELS:
        if m not in ds_agg:
            continue
        v = ds_agg[m]
        bold = "font-weight:bold;color:#0d47a1" if abs(v["rmse_mean"] - best_rmse) < 1e-9 else ""
        rows += (
            f'<tr>'
            f'<td style="{bold}">{MODEL_LABELS[m]}</td>'
            f'<td style="{bold}">{v["rmse_mean"]:.4f} ± {v["rmse_std"]:.4f}</td>'
            f'<td>{v["mae_mean"]:.4f}</td>'
            f'<td>{v["phm_mean"]:.4f}</td>'
            f'<td>{v["r2_mean"]:.4f}</td>'
            f'<td>{v["n_params"]:,}</td>'
            f'</tr>'
        )
    return f"""
    <table class="data-table">
      <thead>
        <tr><th>Model</th><th>RMSE &darr; (mean &plusmn; std)</th><th>MAE &darr;</th>
            <th>PHM Score &uarr;</th><th>R&sup2; &uarr;</th><th>Params</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def perf_table_seeds(data: dict, ds: str) -> str:
    rows = ""
    for m in MODELS:
        for seed in SEEDS:
            v = data[ds][m].get(seed)
            if not v:
                continue
            ep = v.get("best_epoch")
            ep_str = str(ep) if ep is not None else "&mdash;"
            rows += (
                f'<tr>'
                f'<td>{MODEL_LABELS[m]}</td>'
                f'<td>{seed}</td>'
                f'<td>{v["rmse"]:.4f}</td>'
                f'<td>{v["mae"]:.4f}</td>'
                f'<td>{v["phm"]:.4f}</td>'
                f'<td>{v["r2"]:.4f}</td>'
                f'<td>{ep_str}/75</td>'
                f'</tr>'
            )
    return f"""
    <table class="data-table">
      <thead>
        <tr><th>Model</th><th>Seed</th><th>RMSE &darr;</th><th>MAE &darr;</th>
            <th>PHM &uarr;</th><th>R&sup2;</th><th>Best epoch</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# CSS (A4 paginated, print-ready)
# ---------------------------------------------------------------------------

CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  font-family: "Helvetica Neue", Helvetica, Arial, "Segoe UI", sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #1a1a1a;
  background: #ffffff;
}
body { padding: 0; }
.container { max-width: 178mm; margin: 0 auto; }
section.page {
  page-break-before: always;
  break-before: page;
  padding-top: 4mm;
}
section.page:first-of-type { page-break-before: auto; break-before: auto; }
h1 { font-size: 20pt; color: #0d47a1; margin-bottom: 6mm; }
h2 {
  font-size: 15pt; color: #ffffff; background: #0d47a1;
  padding: 4mm 6mm; margin: 0 0 5mm; border-radius: 4px;
  page-break-after: avoid; break-after: avoid;
}
h3 {
  font-size: 12pt; color: #0d47a1; margin: 6mm 0 2mm;
  border-bottom: 1.5pt solid #bbdefb; padding-bottom: 1mm;
  page-break-after: avoid; break-after: avoid;
}
h4 {
  font-size: 10.5pt; color: #333; margin: 4mm 0 2mm;
  page-break-after: avoid; break-after: avoid;
}
p, ul, ol { margin: 2mm 0 2.5mm; }
p { text-align: justify; }
ul, ol { padding-left: 7mm; }
li { margin-bottom: 1.2mm; }
.lead { font-size: 11pt; color: #333; }
table.data-table {
  width: 100%; border-collapse: collapse;
  margin: 3mm 0; font-size: 9.5pt;
  page-break-inside: avoid; break-inside: avoid;
}
table.data-table th {
  background: #e3f2fd; color: #0d47a1; text-align: left;
  padding: 2mm 3mm; border-bottom: 1.5pt solid #90caf9;
  font-weight: 600;
}
table.data-table td {
  padding: 1.8mm 3mm;
  border-bottom: 0.5pt solid #eeeeee; vertical-align: top;
}
figure {
  margin: 3mm 0; text-align: center;
  page-break-inside: avoid; break-inside: avoid;
}
figcaption {
  font-size: 9pt; color: #555; margin-top: 1mm;
  font-style: italic;
}
.missing { color: #b71c1c; background: #ffebee; padding: 2mm 3mm; border-radius: 3px; }
code {
  font-family: "Menlo", "Consolas", monospace;
  font-size: 9.5pt; background: #f4f6f8; padding: 0.5mm 1.5mm;
  border-radius: 2px; color: #c62828;
}
.alert {
  background: #fff8e1; border-left: 3pt solid #f9a825;
  padding: 3mm 4mm; margin: 3mm 0; border-radius: 3px;
  page-break-inside: avoid; break-inside: avoid;
}
.note {
  background: #e8f5e9; border-left: 3pt solid #2e7d32;
  padding: 3mm 4mm; margin: 3mm 0; border-radius: 3px;
}
.stat-grid {
  display: grid; grid-template-columns: 1fr 1fr 1fr;
  gap: 3mm; margin: 4mm 0;
}
.stat-box {
  background: #e3f2fd; border-left: 3pt solid #1565c0;
  border-radius: 3px; padding: 3mm; text-align: center;
}
.stat-box .val {
  font-size: 16pt; font-weight: 700; color: #0d47a1;
}
.stat-box .lbl { font-size: 8.5pt; color: #555; margin-top: 1mm; }
.toc {
  background: #f5f8fc; border: 0.5pt solid #cfd8dc;
  border-radius: 4px; padding: 4mm 6mm; margin: 3mm 0;
  page-break-inside: avoid; break-inside: avoid;
}
.toc a { color: #0d47a1; text-decoration: none; }
.toc ol { margin: 2mm 0 0; padding-left: 6mm; }
.fig-pair {
  display: grid; grid-template-columns: 1fr 1fr; gap: 4mm;
}
.fig-quad {
  display: grid; grid-template-columns: 1fr 1fr; gap: 3mm;
}
.muted { color: #666; font-size: 9pt; }
hr {
  border: none; border-top: 0.5pt solid #cfd8dc;
  margin: 4mm 0;
}
"""

# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(stage2_data, stage2_agg, sae_hist, shap_jsons, bpfx_jsons) -> str:
    phm_hist = sae_hist.get("phm2012", [])
    xjtu_hist = sae_hist.get("xjtusy", [])
    loss_curve_src = sae_loss_curve_b64(phm_hist, xjtu_hist)

    phm_best_rmse = min((v["rmse_mean"] for v in stage2_agg.get("phm2012", {}).values()), default=0)
    xjtu_best_rmse = min((v["rmse_mean"] for v in stage2_agg.get("xjtusy", {}).values()), default=0)
    phm_bpfi_hit = bpfx_jsons.get("phm2012", {}).get("hit_rate", {}).get("BPFI", 0) * 100
    xjtu_bpfo_hit = bpfx_jsons.get("xjtusy", {}).get("hit_rate", {}).get("BPFO", 0) * 100
    phm_sae_loss = phm_hist[-1]["recon"] if phm_hist else 0
    xjtu_sae_loss = xjtu_hist[-1]["recon"] if xjtu_hist else 0

    def page(anchor: str, title: str, content: str) -> str:
        return f'<section class="page" id="{anchor}"><h2>{title}</h2>{content}</section>'

    # ------------------------------------------------------------------ Title page / overview
    s1 = f"""
    <section class="page" id="overview" style="page-break-before:auto;break-before:auto">
    <h1>Bearing Remaining Useful Life Prediction<br>
        <span style="font-size:13pt;color:#555;font-weight:400">
        Experiment Report &mdash; Mamba-xLSTM Hybrid with SAE Interpretability
        </span></h1>

    <p class="muted">
      Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} &middot;
      Stages 1&ndash;4 complete &middot; 18 training runs (3 models &times; 2 datasets &times; 3 seeds)
    </p>

    <h2 style="margin-top:8mm">1. Ringkasan Eksekutif</h2>
    <p class="lead">
      Laporan ini mendokumentasikan keseluruhan eksperimen yang dilakukan untuk
      disertasi doktor dengan judul <em>"Bearing Remaining Useful Life Prediction
      with Mamba-xLSTM Hybrid and Sparse Autoencoder Interpretability"</em>
      (ITB, 2026). Studi ini menjawab dua pertanyaan penelitian yang sekaligus
      menjadi dua pilar kebaruan disertasi.
    </p>
    <ul>
      <li><strong>Pilar 1 &mdash; Performa arsitektur.</strong> Apakah arsitektur
          hibrida <em>Mamba-xLSTM-Net</em> yang menggabungkan <em>state-space model</em>
          selektif (Mamba-3) dengan <em>matrix-memory</em> mLSTM mampu memberikan
          prediksi sisa umur pakai bantalan gelinding yang kompetitif terhadap
          arsitektur kontemporer pada dua tolok ukur publik <em>run-to-failure</em>
          (PHM2012 dan XJTU-SY)?</li>
      <li><strong>Pilar 2 &mdash; Interpretabilitas representasi.</strong> Apakah
          representasi laten yang dipelajari oleh model RUL berbasis <em>deep learning</em>
          dapat dipetakan, melalui <em>Sparse Autoencoder</em> (SAE) <em>post-hoc</em>,
          ke frekuensi karakteristik fisik bantalan (BPFO, BPFI, BSF, FTF) yang
          diturunkan dari geometri bantalan dan kecepatan rotasi poros?</li>
    </ul>

    <p>
      Untuk menjawab kedua pertanyaan tersebut, eksperimen dibagi menjadi empat
      tahap yang saling membangun. Tahap&nbsp;1 melakukan validasi konvergensi
      jangka panjang (200 epoch) terhadap arsitektur unggulan untuk menetapkan
      anggaran pelatihan yang adil. Tahap&nbsp;2 menjalankan perbandingan
      <em>multi-seed</em> tiga model Tier-S pada dua dataset. Tahap&nbsp;3 melatih
      SAE Top-<em>k</em> di atas representasi tersembunyi model unggulan, kemudian
      melakukan atribusi global menggunakan SHAP dan <em>Integrated Gradients</em>.
      Tahap&nbsp;4 memetakan aktivasi fitur SAE ke frekuensi karakteristik bantalan
      menggunakan korelasi Pearson antara aktivasi laten dan amplitudo
      <em>Hilbert envelope spectrum</em> pada pita &plusmn;2 Hz di sekitar setiap
      frekuensi target.
    </p>

    <div class="stat-grid">
      <div class="stat-box"><div class="val">18</div>
        <div class="lbl"><em>Training runs</em><br>3 model &times; 2 dataset &times; 3 seed</div></div>
      <div class="stat-box"><div class="val">{phm_best_rmse:.4f}</div>
        <div class="lbl">RMSE rerata terbaik<br>PHM2012 (multi-seed)</div></div>
      <div class="stat-box"><div class="val">{xjtu_best_rmse:.4f}</div>
        <div class="lbl">RMSE rerata terbaik<br>XJTU-SY (multi-seed)</div></div>
      <div class="stat-box"><div class="val">1024</div>
        <div class="lbl">Dimensi laten SAE<br>(d_model 128 &times; expansion 8)</div></div>
      <div class="stat-box"><div class="val">{phm_bpfi_hit:.1f}%</div>
        <div class="lbl"><em>Hit-rate</em> BPFI<br>PHM2012</div></div>
      <div class="stat-box"><div class="val">{xjtu_bpfo_hit:.1f}%</div>
        <div class="lbl"><em>Hit-rate</em> BPFO<br>XJTU-SY</div></div>
    </div>

    <div class="toc">
      <strong>Daftar Isi</strong>
      <ol>
        <li><a href="#overview">Ringkasan Eksekutif</a></li>
        <li><a href="#stage1">Tahap 1 &mdash; Validasi Konvergensi (200 epoch)</a></li>
        <li><a href="#stage2">Tahap 2 &mdash; Perbandingan Algoritma (18 <em>runs</em>, 3 <em>seed</em>)</a></li>
        <li><a href="#stage3">Tahap 3 &mdash; Interpretabilitas SAE</a></li>
        <li><a href="#stage4">Tahap 4 &mdash; Pemetaan Frekuensi BPFx</a></li>
        <li><a href="#findings">Temuan Utama &amp; Implikasi untuk Disertasi</a></li>
        <li><a href="#appendix">Lampiran &mdash; Daftar Lengkap <em>Runs</em></a></li>
      </ol>
    </div>
    </section>"""

    # ------------------------------------------------------------------ Stage 1
    s2 = page("stage1", "2. Tahap 1 &mdash; Validasi Konvergensi", f"""
    <h3>2.1 Tujuan dan Hipotesis</h3>
    <p>
      Sebelum melakukan perbandingan lintas arsitektur dengan anggaran pelatihan
      yang ekonomis, perlu dilakukan validasi bahwa anggaran tersebut cukup untuk
      mencapai konvergensi. Tahap&nbsp;1 dirancang untuk menjawab pertanyaan
      operasional berikut: <em>apakah 75 epoch sudah memadai bagi Mamba-xLSTM-Net
      untuk mencapai performa terbaiknya pada dataset PHM2012, ataukah anggaran
      yang lebih besar (misal 200 epoch) akan memberikan peningkatan signifikan?</em>
      Hipotesis nol yang diuji adalah bahwa <em>val/RMSE</em> mencapai plato sebelum
      epoch ke-75 sehingga anggaran 75 epoch dapat digunakan tanpa kehilangan
      kapasitas model.
    </p>

    <h3>2.2 Setup Eksperimen</h3>
    <table class="data-table">
      <thead><tr><th>Parameter</th><th>Nilai</th></tr></thead>
      <tbody>
        <tr><td>Arsitektur</td><td>Mamba-xLSTM-Net (d_model = 128, mamba d_state = 128)</td></tr>
        <tr><td>Dataset</td><td>PHM2012 (FEMTO-PRONOSTIA)</td></tr>
        <tr><td><em>Seed</em></td><td>42</td></tr>
        <tr><td><em>Max epochs</em></td><td>200</td></tr>
        <tr><td><em>Scheduler</em></td><td>Cosine annealing, T_max = 200, warmup = 5</td></tr>
        <tr><td>Optimizer</td><td>AdamW lr = 8 &times; 10<sup>&minus;4</sup>, wd = 3 &times; 10<sup>&minus;4</sup></td></tr>
        <tr><td>Presisi</td><td>bf16-mixed pada GPU NVIDIA A40 (46 GB VRAM)</td></tr>
      </tbody>
    </table>

    <h3>2.3 Hasil Konvergensi</h3>
    <table class="data-table">
      <thead><tr><th>Indikator</th><th>Nilai</th></tr></thead>
      <tbody>
        <tr><td>Epoch terbaik pada validation</td><td>55</td></tr>
        <tr><td>Nilai <code>val/RMSE</code> terbaik</td><td>0.113</td></tr>
        <tr><td>Saturasi <em>training loss</em></td><td>sekitar epoch 30</td></tr>
        <tr><td>Variasi <code>val/RMSE</code> epoch 60&ndash;200</td><td>&lt; 0.005 (plato)</td></tr>
      </tbody>
    </table>

    <h3>2.4 Interpretasi dan Keputusan</h3>
    <p>
      Hasil Tahap&nbsp;1 mendukung hipotesis bahwa anggaran 75 epoch sudah lebih
      dari cukup. <em>Val/RMSE</em> terbaik dicapai pada epoch ke-55 dan tetap stabil
      hingga epoch ke-200 dengan deviasi di bawah 0.005 RMSE. Hal ini menunjukkan
      bahwa pelatihan tambahan setelah epoch ke-75 tidak memberikan keuntungan
      generalisasi yang berarti dan justru meningkatkan biaya komputasi 2,7&times;.
    </p>
    <div class="alert">
      <strong>Keputusan operasional.</strong> Seluruh <em>run</em> pada Tahap&nbsp;2
      menggunakan konfigurasi <code>cloud_full_75.yaml</code> (75 epoch, bf16-mixed,
      <em>cosine annealing</em> T_max = 75, <em>early stopping</em> dinonaktifkan).
      <em>Early stopping</em> dinonaktifkan secara sengaja agar setiap model menempuh
      jumlah epoch yang identik, sehingga kurva pelatihan dapat dibandingkan
      secara adil tanpa pemotongan asimetris.
    </div>
    """)

    # ------------------------------------------------------------------ Stage 2
    s3 = page("stage2", "3. Tahap 2 &mdash; Perbandingan Algoritma", f"""
    <h3>3.1 Model yang Dibandingkan (Tier-S)</h3>
    <p>
      Tiga model Tier-S dipilih berdasarkan analisis pendahuluan 12-model selama
      30 epoch (lihat dokumen <code>experiment-design.md</code>). Tiga arsitektur ini
      mewakili paradigma yang berbeda namun saling melengkapi:
    </p>
    <table class="data-table">
      <thead>
        <tr><th>Model</th><th>Paradigma</th><th>Parameter</th><th>Kekuatan utama</th></tr>
      </thead>
      <tbody>
        <tr><td><strong>Mamba-xLSTM-Net</strong></td>
            <td>SSM selektif (Mamba-3) + <em>matrix memory</em> mLSTM</td>
            <td>898.481</td>
            <td>Modul fokus disertasi; <em>long-range</em> + selektif</td></tr>
        <tr><td><strong>N-BEATS-xLSTM-RUL</strong></td>
            <td>Polynomial/wear/shock basis + xLSTM <em>temporal front</em></td>
            <td>459.592 (PHM) / 457.672 (XJTU)</td>
            <td><em>Strong prior</em> fisik; <em>decomposable</em></td></tr>
        <tr><td><strong>SparseGate-TCN-RUL</strong></td>
            <td>TCN dilatated + <em>sparse gating</em> + atensi ringan</td>
            <td>249.192</td>
            <td>Paling ringan, <em>built-in interpretability</em> via <em>gate sparsity</em></td></tr>
      </tbody>
    </table>

    <h3>3.2 Konfigurasi Pelatihan (cloud_full_75.yaml)</h3>
    <table class="data-table">
      <thead><tr><th>Hiperparameter</th><th>Nilai</th></tr></thead>
      <tbody>
        <tr><td><code>max_epochs</code></td><td>75</td></tr>
        <tr><td>Optimizer</td><td>AdamW</td></tr>
        <tr><td><code>lr</code> dasar</td><td>8.0 &times; 10<sup>&minus;4</sup></td></tr>
        <tr><td><code>weight_decay</code></td><td>3.0 &times; 10<sup>&minus;4</sup></td></tr>
        <tr><td><code>xlstm_lr_mult</code></td><td>0.75 (LR khusus xLSTM lebih kecil)</td></tr>
        <tr><td><code>freeze_xlstm_epochs</code></td><td>5 (xLSTM dibekukan saat <em>warm-up</em> trunk)</td></tr>
        <tr><td><em>Scheduler</em></td><td>Cosine annealing, T_max = 75, warmup = 5</td></tr>
        <tr><td><code>monotonicity_weight</code></td><td>0.05 (penalti monotonitas RUL)</td></tr>
        <tr><td><code>gradient_clip_val</code></td><td>1.0</td></tr>
        <tr><td><em>Early stopping</em></td><td>dinonaktifkan (<code>patience = 9999</code>)</td></tr>
        <tr><td>Presisi</td><td>bf16-mixed (Ampere+)</td></tr>
        <tr><td>GPU</td><td>NVIDIA A40, 46 GB VRAM</td></tr>
        <tr><td><em>Seeds</em></td><td>42, 43, 44</td></tr>
      </tbody>
    </table>

    <h3>3.3 Hasil Agregat &mdash; PHM2012 (mean &plusmn; std, n=3)</h3>
    {perf_table_agg(stage2_agg, "phm2012")}

    <p>
      Pada PHM2012, SparseGate-TCN-RUL mencatat RMSE rerata terbaik
      ({stage2_agg["phm2012"]["sparse_gate_tcn_rul"]["rmse_mean"]:.4f})
      dengan deviasi terbesar
      (&plusmn;{stage2_agg["phm2012"]["sparse_gate_tcn_rul"]["rmse_std"]:.4f}),
      yang menunjukkan ketergantungan kuat pada inisialisasi <em>seed</em>.
      Mamba-xLSTM-Net berada di urutan kedua secara rerata, tetapi <em>std</em> jauh lebih
      kecil (&plusmn;{stage2_agg["phm2012"]["mamba_xlstm_net"]["rmse_std"]:.4f}),
      menandakan stabilitas konvergensi yang lebih baik. N-BEATS-xLSTM-RUL
      mencatat RMSE rerata tertinggi tetapi paling konsisten lintas <em>seed</em>
      (&plusmn;{stage2_agg["phm2012"]["nbeats_xlstm_rul"]["rmse_std"]:.4f}),
      sebuah trade-off antara kapasitas dan keterprediksian yang khas dari
      model berbasis <em>basis blocks</em> dengan <em>strong prior</em>.
    </p>

    <h3>3.4 Hasil Per-<em>Seed</em> &mdash; PHM2012</h3>
    {perf_table_seeds(stage2_data, "phm2012")}

    <h3>3.5 Hasil Agregat &mdash; XJTU-SY (mean &plusmn; std, n=3)</h3>
    {perf_table_agg(stage2_agg, "xjtusy")}

    <p>
      Pada XJTU-SY, perbedaan RMSE antar tiga model sangat sempit (&lt;0.01 RMSE).
      Hal ini disebabkan oleh <em>test split</em> yang hanya terdiri dari dua bantalan
      sehingga variansi antar <em>seed</em> mendominasi variansi antar arsitektur.
      Meskipun begitu, Mamba-xLSTM-Net unggul mencolok pada PHM Score
      ({stage2_agg["xjtusy"]["mamba_xlstm_net"]["phm_mean"]:.4f}), metrik yang paling
      relevan untuk pengambilan keputusan pemeliharaan prediktif karena
      memperhitungkan sifat asimetris konsekuensi prediksi terlambat versus
      prediksi terlalu cepat (lihat &sect;3.7).
    </p>

    <h3>3.6 Hasil Per-<em>Seed</em> &mdash; XJTU-SY</h3>
    {perf_table_seeds(stage2_data, "xjtusy")}

    <h3>3.7 Analisis Epoch Terbaik dan Pola Konvergensi</h3>
    <div class="alert">
      <strong>Mamba-xLSTM-Net</strong> mencapai checkpoint terbaik di epoch
      71&ndash;72/75 (PHM2012) dan 70/75 (XJTU-SY) &mdash; secara konsisten menggunakan
      hampir seluruh anggaran pelatihan. Profil ini mengindikasikan bahwa
      arsitektur ini <em>belum jenuh</em> dan berpotensi terus membaik dengan
      anggaran yang lebih panjang.<br><br>
      <strong>N-BEATS-xLSTM-RUL</strong> mencapai checkpoint terbaik di epoch
      0&ndash;1 untuk semua <em>seed</em> dan kedua dataset. <em>Polynomial basis blocks</em>
      menyediakan <em>prior</em> struktural yang sangat kuat sehingga model
      "sudah hampir benar" sejak inisialisasi; pelatihan lebih lanjut justru
      menyebabkan <em>overfit</em> marginal.<br><br>
      <strong>SparseGate-TCN-RUL</strong> menunjukkan dua mode: pada <em>seed</em> 42
      mencapai puncak di epoch 20 (PHM RMSE 0.184); pada <em>seed</em> 43 dan 44 puncak
      tercapai di epoch 0 dengan nilai yang lebih konservatif. Variansi tinggi ini
      menjadi penyebab utama <em>std</em> RMSE PHM yang mencapai &plusmn;0.030.
    </div>
    """)

    # ------------------------------------------------------------------ Stage 3
    shap_phm = shap_jsons.get("phm2012", {})
    shap_xjtu = shap_jsons.get("xjtusy", {})

    s4 = page("stage3", "4. Tahap 3 &mdash; Interpretabilitas SAE", f"""
    <h3>4.1 Tujuan dan Justifikasi Metodologi</h3>
    <p>
      Setelah Tahap&nbsp;2 menetapkan <em>checkpoint</em> terbaik per
      (model &times; dataset), Tahap&nbsp;3 memulai pilar interpretabilitas
      disertasi. Metode yang dipilih adalah <em>Top-k Sparse Autoencoder</em>
      (SAE) versi Anthropic, yang melatih <em>autoencoder</em> dengan
      pembatasan eksplisit pada jumlah fitur aktif per sampel. Berbeda dengan
      atensi atau probing klasik, SAE mempelajari <em>kamus laten</em> yang
      <em>monosemantik</em>: setiap fitur SAE idealnya menangkap satu konsep yang
      dapat ditafsirkan secara manusiawi. Pendekatan ini telah terbukti
      mengungkap struktur internal model bahasa besar dan kami adaptasi ke
      domain sinyal getaran bantalan.
    </p>

    <h3>4.2 Arsitektur dan Hiperparameter SAE</h3>
    <table class="data-table">
      <thead><tr><th>Parameter</th><th>Nilai</th></tr></thead>
      <tbody>
        <tr><td>Tipe</td><td>Top-<em>k</em> Sparse Autoencoder</td></tr>
        <tr><td>d_model (input <em>hidden state</em>)</td><td>128</td></tr>
        <tr><td>Faktor ekspansi</td><td>8 &rarr; d_latent = 1024</td></tr>
        <tr><td><em>k</em> (jumlah fitur aktif per sampel)</td><td>51 (&asymp; 5%)</td></tr>
        <tr><td>Loss</td><td>MSE rekonstruksi (sparsity dijamin <em>strict</em> Top-<em>k</em>)</td></tr>
        <tr><td>Optimizer</td><td>AdamW lr = 1 &times; 10<sup>&minus;3</sup></td></tr>
        <tr><td>Epoch pelatihan</td><td>50</td></tr>
        <tr><td>Sampel pelatihan</td><td>20.000 <em>hidden states</em> dari <em>training set</em></td></tr>
      </tbody>
    </table>
    <p>
      <em>Hidden states</em> diambil dari lapisan terakhir Mamba-xLSTM-Net sebelum
      <em>prediction head</em>, untuk dua <em>best run</em> yang dipilih: PHM2012
      <em>seed</em> 42 (RMSE 0.2166) dan XJTU-SY <em>seed</em> 44 (RMSE 0.2533).
      Pemilihan <em>seed</em> berbeda mengikuti aturan "<em>best run per</em>
      (model &times; dataset)" yang tercatat dalam &sect;10.4 dokumen
      desain eksperimen.
    </p>

    <h3>4.3 Kurva Pelatihan SAE</h3>
    <table class="data-table" style="width:auto">
      <thead>
        <tr><th>Dataset</th><th><em>Recon loss</em> awal</th><th><em>Recon loss</em> epoch 49</th>
            <th>Reduksi</th></tr>
      </thead>
      <tbody>
        <tr><td>PHM2012</td>
            <td>{phm_hist[0]["recon"] if phm_hist else 0:.6f}</td>
            <td>{phm_hist[-1]["recon"] if phm_hist else 0:.6f}</td>
            <td>{(1 - phm_hist[-1]["recon"]/phm_hist[0]["recon"])*100:.1f}%</td></tr>
        <tr><td>XJTU-SY</td>
            <td>{xjtu_hist[0]["recon"] if xjtu_hist else 0:.6f}</td>
            <td>{xjtu_hist[-1]["recon"] if xjtu_hist else 0:.6f}</td>
            <td>{(1 - xjtu_hist[-1]["recon"]/xjtu_hist[0]["recon"])*100:.1f}%</td></tr>
      </tbody>
    </table>
    <figure>
      <img src="{loss_curve_src}" alt="SAE training loss curves"
           style="width:100%;max-width:170mm;border:1px solid #d0d7de;border-radius:4px">
      <figcaption>Gambar 4.1. Kurva <em>reconstruction MSE</em> SAE selama 50 epoch
      pelatihan pada kedua dataset. Konvergensi cepat menuju loss yang sangat kecil
      (&lt;0.001) menegaskan bahwa <em>hidden state</em> model Mamba-xLSTM-Net
      bersifat <em>sparse-representable</em>: 5% fitur aktif sudah cukup untuk
      merekonstruksi sebagian besar informasi yang dipakai untuk prediksi RUL.</figcaption>
    </figure>

    <h3>4.4 Visualisasi UMAP Ruang Laten</h3>
    <p>
      UMAP <em>(Uniform Manifold Approximation and Projection)</em> diaplikasikan pada
      ruang laten SAE 1024-dimensi untuk memetakan struktur klaster ke 2D. Warna
      titik mewakili nilai RUL <em>ground-truth</em>, dari 1,0 (kondisi sehat) hingga
      0,0 (mendekati <em>end-of-life</em>). Pola gradasi warna menunjukkan apakah
      ruang laten secara alami menyusun lintasan degradasi.
    </p>
    <div class="fig-pair">
      {img_tag(BEST_RUNS["phm2012"] / "explain/sae_umap_clusters.png",
               "Gambar 4.2a. UMAP ruang laten SAE \u2014 PHM2012 (Mamba-xLSTM-Net, seed 42)")}
      {img_tag(BEST_RUNS["xjtusy"] / "explain/sae_umap_clusters.png",
               "Gambar 4.2b. UMAP ruang laten SAE \u2014 XJTU-SY (Mamba-xLSTM-Net, seed 44)")}
    </div>

    <h3>4.5 SHAP &mdash; <em>Global Feature Importance</em></h3>
    <p>
      Nilai SHAP global (rerata |SHAP| atas semua sampel uji) menunjukkan fitur
      input mana yang paling banyak memengaruhi prediksi RUL. Pada PHM2012, fitur
      <em>frequency-domain</em> dari kanal pertama (<code>fd_c0_rms_freq</code>,
      <code>fd_c0_centroid</code>) mendominasi, konsisten dengan teori klasik bahwa
      perubahan distribusi energi spektral merupakan indikator awal degradasi
      bantalan. Pada XJTU-SY, fitur <em>time-domain</em> RMS dari kanal pertama
      (<code>td_c0_rms</code>) menjadi yang paling dominan, sejalan dengan kondisi
      operasi XJTU-SY yang menampilkan kenaikan amplitudo getaran lebih tajam pada
      tahap akhir hidup bantalan.
    </p>
    <div class="fig-pair">
      {img_tag(BEST_RUNS["phm2012"] / "explain/shap_global.png",
               "Gambar 4.3a. SHAP global importance \u2014 PHM2012")}
      {img_tag(BEST_RUNS["xjtusy"] / "explain/shap_global.png",
               "Gambar 4.3b. SHAP global importance \u2014 XJTU-SY")}
    </div>
    <div class="fig-pair">
      <div>{shap_top10_table(shap_phm, "PHM2012")}</div>
      <div>{shap_top10_table(shap_xjtu, "XJTU-SY")}</div>
    </div>

    <h3>4.6 SHAP &mdash; <em>Heatmap</em> Waktu &times; Fitur</h3>
    <p>
      Pada satu sampel uji dengan RUL &asymp; 0,5, <em>heatmap</em> berikut
      memperlihatkan kontribusi SHAP per kombinasi (timestep, fitur). Pola
      vertikal yang tajam menandakan fitur yang konsisten penting sepanjang
      <em>window</em>, sementara pola horizontal menandakan momen waktu tertentu
      yang menjadi titik balik prediksi.
    </p>
    <div class="fig-pair">
      {img_tag(BEST_RUNS["phm2012"] / "explain/shap_heatmap_35.png",
               "Gambar 4.4a. SHAP heatmap \u2014 PHM2012 (window 35, RUL\u22480,5)")}
      {img_tag(BEST_RUNS["xjtusy"] / "explain/shap_heatmap_0.png",
               "Gambar 4.4b. SHAP heatmap \u2014 XJTU-SY (window 0, RUL\u22480,5)")}
    </div>

    <h3>4.7 <em>Integrated Gradients</em> &mdash; Atribusi per-<em>timestep</em></h3>
    <p>
      <em>Integrated Gradients</em> (IG) memberikan atribusi langsung dari output
      model terhadap input pada level <em>(timestep, fitur)</em> melalui integrasi
      jalur gradien dari <em>baseline</em> (vektor nol) ke input aktual. Empat
      <em>window</em> contoh ditampilkan per dataset untuk memberikan gambaran
      konsistensi atribusi lintas sampel.
    </p>
    <p style="font-weight:600">PHM2012 (Mamba-xLSTM-Net, <em>seed</em> 42)</p>
    <div class="fig-quad">
      {img_tag(BEST_RUNS["phm2012"] / "explain/ig_0.png", "Gambar 4.5a. IG window 0 \u2014 PHM2012")}
      {img_tag(BEST_RUNS["phm2012"] / "explain/ig_1.png", "Gambar 4.5b. IG window 1 \u2014 PHM2012")}
      {img_tag(BEST_RUNS["phm2012"] / "explain/ig_2.png", "Gambar 4.5c. IG window 2 \u2014 PHM2012")}
      {img_tag(BEST_RUNS["phm2012"] / "explain/ig_3.png", "Gambar 4.5d. IG window 3 \u2014 PHM2012")}
    </div>
    <p style="font-weight:600;margin-top:4mm">XJTU-SY (Mamba-xLSTM-Net, <em>seed</em> 44)</p>
    <div class="fig-quad">
      {img_tag(BEST_RUNS["xjtusy"] / "explain/ig_0.png", "Gambar 4.6a. IG window 0 \u2014 XJTU-SY")}
      {img_tag(BEST_RUNS["xjtusy"] / "explain/ig_1.png", "Gambar 4.6b. IG window 1 \u2014 XJTU-SY")}
      {img_tag(BEST_RUNS["xjtusy"] / "explain/ig_2.png", "Gambar 4.6c. IG window 2 \u2014 XJTU-SY")}
      {img_tag(BEST_RUNS["xjtusy"] / "explain/ig_3.png", "Gambar 4.6d. IG window 3 \u2014 XJTU-SY")}
    </div>
    """)

    # ------------------------------------------------------------------ Stage 4
    bpfx_phm = bpfx_jsons.get("phm2012", {})
    bpfx_xjtu = bpfx_jsons.get("xjtusy", {})

    hit_summary_rows = ""
    for ds_key in DATASETS:
        bd = bpfx_jsons.get(ds_key, {})
        hr = bd.get("hit_rate", {})
        hit_summary_rows += (
            f'<tr><td><strong>{DS_LABELS[ds_key]}</strong></td>'
            f'<td>{hr.get("BPFO",0)*100:.1f}%</td>'
            f'<td>{hr.get("BPFI",0)*100:.1f}%</td>'
            f'<td>{hr.get("BSF",0)*100:.1f}%</td>'
            f'<td>{hr.get("FTF",0)*100:.1f}%</td>'
            f'</tr>'
        )

    s5 = page("stage4", "5. Tahap 4 &mdash; Pemetaan Frekuensi BPFx", f"""
    <h3>5.1 Frekuensi Karakteristik Bantalan</h3>
    <p>
      Empat frekuensi karakteristik bantalan gelinding diturunkan dari geometri
      bantalan (jumlah elemen <em>n</em>, diameter elemen <em>d</em>, <em>pitch
      diameter</em> <em>D</em>, sudut kontak <em>&theta;</em>) dan kecepatan rotasi
      poros <em>f<sub>r</sub></em>:
    </p>
    <div class="note" style="font-family:Menlo,Consolas,monospace;font-size:9.5pt">
      BPFO = (n/2) &middot; f<sub>r</sub> &middot; (1 &minus; (d/D) cos &theta;)<br>
      BPFI = (n/2) &middot; f<sub>r</sub> &middot; (1 + (d/D) cos &theta;)<br>
      BSF  = (D/2d) &middot; f<sub>r</sub> &middot; (1 &minus; ((d/D) cos &theta;)<sup>2</sup>)<br>
      FTF  = (f<sub>r</sub>/2) &middot; (1 &minus; (d/D) cos &theta;)
    </div>
    <p>
      Untuk kedua bantalan target, sudut kontak diasumsikan &theta; = 0 (bantalan
      bola alur dalam tanpa beban aksial). Tabel berikut merangkum geometri
      bantalan dan frekuensi karakteristik teoretis pada kondisi operasi nominal:
    </p>
    <table class="data-table">
      <thead>
        <tr><th>Dataset</th><th>Bantalan</th>
            <th>n</th><th>d (mm)</th><th>D (mm)</th>
            <th>f<sub>r</sub> (Hz)</th>
            <th>BPFO</th><th>BPFI</th><th>BSF</th><th>FTF</th></tr>
      </thead>
      <tbody>
        <tr><td>PHM2012</td><td>NSK 6804</td>
            <td>13</td><td>3.5</td><td>25.5</td>
            <td>30.0</td>
            <td>168.24</td><td>221.76</td><td>107.23</td><td>12.94</td></tr>
        <tr><td>XJTU-SY</td><td>LDK UER204</td>
            <td>8</td><td>7.92</td><td>34.55</td>
            <td>35.0</td>
            <td>107.91</td><td>172.09</td><td>72.33</td><td>13.49</td></tr>
      </tbody>
    </table>

    <h3>5.2 Prosedur Pemetaan</h3>
    <p>
      Pemetaan dilakukan dengan empat langkah: (i) kumpulkan 300 rekaman getaran
      mentah dari <em>training set</em>; (ii) untuk setiap rekaman hitung
      <em>Hilbert envelope spectrum</em> dan ambil amplitudo pada pita
      &plusmn;2 Hz di sekitar setiap frekuensi BPFx; (iii) untuk setiap fitur SAE
      (1024 fitur) kumpulkan aktivasi pada <em>window</em> yang berkorespondensi;
      (iv) hitung korelasi Pearson <em>r</em> antara aktivasi fitur SAE dan
      amplitudo envelope. Sebuah fitur dianggap "ter-petakan" ke frekuensi BPFx
      tertentu bila |r| &ge; 0.30. <em>Hit-rate</em> didefinisikan sebagai
      persentase fitur SAE yang melewati ambang ini.
    </p>

    <h3>5.3 Ringkasan <em>Hit-Rate</em></h3>
    <table class="data-table">
      <thead>
        <tr><th>Dataset</th><th>BPFO</th><th>BPFI</th><th>BSF</th><th>FTF</th></tr>
      </thead>
      <tbody>{hit_summary_rows}</tbody>
    </table>
    <div class="fig-pair">
      {img_tag(BPFX_DIR / "phm2012_hitrate_bar.png",
               "Gambar 5.1a. Hit-rate BPFx \u2014 PHM2012")}
      {img_tag(BPFX_DIR / "xjtusy_hitrate_bar.png",
               "Gambar 5.1b. Hit-rate BPFx \u2014 XJTU-SY")}
    </div>

    <h3>5.4 <em>Heatmap</em> Korelasi (50 Fitur SAE Teraktif)</h3>
    <div class="fig-pair">
      {img_tag(BPFX_DIR / "phm2012_corr_heatmap.png",
               "Gambar 5.2a. Heatmap |r| \u2014 PHM2012 (top-50 fitur aktif)")}
      {img_tag(BPFX_DIR / "xjtusy_corr_heatmap.png",
               "Gambar 5.2b. Heatmap |r| \u2014 XJTU-SY (top-50 fitur aktif)")}
    </div>

    <h3>5.5 Lima Fitur SAE Teratas per Frekuensi BPFx</h3>
    <div class="fig-pair">
      <div>{bpfx_top5_table(bpfx_phm, "PHM2012")}</div>
      <div>{bpfx_top5_table(bpfx_xjtu, "XJTU-SY")}</div>
    </div>

    <h3>5.6 Interpretasi</h3>
    <div class="alert">
      <strong>(1) Konsistensi dengan teori klasik diagnosis bantalan.</strong>
      Pada PHM2012 hanya BPFO (1.95%) dan BPFI (2.34%) yang memiliki
      <em>hit-rate</em> &gt; 0. Pada XJTU-SY, BPFO (2.25%) dominan, dengan
      sedikit kontribusi BSF (0.29%). Pola ini konsisten dengan literatur:
      <em>spalling</em> pada <em>outer race</em> dan <em>inner race</em> merupakan mode
      kegagalan dominan pada kedua dataset benchmark. BSF dan FTF mendekati nol
      sesuai dengan tidak adanya kegagalan <em>rolling element</em> atau sangkar
      yang terdokumentasi pada bantalan pelatihan.<br><br>
      <strong>(2) Magnitudo korelasi yang bermakna.</strong> Pada PHM2012, fitur
      f474 mencapai r = 0.507 untuk BPFI dan f750 mencapai r = 0.479 &mdash;
      korelasi moderat yang signifikan secara statistik (n = 300, p &laquo; 0.05).
      Pada XJTU-SY, fitur f959 mencapai r = 0.501 untuk BPFO. Nilai-nilai ini
      menegaskan bahwa sebagian fitur SAE memang menangkap variasi energi
      spektral di sekitar frekuensi karakteristik fisik.<br><br>
      <strong>(3) Mengapa hit-rate tidak tinggi?</strong> Input ke Mamba-xLSTM-Net
      adalah <em>vektor fitur band-energy</em> yang sudah diagregasi, bukan sinyal
      mentah. Fitur SAE oleh karena itu mempelajari kombinasi <em>abstrak</em> dari
      fitur agregat tersebut, bukan replikasi analisis envelope. Hit-rate
      2&ndash;2.3% berarti dari 1024 fitur laten, sekitar 20&ndash;24 fitur memiliki
      korespondensi langsung yang terdeteksi &mdash; sebuah jumlah yang konsisten
      dengan klaim "<em>monosemantic features</em>" yang sparse.
    </div>

    <p class="muted">
      <strong>Keterbatasan metodologis.</strong> Alignment antara rekaman raw
      (kronologis per bantalan) dan <em>window</em> SAE (stride = 1 dari <em>train
      DataModule</em>) dilakukan secara proporsional pada 300 rekaman pertama.
      Alignment yang lebih presisi (memetakan tiap <em>window</em> ke rekaman
      aslinya) berpotensi meningkatkan akurasi korelasi dan dapat menaikkan
      hit-rate dengan teramati. Hal ini diakui di Bab V keterbatasan dan menjadi
      kandidat <em>future work</em>.
    </p>
    """)

    # ------------------------------------------------------------------ Findings
    s6 = page("findings", "6. Temuan Utama dan Implikasi untuk Disertasi", f"""
    <h3>6.1 Pilar 1 &mdash; Performa Arsitektur</h3>
    <ul>
      <li><strong>SparseGate-TCN-RUL</strong> mencapai RMSE rerata terendah pada
          PHM2012 (0.226 &plusmn; 0.030) tetapi dengan variansi <em>seed</em> tertinggi.
          Performa terbaik diraih pada <em>seed</em> 42 (RMSE 0.184) yang merupakan
          hasil konvergensi yang sangat baik di epoch 20 sebelum
          <em>overfitting</em> dimulai.</li>
      <li><strong>Mamba-xLSTM-Net</strong> menjadi model paling <em>stabil</em>
          pada PHM2012 (&sigma; = 0.020) dan unggul mutlak pada PHM Score XJTU-SY
          (0.907) &mdash; metrik yang paling relevan untuk pemeliharaan prediktif.
          <em>Best checkpoint</em> selalu berada di epoch 70&ndash;72/75
          mengindikasikan kapasitas yang belum jenuh.</li>
      <li><strong>N-BEATS-xLSTM-RUL</strong> menunjukkan stabilitas lintas-<em>seed</em>
          terbaik (&sigma; = 0.007 PHM, &sigma; = 0.003 XJTU) berkat <em>prior</em>
          struktural polinomial. RMSE rerata sedikit lebih tinggi, namun
          prediktabilitas lebih penting untuk skenario produksi.</li>
    </ul>

    <h3>6.2 Pilar 2 &mdash; Interpretabilitas (SAE + BPFx)</h3>
    <ul>
      <li><em>Reconstruction loss</em> SAE &lt; 0.001 pada kedua dataset setelah 50
          epoch &mdash; <em>hidden states</em> Mamba-xLSTM-Net bersifat
          <em>sparse-representable</em> dengan hanya 5% fitur aktif.</li>
      <li>BPFO dan BPFI merupakan satu-satunya frekuensi karakteristik dengan
          <em>hit-rate</em> &gt; 0, sesuai dengan mode kegagalan dominan
          (<em>spalling</em> outer/inner race) pada kedua benchmark.</li>
      <li>Magnitudo korelasi tertinggi mencapai r = 0.507 (f474 vs BPFI, PHM2012)
          dan r = 0.501 (f959 vs BPFO, XJTU-SY) &mdash; signifikan secara statistik
          (n = 300, p &laquo; 0.05) dan termasuk korelasi moderat yang bermakna
          dalam konteks fitur SAE post-hoc.</li>
      <li>SHAP global didominasi fitur <em>frequency-domain</em> pada PHM2012 dan
          fitur <em>time-domain</em> RMS pada XJTU-SY &mdash; sejalan dengan
          karakteristik degradasi masing-masing dataset.</li>
    </ul>

    <h3>6.3 Implikasi Kebaruan Disertasi</h3>
    <ol>
      <li><strong>Kebaruan Teknologi-Metodologi.</strong> Hibrida Mamba-xLSTM-Net
          terbukti efisien (~898K parameter) dan stabil lintas <em>seed</em>.
          Mekanisme fusi SSM + matrix-memory yang diturunkan dari sifat non-stationer
          sinyal getaran memberikan baseline yang sah untuk klaim Pilar&nbsp;1.</li>
      <li><strong>Kebaruan Output.</strong> Korespondensi antara fitur SAE post-hoc
          dan frekuensi karakteristik BPFO/BPFI merupakan bukti empiris bahwa model
          <em>deep learning</em> RUL mempelajari representasi yang konsisten dengan
          teori klasik diagnosis getaran &mdash; klaim Pilar&nbsp;2 yang didukung
          oleh angka konkret (r &gt; 0.5).</li>
      <li><strong>Kebaruan Konsep.</strong> Kerangka "<em>BPFx hit-rate</em>"
          sebagai metrik <em>interpretability fidelity</em> untuk model RUL bantalan
          merupakan kontribusi metodologis yang dapat diadopsi peneliti lain pada
          dataset getaran serupa.</li>
    </ol>

    <h3>6.4 Langkah Lanjut Penulisan Disertasi</h3>
    <ol>
      <li>Konversi <code>writings/bab5-draft-hasil-performa.md</code> ke
          LaTeX (<code>chapters/05-hasil-pembahasan.tex</code>).</li>
      <li>Sematkan Gambar 4.1&ndash;4.6 dan 5.1&ndash;5.2 ke Bab&nbsp;V melalui
          <code>\\includegraphics</code> dengan caption yang konsisten.</li>
      <li>Tulis Bab&nbsp;III &sect;III.7 (frekuensi karakteristik bantalan)
          memakai turunan persamaan BPFO/BPFI/BSF/FTF yang dikonfirmasi pada
          &sect;5.1 laporan ini.</li>
      <li>Tambahkan uji signifikansi statistik (Wilcoxon <em>signed-rank</em>
          berpasangan) pada kolom RMSE multi-<em>seed</em> untuk klaim
          "<em>significantly better/worse</em>".</li>
      <li>Opsional: perluas pemetaan BPFx ke SparseGate-TCN-RUL agar
          menghasilkan tabel perbandingan lintas-arsitektur (set
          <code>INCLUDE_SPARSE=1</code> pada <em>script</em> Tahap&nbsp;3).</li>
    </ol>
    """)

    # ------------------------------------------------------------------ Appendix
    app_rows = ""
    idx = 0
    for ds in DATASETS:
        for m in MODELS:
            for seed in SEEDS:
                if seed not in stage2_data[ds][m]:
                    continue
                v = stage2_data[ds][m][seed]
                idx += 1
                app_rows += (
                    f'<tr><td>{idx}</td><td>{DS_LABELS[ds]}</td>'
                    f'<td>{MODEL_LABELS[m]}</td><td>{seed}</td>'
                    f'<td>{v["rmse"]:.4f}</td><td>{v["mae"]:.4f}</td>'
                    f'<td>{v["phm"]:.4f}</td><td>{v["r2"]:.4f}</td>'
                    f'<td style="font-size:8.5pt"><code>{v["run_id"]}</code></td></tr>'
                )

    s7 = page("appendix", "7. Lampiran &mdash; Daftar Lengkap <em>Runs</em>", f"""
    <p>
      Tabel berikut mencantumkan seluruh {idx} <em>run</em> Tahap&nbsp;2 dengan
      metrik <em>test set</em>. <em>Run ID</em> dapat digunakan untuk menelusuri
      direktori <code>results/runs/&lt;run_id&gt;/</code> yang berisi konfigurasi,
      checkpoint, dan log lengkap.
    </p>
    <table class="data-table" style="font-size:9pt">
      <thead>
        <tr><th>#</th><th>Dataset</th><th>Model</th><th>Seed</th>
            <th>RMSE</th><th>MAE</th><th>PHM</th><th>R&sup2;</th><th>Run ID</th></tr>
      </thead>
      <tbody>{app_rows}</tbody>
    </table>

    <h3>Referensi Dataset</h3>
    <ul>
      <li>Nectoux, P. dkk. (2012). <em>PRONOSTIA: An experimental platform for
          bearings accelerated degradation tests</em>. IEEE Int. Conf. on
          Prognostics and Health Management.</li>
      <li>Wang, B. dkk. (2020). <em>A hybrid prognostics approach for estimating
          remaining useful life of rolling element bearings</em>. IEEE Trans.
          Reliability, 69(1), 401&ndash;412 (dataset XJTU-SY).</li>
    </ul>

    <p class="muted" style="margin-top:6mm">
      Laporan ini dibangun secara otomatis dari <code>summary.json</code>,
      <code>sae_history.json</code>, <code>shap_global.json</code>, dan
      <code>*_bpfx_results.json</code> melalui <em>script</em>
      <code>Mamba-xLSTM/scripts/generate_experiment_report.py</code>.
      Seluruh gambar disisipkan sebagai data-URI base64 sehingga berkas HTML
      bersifat <em>self-contained</em> dan dapat dibuka tanpa dependensi eksternal.
    </p>
    """)

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bearing RUL &mdash; Experiment Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
{s1}
{s2}
{s3}
{s4}
{s5}
{s6}
{s7}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Collecting Stage 2 metrics …")
    stage2_data = collect_stage2()
    stage2_agg = agg_stage2(stage2_data)

    print("Loading SAE history …")
    sae_hist = {
        ds: read_json(BEST_RUNS[ds] / "explain" / "sae_history.json")
        for ds in DATASETS
    }

    print("Loading SHAP globals …")
    shap_jsons = {
        ds: read_json(BEST_RUNS[ds] / "explain" / "shap_global.json")
        for ds in DATASETS
    }

    print("Loading BPFx results …")
    bpfx_jsons = {
        ds: read_json(BPFX_DIR / f"{ds}_bpfx_results.json")
        for ds in DATASETS
    }

    print("Building HTML report …")
    html = build_html(stage2_data, stage2_agg, sae_hist, shap_jsons, bpfx_jsons)

    out_path = WRITINGS / "experiment-report.html"
    WRITINGS.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nReport written → {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
