# RUL Streaming Inference Engine

Simulates a live sensor stream from real PHM2012 / XJTU-SY bearing data (and optional PT SKF industrial trending) and runs the dissertation **Mamba-xLSTM-Net** checkpoint on each acquisition. Results are shown in a local web dashboard.

Runs can also be **recorded, saved, and replayed offline** with an automatic **failure analysis** that locates the significant predicted-health drops and explains each one (see [Saveable runs and failure analysis](#saveable-runs-and-failure-analysis)).

## Prerequisites

- `Mamba-xLSTM/.venv` with trained checkpoints under `Mamba-xLSTM/results/runs/`
- `data-bearing/` populated:
  - **PHM2012 + XJTU-SY** — required for benchmark streams
  - **`skf-ch15-or1-6m/`** — optional PT SKF CH-15 OR-1 six-month plant export

## Install (web deps only)

From `research/`:

```bash
Mamba-xLSTM/.venv/bin/python -m pip install -r inference-engine/requirements.txt
```

## Run dashboard

From `research/`:

```bash
./inference-engine/run.sh
```

Open **http://localhost:8800**

## Smoke test (headless)

```bash
cd inference-engine
../Mamba-xLSTM/.venv/bin/python scripts/smoke_test.py
```

## Architecture

| Module | Role |
|--------|------|
| `app/model_registry.py` | Dataset → run dir, checkpoint, HI scaler, stream metadata |
| `app/skf_loader.py` | PT SKF Observer HTML-XLS trending parser (CH-15 OR-1, 6-month export) |
| `app/engine.py` | HI extraction, scaler, EMA, rolling window, inference, WebSocket payloads |
| `app/explain.py` | Live saliency + on-demand Integrated Gradients |
| `app/replay_store.py` | Record full runs, persist them, detect + explain significant RUL drops |
| `app/server.py` | FastAPI REST + WebSocket stream + run-library endpoints |
| `web/` | Dashboard (industrial instrument-panel UI) |

## Replay library and failure analysis

The dashboard is **replay-centric**: the model runs **once per bearing** during recording (real inference over the whole run-to-failure), and everything you watch afterwards is aligned playback of the saved result — so the RUL curve, plain-language insight, drivers, Integrated Gradients, and the significant drops are all present and aligned from acquisition 0.

**Record the whole library at once.** Press **⏺ Record All Runs** in the **Run Library** panel (or run the CLI below). It records every dataset+bearing sequentially, skipping ones already saved, with a combined progress bar over the WebSocket. Works for **PHM2012, XJTU-SY, and the SKF plant transfer**.

```bash
cd inference-engine
../Mamba-xLSTM/.venv/bin/python scripts/record_all.py          # skips existing
../Mamba-xLSTM/.venv/bin/python scripts/record_all.py --force  # re-record all
```

**Replay.** Pick a run in **Replay Control → Recorded Run** and it loads and plays from acquisition 0. **Play / Pause / Step / Reset** are the transport; the **Seek / Review** slider scrubs to any acquisition and reads its exact values, waveform, drivers, and time. No model is in the loop — replay is pure client-side playback of the saved JSON.

**Significant-drop detection.** `detect_drops` locates the moments where the smoothed predicted RUL falls sharply (default ≥ 8 percentage points over a short lookback, non-max-suppressed by magnitude). The **Failure Analysis — Significant Drops** zone lists every drop and marks them on the RUL chart; clicking one seeks there and shows its saved explanation:

- **Integrated Gradients** at the drop window (the model's input attribution),
- the **fusion gate** balance (xLSTM vs Mamba) and **top drivers** at that point,
- the **raw HI feature deltas** across the drop — ranked by change *relative to each feature's normal variability*, so the panel surfaces the features that genuinely moved (e.g. high-mid-band energy rising, spectral centroid falling — a classic fault signature) rather than near-zero noise.

**Persistence.** Runs are written to `inference-engine/runs/<dataset>__<bearing>__<timestamp>.json` (git-ignored, local only). REST: `GET /api/runs`, `GET /api/runs/{id}`, `DELETE /api/runs/{id}`. Recording is the WebSocket `record` (single) / `record_all` (batch) action.

**Verify in a real browser** (after the library is recorded):

```bash
cd inference-engine && bash scripts/run_browser_test.sh
```

### Datasets and models

| Dashboard key | Data source | Model checkpoint | RUL on chart |
|---------------|-------------|------------------|--------------|
| `phm2012` | PHM2012 test bearings | Mamba-xLSTM-Net trained on PHM2012 (seed 42) | Neural-network prediction vs ground truth |
| `xjtusy` | XJTU-SY test bearings | Mamba-xLSTM-Net trained on XJTU-SY (seed 42) | Neural-network prediction vs ground truth |
| `skf_ch15_or1` | PT SKF Observer trending (`skf-ch15-or1-6m/`) | **Same PHM2012 checkpoint** (transfer demo) | **Causal envelope RUL, no ground truth** (see below) |

PHM2012 run directory (also used for SKF transfer):

`Mamba-xLSTM/results/runs/20260515_174110_algorithm_comparison_phm2012_mamba_xlstm_net_s42`

### Industrial plant stream (PT SKF CH-15 OR-1)

Place the SKF Observer trending exports under `data-bearing/skf-ch15-or1-6m/`. The active export contains **six HTML `.xls` files** (Observer saves tables as HTML):

| Stream ID | Channel | Files |
|-----------|---------|-------|
| `ch1_01_nde` | Channel 1-01 NDE (grinding, non-drive end) | `Ch1-01-R-A-NDE.xls`, `CH1-01-R-V-NDE.xls`, `Ch1-01-R-ENV-NDE.xls` |
| `ch3_02_de` | Channel 3-02 DE (grinding, drive end) | `Ch3-02-R-A-DE.xls`, `CH3-02-R-V-DE.xls`, `Ch3-02-R-ENV-DE.xls` |

Select **PT SKF Indonesia — CH-15 OR-1** in the dashboard, then pick a stream.

#### Coverage and cadence

- **Span:** ~5.3 months (01 Apr → 31 Aug 2023), early monitoring through the field failure.
- **Cadence:** **non-uniform** — roughly **daily** in the early phase (Apr–late Aug), then **hourly** once the sustained fault regime begins (~28 Aug).
- **EOL anchor:** field failure event (envelope `gE` spike collapse, ~31 Aug 2023 for NDE; ~01 Sep 2023 for DE). Post-repair September baseline is segmented out of the run-to-failure replay.
- **Note on the early phase:** it is *not* a clean healthy baseline. The envelope already shows intermittent fault-level spikes (2.7–6.3 gE) from early April, including several consecutive days in May. The sustained, continuously-high regime only starts ~28 Aug. This shapes the RUL curve below.

Because cadence varies, all timing (`elapsed_s`, predicted time-to-failure, predicted EOL) uses **per-acquisition timestamps**, not a fixed interval × step index.

#### What runs for SKF (transfer demo)

SKF exports contain **trended scalars only** (Overall A / V / ENV per measurement point), not raw waveforms. The pipeline therefore:

1. Parses the three Overall trends and aligns them on a common time axis (`skf_loader.py`).
2. Synthesizes a pseudo two-channel waveform from velocity + envelope (`synthetic_acquisition`).
3. Extracts HI features with the **PHM2012 `hi_scaler.json`** and feeds a rolling window into **Mamba-xLSTM-Net** for branch gate, top drivers, and Integrated Gradients.

**No ground truth.** Plant data carries no labelled RUL, so the chart shows a **single predicted curve with no truth overlay** (`has_gt_rul=False`). The dashboard hides the "Truth" line automatically and reports time-left as "N/A (plant)".

**Predicted RUL curve.** For SKF, `engine.py` sets the displayed RUL to the **causal envelope (gE) health index** (`envelope_rul` in `skf_loader.py`): a cumulative-max of the envelope, baseline/peak-normalized, returned as `1 − HI` (1 = healthy, 0 = at/after EOL). This is causal (no peeking at the failure time, unlike the old `rul_fraction`). The PHM2012-trained Mamba-xLSTM-Net still runs every step for the **fusion gate, driver attribution, and Integrated Gradients**, but its raw RUL output is a flat ~0.3 band on this cross-domain data and is therefore not used as the displayed RUL.

There is **no SKF-specific trained checkpoint**; this is explicitly a cross-domain transfer / sanity-check setup aligned with Lampiran D.

> Because the early phase already contains intermittent fault spikes, the cumulative-max envelope RUL drops within the first weeks and then plateaus until the final collapse, rather than declining smoothly from 1.0. That is a faithful, causal reflection of this particular bearing's history, not a bug.

#### Expected scale (after EOL segmentation)

| Stream | Run-to-failure points | Wall-clock span | EOL |
|--------|----------------------|-----------------|-----|
| `ch1_01_nde` | ~280 | ~151 days | 2023-08-31 07:00 |
| `ch3_02_de` | ~200 | ~152 days | 2023-09-01 06:44 |

Exact counts depend on alignment tolerances when parsing the HTML exports.
