# RUL Streaming Inference Engine

Simulates a live sensor stream from real PHM2012 / XJTU-SY bearing data and runs the dissertation **Mamba-xLSTM-Net** checkpoint on each acquisition. Results are shown in a local web dashboard.

## Prerequisites

- `Mamba-xLSTM/.venv` with trained checkpoints under `Mamba-xLSTM/results/runs/`
- `data-bearing/` populated (PHM2012 + XJTU-SY + optional `skf-ch15-or1/` industrial plant data)

## Install (web deps only)

```bash
Mamba-xLSTM/.venv/bin/python -m pip install -r inference-engine/requirements.txt
```

## Run dashboard

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

- `app/model_registry.py` — dataset → run dir, checkpoint, config
- `app/skf_loader.py` — PT SKF Observer HTML-XLS trending parser (CH-15 OR-1)
- `app/engine.py` — HI extraction, scaler, EMA, rolling window, inference
- `app/server.py` — FastAPI REST + WebSocket stream
- `web/` — canvas dashboard (industrial instrument-panel UI)

### Industrial plant stream (PT SKF CH-15 OR-1)

Place SKF Observer exports under `data-bearing/skf-ch15-or1/` (HTML `.xls` trending + `ObserverExport.xmd`). Select **PT SKF Indonesia — CH-15 OR-1** in the dashboard. RUL uses the PHM2012-trained model as a **transfer demo** (no ground-truth RUL on plant data).
