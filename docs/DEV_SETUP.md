# Development Setup: Local Frontend + AutoDL Remote Backend

This document describes the validated integration flow used in this repository:
- backend runs on AutoDL Linux
- frontend runs locally with Vite
- local machine reaches remote backend through an SSH tunnel

Related reference:
- [ENV_VARS.md](ENV_VARS.md)
- [SMOKE_TESTS.md](SMOKE_TESTS.md)

---

## 1. Prerequisites

### Local machine
- Node.js and npm available
- SSH client available
- repository cloned locally

### AutoDL Linux server
- Python environment available
- repository cloned on server
- backend port `8000` can be listened on server side

---

## 2. Start Backend on AutoDL

Run from repository root on AutoDL:

```bash
cd /path/to/VibeCoding
./scripts/start_backend.sh
```

What this script does:
- ensures minimal deps (`fastapi`, `uvicorn`, `pydantic`)
- starts `backend_adapter.main:app` on `0.0.0.0:8000`
- applies CORS defaults for local Vite

Optional override:

```bash
BACKEND_ADAPTER_CORS="http://localhost:5173,http://127.0.0.1:5173" \
BACKEND_ADAPTER_PORT=8000 \
./scripts/start_backend.sh
```

### Check power-118 real-mode prerequisites

Before trying `power-118` real runs on AutoDL, run:

```bash
cd /path/to/VibeCoding
python scripts/check_power118_env.py
```

What to look for:
- `Real mode ready: YES`
- `gurobipy import` shows `PASS`
- `Gurobi model init` shows `PASS`
- `118_data.xls readable` shows `PASS`
- `check_gurobi_runtime call` shows `PASS`
- `real-run preconditions` shows `PASS`

If one of these fails, check in this order:
- `gurobipy` package and Gurobi license
- `pandas` / `xlrd`
- `external/power118/118_data.xls`
- `external/power118/SCUC_118_new.py`

When the check passes, you can continue with:

```bash
curl -X POST "http://127.0.0.1:8000/api/runs" \
  -H "Content-Type: application/json" \
  -d '{"scenarioId":"power-118","runMode":"exact"}'
```

### Build the power-118 ML dataset on AutoDL

Run from repository root on AutoDL:

```bash
cd /path/to/VibeCoding
python scripts/build_power118_ml_dataset.py --num-samples 64 --output-dir backend_adapter/data/power118_dataset
```

Expected output artifact:
- `backend_adapter/data/power118_dataset/power118_ml_dataset.pkl`
- `backend_adapter/data/power118_dataset/dataset_summary.json`

### Train the power-118 ML artifacts on AutoDL

```bash
cd /path/to/VibeCoding
python scripts/train_power118_model.py \
  --dataset-path backend_adapter/data/power118_dataset/power118_ml_dataset.pkl \
  --output-dir backend_adapter/data/power118_model
```

Expected output artifacts:
- `backend_adapter/data/power118_ml_model.joblib`
- `backend_adapter/data/power118_ml_metadata.json`
- versioned training directory under `backend_adapter/data/power118_model/`

### Run offline evaluation

```bash
cd /path/to/VibeCoding
python scripts/eval_power118_modes.py --num-cases 8 --output-dir backend_adapter/data/power118_eval
```

Default evaluation outputs:
- `backend_adapter/data/power118_eval/power118_eval_records.json`
- `backend_adapter/data/power118_eval/power118_eval_records.csv`
- `backend_adapter/data/power118_eval/summary.json`
- `backend_adapter/data/power118_eval/report.md`

To compare warm-start and constraint-aware variants in the current v3 setup:

```bash
cd /path/to/VibeCoding
python scripts/eval_power118_modes.py \
  --num-cases 1 \
  --seed 7 \
  --output-dir backend_adapter/data/power118_eval_constraint_v3 \
  --time-limit-ms 20000 \
  --require-exact-baseline
```

Then validate aggregation consistency:

```bash
cd /path/to/VibeCoding
python scripts/check_power118_eval_consistency.py \
  --records-path backend_adapter/data/power118_eval_constraint_v3/power118_eval_records.json \
  --summary-path backend_adapter/data/power118_eval_constraint_v3/summary.json
```

Important interpretation note:
- if `Exact real baseline available: NO`, the local or current backend environment did not complete a real feasible exact SCUC baseline
- in that case the script still records fallback and compat behavior honestly, but objective-gap conclusions should be treated as unavailable
- if you need the script to fail when no real exact baseline is available, add `--require-exact-baseline`

### One-command remote pipeline

On Linux or AutoDL you can run:

```bash
cd /path/to/VibeCoding
./scripts/run_power118_remote_pipeline.sh
```

This chains:
- environment check
- dataset build
- model training
- offline evaluation

---

## 3. Open SSH Tunnel on Local Machine

### Script way

```bash
./scripts/open_tunnel.sh <user@autodl-host>
```

### Windows PowerShell script way

```powershell
./scripts/open_tunnel.ps1 -HostName "<autodl-host>" -UserName "<ssh-user>"
```

Alternative form:

```powershell
./scripts/open_tunnel.ps1 -Target "<ssh-user>@<autodl-host>"
```

Default mapping:
- local: `127.0.0.1:8000`
- remote: `127.0.0.1:8000`

### Raw command (validated)

```bash
ssh -N -L 8000:127.0.0.1:8000 <user@autodl-host>
```

### Custom SSH port

```bash
./scripts/open_tunnel.sh <user@autodl-host> 8000 8000 22
```

PowerShell custom SSH port:

```powershell
./scripts/open_tunnel.ps1 -HostName "<autodl-host>" -UserName "<ssh-user>" -LocalPort 8000 -RemotePort 8000 -SshPort 22
```

---

## 4. Configure and Start Local Frontend

### Bash / Git Bash

```bash
cd frontend
cp .env.development.example .env.development
npm install
npm run dev
```

### PowerShell

```powershell
cd frontend
Copy-Item .env.development.example .env.development -Force
npm install
npm run dev
```

`frontend/.env.development` should include:

```bash
VITE_API_BASE=http://127.0.0.1:8000
```

This keeps frontend stable while backend host changes over time.

---

## 5. Validate API Connectivity

Run locally (with tunnel active):

```bash
curl http://127.0.0.1:8000/api/scenarios
```

Optional run test:

```bash
curl -X POST "http://127.0.0.1:8000/api/runs" \
  -H "Content-Type: application/json" \
  -d '{"scenarioId":"portfolio"}'
```

Power-118 mode-specific run examples:

```bash
curl -X POST "http://127.0.0.1:8000/api/runs" \
  -H "Content-Type: application/json" \
  -d '{"scenarioId":"power-118","runMode":"hybrid","fallbackToExact":true}'
```

```bash
curl -X POST "http://127.0.0.1:8000/api/runs" \
  -H "Content-Type: application/json" \
  -d '{"scenarioId":"power-118","runMode":"ml","fallbackToExact":true}'
```

Latest run check:

```bash
curl "http://127.0.0.1:8000/api/runs/latest?scenarioId=portfolio"
```

---

## 6. Validate Frontend Integration

Open:
- `http://localhost:5173/workbench`

Expected:
- page loads normally
- `Data source: Backend API` is visible
- scenario list is loaded from `/api/scenarios`

If backend is unreachable, frontend falls back to mock mode with a visible notice.

---

## 7. Common Issues

### Frontend still shows mock fallback
- verify backend process is running on AutoDL
- verify SSH tunnel is active
- verify `VITE_API_BASE=http://127.0.0.1:8000`
- verify local `curl /api/scenarios` succeeds

### Browser CORS error
- set server env var:
  - `BACKEND_ADAPTER_CORS="http://localhost:5173,http://127.0.0.1:5173"`
- restart backend

### `/api/runs/latest` returns 404
- no previous run exists yet
- call `POST /api/runs` first

### `power-118` stays in compat mode
- verify `python scripts/check_power118_env.py` returns `Real mode ready: YES`
- verify `backend_adapter/data/power118_ml_model.joblib` and `backend_adapter/data/power118_ml_metadata.json` exist for `hybrid` or `ml`
- verify the model metadata feature schema matches the current backend feature extraction code
- inspect `fallbackReason`, `modelLoadStatus`, `modelVersion`, and `featureSchemaVersion` from the returned run payload

---

## 8. Recommended Daily Workflow

1. Start backend on AutoDL: `./scripts/start_backend.sh`
2. Start tunnel locally:
   - Bash/Git Bash: `./scripts/open_tunnel.sh <user@host>`
   - PowerShell: `./scripts/open_tunnel.ps1 -HostName "<autodl-host>" -UserName "<ssh-user>"`
3. Start frontend locally: `cd frontend && npm run dev`
4. Verify `/api/scenarios` and Workbench data source status
