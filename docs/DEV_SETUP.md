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
  -d '{"scenarioId":"power-118"}'
```

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

---

## 8. Recommended Daily Workflow

1. Start backend on AutoDL: `./scripts/start_backend.sh`
2. Start tunnel locally:
   - Bash/Git Bash: `./scripts/open_tunnel.sh <user@host>`
   - PowerShell: `./scripts/open_tunnel.ps1 -HostName "<autodl-host>" -UserName "<ssh-user>"`
3. Start frontend locally: `cd frontend && npm run dev`
4. Verify `/api/scenarios` and Workbench data source status
