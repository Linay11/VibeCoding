# Environment Variables Reference

This document summarizes runtime environment variables used by the current repository across frontend, backend adapter, and helper scripts.

Scope:
- local frontend (`frontend/`)
- backend adapter (`backend_adapter/`)
- startup scripts (`scripts/`)

---

## Quick Recommendation (Local Frontend + SSH Tunnel + AutoDL Backend)

Recommended setup for the validated workflow:

- Frontend (`frontend/.env.development`)
  - `VITE_API_BASE=http://127.0.0.1:8000`
- Backend (AutoDL shell)
  - `BACKEND_ADAPTER_CORS=http://localhost:5173,http://127.0.0.1:5173`
  - keep other backend adapter vars as defaults unless you need custom path/port

Why:
- frontend calls local tunnel endpoint (`127.0.0.1:8000`)
- tunnel forwards traffic to remote AutoDL backend
- CORS allows Vite dev origins

---

## Variable Catalog

| Variable | Layer | Purpose | Default | Example | Typical Scenario |
|---|---|---|---|---|---|
| `VITE_API_BASE` | Frontend (Vite) | API base URL prepended to frontend requests | `""` in frontend code (same-origin), but `.env.development.example` is `http://127.0.0.1:8000` | `VITE_API_BASE=http://127.0.0.1:8000` | Local frontend + SSH tunnel to remote backend |
| `BACKEND_ADAPTER_CORS` | Backend adapter | Comma-separated allowed origins for CORS | `*` in `backend_adapter/main.py`; `scripts/start_backend.sh` defaults to local Vite origins | `BACKEND_ADAPTER_CORS=http://localhost:5173,http://127.0.0.1:5173` | Enable browser access from local Vite |
| `BACKEND_ADAPTER_STORE` | Backend adapter | Path to latest-run JSON store used by `/api/runs/latest` | `backend_adapter/data/latest_runs.json` | `BACKEND_ADAPTER_STORE=/data/vibecoding/latest_runs.json` | Persist run history file in custom location |
| `BACKEND_ADAPTER_HOST` | Backend start script | Host/IP for `uvicorn` bind address | `0.0.0.0` | `BACKEND_ADAPTER_HOST=0.0.0.0` | Expose backend on AutoDL VM |
| `BACKEND_ADAPTER_PORT` | Backend start script | Port for `uvicorn` server | `8000` | `BACKEND_ADAPTER_PORT=8000` | Standard backend adapter port |

---

## Where Variables Are Read

- Frontend:
  - `frontend/src/services/optimizerApi.js` reads `import.meta.env.VITE_API_BASE`.
  - `frontend/.env.development.example` provides tunnel-mode default.
- Backend adapter:
  - `backend_adapter/main.py` reads:
    - `BACKEND_ADAPTER_CORS`
    - `BACKEND_ADAPTER_STORE`
- Script layer:
  - `scripts/start_backend.sh` reads:
    - `BACKEND_ADAPTER_HOST`
    - `BACKEND_ADAPTER_PORT`
    - `BACKEND_ADAPTER_CORS`
    - `BACKEND_ADAPTER_STORE`
  - `scripts/start_frontend.sh` reads:
    - `VITE_API_BASE`

---

## Usage Examples

### 1) Frontend `.env` file

```bash
cd frontend
cp .env.development.example .env.development
# .env.development
VITE_API_BASE=http://127.0.0.1:8000
```

### 2) Start backend with explicit CORS and port (AutoDL/Linux)

```bash
BACKEND_ADAPTER_CORS="http://localhost:5173,http://127.0.0.1:5173" \
BACKEND_ADAPTER_PORT=8000 \
./scripts/start_backend.sh
```

### 3) PowerShell temporary env for frontend

```powershell
$env:VITE_API_BASE = "http://127.0.0.1:8000"
cd frontend
npm run dev
```

---

## Notes

- For current dev mode, keep `VITE_API_BASE` pointed to local tunnel endpoint (`127.0.0.1:8000`), not directly to remote AutoDL public IP.
- Avoid `BACKEND_ADAPTER_CORS=*` in stricter environments; use explicit origins.
