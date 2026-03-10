# backend_adapter (Minimal FastAPI Layer)

This folder adds a thin API adapter without changing existing core logic in `mlopt/` and `online_optimization/`.

## Implemented endpoints

- `GET /api/scenarios`
- `POST /api/runs`
- `GET /api/runs/latest`

## Run payload stability

`POST /api/runs` and `GET /api/runs/latest` now normalize run payloads to a stable shape before returning:

- `runId`, `scenarioId`, `generatedAt`
- `metrics.solveTimeMs`, `metrics.infeasibilityRate`, `metrics.suboptimality`
- `strategies[]`, `trend[]`, `comparison[]`
- `adapterMode` (`real` or `compat`)
- `adapterNote` (diagnostic reason/details)

Latest-run reads are self-healing: legacy stored payloads are normalized on read and written back.

## Error behavior

The adapter returns consistent error responses:

- `400` for invalid parameters
- `404` when latest run data is missing
- `500` for runtime failures

Error shape:

```json
{
  "error": {
    "code": "INVALID_ARGUMENT",
    "message": "..."
  }
}
```

## Local run (minimal)

1. Install dependencies (example):

```bash
pip install fastapi uvicorn pydantic
```

2. From repository root, start server:

```bash
uvicorn backend_adapter.main:app --reload --host 0.0.0.0 --port 8000
```

Or use repository script:

```bash
./scripts/start_backend.sh
```

3. Check APIs:

- `http://localhost:8000/api/scenarios`
- `http://localhost:8000/api/runs/latest?scenarioId=portfolio`

4. Trigger one run:

```bash
curl -X POST "http://localhost:8000/api/runs" \
  -H "Content-Type: application/json" \
  -d "{\"scenarioId\":\"portfolio\"}"
```

## Smoke tests

Test file:
- `backend_adapter/tests/test_run_endpoints.py`

Install minimal test deps if needed:

```bash
pip install pytest httpx
```

Run smoke tests:

```bash
python -m pytest backend_adapter/tests/test_run_endpoints.py -q
```

## Environment variables

- `BACKEND_ADAPTER_STORE`: path for latest-run JSON store  
  default: `backend_adapter/data/latest_runs.json`
- `BACKEND_ADAPTER_CORS`: comma-separated CORS origins  
  default: `*`
  - Example for local Vite dev:
    - `BACKEND_ADAPTER_CORS=http://localhost:5173,http://127.0.0.1:5173`
  - Use `*` to allow all origins in development.

## Notes

- Scenario runners try to reuse existing problem builders first.
- If runtime dependencies/solvers are unavailable, adapter returns a minimal compatibility run payload (`adapterMode=compat`) instead of failing hard.
