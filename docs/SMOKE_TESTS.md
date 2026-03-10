# Smoke Tests (Minimal but Effective)

This document defines the current minimal smoke-test loop for this repository.

Goal:
- quickly detect regressions on main API/UI path
- keep tooling lightweight
- avoid touching `mlopt/` and `online_optimization/` core logic

---

## 1) Backend Adapter Smoke Tests (Automated)

### Scope

Current automated smoke tests cover:
- `GET /api/scenarios`
- `POST /api/runs`
- `GET /api/runs/latest`
- key error branches (`400`, `404`)
- stable run payload fields

Test file:
- `backend_adapter/tests/test_run_endpoints.py`

### Dependencies

If your local Python env does not include test packages:

```bash
pip install pytest httpx
```

Notes:
- `pytest` is required to run tests.
- `httpx` is required by FastAPI/Starlette `TestClient`.

### Run command

From repository root:

```bash
python -m pytest backend_adapter/tests/test_run_endpoints.py -q
```

### What should pass

- scenarios endpoint returns non-empty list with `id/name/description`
- post run returns stable `run` contract
- latest returns same stable shape after post
- latest returns `404 NOT_FOUND` before any run exists
- invalid requests return `400 INVALID_ARGUMENT`

---

## 2) Frontend Smoke Verification (Minimal)

Frontend now includes focused component smoke tests (Vitest + React Testing Library),
plus existing `lint + build` checks.

### Automated baseline

From `frontend/`:

```bash
npm run lint
npm run build
npm run test:smoke
```

What `test:smoke` covers:
- Workbench fallback response rendering path
- Workbench compat response rendering path
- Result Summary correctly displays:
  - data source
  - run mode
  - mode reason
- compat mode reason is consistent between summary and state panel
- `Run Experiment` state transition:
  - click -> `running`
  - button disabled/loading text
  - resolve -> `result` rendered
- `Refresh Latest` error branches:
  - `404 no latest` fallback rendering
  - network failure fallback rendering
  - summary / state panel / notice remain consistent without conflicting messages
- `Refresh Latest` success branches:
  - compat success refresh: `Refreshing...` -> `Refresh Latest` and timestamp update
  - real success refresh: `Refreshing...` -> `Refresh Latest` and timestamp update
  - button disabled state stays consistent during refresh
- Scenario switch linkage:
  - changing scenario triggers latest fetch for the new scenario id
  - Result Summary updates with new scenario context
  - state panel/mode switches to match the new scenario response
  - race-safe behavior: older latest response resolving late does not override the last selected scenario state
  - this guards the core Scenario -> latest -> summary/state linkage against regression

Why this minimal set now:
- highest-value demo path with minimal setup overhead
- protects runtime explainability and interaction stability without large test framework expansion

### Manual Workbench smoke checklist

Start stack (see `docs/DEV_SETUP.md` for full setup):
1. Start backend adapter
2. Open SSH tunnel (if remote backend)
3. Start frontend dev server
4. Open `http://localhost:5173/workbench`

Checklist:

1. Workbench renders with:
- page title and summary panel visible
- summary shows: scenario, data source, run mode, mode reason, generated time

2. Result Summary fields update after click:
- click `Run Experiment`
- `generated time` and `runId` update
- `data source` and `run mode` badges reflect returned mode

3. State rendering is clear:
- loading: visible while initial load / refresh / run
- empty: visible before run data exists
- backend real: state panel indicates real backend run
- backend compat: state panel indicates compatibility mode
- frontend fallback: state panel indicates fallback mode
- error: state panel indicates action error

4. Metrics and analysis:
- solve time / infeasibility rate / suboptimality show helper text
- trend chart and comparison chart render
- strategy table rows render with feasible/cost/rank

5. Fallback behavior:
- stop backend (or break API base/tunnel)
- click `Refresh Latest` or `Run Experiment`
- confirm fallback state + notice + mode reason are visible

---

## 3) Integration Sanity Checks (Frontend + Backend)

Use these quick endpoint checks from local machine:

```bash
curl http://127.0.0.1:8000/api/scenarios
curl -X POST "http://127.0.0.1:8000/api/runs" -H "Content-Type: application/json" -d '{"scenarioId":"portfolio"}'
curl "http://127.0.0.1:8000/api/runs/latest?scenarioId=portfolio"
```

Checkpoints:
- all responses are JSON
- `run.adapterMode` and `run.adapterNote` always exist on run payload
- frontend summary and state panel align with returned mode

---

## 4) Current Limitations

- frontend automated coverage is intentionally narrow (single high-value component smoke path).
- backend tests are smoke-level only; no exhaustive solver behavior assertions.
- real vs compat depends on runtime dependencies (solver/modules available in current environment).
