# Frontend Backend Integration Spec

This document captures the current contract expected by the Workbench frontend (`frontend/src/services/optimizerApi.js`).

## 1. Base URL and Transport

- API base is read from `VITE_API_BASE`.
- Effective request URL is `${VITE_API_BASE}<path>`.
- Requests use `Content-Type: application/json`.
- Non-2xx responses are treated as API failures by frontend.

## 2. Endpoints Used by Frontend

1. `GET /api/scenarios`
2. `POST /api/runs`
3. `GET /api/runs/latest?scenarioId=<id>`

No other backend endpoints are currently used by Workbench.

---

## 3. Stable Run Payload (Target Contract)

Both `POST /api/runs` and `GET /api/runs/latest` should return:

```json
{
  "run": {
    "runId": "run-portfolio-20260310120000000000",
    "scenarioId": "portfolio",
    "generatedAt": "2026-03-10T12:00:00.000000+00:00",
    "metrics": {
      "solveTimeMs": 42.5,
      "infeasibilityRate": 0.0,
      "suboptimality": 0.01
    },
    "strategies": [
      {
        "id": "strategy-1",
        "name": "SolverSolution",
        "feasible": true,
        "cost": 8.123,
        "rank": 1
      }
    ],
    "trend": [
      { "label": "R-6", "value": 48.0 }
    ],
    "comparison": [
      { "label": "SolverSolution", "value": 8.123 }
    ],
    "adapterMode": "real",
    "adapterNote": "portfolio solved via OSQP"
  }
}
```

### Required fields

- `run.runId`
- `run.scenarioId`
- `run.generatedAt`
- `run.metrics.solveTimeMs`
- `run.metrics.infeasibilityRate`
- `run.metrics.suboptimality`
- `run.strategies` (array)
- `run.trend` (array)
- `run.comparison` (array)
- `run.adapterMode` (`real` or `compat`)
- `run.adapterNote` (string, can be diagnostic)

### Optional/compatibility fields accepted by frontend

Frontend still tolerates older shapes for resilience, but backend should avoid relying on them:

- top-level `data` wrapper or direct run object
- `run.id` as fallback for `runId`
- `solveTimeMs` / `timeMs` at top level
- `comparison` alternative key: `comparisons`

---

## 4. Scenario List Contract

`GET /api/scenarios` should return:

```json
{
  "scenarios": [
    {
      "id": "portfolio",
      "name": "Portfolio Optimization",
      "description": "..."
    }
  ]
}
```

### Required fields

- top-level `scenarios` array
- each item should include `id`, `name`, `description`

Frontend still tolerates legacy forms (`data` wrapper, raw array, string items), but this is not recommended for backend.

---

## 5. Error Contract

Backend errors should follow:

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "no latest run for scenarioId: portfolio"
  }
}
```

Recommended status mapping:
- `400` invalid arguments (`INVALID_ARGUMENT`)
- `404` not found (`NOT_FOUND`)
- `500` runtime failure (`RUN_FAILED`)

---

## 6. Mode Semantics (UI)

Workbench displays three execution/data modes:

- `real`: backend actually solved scenario using runtime solver stack
- `compat`: backend returned compatibility run (still from backend adapter), usually due solver/dependency/runtime constraints
- `fallback`: frontend generated local mock run because API request failed or latest data was unavailable

Mode source:
- API success: mode comes from `run.adapterMode`
- API failure: mode is frontend-local `fallback`

Reason source:
- API success: `run.adapterNote`
- API failure: classified frontend failure reason (network / backend_failed / no_latest / request_invalid)

---

## 7. Frontend Fallback-Derived Fields (Still Present, Should Be Rare)

If backend payload is incomplete, frontend may derive:

- `runId` from scenario + timestamp
- `generatedAt` from current timestamp
- `trend` from `metrics.solveTimeMs`
- `comparison` from top strategy rows

These are resilience-only fallbacks. Backend should provide full stable run payload to minimize derivation.
