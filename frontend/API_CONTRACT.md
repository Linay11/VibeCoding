# Frontend Backend Integration Spec

This document captures the **current API contract expected by the frontend** (as implemented in `frontend/src/services/optimizerApi.js` and consumed in `frontend/src/pages/WorkbenchPage.jsx`).

## 1. Scope and Base URL

- Frontend API base: `VITE_API_BASE` (env var), default `""` (same origin).
- Effective request URL: `${VITE_API_BASE}<endpoint_path>`.
- All requests are sent with header:
  - `Content-Type: application/json`
- Any non-2xx response is treated as failure and triggers frontend fallback behavior.

## 2. Endpoints Currently Used / Implied

1. `GET /api/scenarios`
2. `GET /api/runs/latest?scenarioId=<id>`
3. `POST /api/runs`

No other backend endpoints are currently called by the frontend.

---

## 3. Shared Frontend-Normalized Models

These are the normalized shapes used by UI components after adapter processing.

### 3.1 Scenario (normalized)

```ts
type Scenario = {
  id: string
  name: string
  description: string
}
```

### 3.2 RunData (normalized)

```ts
type RunData = {
  runId: string
  scenarioId: string
  generatedAt: string // ISO timestamp
  metrics: {
    solveTimeMs: number
    infeasibilityRate: number
    suboptimality: number
  }
  strategies: Array<{
    id: string
    name: string
    feasible: boolean
    cost: number
    rank: number
  }>
  trend: Array<{
    label: string
    value: number
  }>
  comparison: Array<{
    label: string
    value: number
  }>
}
```

---

## 4. Endpoint Contract Details

## 4.1 `GET /api/scenarios`

### Method
- `GET`

### Request body
- None

### Accepted response schema (raw)
Frontend accepts **any one** of:

```json
{
  "scenarios": [ ... ]
}
```

or

```json
{
  "data": [ ... ]
}
```

or directly:

```json
[ ... ]
```

Where each array item can be:

1. String form
```json
"portfolio"
```

2. Object form
```json
{
  "id": "portfolio",
  "name": "Portfolio Optimization",
  "description": "..."
}
```

### Required fields (for successful API-mode usage)
- Top-level must resolve to an array via `payload.scenarios`, `payload.data`, or `payload`.
- Array must normalize to at least one valid scenario; otherwise frontend treats it as failure and falls back.

### Optional fields
- `id`, `name`, `description` are optional in object form (frontend derives defaults).

### Fallback-derived fields currently handled in frontend
- If item is string:
  - `id = item`
  - `name = item`
  - `description = "Scenario <index>"`
- If item object is missing fields:
  - `id = item.id ?? item.name ?? "scenario-<index>"`
  - `name = item.name ?? item.id ?? "Scenario <index>"`
  - `description = item.description ?? "Optimization scenario"`
- If endpoint fails / invalid / empty:
  - Uses local `mockScenarios`
  - Emits UI notice banner with failure reason

---

## 4.2 `GET /api/runs/latest?scenarioId=<id>`

### Method
- `GET`

### Request query
- `scenarioId` (string), provided by selected scenario in UI.

### Request body
- None

### Accepted response schema (raw)
Frontend accepts run object from:

```json
{
  "run": { ... }
}
```

or

```json
{
  "data": { ... }
}
```

or directly:

```json
{ ... }
```

### Run object fields (raw, any subset accepted)
```json
{
  "runId": "run-123",
  "id": "run-123",
  "scenarioId": "portfolio",
  "generatedAt": "2026-03-10T08:00:00.000Z",
  "metrics": {
    "solveTimeMs": 42.1,
    "infeasibilityRate": 0.01,
    "suboptimality": 0.02
  },
  "solveTimeMs": 42.1,
  "timeMs": 42.1,
  "infeasibilityRate": 0.01,
  "suboptimality": 0.02,
  "strategies": [
    {
      "id": "s1",
      "name": "Strategy A",
      "feasible": true,
      "cost": 8.9,
      "rank": 1
    }
  ],
  "trend": [
    { "label": "R-6", "value": 51.3 }
  ],
  "comparison": [
    { "label": "Strategy A", "value": 8.9 }
  ],
  "comparisons": [
    { "label": "Strategy A", "cost": 8.9 }
  ]
}
```

### Required fields (for successful API-mode usage)
- Response must resolve to an object (`run`, `data`, or direct object).
- No strictly required inner fields due normalization defaults.

### Optional fields
- All fields listed above are optional from frontend perspective.

### Fallback-derived fields currently handled in frontend
- `runId = raw.runId ?? raw.id ?? "run-<scenarioId>-<timestamp>"`
- `scenarioId = raw.scenarioId ?? request.scenarioId`
- `generatedAt = raw.generatedAt ?? new Date().toISOString()`
- Metrics:
  - `solveTimeMs = raw.metrics.solveTimeMs ?? raw.solveTimeMs ?? raw.timeMs ?? 0`
  - `infeasibilityRate = raw.metrics.infeasibilityRate ?? raw.infeasibilityRate ?? 0`
  - `suboptimality = raw.metrics.suboptimality ?? raw.suboptimality ?? 0`
- Strategy rows:
  - `id = row.id ?? "strategy-<index>"`
  - `name = row.name ?? "Strategy <index>"`
  - `feasible = Boolean(row.feasible)`
  - `cost = Number(row.cost ?? 0)`
  - `rank = Number(row.rank ?? <index>)`
- Trend rows:
  - from `trend[]`, each `value = value ?? solveTimeMs ?? solve ?? 0`
  - if `trend` missing/empty in runData, UI derives synthetic trend from `metrics.solveTimeMs`
- Comparison rows:
  - from `comparison[]` or `comparisons[]`, each `value = value ?? cost ?? 0`
  - if missing/empty, UI derives comparison from top 4 strategy rows (`rank`, `cost`)
- If endpoint fails / invalid:
  - Uses local `buildMockRun(scenarioId)`
  - Emits UI notice banner with failure reason

---

## 4.3 `POST /api/runs`

### Method
- `POST`

### Request body
```json
{
  "scenarioId": "portfolio"
}
```

### Required request fields
- `scenarioId: string`

### Accepted response schema
- Same as `GET /api/runs/latest` (Section 4.2).

### Required response fields (for successful API-mode usage)
- Same as `GET /api/runs/latest`: response must resolve to an object.

### Optional response fields
- Same as `GET /api/runs/latest`.

### Fallback-derived fields currently handled in frontend
- Same normalization and field derivation as `GET /api/runs/latest`.
- On failure:
  - Uses `buildMockRun(scenarioId)`
  - Emits UI notice banner with failure reason

---

## 5. Integration Notes for Backend Team

1. To avoid fallback mode, ensure all three endpoints return 2xx and JSON.
2. For best UX, return:
   - stable `runId`
   - fully populated `metrics`
   - `strategies` array
   - optional `trend` and `comparison` for chart precision
3. Wrapper compatibility is flexible:
   - array/object can be under `data` or `run`, or returned directly.
4. Frontend behavior is resilient, but missing values may show as `0`, generated IDs, or derived chart data.
