# mlopt Frontend (React + Vite)

This is a minimal runnable frontend for the `mlopt` repository.

## Included in this MVP

- Home page
- Main feature page (Optimization Workbench)
- Top navigation
- Polished responsive UI with accessibility basics
- Data adapter that prefers backend API and falls back to mock data
- Lightweight chart panel (trend + result comparison)

## API strategy

The frontend tries these endpoints first:

- `GET /api/scenarios`
- `GET /api/runs/latest?scenarioId=<id>`
- `POST /api/runs`

If any request fails, UI automatically uses mock data so the app remains usable.

## Run locally

```bash
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite (default `http://localhost:5173`).

## Quick access checklist

After startup, verify these pages:

- Home: `http://localhost:5173/`
- Workbench: `http://localhost:5173/workbench`

Navigation links in the header should switch between both pages.

## Build and lint

```bash
npm run lint
npm run build
```

## Optional environment variable

Use a backend base URL if API is not on the same origin:

```bash
VITE_API_BASE=http://localhost:8000
```

Windows PowerShell:

```powershell
$env:VITE_API_BASE="http://localhost:8000"
npm run dev
```

For AutoDL remote backend (recommended for local-frontend + remote-backend integration):

```powershell
$env:VITE_API_BASE="http://<your-autodl-ip-or-domain>:8000"
npm run dev
```

You can also copy and edit:

```bash
cp .env.development.example .env.development
```

Default `.env.development.example` is already set for SSH tunnel mode:

```bash
VITE_API_BASE=http://127.0.0.1:8000
```

## API fallback behavior

- UI first tries backend endpoints:
  - `GET /api/scenarios`
  - `GET /api/runs/latest?scenarioId=<id>`
  - `POST /api/runs`
- If a request fails, the page remains functional with mock data.
- A visible notice banner explains why fallback happened.

## Chart data source

Workbench chart panel has two charts:

- Solve Time Trend
- Result Comparison

Data source priority:

1. Use API payload fields if present:
   - `run.trend[]` (or `data.trend[]`) for trend chart
   - `run.comparison[]` / `run.comparisons[]` for comparison chart
2. If API does not provide chart fields:
   - Trend is derived from latest run metrics
   - Comparison is derived from strategy cost rows
3. If API request fails:
   - Full run data (including chart fields) is generated from mock data

## Interaction checklist

- Scenario selector updates run context.
- `Refresh Latest` pulls latest run data.
- `Run Experiment` triggers a run action (or mock run if API unavailable).
- Buttons disable during loading/running to prevent duplicate actions.
- Empty and no-data states include actionable guidance.

## Known limitations in this MVP

- No backend auth/session flow yet.
- No long-running job queue/progress polling yet.
- Chart panel is lightweight and static (no zoom/brush or multi-series controls yet).
- No automated E2E tests yet.
