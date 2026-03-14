# Power-118 Remote Runbook

This runbook is for Linux or AutoDL environments where real Gurobi-backed `exact` and `hybrid` validation is expected.

## Prerequisites

- repository cloned on the remote machine
- Python environment with project dependencies installed
- valid Gurobi license
- `external/power118/118_data.xls` available

Validate the exact runtime first:

```bash
cd /path/to/VibeCoding
python scripts/check_power118_env.py
```

If `Real mode ready: YES` is not shown, do not trust `exact` or `hybrid` evaluation results yet.

## 1. Build Dataset

```bash
cd /path/to/VibeCoding
python scripts/build_power118_ml_dataset.py \
  --num-samples 64 \
  --seed 7 \
  --output-dir backend_adapter/data/power118_dataset
```

Expected artifacts:
- `backend_adapter/data/power118_dataset/power118_ml_dataset.pkl`
- `backend_adapter/data/power118_dataset/dataset_summary.json`

## 2. Train Model

```bash
cd /path/to/VibeCoding
python scripts/train_power118_model.py \
  --dataset-path backend_adapter/data/power118_dataset/power118_ml_dataset.pkl \
  --output-dir backend_adapter/data/power118_model \
  --random-state 7
```

Expected artifacts:
- versioned directory under `backend_adapter/data/power118_model/`
- published default artifacts:
  - `backend_adapter/data/power118_ml_model.joblib`
  - `backend_adapter/data/power118_ml_metadata.json`

## 3. Run Offline Evaluation

```bash
cd /path/to/VibeCoding
python scripts/eval_power118_modes.py \
  --num-cases 8 \
  --output-dir backend_adapter/data/power118_eval \
  --require-exact-baseline
```

Expected artifacts:
- `backend_adapter/data/power118_eval/power118_eval_records.json`
- `backend_adapter/data/power118_eval/power118_eval_records.csv`
- `backend_adapter/data/power118_eval/summary.json`
- `backend_adapter/data/power118_eval/report.md`

## 4. Start Backend

```bash
cd /path/to/VibeCoding
./scripts/start_backend.sh
```

## 5. Validate Exact / Hybrid / ML

Exact:

```bash
curl -X POST "http://127.0.0.1:8000/api/runs" \
  -H "Content-Type: application/json" \
  -d '{"scenarioId":"power-118","runMode":"exact","fallbackToExact":true}'
```

Hybrid:

```bash
curl -X POST "http://127.0.0.1:8000/api/runs" \
  -H "Content-Type: application/json" \
  -d '{"scenarioId":"power-118","runMode":"hybrid","fallbackToExact":true}'
```

ML Only:

```bash
curl -X POST "http://127.0.0.1:8000/api/runs" \
  -H "Content-Type: application/json" \
  -d '{"scenarioId":"power-118","runMode":"ml","fallbackToExact":true}'
```

What to check in the returned payload:
- `requestedRunMode`
- `solverModeUsed`
- `fallbackReason`
- `modelVersion`
- `featureSchemaVersion`
- `runtimeMs`
- `objectiveValue`
- `feasible`

## Common Failure Points

- Gurobi or license unavailable:
  - `check_power118_env.py` fails
  - `adapterMode` becomes `compat`
  - `fallbackReason` mentions runtime blockage
- Model artifacts missing:
  - `backend_adapter/data/power118_ml_model.joblib` or metadata JSON not found
  - `hybrid` or `ml` downgrades with explicit `fallbackReason`
- Metadata or schema mismatch:
  - model loads but inference is blocked
  - `fallbackReason` mentions feature schema mismatch
- Exact baseline unavailable during evaluation:
  - `summary.json` and `report.md` will mark exact baseline as unavailable
  - objective-gap metrics should be treated as unavailable rather than trusted
