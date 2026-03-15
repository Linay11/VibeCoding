# Power118 Minimal Ablation Commands (AutoDL/Gurobi)

## 1) Build Dataset (once)

```bash
python scripts/build_power118_ml_dataset.py \
  --num-samples 64 \
  --seed 7 \
  --output-dir backend_adapter/data/power118_dataset
```

Expected dataset:
- `backend_adapter/data/power118_dataset/power118_ml_dataset.pkl`

## 2) Batch Training Command List

Set one run root first:

```bash
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="backend_adapter/data/power118_ablation_runs/${RUN_TAG}"
MODELS_ROOT="${RUN_ROOT}/models"
DATASET_PATH="backend_adapter/data/power118_dataset/power118_ml_dataset.pkl"
```

Naming convention:
- Variant archive dir: `${MODELS_ROOT}/<objective>__<feature-mode>/`
- Example: `proxy-only__inst-plus-abs`
- Model file: `power118_ml_model.joblib`
- Metadata file: `power118_ml_metadata.json`

Train objective ablation (`inst+abs`):

```bash
python scripts/train_power118_model.py --dataset-path "$DATASET_PATH" --output-dir "$MODELS_ROOT" --archive-tag "proxy-only__inst-plus-abs" --model-variant "proxy-only__inst-plus-abs" --constraint-training-objective proxy-only --feature-ablation-mode inst+abs --no-publish-default-artifacts
python scripts/train_power118_model.py --dataset-path "$DATASET_PATH" --output-dir "$MODELS_ROOT" --archive-tag "mixed__inst-plus-abs" --model-variant "mixed__inst-plus-abs" --constraint-training-objective mixed --feature-ablation-mode inst+abs --no-publish-default-artifacts
python scripts/train_power118_model.py --dataset-path "$DATASET_PATH" --output-dir "$MODELS_ROOT" --archive-tag "exact-priority__inst-plus-abs" --model-variant "exact-priority__inst-plus-abs" --constraint-training-objective exact-priority --feature-ablation-mode inst+abs --no-publish-default-artifacts
```

Optional representation ablation example (same objective, different features):

```bash
python scripts/train_power118_model.py --dataset-path "$DATASET_PATH" --output-dir "$MODELS_ROOT" --archive-tag "mixed__inst-only" --model-variant "mixed__inst-only" --constraint-training-objective mixed --feature-ablation-mode inst-only --no-publish-default-artifacts
python scripts/train_power118_model.py --dataset-path "$DATASET_PATH" --output-dir "$MODELS_ROOT" --archive-tag "mixed__abs-only" --model-variant "mixed__abs-only" --constraint-training-objective mixed --feature-ablation-mode abs-only --no-publish-default-artifacts
python scripts/train_power118_model.py --dataset-path "$DATASET_PATH" --output-dir "$MODELS_ROOT" --archive-tag "mixed__inst-plus-abs" --model-variant "mixed__inst-plus-abs" --constraint-training-objective mixed --feature-ablation-mode inst+abs --no-publish-default-artifacts
```

## 3) Variant Config Template

Copy and edit:

```bash
cp scripts/power118_variant_config.template.json "${RUN_ROOT}/variant-config.json"
```

Update each `modelPath` / `metadataPath` to point to your trained artifacts under `${MODELS_ROOT}`.

## 4) Batch Evaluation Command List

```bash
python scripts/eval_power118_modes.py \
  --num-cases 8 \
  --seed 7 \
  --output-dir "${RUN_ROOT}/eval" \
  --variant-config-path "${RUN_ROOT}/variant-config.json" \
  --modes exact hybrid_warm_start hybrid_constraint_aware_v2 hybrid_constraint_aware_v3 ml \
  --require-exact-baseline
```

## 5) One-Command Orchestration (recommended)

```bash
python scripts/run_power118_ablation.py \
  --dataset-path "$DATASET_PATH" \
  --run-root "$RUN_ROOT" \
  --objectives proxy-only mixed exact-priority \
  --feature-modes inst+abs \
  --num-cases 8 \
  --modes exact hybrid_warm_start hybrid_constraint_aware_v2 hybrid_constraint_aware_v3 ml \
  --require-exact-baseline
```

To include representation ablation:

```bash
python scripts/run_power118_ablation.py \
  --dataset-path "$DATASET_PATH" \
  --run-root "$RUN_ROOT" \
  --objectives proxy-only mixed exact-priority \
  --feature-modes inst-only abs-only inst+abs \
  --num-cases 8 \
  --modes exact hybrid_warm_start hybrid_constraint_aware_v2 hybrid_constraint_aware_v3 ml \
  --require-exact-baseline
```
