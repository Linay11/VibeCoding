#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/backend_adapter/data/power118_dataset}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/backend_adapter/data/power118_model}"
EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/backend_adapter/data/power118_eval}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
NUM_CASES="${NUM_CASES:-8}"
SEED="${SEED:-7}"

echo "[power118] repo: $ROOT_DIR"
echo "[power118] dataset dir: $DATASET_DIR"
echo "[power118] model dir: $MODEL_DIR"
echo "[power118] eval dir: $EVAL_DIR"
echo "[power118] seed: $SEED"
echo "[power118] dataset samples: $NUM_SAMPLES"
echo "[power118] eval cases: $NUM_CASES"

python scripts/check_power118_env.py
python scripts/build_power118_ml_dataset.py --num-samples "$NUM_SAMPLES" --seed "$SEED" --output-dir "$DATASET_DIR"
python scripts/train_power118_model.py --dataset-path "$DATASET_DIR/power118_ml_dataset.pkl" --output-dir "$MODEL_DIR" --random-state "$SEED"
python scripts/eval_power118_modes.py --num-cases "$NUM_CASES" --seed "$SEED" --output-dir "$EVAL_DIR" --require-exact-baseline
