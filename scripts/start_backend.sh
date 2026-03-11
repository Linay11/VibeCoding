#!/usr/bin/env bash
set -euo pipefail

# Start backend_adapter on Linux/AutoDL.
# Usage:
#   ./scripts/start_backend.sh
# Optional env vars:
#   BACKEND_ADAPTER_HOST (default: 0.0.0.0)
#   BACKEND_ADAPTER_PORT (default: 8000)
#   BACKEND_ADAPTER_CORS (default: http://localhost:5173,http://127.0.0.1:5173)
#   BACKEND_ADAPTER_STORE (default: backend_adapter/data/latest_runs.json)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${BACKEND_ADAPTER_HOST:-0.0.0.0}"
PORT="${BACKEND_ADAPTER_PORT:-8000}"
export BACKEND_ADAPTER_CORS="${BACKEND_ADAPTER_CORS:-http://localhost:5173,http://127.0.0.1:5173}"
export BACKEND_ADAPTER_STORE="${BACKEND_ADAPTER_STORE:-$ROOT_DIR/backend_adapter/data/latest_runs.json}"

echo "[backend] root: $ROOT_DIR"
echo "[backend] host: $HOST port: $PORT"
echo "[backend] cors: $BACKEND_ADAPTER_CORS"
echo "[backend] store: $BACKEND_ADAPTER_STORE"

python -m pip install --disable-pip-version-check -q fastapi uvicorn pydantic pandas xlrd

exec uvicorn backend_adapter.main:app --host "$HOST" --port "$PORT"
