#!/usr/bin/env bash
set -euo pipefail

# Start local frontend against a local tunnel endpoint.
# Usage:
#   ./scripts/start_frontend.sh
# Optional env:
#   VITE_API_BASE (default: http://127.0.0.1:8000)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
cd "$FRONTEND_DIR"

export VITE_API_BASE="${VITE_API_BASE:-http://127.0.0.1:8000}"

echo "[frontend] dir: $FRONTEND_DIR"
echo "[frontend] VITE_API_BASE: $VITE_API_BASE"

if [[ ! -f ".env.development" ]]; then
  cp .env.development.example .env.development
  echo "[frontend] created .env.development from example"
fi

npm install
exec npm run dev

