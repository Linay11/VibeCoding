#!/usr/bin/env bash
set -euo pipefail

# Start local frontend against a local tunnel endpoint.
# Usage:
#   ./scripts/start_frontend.sh
# Optional env:
#   VITE_API_BASE_URL (default: http://127.0.0.1:8000)
#   VITE_API_BASE (legacy fallback, optional)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
cd "$FRONTEND_DIR"

API_BASE_URL="${VITE_API_BASE_URL:-${VITE_API_BASE:-http://127.0.0.1:8000}}"
export VITE_API_BASE_URL="$API_BASE_URL"

echo "[frontend] dir: $FRONTEND_DIR"
echo "[frontend] VITE_API_BASE_URL: $VITE_API_BASE_URL"

if [[ ! -f ".env.development" ]]; then
  cp .env.development.example .env.development
  echo "[frontend] created .env.development from example"
fi

npm install
exec npm run dev
