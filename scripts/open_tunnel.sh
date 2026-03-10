#!/usr/bin/env bash
set -euo pipefail

# Open local SSH tunnel to remote backend.
# Default mapping:
#   local 127.0.0.1:8000 -> remote 127.0.0.1:8000
#
# Usage:
#   ./scripts/open_tunnel.sh user@host [local_port] [remote_port] [ssh_port]
#
# Examples:
#   ./scripts/open_tunnel.sh user@autodl-host
#   ./scripts/open_tunnel.sh user@autodl-host 8000 8000 22

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 user@host [local_port] [remote_port] [ssh_port]"
  exit 1
fi

TARGET="$1"
LOCAL_PORT="${2:-8000}"
REMOTE_PORT="${3:-8000}"
SSH_PORT="${4:-22}"

echo "[tunnel] target: $TARGET"
echo "[tunnel] local : 127.0.0.1:${LOCAL_PORT}"
echo "[tunnel] remote: 127.0.0.1:${REMOTE_PORT}"
echo "[tunnel] ssh port: ${SSH_PORT}"
echo "[tunnel] press Ctrl+C to close"

exec ssh -p "$SSH_PORT" -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$TARGET"

