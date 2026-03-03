#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${TB_LOGDIR:-$SCRIPT_DIR/runs}"
PORT="${TB_PORT:-6006}"
HOST="${TB_HOST:-127.0.0.1}"

echo "Starting TensorBoard"
echo "  logdir: $LOGDIR"
echo "  host:   $HOST"
echo "  port:   $PORT"

exec tensorboard --logdir "$LOGDIR" --host "$HOST" --port "$PORT"