#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${HOT_POKER_PYTHON:-$ROOT/.venv/bin/python}"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"
"$PYTHON_BIN" -m super_poker.automation weekly "$@" \
  >> "$LOG_DIR/weekly-learning.log" 2>&1
