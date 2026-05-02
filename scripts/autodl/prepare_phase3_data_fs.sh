#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ERP_REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PYTHON_BIN="${PHASE3_PYTHON:-}"

cd "$ROOT_DIR"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/autodl/prepare_phase3_data_fs.py" "$@"
