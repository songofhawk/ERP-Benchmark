#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/dataset/200Hz}"
DATASET_DIR="$DATA_ROOT/CESCA-AODD"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing project virtualenv python: $PYTHON_BIN"
  exit 1
fi

if [[ ! -d "$DATASET_DIR/Feature" || ! -d "$DATASET_DIR/Label" ]]; then
  echo "Missing processed dataset at: $DATASET_DIR"
  echo "Expected:"
  echo "  $DATASET_DIR/Feature"
  echo "  $DATASET_DIR/Label"
  exit 1
fi

"$PYTHON_BIN" -u "$ROOT_DIR/run.py" \
  --method EEGNet \
  --task_name supervised \
  --is_training 1 \
  --root_path "$DATA_ROOT/" \
  --model_id S-CESCA-AODD-canonical-balanced-long \
  --model EEGNet \
  --data MultiDatasets \
  --training_datasets CESCA-AODD \
  --testing_datasets CESCA-AODD \
  --batch_size 32 \
  --use_class_weights \
  --des 'Canonical-Balanced-Long' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 5
