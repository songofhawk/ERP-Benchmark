#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <DATASET_NAME>" >&2
  exit 1
fi

DATASET_NAME="$1"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
DATA_ROOT="$ROOT_DIR/dataset/200Hz"
LOG_DIR="$ROOT_DIR/results/phase2_logs"
RUN_LOG="$LOG_DIR/phase2_${DATASET_NAME}_mps.log"
STATUS_FILE="$LOG_DIR/phase2_${DATASET_NAME}_mps.status"
CURRENT_MODEL_FILE="$LOG_DIR/phase2_${DATASET_NAME}_mps.current_model"
CURRENT_MODEL="idle"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing project virtualenv python: $PYTHON_BIN" >&2
  exit 1
fi

precheck_single_class() {
  "$PYTHON_BIN" - <<'PY'
from types import SimpleNamespace
import numpy as np
from data_provider.data_factory import data_provider
import os

dataset_name = os.environ['DATASET_NAME']
args = SimpleNamespace(
    data='MultiDatasets',
    task_name='supervised',
    root_path=os.environ['DATA_ROOT'] + '/',
    pretraining_datasets='TDBRAIN-19',
    training_datasets=dataset_name,
    testing_datasets=dataset_name,
    cross_val='mccv',
    seed=41,
    no_normalize=False,
    segment_length=128,
    overlapping=0.5,
    sampling_rate=200,
    low_cut=0.5,
    high_cut=45,
    batch_size=64,
    num_workers=0,
    seq_len=96,
)
train_data, _ = data_provider(args, 'TRAIN')
unique_labels = np.unique(train_data.y[:, 0])
print(len(unique_labels))
PY
}

mark_failed() {
  local exit_code="$1"
  echo "failed:${CURRENT_MODEL}:${exit_code}" > "$STATUS_FILE"
  echo "$CURRENT_MODEL" > "$CURRENT_MODEL_FILE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL ${CURRENT_MODEL} on ${DATASET_NAME} exit_code=${exit_code}" | tee -a "$RUN_LOG"
}

trap 'code=$?; mark_failed "$code"; exit "$code"' ERR

run_model() {
  local model_name="$1"
  shift
  CURRENT_MODEL="$model_name"
  echo "$CURRENT_MODEL" > "$CURRENT_MODEL_FILE"
  echo "running:${CURRENT_MODEL}" > "$STATUS_FILE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $model_name on $DATASET_NAME" | tee -a "$RUN_LOG"
  "$PYTHON_BIN" -u "$ROOT_DIR/run.py" "$@" 2>&1 | tee -a "$RUN_LOG"
  local exit_code=${PIPESTATUS[0]}
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $model_name on $DATASET_NAME exit_code=$exit_code" | tee -a "$RUN_LOG"
  if [[ $exit_code -ne 0 ]]; then
    echo "failed:$model_name:$exit_code" > "$STATUS_FILE"
    exit $exit_code
  fi
}

echo "running:bootstrap" > "$STATUS_FILE"
echo "$CURRENT_MODEL" > "$CURRENT_MODEL_FILE"
: > "$RUN_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] START Phase2 compare on $DATASET_NAME" | tee -a "$RUN_LOG"

CLASS_COUNT=$(DATASET_NAME="$DATASET_NAME" DATA_ROOT="$DATA_ROOT" precheck_single_class | tail -n 1 | tr -d ' ')
if [[ "$CLASS_COUNT" -lt 2 ]]; then
  CURRENT_MODEL="skipped"
  echo "$CURRENT_MODEL" > "$CURRENT_MODEL_FILE"
  echo "skipped:single_class_train_labels" > "$STATUS_FILE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP $DATASET_NAME because training split has only one class" | tee -a "$RUN_LOG"
  exit 10
fi

run_model EEGFeatures \
  --method EEGFeatures \
  --task_name supervised \
  --is_training 1 \
  --root_path "$DATA_ROOT/" \
  --model_id "S-${DATASET_NAME}-phase2-long-mps" \
  --model EEGFeatures \
  --data MultiDatasets \
  --training_datasets "$DATASET_NAME" \
  --testing_datasets "$DATASET_NAME" \
  --batch_size 64 \
  --use_class_weights \
  --des 'Phase2-Long-MPS' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 5

run_model ERPFeatures \
  --method ERPFeatures \
  --task_name supervised \
  --is_training 1 \
  --root_path "$DATA_ROOT/" \
  --model_id "S-${DATASET_NAME}-phase2-long-mps" \
  --model ERPFeatures \
  --data MultiDatasets \
  --training_datasets "$DATASET_NAME" \
  --testing_datasets "$DATASET_NAME" \
  --batch_size 64 \
  --use_class_weights \
  --des 'Phase2-Long-MPS' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 5

run_model EEGNet \
  --method EEGNet \
  --task_name supervised \
  --is_training 1 \
  --root_path "$DATA_ROOT/" \
  --model_id "S-${DATASET_NAME}-canonical-balanced-long-mps" \
  --model EEGNet \
  --data MultiDatasets \
  --training_datasets "$DATASET_NAME" \
  --testing_datasets "$DATASET_NAME" \
  --batch_size 32 \
  --use_class_weights \
  --des 'Canonical-Balanced-Long-MPS' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 5

run_model EEGConformer \
  --method EEGConformer \
  --task_name supervised \
  --is_training 1 \
  --root_path "$DATA_ROOT/" \
  --model_id "S-${DATASET_NAME}-phase2-long-mps" \
  --model EEGConformer \
  --data MultiDatasets \
  --training_datasets "$DATASET_NAME" \
  --testing_datasets "$DATASET_NAME" \
  --e_layers 6 \
  --batch_size 32 \
  --n_heads 8 \
  --d_model 128 \
  --d_ff 256 \
  --use_class_weights \
  --des 'Phase2-Long-MPS' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 5

CURRENT_MODEL="completed"
echo "$CURRENT_MODEL" > "$CURRENT_MODEL_FILE"
echo "completed" > "$STATUS_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] END Phase2 compare on $DATASET_NAME" | tee -a "$RUN_LOG"
