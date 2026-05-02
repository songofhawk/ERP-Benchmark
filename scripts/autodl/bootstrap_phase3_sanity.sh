#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ERP_REPO_DIR:-/root/ERP-Benchmark}"
REPO_URL="${ERP_REPO_URL:-}"
DATA_ROOT="${PHASE3_DATA_ROOT:-}"
DATA_ARCHIVE_URL="${PHASE3_DATA_ARCHIVE_URL:-}"
DATA_ARCHIVE_NAME="${PHASE3_DATA_ARCHIVE_NAME:-cesca-aodd-200hz.tar.gz}"
DATA_EXTRACT_DIR="${PHASE3_DATA_EXTRACT_DIR:-/root/autodl-fs}"
INSTALL_REQUIREMENTS="${PHASE3_INSTALL_REQUIREMENTS:-0}"
REQUIREMENTS_FILE="${PHASE3_REQUIREMENTS_FILE:-requirements-phase3-sanity.txt}"
PYTHON_BIN="${PHASE3_PYTHON:-}"
LOG_DIR="$ROOT_DIR/results/phase3_logs"
BOOTSTRAP_LOG="$LOG_DIR/autodl_bootstrap_phase3_sanity.log"

if [[ ! -d "$ROOT_DIR/.git" ]]; then
  if [[ -z "$REPO_URL" ]]; then
    echo "Missing repo at $ROOT_DIR and ERP_REPO_URL is not set" >&2
    exit 2
  fi
  git clone "$REPO_URL" "$ROOT_DIR"
fi

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"
: > "$BOOTSTRAP_LOG"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] bootstrap started"
echo "ROOT_DIR=$ROOT_DIR"

if [[ -d "$ROOT_DIR/.git" ]]; then
  git status --short || true
  git pull --ff-only || true
fi

if [[ -z "$DATA_ROOT" ]]; then
  if [[ -d "$ROOT_DIR/dataset/200Hz" ]]; then
    DATA_ROOT="$ROOT_DIR/dataset/200Hz"
  elif [[ -d "/root/autodl-fs/dataset/200Hz" ]]; then
    DATA_ROOT="/root/autodl-fs/dataset/200Hz"
  elif [[ -d "/root/autodl-fs/200Hz" ]]; then
    DATA_ROOT="/root/autodl-fs/200Hz"
  elif [[ -d "/root/autodl-tmp/dataset/200Hz" ]]; then
    DATA_ROOT="/root/autodl-tmp/dataset/200Hz"
  elif [[ -d "/root/autodl-tmp/200Hz" ]]; then
    DATA_ROOT="/root/autodl-tmp/200Hz"
  else
    DATA_ROOT="$ROOT_DIR/dataset/200Hz"
  fi
fi
echo "DATA_ROOT=$DATA_ROOT"

dataset_exists_under() {
  local root="$1"
  local feature_dir="$root/CESCA-AODD/Feature"
  local label_dir="$root/CESCA-AODD/Label"
  [[ -d "$feature_dir" && -d "$label_dir" ]] && find "$feature_dir" -name '*.npy' -print -quit | grep -q .
}

resolve_phase3_data_root() {
  local candidates=(
    "$DATA_ROOT"
    "$DATA_EXTRACT_DIR/dataset/200Hz"
    "$DATA_EXTRACT_DIR/200Hz"
    "$DATA_EXTRACT_DIR/downloads_gdrive/dataset/200Hz"
    "$DATA_EXTRACT_DIR/downloads_gdrive/200Hz"
    "$ROOT_DIR/downloads_gdrive/dataset/200Hz"
    "$ROOT_DIR/downloads_gdrive/200Hz"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if dataset_exists_under "$candidate"; then
      DATA_ROOT="$candidate"
      return 0
    fi
  done
  return 1
}

ensure_phase3_data() {
  if resolve_phase3_data_root; then
    echo "CESCA-AODD data already exists under $DATA_ROOT"
    return 0
  fi

  if [[ -z "$DATA_ARCHIVE_URL" ]]; then
    echo "CESCA-AODD data missing under $DATA_ROOT and PHASE3_DATA_ARCHIVE_URL is not set" >&2
    return 2
  fi

  mkdir -p "$DATA_EXTRACT_DIR"

  if [[ "$DATA_ARCHIVE_URL" == *"drive.google.com/drive/folders/"* ]]; then
    echo "Downloading CESCA-AODD from Google Drive folder to $DATA_EXTRACT_DIR"
    "$PYTHON_BIN" - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("gdown") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown~=5.2.0"])
PY
    "$PYTHON_BIN" "$ROOT_DIR/scripts/download_processed_datasets.py" \
      --url "$DATA_ARCHIVE_URL" \
      --output "$DATA_EXTRACT_DIR" \
      --datasets CESCA-AODD \
      --stop-on-error
    if resolve_phase3_data_root; then
      echo "CESCA-AODD data downloaded under $DATA_ROOT"
      return 0
    fi
    echo "Google Drive download finished, but CESCA-AODD was not found under expected roots" >&2
    return 4
  fi

  local archive_path="/tmp/$DATA_ARCHIVE_NAME"
  echo "Downloading CESCA-AODD archive to $archive_path"
  curl -L "$DATA_ARCHIVE_URL" -o "$archive_path"

  echo "Extracting $archive_path to $DATA_EXTRACT_DIR"
  case "$archive_path" in
    *.tar.gz|*.tgz)
      tar -xzf "$archive_path" -C "$DATA_EXTRACT_DIR"
      ;;
    *.tar)
      tar -xf "$archive_path" -C "$DATA_EXTRACT_DIR"
      ;;
    *.zip)
      "$PYTHON_BIN" - "$archive_path" "$DATA_EXTRACT_DIR" <<'PY'
import sys
import zipfile

archive_path, extract_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(archive_path) as zf:
    zf.extractall(extract_dir)
PY
      ;;
    *)
      echo "Unsupported archive type: $archive_path" >&2
      return 3
      ;;
  esac

  rm -f "$archive_path"
  if ! resolve_phase3_data_root; then
    echo "Data extraction finished, but expected folders are still missing:" >&2
    echo "  $DATA_ROOT/CESCA-AODD/Feature" >&2
    echo "  $DATA_ROOT/CESCA-AODD/Label" >&2
    return 4
  fi
  echo "CESCA-AODD data extracted under $DATA_ROOT"
}

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi
echo "PYTHON_BIN=$PYTHON_BIN"

if [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"
fi

ensure_phase3_data
"$PYTHON_BIN" verify_env.py
"$PYTHON_BIN" scripts/phase3/run_phase3_sanity.py \
  --dataset CESCA-AODD \
  --data-root "$DATA_ROOT" \
  --python "$PYTHON_BIN" \
  --epochs 1 \
  --patience 1 \
  --itr 1 \
  --batch-size "${PHASE3_BATCH_SIZE:-128}" \
  --devices "${PHASE3_DEVICES:-0}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-0}" \
  ${PHASE3_FORCE:+--force}

"$PYTHON_BIN" scripts/phase3/collect_phase3_results.py --dataset CESCA-AODD

ARCHIVE_DIR="$ROOT_DIR/results/phase3_logs"
ARCHIVE_PATH="$ARCHIVE_DIR/phase3_sanity_artifacts_$(date '+%Y%m%d_%H%M%S').tar.gz"
tar -czf "$ARCHIVE_PATH" \
  results/phase3_logs \
  results/TestFormer/supervised/TestFormer/S-CESCA-AODD-phase3-sanity-* \
  checkpoints/TestFormer/supervised/TestFormer/S-CESCA-AODD-phase3-sanity-* \
  2>/dev/null || true

touch "$ROOT_DIR/results/phase3_logs/PHASE3_SANITY_DONE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] bootstrap completed"
echo "archive=$ARCHIVE_PATH"
