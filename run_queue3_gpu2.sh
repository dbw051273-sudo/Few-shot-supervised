#!/usr/bin/env bash
set -euo pipefail

# Strict GPU-2 lock for Workstation 5880 Queue-3 Baselines (CUT, I2SB)
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$(dirname "$SCRIPT_ROOT")/UniPrior_v2}"
PRESET="${1:-queue3}"
if [[ $# -gt 0 ]]; then
  shift
fi
PY_BIN="${PY_BIN:-python}"

mkdir -p "$SCRIPT_ROOT/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$SCRIPT_ROOT/logs/queue3_${PRESET}_gpu2_${TS}.log"

LOCK_FILE="/tmp/fewshot_vstain_gpu2_${USER:-u}.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[ERROR] GPU 2 mutex lock is already held ($LOCK_FILE)." | tee -a "$LOG_FILE" >&2
    exit 1
  fi
fi
ACTIVE_PIDS="$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' || true)"
if [[ -n "$ACTIVE_PIDS" && "${ALLOW_GPU2_SHARING:-0}" != "1" ]]; then
  echo "[ERROR] GPU 2 already has active compute PID(s): $ACTIVE_PIDS." | tee -a "$LOG_FILE" >&2
  exit 1
fi
export VSTAIN_GPU2_LOCK_HELD=1

if [[ "$PRESET" != "smoke" ]]; then
  "$PY_BIN" -u "$SCRIPT_ROOT/scripts/run_queue3_baselines.py" \
    --gpu-id 2 \
    --work-dir "$SCRIPT_ROOT/_smoke_check_q3" \
    --cache-dir "$SCRIPT_ROOT/eval_cache" \
    --data-dir "$DATA_DIR" \
    --preset smoke \
    --no-skip-done 2>&1 | tee -a "$LOG_FILE"
fi

"$PY_BIN" -u "$SCRIPT_ROOT/scripts/run_queue3_baselines.py" \
  --gpu-id 2 \
  --work-dir "$SCRIPT_ROOT" \
  --cache-dir "$SCRIPT_ROOT/eval_cache" \
  --data-dir "$DATA_DIR" \
  --preset "$PRESET" ${1+"$@"} 2>&1 | tee -a "$LOG_FILE"

"$PY_BIN" -u "$SCRIPT_ROOT/scripts/pack_round1_outputs.py" \
  --work-dir "$SCRIPT_ROOT" \
  --tag "queue3_${PRESET}" 2>&1 | tee -a "$LOG_FILE"
