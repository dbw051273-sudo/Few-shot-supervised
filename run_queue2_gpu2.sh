#!/usr/bin/env bash
set -euo pipefail

# Strict GPU-2 lock for Workstation 5880 Queue-2 Official Baselines (Pix2Pix, CycleGAN, UNIStainNet)
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$(dirname "$SCRIPT_ROOT")/UniPrior_v2}"
PRESET="${1:-queue2}"
if [[ $# -gt 0 ]]; then
  shift
fi
PY_BIN="${PY_BIN:-python}"

mkdir -p "$SCRIPT_ROOT/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$SCRIPT_ROOT/logs/queue2_${PRESET}_gpu2_${TS}.log"

echo "[LAUNCH-Q2] WORK_DIR=$SCRIPT_ROOT"
echo "[LAUNCH-Q2] DATA_DIR=$DATA_DIR"
echo "[LAUNCH-Q2] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[LAUNCH-Q2] PRESET=$PRESET | LOG=$LOG_FILE"

LOCK_FILE="/tmp/fewshot_vstain_gpu2_${USER:-u}.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[ERROR] GPU 2 mutex lock is already held ($LOCK_FILE). Wait for active Round-1/Queue-2 job to finish." | tee -a "$LOG_FILE" >&2
    exit 1
  fi
fi
ACTIVE_PIDS="$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' || true)"
if [[ -n "$ACTIVE_PIDS" && "${ALLOW_GPU2_SHARING:-0}" != "1" ]]; then
  echo "[ERROR] GPU 2 already has active compute PID(s): $ACTIVE_PIDS. Aborting (set ALLOW_GPU2_SHARING=1 only if intentional)." | tee -a "$LOG_FILE" >&2
  exit 1
fi
export VSTAIN_GPU2_LOCK_HELD=1
nvidia-smi -i 2 --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

# Step 1: Quick isolated pre-flight smoke check for all requested methods before full run
if [[ "$PRESET" != "smoke" ]]; then
  "$PY_BIN" -u "$SCRIPT_ROOT/scripts/run_queue2_baselines.py" \
    --gpu-id 2 \
    --work-dir "$SCRIPT_ROOT/_smoke_check_q2" \
    --cache-dir "$SCRIPT_ROOT/eval_cache" \
    --data-dir "$DATA_DIR" \
    --preset smoke \
    --no-skip-done 2>&1 | tee -a "$LOG_FILE"
fi

# Step 2: Serialized Queue-2 Official Baselines (Source-Val Best Ckpt + 20-shot + Unified Eval + Visuals)
"$PY_BIN" -u "$SCRIPT_ROOT/scripts/run_queue2_baselines.py" \
  --gpu-id 2 \
  --work-dir "$SCRIPT_ROOT" \
  --cache-dir "$SCRIPT_ROOT/eval_cache" \
  --data-dir "$DATA_DIR" \
  --preset "$PRESET" ${1+"$@"} 2>&1 | tee -a "$LOG_FILE"

# Step 3: Auto-pack metrics, visuals, scripts & generated outputs
"$PY_BIN" -u "$SCRIPT_ROOT/scripts/pack_round1_outputs.py" \
  --work-dir "$SCRIPT_ROOT" \
  --tag "queue2_${PRESET}" 2>&1 | tee -a "$LOG_FILE"
