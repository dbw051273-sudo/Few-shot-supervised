"""Shared GPU-2 mutex/occupancy guard and deterministic config fingerprinting helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

_LOCK_HANDLE = None


def preparse_gpu_and_env(task_label: str = 'Workstation GPU tasks') -> argparse.Namespace:
    """Enforce CUDA_VISIBLE_DEVICES=2 lock and offline HF environment before importing torch."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--gpu-id', type=str, default='2')
    ap.add_argument('--allow-other-gpu', action='store_true', default=False)
    ap.add_argument('--allow-gpu-sharing', action='store_true', default=False)
    ap.add_argument('--online-hf', action='store_true', default=False)
    known, _ = ap.parse_known_args()
    if known.gpu_id != '2' and not known.allow_other_gpu:
        raise RuntimeError(
            f"Safety lock: {task_label} must strictly use GPU 2 (got --gpu-id={known.gpu_id}). "
            "Pass --allow-other-gpu explicitly only for local debugging."
        )
    os.environ['CUDA_VISIBLE_DEVICES'] = known.gpu_id
    if not known.online_hf:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    return known


def _active_gpu_pids(gpu_id: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '-i', str(gpu_id), '--query-compute-apps=pid', '--format=csv,noheader'],
            text=True,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception as exc:
        print(
            f"[WARN] nvidia-smi occupancy check for GPU {gpu_id} failed ({exc}); "
            "please verify GPU 2 is idle manually via `nvidia-smi -i 2`.",
            flush=True,
        )
        return []


def acquire_runtime_gpu_guard(gpu_id: str, allow_sharing: bool = False) -> None:
    """Acquire non-blocking POSIX flock for GPU 2 and verify no competing compute PID is running."""
    global _LOCK_HANDLE
    if gpu_id != '2':
        return

    # 1. File mutex lock across all Python/bash launchers targeting GPU 2 (skipped if parent shell already holds lock)
    if os.environ.get('VSTAIN_GPU2_LOCK_HELD') != '1':
        try:
            import fcntl  # POSIX only (Linux 5880 workstation)
            user = os.environ.get('USER') or os.environ.get('USERNAME') or 'u'
            lock_path = Path(f'/tmp/fewshot_vstain_gpu2_{user}.lock')
            fh = open(lock_path, 'w', encoding='utf-8')
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                fh.close()
                raise RuntimeError(
                    f"GPU 2 mutex lock already held ({lock_path}). Another Round-1 / Queue-2 process is active."
                ) from exc
            fh.write(f'{os.getpid()}\n')
            fh.flush()
            _LOCK_HANDLE = fh
        except ImportError:
            pass

    # 2. Active compute-app occupancy check on GPU 2 (unless caller explicitly permits sharing)
    if not allow_sharing and os.environ.get('VSTAIN_GPU2_LOCK_HELD') != '1':
        pids = _active_gpu_pids(gpu_id)
        if pids:
            raise RuntimeError(
                f"GPU {gpu_id} is currently busy with active compute PIDs {pids}. "
                "Wait until idle or pass --allow-gpu-sharing if intentional."
            )


def stable_hash(payload: dict) -> str:
    """Return compact 12-char SHA-256 fingerprint for config/split dicts."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:12]


def metric_matches_hash(json_path: Path, expected_hash: str) -> bool:
    """Return True iff metric JSON exists, parses cleanly, and carries matching config_hash."""
    if not json_path.exists():
        return False
    try:
        obj = json.loads(json_path.read_text(encoding='utf-8'))
    except Exception:
        return False
    return obj.get('config_hash') == expected_hash
