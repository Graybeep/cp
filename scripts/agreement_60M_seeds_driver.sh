#!/usr/bin/env bash
# Detached driver for the remaining 60.3M arithmetic seeds.
#
# WHY THIS EXISTS (2026-07-27): the first launch of agreement_60M_seeds_batch.sh
# was killed by the task harness ~28 min in, mid-seed-4. The kill took the outer
# loop but NOT the grandchild -- run_supervised.py + train_baseline.py survived
# as orphans and drove seed 4 to completion on their own. Same lifetime cap that
# bit the Week-3 pilot (see run_supervised.py docstring).
#
# Fix: launch this DETACHED (PowerShell Start-Process, not a tool child) so it
# is not in the harness's process group and cannot be killed with it.
#
# Serialises strictly: a 60M run holds ~6.0 GB of the 8.2 GB card, so two
# concurrent runs OOM. Each seed waits for the GPU to be free before starting.
#
# Decision rules are pre-registered:
#     notes/agreement-60M-mixture-replication-preregistration.md
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
LOGDIR=logs/agreement_60M_seeds
mkdir -p "$LOGDIR"

SEEDS="${*:-5 6 7 8}"
MAX_WAIT=7200   # seconds to wait for an in-flight run to clear (2 h)

gpu_busy () {  # 0 if a train_baseline process is alive
  # `ps -W` carries no command line under Git Bash, so ask Windows directly.
  local n
  n=$(powershell -NoProfile -Command \
      "@(Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | \
        Where-Object { \$_.CommandLine -match 'train_baseline' }).Count" \
      2>/dev/null | tr -d '[:space:]')
  [ "${n:-0}" -gt 0 ] 2>/dev/null && return 0
  return 1
}

wait_for_free_gpu () {
  local waited=0
  while gpu_busy; do
    if [ "$waited" -ge "$MAX_WAIT" ]; then
      echo "    !! GPU still busy after ${MAX_WAIT}s - aborting rather than OOM"
      return 1
    fi
    sleep 60; waited=$((waited + 60))
    if [ $((waited % 600)) -eq 0 ]; then
      echo "    ... waiting for in-flight run to finish (${waited}s)"
    fi
  done
  sleep 20   # let VRAM actually release
  return 0
}

echo "###### 60M arith seeds=[${SEEDS}] driver start $(date) ######"
for s in $SEEDS; do
  run_id="arith_delay0_exp20_size60M_seed${s}"
  echo "=== [$(date +%H:%M:%S)] ${run_id} ==="
  if [ -f "checkpoints/${run_id}/step_0005000.pt" ]; then
    echo "    already complete - skipping"; continue; fi
  wait_for_free_gpu || exit 1
  "$PY" scripts/run_supervised.py --skill arith --delay-onset 0.0 \
      --skill-exposure-frac 0.20 --model-size 60M --total-steps 5000 \
      --checkpoint-interval 1250 --seed "$s" --run-name sanity_babylm \
      >> "${LOGDIR}/${run_id}.out" 2>&1
  echo "    exit $? at $(date +%H:%M:%S)"
done

echo "###### all seeds done $(date) ######"
for s in 1 2 3 4 5 6 7 8; do
  f="checkpoints/arith_delay0_exp20_size60M_seed${s}/step_0005000.pt"
  [ -f "$f" ] && echo "  seed${s}: COMPLETE" || echo "  seed${s}: MISSING"
done
