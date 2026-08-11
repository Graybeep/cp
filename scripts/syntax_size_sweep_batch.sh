#!/usr/bin/env bash
# Syntax LD model-size sweep (reposition hypothesis, pre-registered:
# notes/week5-syntax-size-sweep-preregistration.md).
#   3 sizes {11M, 23M, 60M} x 24 seeds, delay-0 only, exp10, pool v1, 5000 steps.
# Same supervisor pattern as seed4-8: each run individually resume-on-kill
# (run_supervised.py); this outer loop just sequences the 72 runs.
#
# SINGLE-DRIVER LOCK (structural, not a procedure): concurrent drivers on the
# same run_id write to the same step_*.pt.tmp path and defeat the atomic-write
# guarantee (corrupt zip -> fails torch.load loudly, wastes compute). After a
# real triple-driver incident on this batch (2026-07-19), the script now REFUSES
# to start a second driver. Stale locks (holder killed via SIGKILL/Stop-Process,
# which skip the EXIT trap) are detected via kill -0 and reclaimed.
#
# STATUS HEARTBEAT: logs/szsweep_status.txt is rewritten at every run boundary
# (completed count, current run, timestamp, driver pid) so "is it stuck?" is
# instantly queryable (scripts/szsweep_status.sh) without reading the log.
set -u
cd /d/EXPERIMENTS/reopen || exit 1

LOCK="logs/.szsweep.lock"
STATUS="logs/szsweep_status.txt"
LOG="logs/syntax_size_sweep_batch.log"
TOTAL=72

acquire_lock() {
  # atomic create via O_EXCL (noclobber). Success => we hold the lock.
  if ( set -o noclobber; printf '%s\n' "$$" > "$LOCK" ) 2>/dev/null; then
    return 0
  fi
  local oldpid; oldpid=$(head -n1 "$LOCK" 2>/dev/null)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "$(date '+%F %T') REFUSING: another szsweep driver is active (pid $oldpid)" >> "$LOG"
    return 1
  fi
  echo "$(date '+%F %T') stale lock (pid ${oldpid:-?} not alive) - reclaiming" >> "$LOG"
  rm -f "$LOCK"
  if ( set -o noclobber; printf '%s\n' "$$" > "$LOCK" ) 2>/dev/null; then
    return 0
  fi
  echo "$(date '+%F %T') REFUSING: lost the lock reclaim race" >> "$LOG"
  return 1
}

# Acquire BEFORE the exec redirect so a refusal is still logged, and BEFORE
# setting the cleanup trap so a refused start never deletes the holder's lock.
acquire_lock || exit 1
trap 'rm -f "$LOCK"' EXIT INT TERM   # SIGKILL can't be trapped -> stale detection above

exec >> "$LOG" 2>&1
echo "########## batch (re)launched $(date '+%F %T') pid=$$ ##########"

write_status() {  # $1=phase  $2=current run label
  local done; done=$(ls checkpoints/*szsweep/step_0005000.pt 2>/dev/null | wc -l)
  { echo "updated=$(date '+%F %T')"
    echo "driver_pid=$$"
    echo "phase=$1"
    echo "completed=$done/$TOTAL"
    echo "current_run=$2"
  } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"
}

for size in 11M 23M 60M; do
  for seed in $(seq 1 24); do
    rid="syntax_delay0_exp10_size${size}_seed${seed}_szsweep"
    write_status "training" "$rid"
    echo "=== $rid start $(date '+%F %T')"
    .venv/Scripts/python scripts/run_supervised.py \
      --skill syntax --skill-exposure-frac 0.10 \
      --delay-onset 0.0 --model-size "$size" --seed "$seed" \
      --run-suffix szsweep \
      --checkpoint-interval 2500 --launch-timeout 5400 \
      || echo "BATCH-FAILURE $rid"
    ckdir="checkpoints/$rid"
    if [ -f "$ckdir/step_0005000.pt" ]; then
      rm -f "$ckdir/step_0002500.pt"   # keep final only, once it exists
      echo "    $rid complete + pruned $(date '+%F %T')"
    else
      echo "    WARN $rid has no step_0005000.pt after supervisor returned"
    fi
    write_status "between" "$rid done"
  done
done
write_status "done" "-"
echo "SZSWEEP-COMPLETE $(date '+%F %T')"
