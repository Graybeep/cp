#!/usr/bin/env bash
# 60M syntax RATE-COLLAPSE test (the experiment §8 names as the cheapest
# outstanding one in the project; pre-registration:
# notes/rate-collapse-60M-preregistration.md).
#
#   24 delayed seeds (delay 0.50, exp10, pool v1, 5000 steps, 60M) against the
#   EXISTING 24-seed delay-0 baseline from the size sweep.
#
# WHY 24 AND NOT THE 14 THE PAPER PRICED: at the derived threshold (0.5593,
# d0 rate 6/24) a 14-seed arm clears alpha only on an exact 0/14 — one high
# seed anywhere and it lands at p=0.18, i.e. 12 GPU-hours for an inconclusive
# result. 24 delayed seeds keep p<0.05 even if one seed comes out high
# (0/24 -> p=0.011, 1/24 -> p=0.049), and match the baseline's n exactly.
#
# Reuses the size-sweep driver pattern verbatim: per-run supervisor
# (resume-on-kill), single-driver lock, status heartbeat, prune-on-complete.
set -u
cd /d/EXPERIMENTS/reopen || exit 1

LOCK="logs/.ratecollapse.lock"
STATUS="logs/ratecollapse_status.txt"
LOG="logs/rate_collapse_60M_batch.log"
TOTAL=24

acquire_lock() {
  if ( set -o noclobber; printf '%s\n' "$$" > "$LOCK" ) 2>/dev/null; then
    return 0
  fi
  local oldpid; oldpid=$(head -n1 "$LOCK" 2>/dev/null)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "$(date '+%F %T') REFUSING: another ratecollapse driver is active (pid $oldpid)" >> "$LOG"
    return 1
  fi
  echo "$(date '+%F %T') stale lock (pid ${oldpid:-?} not alive) - reclaiming" >> "$LOG"
  rm -f "$LOCK"
  ( set -o noclobber; printf '%s\n' "$$" > "$LOCK" ) 2>/dev/null || {
    echo "$(date '+%F %T') REFUSING: lost the lock reclaim race" >> "$LOG"; return 1; }
  return 0
}

acquire_lock || exit 1
trap 'rm -f "$LOCK"' EXIT INT TERM

exec >> "$LOG" 2>&1
echo "########## ratecollapse batch (re)launched $(date '+%F %T') pid=$$ ##########"

write_status() {  # $1=phase  $2=current run label
  local done; done=$(ls checkpoints/syntax_delay50_exp10_size60M_seed*_ratecollapse/step_0005000.pt 2>/dev/null | wc -l)
  { echo "updated=$(date '+%F %T')"
    echo "driver_pid=$$"
    echo "phase=$1"
    echo "completed=$done/$TOTAL"
    echo "current_run=$2"
  } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"
}

for seed in $(seq 1 24); do
  rid="syntax_delay50_exp10_size60M_seed${seed}_ratecollapse"
  write_status "training" "$rid"
  echo "=== $rid start $(date '+%F %T')"
  .venv/Scripts/python scripts/run_supervised.py \
    --skill syntax --skill-exposure-frac 0.10 \
    --delay-onset 0.50 --model-size 60M --seed "$seed" \
    --run-suffix ratecollapse \
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
write_status "done" "-"
echo "RATECOLLAPSE-COMPLETE $(date '+%F %T')"
