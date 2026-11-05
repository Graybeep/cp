#!/usr/bin/env bash
# Tier-1 GPU chain: run the remaining Tier-1 batches back-to-back once the
# 60M rate-collapse batch releases the GPU.
#
# The 4060 is serial — two training processes at once contend for VRAM and both
# slow down (or OOM at 60M). So this waits for the rate-collapse driver to exit,
# then runs the rest in the user's stated priority order:
#
#   T1.3  WSD schedule, phase 1           (6 runs,   ~2.5 h)
#   T1.2  intervention arms -> n=3        (10 forks, ~2.2 h)
#   T1.4  lr_shift level/shape 2x2        (9 forks,  ~2.0 h)
#
# REORDERED 2026-08-13: the session has a hard stop ~07:40, which leaves only
# ~2.4 h of chain time after T1.1 releases the GPU — room for ONE sub-batch.
# T1.3 goes first because it is the only one that can FALSIFY the paper's
# headline mechanism (§4.4): under WSD the LR at the d50 onset is still at
# peak, so the schedule account predicts the deficit largely vanishes. T1.2
# and T1.4 refine findings that already stand at n=3; T1.3 could overturn one.
# Phase 1 trains its early anchors first, so a batch cut short by the stop is
# still interpretable in seed order rather than being wasted.
#
# Each sub-batch is independently resumable and skips already-complete runs, so
# killing this chain and re-running it is safe and costs only the run in flight.
# A sub-batch FAILING does not stop the chain: the items are independent, and a
# failure in one should not idle the GPU for the others. Failures are announced
# in the log with a CHAIN-STEP-FAILED line and surface in the final summary.
set -u
cd "$(dirname "$0")/.." || exit 1

LOCK="logs/.tier1chain.lock"
LOG="logs/tier1_chain.log"
WAIT_ON="logs/.ratecollapse.lock"

if ( set -o noclobber; printf '%s\n' "$$" > "$LOCK" ) 2>/dev/null; then :; else
  oldpid=$(head -n1 "$LOCK" 2>/dev/null)
  if [ -n "${oldpid:-}" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "$(date '+%F %T') REFUSING: chain already running (pid $oldpid)" >> "$LOG"
    exit 1
  fi
  rm -f "$LOCK"
  ( set -o noclobber; printf '%s\n' "$$" > "$LOCK" ) 2>/dev/null || exit 1
fi
# NOTE the two separate traps. The old single `trap ... EXIT INT TERM` was a
# bug, observed live on 2026-08-13: a handler that only removes the lock does
# not terminate the script, so on TERM this chain released its lock and then
# RESUMED its poll loop — still alive, still about to grab the GPU, but no
# longer holding the lock that stops a second chain starting alongside it.
# INT/TERM must exit explicitly; EXIT then fires and does the cleanup.
trap 'rm -f "$LOCK"' EXIT
trap 'echo "$(date "+%F %T") chain interrupted — exiting"; exit 130' INT TERM

exec >> "$LOG" 2>&1
echo "########## tier1 chain start $(date '+%F %T') pid=$$ ##########"

# ---- wait for the GPU ------------------------------------------------------
# Poll the rate-collapse driver's lock. Treat a stale lock (holder dead) as
# released, mirroring the acquire_lock logic in the batch scripts.
while [ -f "$WAIT_ON" ]; do
  holder=$(head -n1 "$WAIT_ON" 2>/dev/null)
  if [ -z "${holder:-}" ] || ! kill -0 "$holder" 2>/dev/null; then
    echo "$(date '+%F %T') rate-collapse lock is stale (pid ${holder:-?} dead) — proceeding"
    break
  fi
  sleep 120
done
echo "$(date '+%F %T') GPU free — starting Tier-1 chain"

FAILED=""
step () {  # label  script
  local label="$1"; local script="$2"
  echo "===== [$(date '+%F %T')] CHAIN STEP: $label ($script) ====="
  if bash "$script"; then
    echo "===== [$(date '+%F %T')] CHAIN STEP OK: $label ====="
  else
    echo "CHAIN-STEP-FAILED $label (exit $?) $(date '+%F %T')"
    FAILED="$FAILED $label"
  fi
}

step "T1.3 WSD schedule phase 1"  scripts/wsd_schedule_batch.sh
step "T1.2 intervention arms n=3" scripts/intervention_arms_seeds23_batch.sh
step "T1.4 lr level/shape 2x2"    scripts/lr_decomposition_batch.sh

if [ -n "$FAILED" ]; then
  echo "TIER1-CHAIN-COMPLETE-WITH-FAILURES:$FAILED $(date '+%F %T')"
else
  echo "TIER1-CHAIN-COMPLETE $(date '+%F %T')"
fi
