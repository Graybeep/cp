#!/usr/bin/env bash
# Week 5-7 arith scale-check — FOLLOW-UPS after the 2026-07-23 seed-1 probe.
# Probe results: logs/week5-7_arith_scale/probe_readout.md
#   11M  d0 0.556 / d50 0.304 / lr_shift 0.726  -> Rule-1 FLOOR -> STOP (main cell)
#   60M  d0 0.798 / d50 0.574 / lr_shift 0.796  -> Rule-1 GO fill, BUT c-b = +1.79
#                                                  breaches the +1.0 forgetting gate
#
# Two independent workloads.  They SHARE ONE GPU (RTX 4060 8GB), so they are run
# STRICTLY SEQUENTIALLY, not concurrently — running both at once only contends
# for SMs and slows each.  11M goes first: it is short (~1.5 h) and answers a
# self-contained question, so a killed driver leaves a complete result rather
# than two partial ones.
#
#   stage 11m  : H1/H2/H3 discriminator, EXPLORATORY (~1.5 h)
#   stage 60m  : Rule-1 fill to n=3 + forgetting drift control (~4.8 h)
#   (default)  : 11m then 60m
#
# ---------------------------------------------------------------------------
# STAGE 11m — EXPLORATORY, NOT the pre-registered 11M cell.
# The Rule-1 FLOOR STOP at 11M stands and is NOT reopened by these runs.  This
# stage answers a DIFFERENT, reframed question (user challenge 2026-07-23):
#   does lr_shift at 11M represent genuine capacity headroom that d0 simply did
#   not reach on its own, or does it boost accuracy past the model's own
#   early-training ceiling regardless of capacity?
# Three hypotheses:
#   H1 genuine headroom d0 didn't reach  -> d0 varies across seeds, sometimes high
#   H2 lr_shift is a generic accuracy booster at 11M, capacity-independent
#   H3 LR-EXPOSURE ALIGNMENT: lr_shift replays high LR over [2501,5000) = exactly
#      the window where the d50 run's arithmetic exposure sits, while d0 spreads
#      arith across all 5000 steps with its second half under decayed LR.
# Arms: repeated d0 (REQUIRED — without it H1/H2 are indistinguishable),
#       repeated d50+lr_shift, and the discriminator:
#       *** d0 + lr_shift *** (fork the EARLY-exposure run, same LR replay)
#         jumps to ~0.73 -> H2 (generic booster; "overshoot refutes capacity" fails)
#         stays  ~0.56  -> H3 (gain requires the delayed-exposure profile)
#
# ---------------------------------------------------------------------------
# STAGE 60m — Rule-1 fill (pre-registered) + a NEW drift control.
# The drift control is a NO-OP FORK: fork d50@2500 and replay the schedule
# segment starting at 2501 — ShiftedLRWindow maps LR to
# schedule(step - at_step + source_start), so source_start == at_step == 2501
# yields schedule(step) exactly.  Same fork machinery, same checkpoint, same
# data stream, UNMODIFIED LR.  It isolates how much of the +1.79 is the
# intervention versus fork mechanics / ordinary ppl movement at this size.
# Needed to separate:
#   (a) lr_shift genuinely costs more general capability at larger scale
#   (b) the fixed +1.0 absolute gate is miscalibrated where the baseline delay
#       cost is ~0 (60M b-a = +0.016 vs 23M's 2.64)
#   (c) HORIZON ARTIFACT — the cost is transient re-convergence from the
#       replayed warmup, measured before it finishes.  Evidence from probe data
#       already on disk: over [3750,5000] the 60M lr_shift arm falls 47.806 ->
#       43.027 (-4.779) while the null falls 42.057 -> 41.237 (-0.820); the arm
#       is still descending 5.8x faster than the null AT the horizon.  A
#       decisive test of (c) needs a HORIZON EXTENSION past 5000, which is NOT
#       wired on the fork path (no --total-steps override) — flagged, not built.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
LOGDIR=logs/week5-7_arith_scale
mkdir -p "$LOGDIR"

baseline () {  # run_id  seed  delay  size
  local run_id="$1"; local s="$2"; local delay="$3"; local size="$4"
  echo "=== [$(date +%H:%M:%S)] BASELINE ${run_id} ==="
  if [ -f "checkpoints/${run_id}/step_0005000.pt" ]; then
    echo "    already complete — skipping"; return 0; fi
  "$PY" scripts/run_supervised.py --skill arith --delay-onset "$delay" \
      --skill-exposure-frac 0.20 --model-size "$size" --total-steps 5000 \
      --checkpoint-interval 1250 --seed "$s" --run-name sanity_babylm \
      > "${LOGDIR}/${run_id}.out" 2>&1
  echo "    exit $?"
}

fork () {  # source_run_id  shift_source_start
  local from="$1"; local src_start="$2"
  local run_id="${from}_ivlrshift${src_start}at2501"
  echo "=== [$(date +%H:%M:%S)] FORK ${run_id} ==="
  if [ -f "checkpoints/${run_id}/step_0005000.pt" ]; then
    echo "    already complete — skipping"; return 0; fi
  if [ ! -f "checkpoints/${from}/step_0002500.pt" ]; then
    echo "    !! fork source checkpoint missing — SKIPPING"; return 1; fi
  "$PY" scripts/run_intervention.py --from-run "$from" --fork-step 2500 \
      --intervention lr_shift --shift-source-start "$src_start" \
      > "${LOGDIR}/${run_id}.out" 2> "${LOGDIR}/${run_id}.err"
  echo "    exit $?"
}

stage_11m () {
  echo "###### 11M EXPLORATORY follow-up (H1/H2/H3) — start $(date) ######"
  for s in 2 3; do
    baseline "arith_delay0_exp20_size11M_seed${s}"  "$s" 0.0 11M
    baseline "arith_delay50_exp20_size11M_seed${s}" "$s" 0.5 11M
    fork     "arith_delay50_exp20_size11M_seed${s}" 0
  done
  # THE DISCRIMINATOR — same LR replay applied to the EARLY-exposure run.
  for s in 1 2 3; do
    fork "arith_delay0_exp20_size11M_seed${s}" 0
  done
  echo "###### 11M follow-up done $(date) ######"
}

stage_60m () {
  echo "###### 60M fill to n=3 + drift control — start $(date) ######"
  for s in 2 3; do
    baseline "arith_delay0_exp20_size60M_seed${s}"  "$s" 0.0 60M
    baseline "arith_delay50_exp20_size60M_seed${s}" "$s" 0.5 60M
    fork     "arith_delay50_exp20_size60M_seed${s}" 0
  done
  # DRIFT CONTROL — exact no-op fork (source_start == at_step == 2501).
  for s in 1 2 3; do
    fork "arith_delay50_exp20_size60M_seed${s}" 2501
  done
  echo "###### 60M fill done $(date) ######"
}

case "${1:-all}" in
  11m) stage_11m ;;
  60m) stage_60m ;;
  all) stage_11m; stage_60m ;;
  *) echo "usage: $0 [11m|60m|all]"; exit 2 ;;
esac
echo "###### ALL FOLLOW-UPS COMPLETE $(date) ######"
