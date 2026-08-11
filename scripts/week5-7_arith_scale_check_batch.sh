#!/usr/bin/env bash
# Week 5-7 arithmetic SCALE CHECK — does the d50 critical period AND the lr_shift
# reopening survive at 11M and 60M?  (23M is established at n=3.)  Resolves the
# scale-asymmetry flagged 2026-07-22: syntax got a full 3-size sweep and failed
# it; arith's Detect/Reopen findings have only ever been tested at 23M.
#
# ALL decision rules (probe go/no-go, per-size gates, capacity-limit guard) are
# pre-registered in NUMBERS before any run:
#     notes/week5-7-arith-scale-check-preregistration.md
# Do not read these results by eyeball — apply the rules in that note.
#
# STAGED.  A human go/no-go gate (Rule 1 in the note) sits BETWEEN the stages:
#     probe : seed 1 only, 11M + 60M, all three arms   (~2.4 h)   [default]
#     fill  : seeds 2 + 3, 11M + 60M, all three arms    (~4.8 h)   [ONLY after
#             the probe clears Rule 1 at that size]
#   Run `... probe`, apply Rule 1 to the probe read-out, THEN decide `... fill`.
#
# Per size per seed = 3 arms sharing compute (~2.5 run-equivalents):
#     d0       : arith delay-0    -> early anchor            (full 5000-step run)
#     d50      : arith delay-0.5  -> delayed null + fork src (full run; retains step_2500)
#     lr_shift : fork d50 @2500, replay early LR over [2500,5000)  (half run)
# Arith config is IDENTICAL to the 23M Week-4 runs (exp 0.20, total 5000, same
# warmup/lr defaults) — only --model-size differs, so cross-size numbers are
# comparable.  Same FINAL vocab.bin (model-agnostic; not regenerated here).
#
# checkpoint-interval 1250 == the onset-retention keep-set {1250,2500,3750,5000}
# (reopen/train.py completion-prune): guarantees the step_2500 fork source
# survives the prune, with minimal transient disk (<=4 ckpts/run: 60M ~2.8 GB).
#
# Idempotent (skips if the run's step_0005000.pt exists) and kill-tolerant
# (run_supervised.py self-resumes; run_intervention.py forks resume own ckpt).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
LOGDIR=logs/week5-7_arith_scale
mkdir -p "$LOGDIR"

STAGE="${1:-probe}"
SIZES="11M 60M"

baseline () {  # run_id  seed  delay  size
  local run_id="$1"; local s="$2"; local delay="$3"; local size="$4"
  echo "=== [$(date +%H:%M:%S)] BASELINE ${run_id} (seed${s} delay${delay} ${size}) ==="
  if [ -f "checkpoints/${run_id}/step_0005000.pt" ]; then
    echo "    already complete — skipping"; return 0; fi
  "$PY" scripts/run_supervised.py --skill arith --delay-onset "$delay" \
      --skill-exposure-frac 0.20 --model-size "$size" --total-steps 5000 \
      --checkpoint-interval 1250 --seed "$s" --run-name sanity_babylm \
      > "${LOGDIR}/${run_id}.out" 2>&1
  echo "    exit $?"
}

fork_lrshift () {  # size  seed
  local size="$1"; local s="$2"
  local from="arith_delay50_exp20_size${size}_seed${s}"
  local run_id="${from}_ivlrshift0at2501"
  echo "=== [$(date +%H:%M:%S)] FORK lr_shift ${run_id} ==="
  if [ -f "checkpoints/${run_id}/step_0005000.pt" ]; then
    echo "    already complete — skipping"; return 0; fi
  # size is inferred from the source checkpoint's stored model_cfg — no --model-size.
  "$PY" scripts/run_intervention.py --from-run "$from" --fork-step 2500 \
      --intervention lr_shift --shift-source-start 0 \
      > "${LOGDIR}/${run_id}.out" 2> "${LOGDIR}/${run_id}.err"
  echo "    exit $?"
}

run_seed () {  # size  seed  — arms strictly sequential so the fork source exists
  local size="$1"; local s="$2"
  baseline "arith_delay0_exp20_size${size}_seed${s}"  "$s" 0.0 "$size"   # early anchor
  baseline "arith_delay50_exp20_size${size}_seed${s}" "$s" 0.5 "$size"   # null + fork source
  fork_lrshift "$size" "$s"                                              # reopening
}

case "$STAGE" in
  probe) SEEDS="1"   ;;
  fill)  SEEDS="2 3" ;;
  *) echo "usage: $0 [probe|fill]"; exit 2 ;;
esac

echo "###### arith scale check — stage=${STAGE} seeds=[${SEEDS}] sizes=[${SIZES}] start $(date) ######"
for size in $SIZES; do
  for s in $SEEDS; do
    run_seed "$size" "$s"
  done
done
echo "###### arith scale check — stage=${STAGE} done $(date) ######"
echo "READ-OUT: pull step-5000 arith_accuracy (+ val_ppl for forgetting) from"
echo "  logs/<run_id>/metrics.csv for every run, then APPLY the pre-registered"
echo "  rules in notes/week5-7-arith-scale-check-preregistration.md (do NOT eyeball)."
