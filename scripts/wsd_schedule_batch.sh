#!/usr/bin/env bash
# Non-cosine schedule generalization: does the arithmetic critical period
# exist under WSD (warmup-stable-decay)?  [T1.3, phase 1]
#
# WHY THIS IS SHARPER THAN A ROBUSTNESS CHECK. The paper's mechanism claim is
# that closure at d50 is a property of the DECAYED learning-rate schedule. Under
# cosine, LR at the d50 onset (step 2500) has already fallen to 3.57e-4, 60% of
# peak. Under WSD it is still at PEAK (6.00e-4) and stays there until step 4000.
# So the schedule account makes a directional, falsifiable prediction:
#
#   schedule account correct -> the d50 deficit should largely VANISH under WSD
#   deficit persists at ~0.21 -> closure is NOT mainly the LR schedule, and
#                                §4.4's central claim needs rewriting
#
# Either outcome is publishable and both are informative. This is the cheapest
# test in the queue that can falsify the paper's headline mechanism.
#
# Phase 1 (this script): WSD x {d0, d50} x 3 seeds = 6 runs, ~25 min each.
# Phase 2 (conditional, launched separately only if a deficit appears):
#   lr_shift forks under WSD. If there is no deficit there is nothing to reopen.
#
# Everything matches the existing cosine arith runs except --lr-schedule:
#   --skill arith --skill-exposure-frac 0.20 --model-size 23M
#   --total-steps 5000 --checkpoint-interval 100
# WSD shares cosine's peak, floor and warmup, so only the shape differs.
# Runs land in their own run_id namespace (..._wsd_size23M_...), so they cannot
# collide with the cosine runs.
#
# Reading rule: notes/wsd-schedule-preregistration.md
set -u
cd "$(dirname "$0")/.." || exit 1

PY=.venv/Scripts/python.exe
LOGDIR=logs/wsd_schedule
mkdir -p "$LOGDIR"

run_one () {  # delay  tag  seed   (tag = delay*100, passed explicitly: no bc
              #                    dependency, and the run_id is grep-able here)
  local delay="$1"; local tag="$2"; local seed="$3"
  local run_id="arith_delay${tag}_exp20_wsd_size23M_seed${seed}"
  echo "=== [$(date +%F' '%H:%M:%S)] WSD delay${tag} seed${seed} -> ${run_id} ==="
  if [ -f "checkpoints/${run_id}/step_0005000.pt" ]; then
    echo "    already complete — skipping"; return 0
  fi
  "$PY" scripts/run_supervised.py \
      --skill arith --skill-exposure-frac 0.20 \
      --delay-onset "$delay" --model-size 23M --seed "$seed" \
      --total-steps 5000 --checkpoint-interval 100 \
      --lr-schedule wsd --launch-timeout 5400 \
      > "${LOGDIR}/${run_id}.out" 2> "${LOGDIR}/${run_id}.err"
  local code=$?
  echo "    exit ${code}"
  return $code
}

echo "###### WSD schedule batch (phase 1) start $(date) ######"

# early anchors first: without them the delayed runs have nothing to be read
# against, so a partial batch is still interpretable in seed order
for seed in 1 2 3; do
  run_one 0.0 0  "$seed" || echo "RUN-FAILURE wsd delay0 seed$seed"
  run_one 0.5 50 "$seed" || echo "RUN-FAILURE wsd delay50 seed$seed"
done

echo "###### WSD schedule batch (phase 1) done $(date) ######"
echo "WSD-PHASE1-COMPLETE $(date '+%F %T')"
