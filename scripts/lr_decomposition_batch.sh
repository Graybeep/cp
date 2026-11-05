#!/usr/bin/env bash
# Decompose lr_shift into LEVEL and SHAPE (T1.4).
#
# lr_shift moves the learning rate's magnitude AND its trajectory (warmup ramp
# + decay curvature) in one lever, so "restoring the schedule reopens the
# deficit" is a black-box claim. This runs the 2x2 that opens it:
#
#                     level=late              level=early
#   shape=flat        lr_flat_control (C)     lr_level_only (A)
#   shape=source      lr_shape_only   (B)     lr_shift  [already n=3]
#
# The fourth cell IS the existing lr_shift arm, so only three are launched.
# Level is matched on the window MEAN (see LRComponentWindow's docstring).
#
# C is not optional: without it, A confounds "higher LR" with "no decay".
#
# 3 arms x 3 seeds = 9 forks from the same common checkpoints the Week 4 arms
# used (arith_delay50_exp20_size23M_seed{S} @ step 2500), ~13 min each.
#
# Reading rule: notes/lr-decomposition-preregistration.md
set -u
cd "$(dirname "$0")/.." || exit 1

PY=.venv/Scripts/python.exe
FORK_STEP=2500
LOGDIR=logs/lr_decomposition
mkdir -p "$LOGDIR"

run_arm () {
  local seed="$1"; local name="$2"; local suffix="$3"
  local src="arith_delay50_exp20_size23M_seed${seed}"
  local run_id="${src}_${suffix}"
  local done_ckpt="checkpoints/${run_id}/step_0005000.pt"
  echo "=== [$(date +%F' '%H:%M:%S)] seed${seed} ARM ${name} -> ${run_id} ==="
  if [ -f "$done_ckpt" ]; then
    echo "    already complete (step_0005000.pt present) — skipping"
    return 0
  fi
  if [ ! -f "checkpoints/${src}/step_000${FORK_STEP}.pt" ]; then
    echo "    MISSING FORK SOURCE checkpoints/${src}/step_000${FORK_STEP}.pt"
    return 1
  fi
  "$PY" scripts/run_intervention.py \
      --from-run "$src" --fork-step "$FORK_STEP" \
      --intervention "$name" --shift-source-start 0 \
      > "${LOGDIR}/seed${seed}_${name}.out" \
      2> "${LOGDIR}/seed${seed}_${name}.err"
  local code=$?
  echo "    exit ${code} (log: ${LOGDIR}/seed${seed}_${name}.out)"
  return $code
}

echo "###### lr_shift level/shape decomposition start $(date) ######"

for seed in 1 2 3; do
  run_arm "$seed" lr_flat_control ivlrflatcontrol2501 || echo "ARM-FAILURE seed$seed lr_flat_control"
  run_arm "$seed" lr_level_only   ivlrlevelonly2501   || echo "ARM-FAILURE seed$seed lr_level_only"
  run_arm "$seed" lr_shape_only   ivlrshapeonly2501   || echo "ARM-FAILURE seed$seed lr_shape_only"
done

echo "###### lr_shift decomposition done $(date) ######"
echo "LRDECOMP-COMPLETE $(date '+%F %T')"
