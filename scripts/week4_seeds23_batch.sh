#!/usr/bin/env bash
# Week 4 — seeds 2-3 replication of the lr_shift reopening result (seed 1:
# 0.506 -> 0.752, +84%). Per the pre-registered rule, an arm crossing threshold
# must replicate at seeds 2-3 before it is believed.
#
# Expensive part: the arith_delay50 baselines for seeds 2-3 do not exist (the
# common checkpoint only existed for seed 1). Each baseline serves as BOTH the
# null arm (read at step 5000) AND the fork source (@ step 2500).
#
# Baselines match seed 1 exactly: --skill arith --delay-onset 0.5
#   --skill-exposure-frac 0.20 --model-size 23M --total-steps 5000
#   --checkpoint-interval 100 (warmup 300 / lr 6e-4 = TrainConfig defaults).
# Baselines run under run_supervised.py (kill-tolerant). Forks self-resume.
# Negative-control (Q2) replication folded in — its syntax_delay50 seed2/3
# sources already exist, so it is just two more cheap forks.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
LOGDIR=logs/week4_arith_pilot
mkdir -p "$LOGDIR"

baseline () {  # seed
  local s="$1"; local run_id="arith_delay50_exp20_size23M_seed${s}"
  echo "=== [$(date +%H:%M:%S)] BASELINE seed${s} -> ${run_id} ==="
  if [ -f "checkpoints/${run_id}/step_0005000.pt" ]; then
    echo "    already complete — skipping"; return 0; fi
  "$PY" scripts/run_supervised.py --skill arith --delay-onset 0.5 \
      --skill-exposure-frac 0.20 --model-size 23M --total-steps 5000 \
      --checkpoint-interval 100 --seed "$s" --run-name sanity_babylm \
      > "${LOGDIR}/seed${s}_baseline.out" 2>&1
  echo "    exit $?"
}

fork () {  # name  from-run  done-run-id  extra-args...
  local name="$1"; local from="$2"; local run_id="$3"; shift 3
  echo "=== [$(date +%H:%M:%S)] FORK ${name} -> ${run_id} ==="
  if [ -f "checkpoints/${run_id}/step_0005000.pt" ]; then
    echo "    already complete — skipping"; return 0; fi
  "$PY" scripts/run_intervention.py --from-run "$from" --fork-step 2500 \
      "$@" > "${LOGDIR}/${name}.out" 2> "${LOGDIR}/${name}.err"
  echo "    exit $?"
}

echo "###### Week 4 seeds 2-3 replication start $(date) ######"

# common checkpoints first (the expensive, gating step)
baseline 2
baseline 3

# lr_shift replication (primary)
fork seed2_lrshift arith_delay50_exp20_size23M_seed2 \
     arith_delay50_exp20_size23M_seed2_ivlrshift0at2501 \
     --intervention lr_shift --shift-source-start 0
fork seed3_lrshift arith_delay50_exp20_size23M_seed3 \
     arith_delay50_exp20_size23M_seed3_ivlrshift0at2501 \
     --intervention lr_shift --shift-source-start 0

# negative-control replication (Q2), sources already exist
fork seed2_negctrl syntax_delay50_exp10_size23M_seed2 \
     syntax_delay50_exp10_size23M_seed2_ivlrshift0at2501 \
     --intervention lr_shift --shift-source-start 0
fork seed3_negctrl syntax_delay50_exp10_size23M_seed3 \
     syntax_delay50_exp10_size23M_seed3_ivlrshift0at2501 \
     --intervention lr_shift --shift-source-start 0

echo "###### Week 4 seeds 2-3 replication done $(date) ######"
