#!/usr/bin/env bash
# Week 4 CLOSE-OUT — the three remaining cheap items, pure execution (no design
# decisions left). All idempotent (skip if step_0005000.pt exists) and
# kill-tolerant (run_supervised self-resumes; forks resume their own ckpt).
# Runs strictly sequentially — one 23M run at a time fits the 8GB GPU.
#
#  1. arith_delay50 seed1 RECHECK from scratch (distinct run_id, --seed 1 =
#     same RNG draw) — converts the seed-1 low-null anomaly answer from
#     "no resume detected" to "reproduced under hardened code". Expect ~0.506.
#  2. cram+lr_shift at seeds 2-3 (the ORIGINAL Week-4 "Q1": dose-cap vs
#     LR-starvation). Forks from the existing arith_delay50 seed2/3 @2500.
#     seed1 was 0.746 ~ lr_shift 0.752; n=1 -> n=3.
#  3. arith_delay0 baselines at seeds 2-3 — the per-seed EARLY anchors that let
#     us compute whether the DELAY-INDUCED DEFICIT (early - null) is itself
#     seed-variable, not just the null endpoint.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
LOGDIR=logs/week4_close
mkdir -p "$LOGDIR"

baseline () {  # run_id_probe  seed  delay  extra-suffix-args...
  local run_id="$1"; local s="$2"; local delay="$3"; shift 3
  echo "=== [$(date +%H:%M:%S)] BASELINE seed${s} delay${delay} -> ${run_id} ==="
  if [ -f "checkpoints/${run_id}/step_0005000.pt" ]; then
    echo "    already complete — skipping"; return 0; fi
  "$PY" scripts/run_supervised.py --skill arith --delay-onset "$delay" \
      --skill-exposure-frac 0.20 --model-size 23M --total-steps 5000 \
      --checkpoint-interval 100 --seed "$s" --run-name sanity_babylm "$@" \
      > "${LOGDIR}/${run_id}.out" 2>&1
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

echo "###### Week 4 close-out start $(date) ######"

# 1. seed-1 recheck (delay 0.5, distinct run_id via --run-suffix)
baseline arith_delay50_exp20_size23M_seed1_recheck 1 0.5 --run-suffix recheck

# 3. per-seed EARLY anchors (delay 0.0)
baseline arith_delay0_exp20_size23M_seed2 2 0.0
baseline arith_delay0_exp20_size23M_seed3 3 0.0

# 2. cram+lr_shift replication (original Q1) at seeds 2-3
fork seed2_cramlrshift arith_delay50_exp20_size23M_seed2 \
     arith_delay50_exp20_size23M_seed2_ivcramlrshift0at2501 \
     --intervention cram_lrshift --shift-source-start 0
fork seed3_cramlrshift arith_delay50_exp20_size23M_seed3 \
     arith_delay50_exp20_size23M_seed3_ivcramlrshift0at2501 \
     --intervention cram_lrshift --shift-source-start 0

echo "###### Week 4 close-out done $(date) ######"
