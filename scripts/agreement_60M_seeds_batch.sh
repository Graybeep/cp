#!/usr/bin/env bash
# +5 arithmetic-mixture d0 baselines at 60.3M (seeds 4-8), to replicate the
# single-cell attraction divergence found at n=3.
#
# ALL decision rules are pre-registered in NUMBERS before this runs:
#     notes/agreement-60M-mixture-replication-preregistration.md
# Primary test = Welch two-sided on 60.3M attraction, arith n=8 vs syntax n=24.
# A null is pre-registered as "closes the divergence as noise" with the MDE
# (~0.10) attached. Do not read these results by eyeball.
#
# Config is IDENTICAL to seeds 1-3 (checked against
# logs/arith_delay0_exp20_size60M_seed1/config.json) -- only --seed differs, so
# the eight seeds form one homogeneous cell. Same final vocab.bin / train.bin.
#
# ~46 min/run => ~3.8 h total. 4 ckpts x 724 MB x 5 runs = ~14 GB transient.
# Idempotent (skips any run whose step_0005000.pt exists) and kill-tolerant
# (run_supervised.py self-resumes from the last checkpoint).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
LOGDIR=logs/agreement_60M_seeds
mkdir -p "$LOGDIR"

SEEDS="${*:-4 5 6 7 8}"

echo "###### arith d0 60M seeds=[${SEEDS}] start $(date) ######"
for s in $SEEDS; do
  run_id="arith_delay0_exp20_size60M_seed${s}"
  echo "=== [$(date +%H:%M:%S)] ${run_id} ==="
  if [ -f "checkpoints/${run_id}/step_0005000.pt" ]; then
    echo "    already complete - skipping"; continue; fi
  "$PY" scripts/run_supervised.py --skill arith --delay-onset 0.0 \
      --skill-exposure-frac 0.20 --model-size 60M --total-steps 5000 \
      --checkpoint-interval 1250 --seed "$s" --run-name sanity_babylm \
      > "${LOGDIR}/${run_id}.out" 2>&1
  echo "    exit $? at $(date +%H:%M:%S)"
done
echo "###### done $(date) ######"
echo "NEXT (scoring, per the pre-registration -- do not vary these flags):"
echo "  $PY scripts/agreement_attraction_quartet.py \\"
echo "      --runs arith_delay0_exp20_size11M_seed{1,2,3} \\"
echo "             arith_delay0_exp20_size23M_seed{1,2,3} \\"
echo "             arith_delay0_exp20_size60M_seed{1,2,3,4,5,6,7,8} \\"
echo "      --n 400 --step 5000 --skip-sentence --tag quartet_scale_arithmix_n8"
