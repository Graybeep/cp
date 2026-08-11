#!/usr/bin/env bash
# Seeds 4-8 delay-0-only baseline batch for syntax LD acquisition-rate
# question (how often does 23M acquire long-distance licensing at all?).
# Same config as seeds 1-3 (exp10, pool v1, 5000 steps, ckpt 500); only
# the seed differs. Run DETACHED so harness task kills can't take the
# driver down; each run is individually supervised (resume-on-kill).
cd /d/EXPERIMENTS/reopen || exit 1
for seed in 4 5 6 7 8; do
  echo "=== seed $seed delay 0.0 start $(date '+%H:%M:%S')"
  .venv/Scripts/python scripts/run_supervised.py \
    --skill syntax --skill-exposure-frac 0.10 \
    --delay-onset 0.0 --seed "$seed" \
    --checkpoint-interval 500 \
    || echo "BATCH-FAILURE seed $seed delay 0.0"
done
echo "SEED48-BATCH-COMPLETE $(date '+%H:%M:%S')"
