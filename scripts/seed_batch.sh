#!/usr/bin/env bash
# Seeds 2-3 replication batch for syntax-local (8 runs, ~3.5h).
# Run DETACHED so harness task kills can't take the driver down; each
# training run is individually supervised (resume-on-kill).
cd /d/EXPERIMENTS/reopen || exit 1
for seed in 2 3; do
  for delay in 0.0 0.25 0.5 0.75; do
    echo "=== seed $seed delay $delay start $(date '+%H:%M:%S')"
    .venv/Scripts/python scripts/run_supervised.py \
      --skill syntax --skill-exposure-frac 0.10 \
      --delay-onset "$delay" --seed "$seed" \
      --checkpoint-interval 500 \
      || echo "BATCH-FAILURE seed $seed delay $delay"
  done
done
echo "SEED-BATCH-COMPLETE $(date '+%H:%M:%S')"
