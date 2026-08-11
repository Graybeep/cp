#!/usr/bin/env bash
# LR-budget replication: hilr50 seeds 2-3 (settle whether the +0.017 LD nudge is stable noise).
set -e
cd /d/EXPERIMENTS/reopen
for s in 2 3; do
  echo "=== hilr50 seed $s starting $(date) ==="
  .venv/Scripts/python.exe -u scripts/train_baseline.py \
    --skill syntax --delay-onset 0.0 --skill-exposure-frac 0.1 \
    --model-size 23M --seed $s \
    --total-steps 5000 --checkpoint-interval 100 --warmup-steps 300 \
    --lr 6e-4 --min-lr-ratio 0.5 \
    --micro-batch-size 8 --grad-accum-steps 4 --num-workers 8 \
    --run-suffix hilr50 > logs/syntax_hilr50_seed${s}.out 2>&1
  echo "=== hilr50 seed $s done $(date) ==="
done
echo "ALL DONE"
