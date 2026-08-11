#!/usr/bin/env bash
set -e
cd /d/EXPERIMENTS/reopen
for s in 1 2; do
  echo "=== circuit-identity seed $s (n=500) $(date) ==="
  .venv/Scripts/python.exe -u scripts/week8_circuit_identity.py --seed $s --n-eval 500 \
    > logs/week8_circuit_identity/seed${s}_full.out 2>&1
done
echo "DIAG CHAIN DONE"
