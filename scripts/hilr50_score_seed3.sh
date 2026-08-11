#!/usr/bin/env bash
cd /d/EXPERIMENTS/reopen
ck=checkpoints/syntax_delay0_exp10_size23M_seed3_hilr50/step_0005000.pt
while [ ! -f "$ck" ]; do sleep 30; done
sleep 5
echo "=== seed3 done; continuation LD baseline vs hilr50 ==="
.venv/Scripts/python.exe scripts/gap_region_interaction.py \
  syntax_delay0_exp10_size23M_seed3 \
  syntax_delay0_exp10_size23M_seed3_hilr50 2>&1 | grep -E "long-distance|@ step"
echo "SEED3 SCORING DONE"
