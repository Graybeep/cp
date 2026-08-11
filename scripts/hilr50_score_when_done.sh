#!/usr/bin/env bash
cd /d/EXPERIMENTS/reopen
# wait for the replication chain to finish
while ! grep -q "ALL DONE" logs/hilr50_replication.log 2>/dev/null; do sleep 30; done
echo "=== replication done; scoring continuation LD ==="
.venv/Scripts/python.exe scripts/gap_region_interaction.py \
  syntax_delay0_exp10_size23M_seed2 \
  syntax_delay0_exp10_size23M_seed2_hilr50 \
  syntax_delay0_exp10_size23M_seed3 \
  syntax_delay0_exp10_size23M_seed3_hilr50 2>&1
echo "=== live final metrics (val_ppl / liveLD / local) ==="
for r in syntax_delay0_exp10_size23M_seed2 syntax_delay0_exp10_size23M_seed2_hilr50 \
         syntax_delay0_exp10_size23M_seed3 syntax_delay0_exp10_size23M_seed3_hilr50; do
  tail -1 logs/$r/metrics.csv | awk -F',' -v r=$r '{printf "%-45s val_ppl=%.2f liveLD=%.4f local=%.4f\n", r,$8,$10,$11}'
done
echo "SCORING DONE"
