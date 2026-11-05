#!/usr/bin/env bash
# Clean stop for the Tier-1 GPU queue (rate-collapse driver + tier1 chain).
#
# SAFE TO RUN AT ANY MOMENT. Nothing here can cost a completed run:
#   - every batch skips runs that already have step_0005000.pt, so finished
#     work is never redone;
#   - the per-run supervisor resumes from the high-water checkpoint, so the
#     run in flight restarts from its last checkpoint, not from step 0;
#   - checkpoint writes are atomic (BUG-1), so a kill mid-save cannot leave a
#     truncated .pt behind.
# The only cost is the training done since the in-flight run's last checkpoint.
#
# ORDER MATTERS. The drivers are killed FIRST and with -9, so that neither loop
# can advance to its next run after we kill the trainer underneath it. A plain
# TERM would not do: bash defers a trap until its foreground child returns, so
# the driver would wake when the trainer died and cheerfully launch the next
# one. -9 skips the EXIT trap that removes the lock, so the locks are removed
# here by hand — and even if that failed, acquire_lock treats a lock whose
# holder is dead as stale and reclaims it, so a leftover file never wedges a
# relaunch.
set -u
cd "$(dirname "$0")/.." || exit 1

echo "###### stop_all $(date '+%F %T') ######"

# Kill the process TREE, not just the lock holders.
#
# The lock-holder-only version of this script was WRONG and let the GPU keep
# running after it reported success: the queue is three levels deep
# (tier1_chain.sh -> <sub>_batch.sh -> python), only the top level writes a
# lock, and -9 on the top level orphans the sub-batch shell that actually
# launches trainers. See scripts/stop_all.ps1 for the full account and the
# sweep that closes the spawn race.
echo "-- processes (shells first, then trainers, swept until clear) --"
powershell -NoProfile -File scripts/stop_all.ps1
ps_rc=$?

echo "-- locks --"
for lock in logs/.ratecollapse.lock logs/.tier1chain.lock; do
  if [ -f "$lock" ]; then
    echo "   removing $lock (holder $(head -n1 "$lock" 2>/dev/null))"
    rm -f "$lock"
  else
    echo "   $lock: absent"
  fi
done

echo "-- state at stop --"
rc=$(ls checkpoints/syntax_delay50_exp10_size60M_seed*_ratecollapse/step_0005000.pt 2>/dev/null | wc -l)
echo "   T1.1 rate-collapse : ${rc}/24 seeds complete"
for pat in 'arith_delay*_exp20_wsd_size23M_seed*' \
           'arith_delay50_exp20_size23M_seed[23]_iv*' \
           'arith_delay50_exp20_size23M_seed*_ivlr{flatcontrol,levelonly,shapeonly}2501'; do
  n=$(ls -d checkpoints/$pat 2>/dev/null | wc -l)
  echo "   dirs matching $pat : $n"
done
echo "   (a dir without step_0005000.pt is a partial run; it will resume)"
if [ "${ps_rc:-1}" -ne 0 ]; then
  echo "###### STOP INCOMPLETE $(date '+%F %T') — see RESULT line above ######"
  exit 1
fi
echo "###### stopped $(date '+%F %T') ######"
