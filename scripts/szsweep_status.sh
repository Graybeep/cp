#!/usr/bin/env bash
# Instant "is the size sweep alive / stuck / done?" query. No log-reading by hand.
#   bash scripts/szsweep_status.sh
cd /d/EXPERIMENTS/reopen || exit 1
LOCK="logs/.szsweep.lock"; STATUS="logs/szsweep_status.txt"; LOG="logs/syntax_size_sweep_batch.log"

echo "== szsweep status @ $(date '+%F %T') =="
if [ -f "$LOCK" ]; then
  pid=$(head -n1 "$LOCK" 2>/dev/null)
  if kill -0 "$pid" 2>/dev/null; then
    echo "driver : ALIVE (pid $pid)"
  else
    echo "driver : STALE LOCK (pid $pid dead) - a relaunch will reclaim it"
  fi
else
  echo "driver : NO LOCK (not running)"
fi

done=$(ls checkpoints/*szsweep/step_0005000.pt 2>/dev/null | wc -l)
echo "done   : $done/72 runs"

if [ -f "$LOG" ]; then
  now=$(date +%s); m=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
  age=$((now - m))
  echo "log    : last write ${age}s ago  ('training live' if < ~120s; a large/growing value = STALL)"
fi

echo "--- heartbeat file ---"
cat "$STATUS" 2>/dev/null || echo "(none yet)"
echo "--- recent events ---"
grep -aE "complete \+ pruned|BATCH-FAILURE|REFUSING|SZSWEEP-COMPLETE|reclaiming" "$LOG" 2>/dev/null \
  | tail -4 | tr -d '\r'
