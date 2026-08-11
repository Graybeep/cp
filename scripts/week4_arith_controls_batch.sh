#!/usr/bin/env bash
# Week 4 — seed-1 control forks that gate the seeds-2-3 replication design.
# Both fork from EXISTING checkpoints (no baseline training cost).
#
# Q2 negative control: lr_shift applied to a DELAYED skill with NO critical
#   period (syntax LD, size-sweep DISCONFIRM). If syntax "recovers" ~84% like
#   arith did, the effect is a generic late-LR boost, not arith-specific
#   reopening. Read syntax LD via scripts/gap_region_interaction.py at step 5000.
# Q1 cram_lrshift: cram's density boost UNDER the restored early LR. Separates
#   "cram dose-capped" from "cram LR-starved". Read arith_accuracy at step 5000.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
LOGDIR=logs/week4_arith_pilot
mkdir -p "$LOGDIR"

run () {  # name  from-run  done-run-id  extra-args...
  local name="$1"; local from="$2"; local run_id="$3"; shift 3
  local done_ckpt="checkpoints/${run_id}/step_0005000.pt"
  echo "=== [$(date +%H:%M:%S)] CONTROL ${name} -> ${run_id} ==="
  if [ -f "$done_ckpt" ]; then echo "    already complete — skipping"; return 0; fi
  "$PY" scripts/run_intervention.py --from-run "$from" --fork-step 2500 \
      "$@" > "${LOGDIR}/${name}.out" 2> "${LOGDIR}/${name}.err"
  echo "    exit $? (log: ${LOGDIR}/${name}.out)"
}

echo "###### Week 4 control forks start $(date) ######"

# Q2 negative control (syntax has no critical period; expect NO recovery)
run q2_syntax_negctrl \
    syntax_delay50_exp10_size23M_seed1 \
    syntax_delay50_exp10_size23M_seed1_ivlrshift0at2501 \
    --intervention lr_shift --shift-source-start 0

# Q1 cram + lr_shift combined (arith d50)
run q1_cram_lrshift \
    arith_delay50_exp20_size23M_seed1 \
    arith_delay50_exp20_size23M_seed1_ivcramlrshift0at2501 \
    --intervention cram_lrshift --shift-source-start 0

echo "###### Week 4 control forks done $(date) ######"
