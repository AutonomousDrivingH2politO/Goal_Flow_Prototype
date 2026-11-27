#!/bin/bash
# Final run of the frozen configs on the fresh test seeds 400000-400499, L0-L4, one at a time (approved by Can
# 2026-11-27), then FINAL_COMPARISON.md. Nothing starts after it.
cd "$(dirname "$0")/.."
D=reports/lidar_demo/final_test; LOG=$D/chain.log; mkdir -p $D
if pgrep -f eval_corridor.py >/dev/null; then echo "another eval_corridor is running; not starting"; exit 1; fi
echo "start $(date) head $(git rev-parse --short HEAD)" >> $LOG
run() {  # name planner workers
  [ -f $D/$1/summary.md ] && { echo "$1 already done" >> $LOG; return; }
  echo "$1 ($2) start $(date)" >> $LOG
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $2 --levels L0,L1,L2,L3,L4 --seeds 400000-400499 \
      --workers $3 --out $D/$1 > $D/$1.log 2>&1
  echo "$1 exit $? end $(date)" >> $LOG
}
run goalflow_final goalflow_final 5
run observed_stubfix_filter observed_stubfix_filter 7
run a observed_stubfix 7
run b observed_l3 7
run c observed_l3_noedge 7
.venv/bin/python scripts/build_final_comparison.py $D >> $LOG 2>&1
echo "end $(date)" >> $LOG
