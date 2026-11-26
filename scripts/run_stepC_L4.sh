#!/bin/bash
# Step C L4 half with distance-proportional drift: exact / jitter / refine, seeds 0-39, L4. Run detached.
cd "$(dirname "$0")/.."
LOG=reports/lidar_demo/l3_goalflow/stepC_L4_chain.log
echo "start $(date) head $(git rev-parse --short HEAD)" >> $LOG
for v in goalflow_l3_exact goalflow_l3_exact_jitter goalflow_l3_exact_refine; do
  OUT=reports/lidar_demo/l3_goalflow/smoke_L4dist_$v
  [ -f $OUT/summary.md ] && { echo "$v already done" >> $LOG; continue; }
  echo "$v start $(date)" >> $LOG
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $v --levels L4 --seeds 0-39 --workers 3 --out $OUT > $OUT.log 2>&1
  echo "$v exit $? end $(date)" >> $LOG
done
echo "done $(date)" >> $LOG
