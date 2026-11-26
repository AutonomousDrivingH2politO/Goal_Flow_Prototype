#!/bin/bash
# 2026-11-26: (1) step C L0 half (goalflow_l3_exact_jitter / _refine, seeds 0-39, L0); (2) baselines (a)-(d) at
# L1-L4, seeds 0-499, with odometry drift proportional to distance driven. Resumable; run detached.
cd "$(dirname "$0")/.."
LOG=reports/lidar_demo/l0_rerun_distdrift/chain.log
mkdir -p reports/lidar_demo/l0_rerun_distdrift
echo "start $(date) head $(git rev-parse --short HEAD)" >> $LOG
for v in goalflow_l3_exact_jitter goalflow_l3_exact_refine; do
  OUT=reports/lidar_demo/l3_goalflow/smoke_L0_$v
  [ -f $OUT/summary.md ] && { echo "$v L0 already done" >> $LOG; continue; }
  echo "$v L0 start $(date)" >> $LOG
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $v --levels L0 --seeds 0-39 --workers 3 --out $OUT > $OUT.log 2>&1
  echo "$v L0 exit $? end $(date)" >> $LOG
done
for cfg in observed_stubfix:a observed_l3:b observed_l3_noedge:c observed_l3_notrack:d; do
  name=${cfg%%:*}; tag=${cfg##*:}; OUT=reports/lidar_demo/l0_rerun_distdrift/$tag
  [ -f $OUT/summary.md ] && { echo "config $tag already done" >> $LOG; continue; }
  echo "config $tag ($name) start $(date)" >> $LOG
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $name --levels L1,L2,L3,L4 --seeds 0-499 \
      --workers 7 --out $OUT > $OUT.log 2>&1
  echo "config $tag exit $? end $(date)" >> $LOG
done
echo "done $(date)" >> $LOG
