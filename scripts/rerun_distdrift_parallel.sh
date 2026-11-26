#!/bin/bash
# Baselines (a)-(d) at L1-L4, seeds 0-499, distance-proportional drift, in two parallel streams (7 workers total).
# Stream 1 (a, c; 3 workers) waits for the step C refine L0 smoke to finish. Resumable; run detached.
cd "$(dirname "$0")/.."
LOG=reports/lidar_demo/l0_rerun_distdrift/chain.log
run() { name=$1; tag=$2; w=$3; OUT=reports/lidar_demo/l0_rerun_distdrift/$tag
  [ -f $OUT/summary.md ] && { echo "config $tag already done" >> $LOG; return; }
  echo "config $tag ($name, $w workers) start $(date)" >> $LOG
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $name --levels L1,L2,L3,L4 --seeds 0-499 \
      --workers $w --out $OUT > $OUT.log 2>&1
  echo "config $tag exit $? end $(date)" >> $LOG; }
echo "--- relaunched as two parallel streams $(date) head $(git rev-parse --short HEAD)" >> $LOG
( run observed_l3 b 4; run observed_l3_notrack d 4 ) &
( while pgrep -f "goalflow_l3_exact_refine" >/dev/null; do sleep 20; done
  [ -f reports/lidar_demo/l3_goalflow/smoke_L0_goalflow_l3_exact_refine/summary.md ] && echo "goalflow_l3_exact_refine L0 end $(date)" >> $LOG
  run observed_stubfix a 3; run observed_l3_noedge c 3 ) &
wait
echo "done $(date)" >> $LOG
