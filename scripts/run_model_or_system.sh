#!/bin/bash
# Model or system? (DEVIATIONS 2026-11-27): goalflow_l3_circles with per-scan logging, filter on and filter off,
# seeds 0-99 at L0/L2/L4. Waits for the EMX baselines to finish (chain.log "done", no eval_corridor running).
cd "$(dirname "$0")/.."
OUTD=reports/lidar_demo/l3_goalflow/model_or_system; LOG=$OUTD/chain.log
mkdir -p $OUTD
until grep -q "^done" reports/lidar_demo/emx_baselines/chain.log && ! pgrep -f eval_corridor.py >/dev/null; do sleep 60; done
echo "start $(date) head $(git rev-parse --short HEAD)" >> $LOG
for name in goalflow_l3_circles_scanlog goalflow_l3_circles_filteroff; do
  OUT=$OUTD/$name
  [ -f $OUT/summary.md ] && { echo "$name already done" >> $LOG; continue; }
  echo "$name start $(date)" >> $LOG
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $name --levels L0,L2,L4 --seeds 0-99 --workers 5 \
      --out $OUT > $OUT.log 2>&1
  echo "$name exit $? end $(date)" >> $LOG
done
echo "done $(date)" >> $LOG
