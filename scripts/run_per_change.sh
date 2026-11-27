#!/bin/bash
# Pre-registered per-change runs (DEVIATIONS 2026-11-27): false-alarm diagnostic (replays), then each change alone on
# seeds 0-99 at L0/L2/L4: (a) entry2, (b) lambda3, (c) edge critic, (d) refinement. Waits for the beams-only runs.
cd "$(dirname "$0")/.."
D=reports/lidar_demo/l3_goalflow/per_change; LOG=$D/chain.log; mkdir -p $D
until grep -q "^done" reports/lidar_demo/emx_art/chain.log 2>/dev/null && ! pgrep -f "eval_corridor.py|diagnose_false" >/dev/null; do sleep 60; done
echo "start $(date) head $(git rev-parse --short HEAD)" >> $LOG
if [ ! -f reports/lidar_demo/l3_goalflow/false_alarms/SUMMARY.md ]; then
  echo "false-alarm diagnostic start $(date)" >> $LOG
  caffeinate -i .venv/bin/python scripts/diagnose_false_alarms.py --workers 5 > $D/false_alarms.log 2>&1
  echo "false-alarm diagnostic exit $? end $(date)" >> $LOG
fi
for name in goalflow_l3_circles_entry2 goalflow_l3_circles_lambda3 goalflow_l3_circles_edge goalflow_l3_circles_refine; do
  OUT=$D/$name
  [ -f $OUT/summary.md ] && { echo "$name already done" >> $LOG; continue; }
  echo "$name start $(date)" >> $LOG
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $name --levels L0,L2,L4 --seeds 0-99 --workers 5 \
      --out $OUT > $OUT.log 2>&1
  echo "$name exit $? end $(date)" >> $LOG
done
echo "done $(date)" >> $LOG
