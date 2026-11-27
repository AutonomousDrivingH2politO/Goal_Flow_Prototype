#!/bin/bash
# (a2) memory fix alone (DEVIATIONS 2026-11-27), seeds 0-99 at L0/L2/L4; after the four per-change runs.
cd "$(dirname "$0")/.."
D=reports/lidar_demo/l3_goalflow/per_change; LOG=$D/chain.log
until grep -q "^done" $LOG 2>/dev/null && ! pgrep -f eval_corridor.py >/dev/null; do sleep 60; done
OUT=$D/goalflow_l3_circles_memfix
[ -f $OUT/summary.md ] && exit 0
echo "goalflow_l3_circles_memfix start $(date) head $(git rev-parse --short HEAD)" >> $LOG
caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners goalflow_l3_circles_memfix --levels L0,L2,L4 \
    --seeds 0-99 --workers 5 --out $OUT > $OUT.log 2>&1
echo "goalflow_l3_circles_memfix exit $? end $(date)" >> $LOG
echo "memfix done $(date)" >> $LOG
