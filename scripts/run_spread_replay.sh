#!/bin/bash
# lambda3 scale (DEVIATIONS 2026-11-27): deterministic rerun of goalflow_l3_circles_scanlog (seeds 0-99, L0/L2/L4) with
# the per-scan top-3 goal-group score spread logged. Waits for the model-or-system chain to finish.
cd "$(dirname "$0")/.."
D=reports/lidar_demo/l3_goalflow/model_or_system
until grep -q "^done" $D/chain.log 2>/dev/null && ! pgrep -f eval_corridor.py >/dev/null; do sleep 60; done
OUT=$D/scanlog_spread
[ -f $OUT/summary.md ] && exit 0
echo "spread replay start $(date) head $(git rev-parse --short HEAD)" >> $D/chain.log
caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners goalflow_l3_circles_scanlog --levels L0,L2,L4 \
    --seeds 0-99 --workers 5 --out $OUT > $OUT.log 2>&1
echo "spread replay exit $? end $(date)" >> $D/chain.log
