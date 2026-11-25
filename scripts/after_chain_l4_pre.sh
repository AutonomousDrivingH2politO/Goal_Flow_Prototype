#!/bin/bash
# After the overnight chain: the 50e9ad9 planner (eval_corridor 'observed' = pre-L3 defaults, no stub fix) at L4,
# seeds 0-499, to measure how much of the L4 gain comes from the edge-fill fix alone.
cd "$(dirname "$0")/.."
OUT=reports/lidar_demo/l0_rerun
until grep -q "^done" $OUT/chain.log; do sleep 60; done
[ -f $OUT/pre_L4/summary.md ] && exit 0
echo "pre_L4 (observed, 50e9ad9 planner) start $(date)" >> $OUT/chain.log
caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners observed --levels L4 --seeds 0-499 \
    --workers 6 --out $OUT/pre_L4 > $OUT/pre_L4.log 2>&1
echo "pre_L4 exit $? end $(date)" >> $OUT/chain.log
