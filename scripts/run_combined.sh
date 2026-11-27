#!/bin/bash
# Combined config (DEVIATIONS 2026-11-27): goalflow_l3_combined = circles + (a2) memory fix + (c) edge critic,
# seeds 0-499 x L0-L4, then COMPARISON.md. Launch only on Can's go; refuses if another eval is running.
cd "$(dirname "$0")/.."
OUT=reports/lidar_demo/l3_goalflow/full_goalflow_l3_combined
if pgrep -f eval_corridor.py >/dev/null; then echo "another eval_corridor is running; not starting"; exit 1; fi
mkdir -p $OUT
echo "start $(date) head $(git rev-parse --short HEAD) workers 5" >> $OUT.chain.log
caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners goalflow_l3_combined --levels L0,L1,L2,L3,L4 \
    --seeds 0-499 --workers 5 --out $OUT > $OUT.log 2>&1
echo "eval exit $? $(date)" >> $OUT.chain.log
.venv/bin/python scripts/build_comparison.py $OUT >> $OUT.chain.log 2>&1
echo "end $(date)" >> $OUT.chain.log
