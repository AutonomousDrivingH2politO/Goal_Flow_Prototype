#!/bin/bash
# Beams-only comparison (DEVIATIONS 2026-11-27): baselines (a) and (c) at emx_L2art and emx_L4art, seeds 0-499.
# Runs after the lambda3 spread replay (itself after the filter-on/off runs); before the per-change runs.
cd "$(dirname "$0")/.."
D=reports/lidar_demo/emx_art; LOG=$D/chain.log; mkdir -p $D
until grep -q "spread replay exit" reports/lidar_demo/l3_goalflow/model_or_system/chain.log 2>/dev/null \
      && ! pgrep -f eval_corridor.py >/dev/null; do sleep 60; done
echo "start $(date) head $(git rev-parse --short HEAD)" >> $LOG
for cfg in observed_stubfix:a observed_l3_noedge:c; do
  name=${cfg%%:*}; tag=${cfg##*:}; OUT=$D/$tag
  [ -f $OUT/summary.md ] && { echo "config $tag already done" >> $LOG; continue; }
  echo "config $tag ($name) start $(date)" >> $LOG
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $name --levels emx_L2art,emx_L4art --seeds 0-499 \
      --workers 7 --out $OUT > $OUT.log 2>&1
  echo "config $tag exit $? end $(date)" >> $LOG
done
echo "done $(date)" >> $LOG
