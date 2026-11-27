#!/bin/bash
# Baselines at the EMX sensor level (front-axle mount), seeds 0-499: (b), (a), (c). Resumable; run detached.
cd "$(dirname "$0")/.."
LOG=reports/lidar_demo/emx_baselines/chain.log
mkdir -p reports/lidar_demo/emx_baselines
echo "start $(date) head $(git rev-parse --short HEAD)" >> $LOG
for cfg in observed_l3:b observed_stubfix:a observed_l3_noedge:c; do
  name=${cfg%%:*}; tag=${cfg##*:}; OUT=reports/lidar_demo/emx_baselines/$tag
  [ -f $OUT/summary.md ] && { echo "config $tag already done" >> $LOG; continue; }
  echo "config $tag ($name) start $(date)" >> $LOG
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $name --levels EMX --seeds 0-499 --workers 7 \
      --out $OUT > $OUT.log 2>&1
  echo "config $tag exit $? end $(date)" >> $LOG
done
echo "done $(date)" >> $LOG
