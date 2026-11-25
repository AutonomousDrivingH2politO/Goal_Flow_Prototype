#!/bin/bash
# Stall ablation of config (b) at L4, seeds 0-499 (2026-11-25). Resumable; run detached:
#   nohup scripts/ablation_stalls_l4.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
OUT=reports/lidar_demo/l0_rerun/ablation_L4
mkdir -p $OUT
echo "start $(date) head $(git rev-parse --short HEAD)" >> $OUT/chain.log
for name in observed_l3_nocache observed_l3_nohold observed_l3_nocache_nohold observed_l3_nohyst; do
  if [ -f $OUT/$name/summary.md ]; then echo "$name already done, skipped" >> $OUT/chain.log; continue; fi
  echo "$name start $(date)" >> $OUT/chain.log
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $name --levels L4 --seeds 0-499 \
      --workers 7 --out $OUT/$name > $OUT/$name.log 2>&1
  echo "$name exit $? end $(date)" >> $OUT/chain.log
done
echo "done $(date)" >> $OUT/chain.log
