#!/bin/bash
# Overnight observed corridor-lattice rerun (2026-11-25): configs (a) (b) (c) (d) in that order, L0-L4,
# seeds 0-499, no time budget; expansions and ms per replan in plan_times.csv; guard refusals in summary.md.
# Resumable: configs with a summary.md are skipped. Run detached so it survives session restarts:
#   nohup scripts/overnight_l0_rerun.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
OUT=reports/lidar_demo/l0_rerun
echo "start $(date) head $(git rev-parse --short HEAD)" >> $OUT/chain.log
for cfg in observed_stubfix:a observed_l3:b observed_l3_noedge:c observed_l3_notrack:d; do
  name=${cfg%%:*}; tag=${cfg##*:}
  if [ -f $OUT/$tag/summary.md ]; then echo "config $tag already done, skipped" >> $OUT/chain.log; continue; fi
  echo "config $tag ($name) start $(date)" >> $OUT/chain.log
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $name --levels L0,L1,L2,L3,L4 \
      --seeds 0-499 --workers 7 --out $OUT/$tag > $OUT/$tag.log 2>&1
  echo "config $tag exit $? end $(date)" >> $OUT/chain.log
done
echo "done $(date)" >> $OUT/chain.log
