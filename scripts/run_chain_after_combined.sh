#!/bin/bash
# Chain after the combined run (approved by Can 2026-11-27): one at a time, no review stops. Each item logs and
# appends a progress line to HANDOFF.md. A failing item is logged and the chain continues.
cd "$(dirname "$0")/.."
LOG=reports/lidar_demo/chain_after_combined.log
note() { echo "- $(date '+%m-%d %H:%M') $1" >> HANDOFF.md; echo "$(date) $1" >> $LOG; }
until grep -q "^end" reports/lidar_demo/l3_goalflow/full_goalflow_l3_combined.chain.log 2>/dev/null \
      && ! pgrep -f "eval_corridor.py" >/dev/null; do sleep 60; done
echo "start $(date) head $(git rev-parse --short HEAD)" >> $LOG
PY=.venv/bin/python
# 2: McNemar + stall overlap (COMPARISON.md of the combined and the full circles run)
$PY scripts/build_comparison.py reports/lidar_demo/l3_goalflow/full_goalflow_l3_combined > /dev/null 2>>$LOG \
  && $PY scripts/build_comparison.py reports/lidar_demo/l3_goalflow/full_goalflow_l3_circles > /dev/null 2>>$LOG
note "item 2 McNemar + stall overlap: exit $? (COMPARISON.md of combined and circles runs)"
# 3: the two L4 collisions
caffeinate -i $PY scripts/diagnose_l4_collisions.py >> $LOG 2>&1
note "item 3 L4 collisions diagnosed: exit $? (reports/lidar_demo/l3_goalflow/l4_collisions/)"
# (a)+filter
OUT=reports/lidar_demo/a_filter
caffeinate -i $PY scripts/eval_corridor.py --planners observed_stubfix_filter --seeds 0-499 --workers 7 --out $OUT > $OUT.log 2>&1
note "(a)+filter run: exit $? ($OUT/)"
# 4: latency profile, single worker, nothing else running
caffeinate -i $PY scripts/profile_latency.py --episodes 12 >> $LOG 2>&1
note "item 4 latency profile: exit $? (reports/lidar_demo/l3_goalflow/latency/)"
# 6: EMX pitch check at the roof mount
$PY scripts/pitch_check.py --scenes 60 >> $LOG 2>&1
note "item 6 EMX pitch check: exit $? (reports/lidar_demo/mounting/pitch_check.md)"
# 5: juno_bangbang
J=reports/lidar_demo/juno
mkdir -p $J
caffeinate -i $PY scripts/eval_corridor.py --planners goalflow_l3_combined --levels L0,L2,L4 --seeds 0-99 --workers 5 \
    --speed-mode juno_bangbang --out $J/goalflow_l3_combined > $J/goalflow_l3_combined.log 2>&1
e1=$?
for cfg in observed_l3:b observed_stubfix:a observed_l3_noedge:c; do
  caffeinate -i $PY scripts/eval_corridor.py --planners ${cfg%%:*} --levels L0,L2,L4 --seeds 0-99 --workers 7 \
      --speed-mode juno_bangbang --out $J/${cfg##*:} > $J/${cfg##*:}.log 2>&1
done
note "item 5 juno_bangbang: GoalFlow exit $e1, baselines last exit $? ($J/)"
# 8: scene overlap
$PY scripts/scene_overlap.py >> $LOG 2>&1
note "item 8 scene overlap: exit $? (reports/lidar_demo/scene_overlap/)"
echo "done $(date)" >> $LOG
note "chain done"
