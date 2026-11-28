#!/bin/bash
# Follow-ups after the final run (approved 2026-11-28; DEVIATIONS): one at a time, 6 workers for the classical runs.
cd "$(dirname "$0")/.."
D=reports/lidar_demo/followups; LOG=$D/chain.log; mkdir -p $D
if pgrep -f eval_corridor.py >/dev/null; then echo "another eval_corridor is running; not starting"; exit 1; fi
note() { echo "- $(date '+%m-%d %H:%M') $1" >> HANDOFF.md; echo "$(date) $1" >> $LOG; }
echo "start $(date) head $(git rev-parse --short HEAD)" >> $LOG
run() {  # out planner workers levels seeds extra-args...
  out=$1; pl=$2; w=$3; lv=$4; sd=$5; shift 5
  [ -f $D/$out/summary.md ] && { echo "$out already done" >> $LOG; return; }
  echo "$out start $(date)" >> $LOG
  caffeinate -i .venv/bin/python scripts/eval_corridor.py --planners $pl --levels $lv --seeds $sd --workers $w \
      --out $D/$out "$@" > $D/$out.log 2>&1
  echo "$out exit $? end $(date)" >> $LOG
}
# 1. nofix memory variant on dev seeds 0-499 x L0-L4
run nofix_dev observed_stubfix_filter_nofix 6 L0,L1,L2,L3,L4 0-499
note "follow-up 1 nofix on dev 0-499 x L0-L4 done ($D/nofix_dev/)"
# 2. juno profile for the car candidate (vs constant from a_filter and nofix_dev)
run juno_a_filter observed_stubfix_filter 6 L0,L2,L4 0-99 --speed-mode juno_bangbang
run juno_nofix observed_stubfix_filter_nofix 6 L0,L2,L4 0-99 --speed-mode juno_bangbang
note "follow-up 2 juno for (a)+filter and nofix done"
# 3. GoalFlow juno cause test (goalflow_final, with its own references on the same seeds)
run gf_const3 goalflow_final 5 L0,L2,L4 0-99
run gf_juno goalflow_final 5 L0,L2,L4 0-99 --speed-mode juno_bangbang
run gf_const2 goalflow_final_2ms 5 L0,L2,L4 0-99 --spawn-speed 2.0
run gf_juno_nodelay goalflow_final 5 L0,L2,L4 0-99 --speed-mode juno_bangbang_nobrakedelay
note "follow-up 3 GoalFlow juno cause test done"
echo "done $(date)" >> $LOG
note "follow-ups done"
