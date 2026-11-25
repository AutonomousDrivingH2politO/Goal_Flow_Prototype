#!/bin/bash
# Keeps at least MIN_IDLE % CPU idle by pausing (SIGSTOP) and resuming (SIGCONT) the dataset
# generator's pool workers. Only the generator is throttled; training and everything else are left
# alone. Paused workers hold their shard and lose nothing. All workers are resumed on exit, and the
# governor exits when the generator finishes.
#   nohup bash scripts/cpu_governor.sh > data/governor.log 2>&1 &
MIN_IDLE=${MIN_IDLE:-5}      # pause one worker below this idle %
RESUME_IDLE=${RESUME_IDLE:-15}  # resume one worker above this idle %
PERIOD=${PERIOD:-15}
PAUSED=()
resume_all() { for p in "${PAUSED[@]}"; do kill -CONT "$p" 2>/dev/null; done; PAUSED=(); }
trap 'resume_all; echo "$(date +%T) exit: all workers resumed"; exit 0' EXIT INT TERM
# the generator's PID is fixed once at start (a later shell command containing the same text must
# not be mistaken for it); pass GEN_PID to override
GEN_PID=${GEN_PID:-$(pgrep -f "python scripts/gen_sim_dataset.py" | while read p; do
  ps -o command= -p "$p" | grep -q "^[^ ]*python scripts/gen_sim_dataset.py" && echo "$p"; done | head -1)}
gen_main() { kill -0 "$GEN_PID" 2>/dev/null && echo "$GEN_PID"; }
workers() {   # generator pool workers (not the resource tracker), running ones only
  local m; m=$(gen_main); [ -z "$m" ] && return
  for c in $(pgrep -P "$m"); do
    ps -o command= -p "$c" | grep -q spawn_main || continue
    case " ${PAUSED[*]} " in *" $c "*) ;; *) echo "$c";; esac
  done
}
idle() { top -l 2 -s 3 -n 0 2>/dev/null | awk '/CPU usage/{v=$7} END{gsub("%","",v); print int(v+0.5)}'; }
echo "$(date +%T) generator PID ${GEN_PID:-none}"
echo "$(date +%T) governor: keep >= ${MIN_IDLE}% idle, resume above ${RESUME_IDLE}%"
while [ -n "$(gen_main)" ]; do
  # forget paused PIDs that no longer exist
  keep=(); for p in "${PAUSED[@]}"; do kill -0 "$p" 2>/dev/null && keep=("${keep[@]}" "$p"); done; PAUSED=("${keep[@]}")
  i=$(idle)
  if [ "$i" -lt "$MIN_IDLE" ]; then
    w=$(workers | tail -1)
    if [ -n "$w" ]; then kill -STOP "$w" && PAUSED=("${PAUSED[@]}" "$w") && echo "$(date +%T) idle ${i}% -> paused $w (${#PAUSED[@]} paused)"; fi
  elif [ "$i" -gt "$RESUME_IDLE" ] && [ ${#PAUSED[@]} -gt 0 ]; then
    last=$(( ${#PAUSED[@]} - 1 ))      # macOS bash 3.2: no negative array indices
    p=${PAUSED[$last]}; unset "PAUSED[$last]"; PAUSED=("${PAUSED[@]}")
    kill -CONT "$p" && echo "$(date +%T) idle ${i}% -> resumed $p (${#PAUSED[@]} paused)"
  fi
  sleep "$PERIOD"
done
echo "$(date +%T) generator finished"
