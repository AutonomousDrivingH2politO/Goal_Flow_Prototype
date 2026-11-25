#!/bin/bash
# Resumes the dataset generator (paused for training) and restarts the CPU governor once the
# overnight training has finished.
cd "$(dirname "$0")/.."
until grep -q "training finished" runs/l2_full_launcher.log 2>/dev/null; do sleep 60; done
for p in $(cat data/generator_paused.pids); do kill -CONT "$p" 2>/dev/null; done
echo "$(date) training finished: generator resumed ($(cat data/generator_paused.pids))"
echo "--- restart $(date +%T) after training" >> data/governor.log
nohup bash scripts/cpu_governor.sh >> data/governor.log 2>&1 &
echo "$(date) governor restarted"
