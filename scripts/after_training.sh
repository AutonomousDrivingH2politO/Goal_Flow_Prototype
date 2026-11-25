#!/bin/bash
# Waits for the overnight training to finish, then renders the fixed-rule validation samples
# into reports/lidar_demo/l2_full/.
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1
until grep -q "training finished" runs/l2_full_launcher.log 2>/dev/null; do sleep 120; done
echo "$(date) training finished; rendering"
nice -n 10 .venv/bin/python scripts/render_l2_samples.py --run runs/l2_full_seed0 --n 12 \
  --out reports/lidar_demo/l2_full
echo "$(date) render exit code $?"
