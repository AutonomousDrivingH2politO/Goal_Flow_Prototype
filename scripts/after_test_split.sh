#!/bin/bash
# When the test split and its scene index are complete: repack to memmap, run the open-loop test evaluation.
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1
until [ "$(ls data/sim/test_*.npz 2>/dev/null | wc -l | tr -d ' ')" -ge 30 ] && [ -f data/sim/scenes_test.npz ]; do sleep 60; done
echo "$(date) test split complete; repacking"
nice -n 10 .venv/bin/python scripts/prepare_training.py --splits test --memmap
echo "$(date) evaluating"
nice -n 10 .venv/bin/python scripts/eval_open_loop.py --run runs/l2_full_seed0 --split test
echo "$(date) test evaluation exit code $?"
