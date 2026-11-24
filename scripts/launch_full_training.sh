#!/bin/bash
# Waits for the L1 train split (300 shards) and its scene index, repacks the train split to memmap,
# then runs the overnight L2 training on the Mac. Survives the terminal (nohup) and keeps the Mac
# awake (caffeinate). Usage: nohup caffeinate -ims bash scripts/launch_full_training.sh > runs/l2_full_launcher.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/l2_full_seed0}
SEED=${SEED:-0}
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1
echo "$(date) waiting for 300 train shards and scenes_train.npz"
while true; do
  n=$(ls data/sim/train_*.npz 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -ge 300 ] && [ -f data/sim/scenes_train.npz ] && [ -f data/sim/scenes_val.npz ]; then break; fi
  sleep 60
done
echo "$(date) train split complete ($n shards); repacking to memmap"
nice -n 10 .venv/bin/python scripts/prepare_training.py --splits train,val --memmap || exit 1
echo "$(date) starting training -> $OUT"
mkdir -p "$OUT"
nice -n 10 .venv/bin/python -m gfsem.train \
  --train mm:data/sim/mm_train --val mm:data/sim/mm_val \
  --scene-index data/sim/scenes_train.npz,data/sim/scenes_val.npz \
  --stages A,B,reg --steps 300000 --hours-A 3.0 --hours-B 3.5 --hours-reg 1.5 \
  --bs 64 --workers 5 --threads 2 --eval-every 1000000 --eval-minutes 30 \
  --val-frames 4000 --val-batches 1000 --vocab-n 1024 --seed "$SEED" --log-every 50 \
  --plots 8 --resume --out "$OUT" > "$OUT/train.log" 2>&1
echo "$(date) training finished with exit code $?"
