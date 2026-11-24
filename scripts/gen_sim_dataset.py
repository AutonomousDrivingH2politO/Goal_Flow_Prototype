#!/usr/bin/env python
"""Phase L1: generate the simulator imitation dataset (teacher = privileged corridor-lattice).

    python scripts/gen_sim_dataset.py --splits val,train,test --workers 7

Shards go to $GFSEM_DATA/sim (default: <repo>/data/sim, git-ignored). Existing shards are
skipped, so the run can be stopped and resumed. Every shard depends only on its seeds.
manifest.jsonl gets one line per shard (frames, bytes, seconds, homotopy classes per scene).
"""
from __future__ import annotations

import os

# One BLAS/OpenMP thread per process: workers are parallel already (set before numpy loads;
# spawned workers inherit the environment).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gfsem.data.sim_dataset import DatasetConfig, build_shard, data_root, n_shards  # noqa: E402


def _job(args):
    split, shard, out = args
    return build_shard(split, shard, Path(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="val,train,test")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-shards", type=int, default=None, help="per split, for quick runs")
    a = ap.parse_args()
    out = a.out or data_root()
    out.mkdir(parents=True, exist_ok=True)
    dcfg = DatasetConfig()
    jobs = []
    for split in a.splits.split(","):
        n = n_shards(split, dcfg) if a.max_shards is None else min(a.max_shards, n_shards(split, dcfg))
        for i in range(n):
            if not (out / f"{split}_{i:04d}.npz").exists():
                jobs.append((split, i, str(out)))
    print(f"{len(jobs)} shards to build into {out}", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool, open(out / "manifest.jsonl", "a") as man:
        for k, r in enumerate(pool.imap_unordered(_job, jobs)):
            man.write(json.dumps(r) + "\n")
            man.flush()
            el = time.time() - t0
            print(f"[{k + 1}/{len(jobs)}] {r['split']}_{r['shard']:04d}: {r['frames']} frames, "
                  f"{r['bytes'] / 1e6:.1f} MB, {r['seconds']:.0f}s; elapsed {el / 60:.1f} min, "
                  f"eta {el / (k + 1) * (len(jobs) - k - 1) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
