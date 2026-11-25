#!/usr/bin/env python
"""Prepare a split for training: (1) true-scene index per seed, (2) memory-mapped arrays.

    python scripts/prepare_training.py --splits val,train --scenes --workers 2   # scene index only
    python scripts/prepare_training.py --splits val,train --memmap               # repack shards

Scene index: $GFSEM_DATA/sim/scenes_<split>.npz (resumable through per-chunk files).
Memmap: $GFSEM_DATA/sim/mm_<split>/<array>.npy, all shards concatenated with offsets rebased, so
DataLoader workers share one page-cached copy instead of pickling every shard.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gfsem.data.scene_index import save_index, scene_record  # noqa: E402
from gfsem.data.sim_dataset import SPLITS, data_root  # noqa: E402

OFFSET_KEYS = {"points_off": ("points",), "edge_off": ("edge",), "targets_off": ("targets",)}
PER_FRAME = ("scan", "goal", "ego", "odom", "seed", "t", "pose", "recovery", "level")
RAGGED = {"points": "points_off", "labels": "points_off", "edge": "edge_off", "edge_side": "edge_off",
          "targets": "targets_off", "sigs": "targets_off"}


def _chunk(args):
    seeds, path = args
    if Path(path).exists():
        return path
    recs = [scene_record(s) for s in seeds]
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(recs, f)
    os.replace(tmp, path)
    return path


def build_scene_index(split: str, root: Path, workers: int, chunk: int = 100):
    base, n = SPLITS[split]
    out = root / f"scenes_{split}.npz"
    if out.exists():
        print(f"{out} exists", flush=True)
        return
    cdir = root / f"scenes_{split}_chunks"
    cdir.mkdir(parents=True, exist_ok=True)
    jobs = [(list(range(base + a, base + min(a + chunk, n))), str(cdir / f"{a:06d}.pkl")) for a in range(0, n, chunk)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_chunk, jobs)):
            if (k + 1) % 10 == 0:
                print(f"  scenes {split}: {k + 1}/{len(jobs)} chunks, {time.time() - t0:.0f}s", flush=True)
    recs = []
    for _, p in jobs:
        with open(p, "rb") as f:
            recs += pickle.load(f)
    save_index(recs, out)
    print(f"wrote {out} ({len(recs)} scenes)", flush=True)


def build_memmap(split: str, root: Path):
    files = sorted(glob.glob(str(root / f"{split}_*.npz")))
    out = root / f"mm_{split}"
    if (out / "DONE").exists():
        print(f"{out} exists", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list] = {}
    offs: dict[str, list] = {k: [] for k in OFFSET_KEYS}
    base = {k: 0 for k in OFFSET_KEYS}
    for f in files:
        a = np.load(f)
        for k in PER_FRAME:
            cols.setdefault(k, []).append(a[k])
        for k in RAGGED:
            cols.setdefault(k, []).append(a[k])
        for k in OFFSET_KEYS:
            o = a[k]
            offs[k].append((o[:-1] if len(offs[k]) < len(files) - 1 else o) + base[k])
            base[k] += int(o[-1])
    n = 0
    for k, v in cols.items():
        arr = np.concatenate(v)
        np.save(out / f"{k}.npy", arr)
        n = max(n, len(arr) if k == "seed" else n)
    for k, v in offs.items():
        np.save(out / f"{k}.npy", np.concatenate(v).astype(np.int64))
    (out / "DONE").write_text(f"{len(files)} shards, {n} frames\n")
    print(f"wrote {out}: {len(files)} shards, {n} frames", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="val,train")
    ap.add_argument("--scenes", action="store_true")
    ap.add_argument("--memmap", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    root = data_root()
    for split in a.splits.split(","):
        if a.scenes:
            build_scene_index(split, root, a.workers)
        if a.memmap:
            build_memmap(split, root)


if __name__ == "__main__":
    main()
