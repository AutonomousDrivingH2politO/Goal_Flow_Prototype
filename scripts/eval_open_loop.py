#!/usr/bin/env python
"""Open-loop evaluation of trained checkpoints on a whole split (frozen models, fixed sample noise).

    python scripts/eval_open_loop.py --run runs/l2_full_seed0 --split test

Metrics are the training-time validation metrics (goal multimodality and DAC diagnostics, curvature
share, ADE/FDE for raw and executed paths), computed over every frame of the split. The test split is
for reporting only: never tune on it.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from gfsem.data.scene_index import SceneIndex  # noqa: E402
from gfsem.data.sim_torch import MemmapSplit, SimFrames, load_shards  # noqa: E402
from gfsem.models.common import pick_device  # noqa: E402
from gfsem.train import evaluate, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/l2_full_seed0"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/l2_full"))
    a = ap.parse_args()
    dev = pick_device("auto")
    root = Path("data/sim")
    mm = root / f"mm_{a.split}"
    src = [MemmapSplit(mm)] if (mm / "DONE").exists() else load_shards(str(root / f"{a.split}_*.npz"))
    si = SceneIndex([root / f"scenes_{a.split}.npz"])
    gf = load(a.run / "goalflow_B_best.pt", dev)
    rg = load(a.run / "regression_best.pt", dev)
    ds = SimFrames(src, vocab=gf.scorer.vocab.cpu().numpy(), scene_index=si)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers,
                    multiprocessing_context="spawn" if a.workers else None)
    args = argparse.Namespace(T=0.5, w5=0.1, w_aux=1.0)
    n_batches = len(dl)
    print(f"{a.split}: {len(ds)} frames, {n_batches} batches, device {dev}", flush=True)
    rep, t0 = {}, time.time()
    for name, model, stage in (("goal", gf, "A"), ("trajectory", gf, "B"), ("regression", rg, "reg")):
        torch.manual_seed(a.seed)
        rep[name] = evaluate(model, dl, stage, args, dev, max_batches=n_batches)
        print(f"{name}: done ({time.time() - t0:.0f}s)", flush=True)
    rep["frames"] = len(ds)
    rep["split"] = a.split
    rep["checkpoints"] = {"goalflow": str(a.run / "goalflow_B_best.pt"), "regression": str(a.run / "regression_best.pt")}
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"{a.split}_report.json"
    f.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
