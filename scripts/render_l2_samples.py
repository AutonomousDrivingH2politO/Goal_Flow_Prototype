#!/usr/bin/env python
"""Render validation frames with 2+ teacher modes, chosen by a fixed rule: the first N such frames of
val_0000 in stored order. Teacher modes, goal scores, 64 flow samples (executed), selected path
(executed and raw), regression (executed and raw).

    python scripts/render_l2_samples.py --run runs/l2_full_seed0 --out reports/lidar_demo/l2_full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from gfsem.data.sim_torch import SimFrames, load_shards  # noqa: E402
from gfsem.models.common import pick_device  # noqa: E402
from gfsem.train import load  # noqa: E402
from gfsem.viz import plot_samples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/l2_full_seed0"))
    ap.add_argument("--shard", default="data/sim/val_0000.npz")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/l2_full"))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = pick_device(a.device)
    gf = a.run / "goalflow_B_best.pt"
    gf = gf if gf.exists() else a.run / "goalflow_B.pt"
    rg = a.run / "regression_best.pt"
    rg = rg if rg.exists() else a.run / "regression.pt"
    model = load(gf, dev)
    reg = load(rg, dev) if rg.exists() else None
    shards = load_shards(a.shard)
    off = shards[0]["targets_off"]
    multi = [i for i in range(len(off) - 1) if off[i + 1] - off[i] >= 2][: a.n]
    print(f"rule: first {a.n} frames of {a.shard} with >= 2 teacher modes -> {multi}", flush=True)
    ds = SimFrames(shards)
    a.out.mkdir(parents=True, exist_ok=True)
    plot_samples(model, ds, a.out, device=dev, reg_model=reg, M=64, indices=multi, prefix="val0000")
    (a.out / "RULE.txt").write_text(f"First {a.n} frames of {a.shard} (stored order) with >= 2 teacher modes: "
                                    f"{multi}\nGoalFlow: {gf}\nRegression: {rg}\n")
    print(f"wrote {len(multi)} images to {a.out}", flush=True)


if __name__ == "__main__":
    main()
