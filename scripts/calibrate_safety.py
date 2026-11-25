#!/usr/bin/env python
"""Calibrate the observed-hit safety filter's cell slack on validation (frozen seed-0 GoalFlow).

    python scripts/calibrate_safety.py

For cell_slack in {0.0, 0.05, 0.1} m, on 1,600 frames drawn from the fixed 4,000-frame validation subset:
  * false-alarm rate on the driven teacher route and on all teacher routes (they are collision-free)
  * recall: share of executed M3 candidates (M = 128) with true path clearance < 0 that the filter flags
  * false-alarm rate on executed candidates with true clearance >= 0
Rule (fixed 2026-11-25 before running): use the largest slack whose driven-route false-alarm rate is
<= 1.0%; if none qualifies, 0.0. The result goes to safety_calibration.json and into eval_selection.py.
Validation only.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from eval_selection import path_clearance  # noqa: E402
from gfsem.data.scene_index import SceneIndex  # noqa: E402
from gfsem.data.sim_torch import MemmapSplit, SimFrames  # noqa: E402
from gfsem.models.common import pick_device  # noqa: E402
from gfsem.models.goalflow import hits_distance, observed_collision  # noqa: E402
from gfsem.sim.config import VehicleConfig  # noqa: E402
from gfsem.train import load  # noqa: E402

SLACKS = (0.0, 0.05, 0.1)
MAX_DRIVEN_FA = 0.01


def main():
    dev = pick_device("auto")
    mm = MemmapSplit(Path("data/sim/mm_val"))
    si = SceneIndex([Path("data/sim/scenes_val.npz")])
    gf = load(Path("runs/l2_full_seed0/goalflow_B_best.pt"), dev).eval()
    ds = SimFrames([mm], vocab=None, scene_index=si, driven_only=True)
    sub = np.random.default_rng(1234).choice(len(ds), 4000, replace=False)
    pick = sorted(np.random.default_rng(7).choice(sub, 1600, replace=False).tolist())
    dl = DataLoader(Subset(ds, pick), batch_size=32, num_workers=4, multiprocessing_context="spawn")
    veh = VehicleConfig.from_yaml()
    acc = {s: dict(driven=0, driven_n=0, routes=0, routes_n=0, col=0, col_flag=0, free=0, free_flag=0) for s in SLACKS}
    with torch.no_grad():
        for bi, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            modes, nm = b["modes"], b["n_modes"]
            valid = torch.arange(8, device=dev)[None] < nm[:, None]
            torch.manual_seed(bi)
            r = gf.plan(b["bev"], b["ego"], M=128, n_steps=1)
            S = r["samples"]
            B, M = S.shape[:2]
            clr = path_clearance(S.reshape(-1, S.shape[2], 2), b["obstacles"].repeat_interleave(M, 0),
                                 b["n_obstacles"].repeat_interleave(M, 0), veh).view(B, M)
            col = clr < 0
            dist = hits_distance(b["bev"][:, 0])
            for s in SLACKS:
                tf = observed_collision(modes, b["bev"][:, 0], dist=dist, cell_slack=s)
                cf = observed_collision(S, b["bev"][:, 0], dist=dist, cell_slack=s)
                A = acc[s]
                A["driven"] += int(tf[:, 0].sum())
                A["driven_n"] += B
                A["routes"] += int((tf & valid).sum())
                A["routes_n"] += int(valid.sum())
                A["col"] += int(col.sum())
                A["col_flag"] += int((col & cf).sum())
                A["free"] += int((~col).sum())
                A["free_flag"] += int((~col & cf).sum())
            print(f"batch {bi + 1}/{len(dl)}", flush=True)
    out = {"frames": len(pick), "rule": f"largest slack with driven false alarm <= {MAX_DRIVEN_FA:.1%}", "slacks": {}}
    chosen = 0.0
    for s in SLACKS:
        A = acc[s]
        row = {"false_alarm_driven": A["driven"] / A["driven_n"], "false_alarm_all_routes": A["routes"] / A["routes_n"],
               "recall_true_collisions": A["col_flag"] / max(A["col"], 1),
               "false_alarm_free_candidates": A["free_flag"] / max(A["free"], 1), **A}
        out["slacks"][f"{s:g}"] = row
        if row["false_alarm_driven"] <= MAX_DRIVEN_FA:
            chosen = max(chosen, s)
    out["chosen_cell_slack"] = chosen
    p = Path("reports/lidar_demo/l2_selection/safety_calibration.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
