#!/usr/bin/env python
"""Why the observed-hit safety filter misses some truly colliding candidates (validation).

    python scripts/analyze_safety_misses.py

Same 1,600 frames and sample noise as calibrate_safety.py, cell_slack = 0. For every executed M3 candidate
whose true clearance is < 0 but which the filter does not flag: the first colliding pose (2x densified),
the obstacle it hits, that obstacle's number of hit cells in this scan (hit cells within r + 0.15 m of its
centre), and the distance from the collision point (the point of the disc nearest Juno's box) to the
nearest hit cell. Split: "unseen this scan" (no hit cell on the obstacle) vs "seen but clipped on an
unobserved side". Validation only.
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

from eval_selection import densify  # noqa: E402
from gfsem.data.scene_index import SceneIndex  # noqa: E402
from gfsem.data.sim_torch import MemmapSplit, SimFrames  # noqa: E402
from gfsem.models.common import pick_device  # noqa: E402
from gfsem.models.goalflow import hits_distance, observed_collision  # noqa: E402
from gfsem.sim.config import VehicleConfig  # noqa: E402
from gfsem.sim.raster import RasterConfig  # noqa: E402
from gfsem.train import load  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    dev = pick_device(ap.parse_args().device)
    mm = MemmapSplit(Path("data/sim/mm_val"))
    si = SceneIndex([Path("data/sim/scenes_val.npz")])
    gf = load(Path("runs/l2_full_seed0/goalflow_B_best.pt"), dev).eval()
    ds = SimFrames([mm], vocab=None, scene_index=si, driven_only=True)
    sub = np.random.default_rng(1234).choice(len(ds), 4000, replace=False)
    pick = sorted(np.random.default_rng(7).choice(sub, 1600, replace=False).tolist())
    dl = DataLoader(Subset(ds, pick), batch_size=32, num_workers=4, multiprocessing_context="spawn")
    veh, rc = VehicleConfig.from_yaml(), RasterConfig()
    x0, x1 = veh.body_x
    h = veh.width / 2
    rows = []
    n_col = 0
    fr_col = fr_flag_any = fr_flag_all = 0           # per-frame recall
    with torch.no_grad():
        for bi, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            torch.manual_seed(bi)
            r = gf.plan(b["bev"], b["ego"], M=128, n_steps=1)
            S = r["samples"]
            B, M, T, _ = S.shape
            dist = hits_distance(b["bev"][:, 0])
            flag = observed_collision(S, b["bev"][:, 0], dist=dist, cell_slack=0.0).cpu().numpy()
            obst = b["obstacles"].cpu().numpy()
            nob = b["n_obstacles"].cpu().numpy()
            hits = (b["bev"][:, 0] > 0.5).cpu().numpy()
            D = dist.cpu().numpy()
            X = densify(S.reshape(-1, T, 2)).view(B, M, -1, 3).cpu().numpy()     # (B, M, K, 3)
            i_idx, j_idx = np.meshgrid(np.arange(rc.size), np.arange(rc.size), indexing="ij")
            cell_x = rc.x_max - (i_idx + 0.5) * rc.res
            cell_y = rc.y_max - (j_idx + 0.5) * rc.res
            for f in range(B):
                O = obst[f, :nob[f]]
                if not len(O):
                    continue
                P = X[f]                                                          # (M, K, 3)
                c, s = np.cos(P[..., 2])[..., None], np.sin(P[..., 2])[..., None]
                dx = O[None, None, :, 0] - P[..., 0:1]
                dy = O[None, None, :, 1] - P[..., 1:2]
                lx, ly = c * dx + s * dy, -s * dx + c * dy
                qx, qy = np.clip(lx, x0, x1), np.clip(ly, -h, h)
                d = np.hypot(lx - qx, ly - qy) - O[None, None, :, 2]                 # (M, K, O)
                col = (d < 0).any((1, 2))
                n_col += int(col.sum())
                if col.any():
                    fr_col += 1
                    fr_flag_any += int((col & flag[f]).any())
                    fr_flag_all += int(not (col & ~flag[f]).any())
                for m in np.flatnonzero(col & ~flag[f]):
                    k, o = np.argwhere(d[m] < 0)[0]                                 # first colliding pose
                    # nearest box point to the centre, in world (ego) coordinates
                    bx = P[m, k, 0] + c[m, k, 0] * qx[m, k, o] - s[m, k, 0] * qy[m, k, o]
                    by = P[m, k, 1] + s[m, k, 0] * qx[m, k, o] + c[m, k, 0] * qy[m, k, o]
                    cx_, cy_, rr = O[o]
                    v = np.array([bx - cx_, by - cy_])
                    nv = np.linalg.norm(v)
                    pt = np.array([cx_, cy_]) + (rr * v / nv if nv > 1e-6 else 0.0)
                    near = np.hypot(cell_x - cx_, cell_y - cy_) <= rr + 0.15
                    n_hits = int((hits[f] & near).sum())
                    hc = np.argwhere(hits[f])
                    if len(hc):
                        hx = rc.x_max - (hc[:, 0] + 0.5) * rc.res
                        hy = rc.y_max - (hc[:, 1] + 0.5) * rc.res
                        dmin = float(np.hypot(hx - pt[0], hy - pt[1]).min())
                    else:
                        dmin = float("inf")
                    rows.append(dict(frame=int(pick[bi * 32 + f]), cand=int(m), obstacle=int(o), radius=float(rr),
                                     obst_x=float(cx_), obst_y=float(cy_), n_hit_cells=n_hits,
                                     dist_collision_point_to_nearest_hit=dmin, depth=float(-d[m, k, o]),
                                     max_depth=float(-d[m, :, o].min()),
                                     first_contact_t=float(k / (P.shape[1] - 1) * 4.0)))
            print(f"batch {bi + 1}/{len(dl)}  misses so far {len(rows)}", flush=True)
    unseen = [r for r in rows if r["n_hit_cells"] == 0]
    seen = [r for r in rows if r["n_hit_cells"] > 0]

    def stats(rs, key):
        x = np.array([r[key] for r in rs if np.isfinite(r[key])])
        return None if not len(x) else {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
                                        "p90": float(np.percentile(x, 90))}
    out = {"frames": len(pick), "true_collisions": n_col, "missed": len(rows),
           "per_frame": {"frames_with_a_truly_colliding_candidate": fr_col,
                         "share_where_filter_flags_some_colliding_candidate": fr_flag_any / max(fr_col, 1),
                         "share_where_filter_flags_every_colliding_candidate": fr_flag_all / max(fr_col, 1)},
           "missed_depth": stats(rows, "max_depth"), "missed_first_contact_s": stats(rows, "first_contact_t"),
           "unseen_this_scan": {"n": len(unseen), "dist_to_nearest_hit": stats(unseen, "dist_collision_point_to_nearest_hit"),
                                "obstacle_range_x": stats(unseen, "obst_x")},
           "seen_but_clipped": {"n": len(seen), "hit_cells": stats(seen, "n_hit_cells"),
                                "dist_to_nearest_hit": stats(seen, "dist_collision_point_to_nearest_hit"),
                                "depth": stats(seen, "depth")},
           "distinct_obstacles_missed": len({(r["frame"], r["obstacle"]) for r in rows}),
           "rows": rows}
    p = Path("reports/lidar_demo/l2_selection/safety_misses.json")
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
