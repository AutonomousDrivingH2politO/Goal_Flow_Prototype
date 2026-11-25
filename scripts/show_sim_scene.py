#!/usr/bin/env python
"""Show one simulator scene: the true scene with doors, walls and the teacher paths, next to the
six BEV channels the learned planner will see.

    python scripts/show_sim_scene.py --seed 3 --noise L2 --t 2.0
    python scripts/show_sim_scene.py --seeds 0-19 --noise L3 --out reports/lidar_demo/viewer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from gfsem.planners.corridor import (CorridorPlanner, input_from_observation,  # noqa: E402
                                     input_from_privileged)
from gfsem.planners.corridor_lattice import LatticeCorridorPlanner  # noqa: E402
from gfsem.sim.closed_loop import CorridorObserved, CorridorPrivileged, run_episode  # noqa: E402
from gfsem.sim.config import BOLLARD, SimConfig  # noqa: E402
from gfsem.sim.geometry import footprint_corners, from_frame, poses_from_frame  # noqa: E402
from gfsem.sim.lidar import privileged_view  # noqa: E402
from gfsem.sim.raster import CHANNELS, RasterConfig, rasterize  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def draw_graph(ax, G, pose, colour_door="tab:green", colour_wall="0.75", lw=0.6):
    """Delaunay edges in world coordinates: doors green, walls grey."""
    if G is None:
        return
    pl = CorridorPlanner()
    S, V = G["S"], from_frame(G["V"], pose)
    seen = set()
    for t, simp in enumerate(S):
        if not G["valid"][t]:
            continue
        for k in range(3):
            a, b = sorted((int(simp[k]), int(simp[(k + 1) % 3])))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            kind, _ = pl._edge_kind(G, a, b)
            if kind == "door":
                ax.plot(V[[a, b], 0], V[[a, b], 1], color=colour_door, lw=lw, alpha=0.7, zorder=1)
            elif kind == "wall":
                ax.plot(V[[a, b], 0], V[[a, b], 1], color=colour_wall, lw=lw * 0.6, zorder=0)


def teacher_paths(scene, pose, K: int):
    """Multimodal teacher labels: the lattice inside each of the top-K distinct A* routes."""
    pl = LatticeCorridorPlanner(mode="privileged")
    paths = pl.teacher_paths(input_from_privileged(privileged_view(scene, pose)), K)
    return [poses_from_frame(p, pose) for p in paths]


def make(kind: str, mode: str):
    if kind == "funnel":
        P = CorridorPlanner(mode=mode)
    else:
        P = LatticeCorridorPlanner(mode=mode)
    return (CorridorPrivileged if mode == "privileged" else CorridorObserved)(P)


def show(seed: int, noise: str, t_snap: float, K: int, out: Path | None, cfg: SimConfig,
         kind: str = "lattice"):
    import matplotlib.pyplot as plt
    scene = generate_scene(seed, cfg.scene, cfg.vehicle)
    tr = scene.track
    priv = run_episode(scene, make(kind, "privileged"), "L0", cfg, record=True)
    obs_ep = run_episode(scene, make(kind, "observed"), noise, cfg, record=True)
    # snapshot from the observed run
    recs = obs_ep.records
    i = min(range(len(recs)), key=lambda j: abs(recs[j]["t"] - t_snap))
    rec = recs[i]
    pose = rec["pose"]
    obs = rec["obs"]
    R = rasterize(obs, RasterConfig())

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.3, 1.3, 1])
    ax = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    poly = tr.polygon()
    for a in (ax, ax2):
        a.fill(poly[:, 0], poly[:, 1], color="0.95", zorder=-2)
        a.plot(*np.vstack([poly, poly[:1]]).T, color="0.3", lw=1)
        g = scene.goal_line
        a.plot(g[:, 0], g[:, 1], "k--", lw=1.5)
        for c, r, k in zip(scene.centres, scene.radii, scene.classes):
            a.add_patch(plt.Circle(c, r, color="tab:blue" if k == BOLLARD else "tab:orange", zorder=3))
        a.set_aspect("equal")

    # left panel: true scene, doors/walls from the spawn triangulation, teacher paths
    p0 = tr.pose(0.0)
    G = CorridorPlanner()._build(input_from_privileged(privileged_view(scene, p0)))
    draw_graph(ax, G, p0)
    tps = teacher_paths(scene, p0, K)
    for j, pth in enumerate(tps):
        ax.plot(pth[:, 0], pth[:, 1], color=plt.cm.viridis(j / max(K - 1, 1)), lw=1.2, alpha=0.8, zorder=4)
    if scene.oracle_path is not None:
        ax.plot(*scene.oracle_path[:, :2].T, color="m", lw=0.8, ls=":", zorder=4, label="oracle")
    ax.plot(priv.trajectory[:, 0], priv.trajectory[:, 1], "g-", lw=2, zorder=5,
            label=f"{kind} privileged (teacher): {priv.outcome}")
    ax.set_title(f"seed {seed}  layout={scene.layout}  feasible={scene.feasible}\n"
                 f"true scene, doors (green) / walls (grey), {len(tps)} distinct teacher routes (top-{K})")
    ax.legend(loc="lower right", fontsize=8)

    # middle panel: observed run, LiDAR hits and fitted obstacles at the snapshot
    ax2.plot(obs_ep.trajectory[:, 0], obs_ep.trajectory[:, 1], "r-", lw=2, zorder=5,
             label=f"{kind} observed {noise}: {obs_ep.outcome}")
    pts_w = from_frame(obs.points, pose)
    ax2.scatter(pts_w[:, 0], pts_w[:, 1], s=3, c="k", zorder=6, label="LiDAR hits")
    e_w = from_frame(obs.edge_points, pose)
    ax2.scatter(e_w[:, 0], e_w[:, 1], s=2, c="tab:green", zorder=6, label="edge points")
    inp = input_from_observation(obs, CorridorPlanner().cfg, cfg.scene)
    for c, r in zip(from_frame(inp.centres, pose).reshape(-1, 2), inp.radii):
        ax2.add_patch(plt.Circle(c, r, fill=False, color="r", lw=1, zorder=7))
    if len(rec["path"]):
        pw = poses_from_frame(rec["path"], pose)
        ax2.plot(pw[:, 0], pw[:, 1], "m-", lw=1.5, zorder=7, label="plan at snapshot")
    fp = footprint_corners(pose[None], cfg.vehicle.body_x, cfg.vehicle.width)[0]
    ax2.fill(fp[:, 0], fp[:, 1], color="r", alpha=0.4, zorder=8)
    ax2.set_title(f"observed corridor-{kind} at {noise}, snapshot t={rec['t']:.1f}s\n"
                  f"flips={obs_ep.flips}, brake={rec['brake']}")
    ax2.legend(loc="lower right", fontsize=8)
    lo = min(poly[:, 0].min(), -5)
    for a in (ax, ax2):
        a.set_xlim(lo, scene.s_goal + 5 if abs(tr.kappa) < 1e-6 else poly[:, 0].max())

    rc = RasterConfig()
    ext = [rc.y_max, rc.y_min, rc.x_min, rc.x_max]  # left is +y
    for c in range(6):
        a = fig.add_subplot(gs[2, c])
        a.imshow(R[c], cmap="magma", vmin=0, vmax=1, extent=ext, origin="upper")
        a.set_title(f"{c}: {CHANNELS[c]}", fontsize=9)
        a.plot([0], [0], "c^", ms=5)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"scene_{seed:04d}_{noise}.png"
        fig.savefig(f, dpi=90)
        plt.close(fig)
        print(f"wrote {f}  priv={priv.outcome} obs={obs_ep.outcome}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default=None, help="e.g. 0-19 (writes PNGs)")
    ap.add_argument("--noise", default="L2")
    ap.add_argument("--t", type=float, default=2.0, help="snapshot time [s]")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--planner", choices=["lattice", "funnel"], default="lattice",
                    help="funnel = corridor-funnel (original design)")
    a = ap.parse_args()
    cfg = SimConfig()
    if a.seeds:
        matplotlib.use("Agg")
        out = a.out or Path("reports/lidar_demo/viewer")
        for s in parse_seeds(a.seeds):
            show(s, a.noise, a.t, a.K, out, cfg, a.planner)
    else:
        if a.out:
            matplotlib.use("Agg")
        show(a.seed, a.noise, a.t, a.K, a.out, cfg, a.planner)


if __name__ == "__main__":
    main()
