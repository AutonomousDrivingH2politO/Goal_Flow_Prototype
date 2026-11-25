#!/usr/bin/env python
"""Closed-loop stills of the observed corridor-lattice planner, configs (a) and (b) side by side.

    python scripts/render_closed_loop.py

(a) = pre-L3 planner + goal-line stub fix; (b) = stub fix + L3 (tracking, edge memory, hysteresis + fallback,
hold, cache; no budget). One still per episode: true track and obstacles, the planner's observed edges (with
fill) and raw edge points at the key scan, the driven path coloured by speed, brake points, and the plan at
the key scan (first brake, else the collision / last scan). Seeds are picked by fixed rules (RULE.txt).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

OUT = Path("reports/lidar_demo/l0_rerun/renders")
GROUPS = {
    "L0_goal_line_stub": ("L0", [27], "lowest seed of the 10 A6 goal-line-stub seeds"),
    "L0_fill_beside_car": ("L0", [64], "lowest seed of the 7 A6 fill-edge seeds"),
    "L0_cone_grazes": ("L0", [287, 451], "the two cone grazes (collision under a, success under b)"),
    "L0_dead_ends": ("L0", [108, 190, 203, 399], "the 4 seeds still failing under (b) (dead-end states); for 190 "
                     "and 399 the blocking door and where it was first seen are marked"),
    "L2_regressions": ("L2", [82, 105, 107, 190, 274, 288, 299, 302, 308, 385],
                       "the L2 seeds that went success -> failure between the pre-fix run and config (a)"),
}
DOORS = {190: (6, 9), 399: (0, 11)}


def make(cfg):
    from gfsem.planners.corridor import CorridorConfig
    from gfsem.planners.corridor_lattice import LatticeCorridorPlanner, l3_config
    if cfg == "a":
        return LatticeCorridorPlanner(mode="observed", cfg=CorridorConfig(goal_cut_guard=1.0))
    return LatticeCorridorPlanner(mode="observed", cfg=CorridorConfig(goal_cut_guard=1.0), lcfg=l3_config())


def episode(seed, level, cfg):
    from gfsem.sim.closed_loop import CorridorObserved, run_episode
    from gfsem.sim.scene import generate_scene

    class Capture(CorridorObserved):
        def __init__(self, p):
            super().__init__(p, "observed")
            self.inputs = []

        def plan(self, obs, scene, tp, st):
            r = super().plan(obs, scene, tp, st)
            self.inputs.append(self.p.last_input)
            return r
    sc = generate_scene(seed)
    p = make(cfg)
    cap = Capture(p)
    r = run_episode(sc, cap, level, record=True)
    return sc, p, cap, r


def panel(ax, sc, p, cap, r, cfg, seed):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from gfsem.sim.geometry import from_frame, poses_from_frame
    recs = r.records
    br = [i for i, x in enumerate(recs) if x["brake"]]
    ki = br[0] if br else len(recs) - 1
    key = recs[ki]
    tr = r.trajectory
    s_all = sc.track.frenet(tr[:, :2])[0]
    s0, s1 = max(sc.track.s_min, s_all.min() - 5), min(sc.track.s_max, max(s_all.max(), sc.s_goal) + 8)
    ss = np.linspace(s0, s1, 300)
    for d in (sc.track.width / 2, -sc.track.width / 2):
        E = np.stack([sc.track.pose(q, d)[:2] for q in ss])
        ax.plot(E[:, 0], E[:, 1], "k-", lw=1.0)
    gl = np.stack([sc.track.pose(sc.s_goal, d)[:2] for d in (-sc.track.width / 2, sc.track.width / 2)])
    ax.plot(gl[:, 0], gl[:, 1], "g-", lw=2.5, label="goal line")
    for c, rr, cl in zip(sc.centres, sc.radii, sc.classes):
        ax.add_patch(plt.Circle(c, rr, color="tab:blue" if cl == 1 else "tab:orange", zorder=4))
    # observed edges (with fill) and raw edge points at the key scan
    bel = key["belief"]
    G = p._build(cap.inputs[ki])
    if G is not None:
        for e, lab in ((G["left"], "observed edges at key scan (incl. fill)"), (G["right"], None)):
            W = from_frame(e, bel)
            ax.plot(W[:, 0], W[:, 1], "-", color="tab:purple", lw=1.2, alpha=0.9, label=lab, zorder=5)
    ep = key["obs"].edge_points
    if len(ep):
        W = from_frame(np.asarray(ep, float), bel)
        ax.plot(W[:, 0], W[:, 1], ".", color="tab:purple", ms=2, label="raw edge points at key scan", zorder=5)
    # driven path by speed
    seg = np.stack([tr[:-1, :2], tr[1:, :2]], 1)
    lc = LineCollection(seg, cmap="viridis", norm=plt.Normalize(0, 3.5), lw=2.2, zorder=6)
    lc.set_array(tr[:-1, 3])
    ax.add_collection(lc)
    bp = np.array([x["pose"][:2] for x in recs if x["brake"]])
    if len(bp):
        ax.plot(bp[:, 0], bp[:, 1], "x", color="red", ms=5, mew=1.2, label="brake (scan)", zorder=7)
    # plan at the key scan (or the last plan before it, if the key scan braked)
    j = ki
    while j >= 0 and len(recs[j]["path"]) == 0:
        j -= 1
    if j >= 0:
        P = poses_from_frame(np.asarray(recs[j]["path"]), recs[j]["belief"])
        ax.plot(P[:, 0], P[:, 1], "--", color="tab:red", lw=1.4,
                label="plan at key scan" if j == ki else f"last plan before key scan (t={recs[j]['t']:.1f} s)", zorder=7)
    ax.plot(*key["pose"][:2], "o", color="red", mfc="none", ms=11, mew=1.5, label=f"key scan t={key['t']:.1f} s", zorder=8)
    if seed in DOORS:
        u, v = DOORS[seed]
        for o in (u, v):
            ax.add_patch(plt.Circle(sc.centres[o], sc.radii[o] + 0.35, fill=False, ec="magenta", lw=2, zorder=8))
        first = None
        for k in range(len(recs)):
            C = cap.inputs[k].centres
            if len(C) and min(np.hypot(*(from_frame(C, recs[k]["pose"]) - sc.centres[o]).T).min() for o in (u, v)) < 0.5:
                first = k
                break
        if first is not None:
            ax.plot(*recs[first]["pose"][:2], "D", color="magenta", ms=8, zorder=8,
                    label=f"car when the blocking door was first seen (t={recs[first]['t']:.1f} s)")
    ax.set_aspect("equal")
    lo, hi = tr[:, :2].min(0) - 6, tr[:, :2].max(0) + 6
    ax.set_xlim(lo[0], max(hi[0], gl[:, 0].max() + 3))
    ax.set_ylim(min(lo[1], gl[:, 1].min() - 1), max(hi[1], gl[:, 1].max() + 1))
    ax.set_title(f"({cfg}) {r.outcome} at {r.time:.1f} s, {r.n_brake_plans} brake scans", fontsize=9)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.85)
    return lc


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for g, (lvl, seeds, text) in GROUPS.items():
        d = OUT / g
        d.mkdir(parents=True, exist_ok=True)
        (d / "RULE.txt").write_text(f"{text}; seeds {seeds} at {lvl}; configs (a) | (b) side by side.\n"
                                    "(a) pre-L3 + goal-line stub fix; (b) stub fix + L3 (no budget).\n"
                                    "Key scan = first brake, else the collision / last scan.\n")
    for g, (lvl, seeds, _) in GROUPS.items():
        for seed in seeds:
            fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
            for ax, cfg in zip(axs, "ab"):
                sc, p, cap, r = episode(seed, lvl, cfg)
                lc = panel(ax, sc, p, cap, r, cfg, seed)
            fig.colorbar(lc, ax=axs, fraction=0.02, label="speed [m/s]")
            fig.suptitle(f"seed {seed} at {lvl}: observed corridor-lattice, (a) vs (b)", fontsize=10)
            fig.savefig(OUT / g / f"seed{seed:03d}_{lvl}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(g, seed, flush=True)


if __name__ == "__main__":
    main()
