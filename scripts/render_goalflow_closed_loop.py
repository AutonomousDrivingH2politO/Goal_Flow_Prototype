#!/usr/bin/env python
"""Closed-loop stills: baseline (a) | GoalFlow goalflow_l3 (picked near-observation memory), smoke-test seeds.

    python scripts/render_goalflow_closed_loop.py

Rule (written to RULE.txt): every feasible goalflow_l3 smoke-test failure (seeds 0-39, L0 and L4), plus the 3
lowest-numbered successful seeds at each level, 12 stills in all. GoalFlow panel: true track and obstacles, driven
path coloured by speed, brake scans, and at the key scan (first brake, else collision / last scan) the 128 executed
candidates (all drawn alike; filter flags are not logged), the chosen path, current hits and remembered hits.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

SMOKE = Path("reports/lidar_demo/l3_goalflow/smoke_goalflow_l3_nearobs/results.csv")
OUT = Path("reports/lidar_demo/l3_goalflow/renders")


def pick():
    R = list(csv.DictReader(open(SMOKE)))
    fails = sorted((r["noise"], int(r["seed"])) for r in R if r["feasible"] == "True" and r["success"] != "True")
    succ = []
    for lvl in ("L0", "L4"):
        s = sorted(int(r["seed"]) for r in R if r["noise"] == lvl and r["success"] == "True")
        succ += [(lvl, x) for x in s[:3]]
    return fails, succ


def gf_panel(ax, sc, P, r):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from gfsem.sim.geometry import from_frame
    tr = r.trajectory
    s_all = sc.track.frenet(tr[:, :2])[0]
    ss = np.linspace(max(sc.track.s_min, s_all.min() - 5), min(sc.track.s_max, max(s_all.max(), sc.s_goal) + 8), 300)
    for d in (sc.track.width / 2, -sc.track.width / 2):
        E = np.stack([sc.track.pose(q, d)[:2] for q in ss])
        ax.plot(E[:, 0], E[:, 1], "k-", lw=1.0)
    gl = np.stack([sc.track.pose(sc.s_goal, d)[:2] for d in (-sc.track.width / 2, sc.track.width / 2)])
    ax.plot(gl[:, 0], gl[:, 1], "g-", lw=2.5, label="goal line")
    for c, rr, cl in zip(sc.centres, sc.radii, sc.classes):
        ax.add_patch(plt.Circle(c, rr, color="tab:blue" if cl == 1 else "tab:orange", zorder=4))
    recs = r.records
    br = [i for i, x in enumerate(recs) if x["brake"]]
    ki = br[0] if br else len(recs) - 1
    L = P.log[ki]
    pose = L["true_pose"]
    flagged = None
    S = L["samples"]
    for q, s in enumerate(S):                              # executed candidates at the key scan
        W = from_frame(s.astype(float), pose)
        ax.plot(W[:, 0], W[:, 1], "-", color="tab:orange", lw=0.3, alpha=0.25, zorder=3)
    if len(L["cur"]):
        W = from_frame(L["cur"], pose)
        ax.plot(W[:, 0], W[:, 1], ".", color="k", ms=2, label="current hits (key scan)", zorder=5)
    if len(L["mem"]):
        W = from_frame(L["mem"], pose)
        ax.plot(W[:, 0], W[:, 1], "s", color="magenta", ms=2.5, mfc="none", label="remembered hits (key scan)", zorder=5)
    seg = np.stack([tr[:-1, :2], tr[1:, :2]], 1)
    lc = LineCollection(seg, cmap="viridis", norm=plt.Normalize(0, 3.5), lw=2.2, zorder=6)
    lc.set_array(tr[:-1, 3])
    ax.add_collection(lc)
    bp = np.array([x["pose"][:2] for x in recs if x["brake"]])
    if len(bp):
        ax.plot(bp[:, 0], bp[:, 1], "x", color="red", ms=5, mew=1.2, label="brake: no safe candidate", zorder=7)
    j = ki
    while j >= 0 and len(recs[j]["path"]) == 0:
        j -= 1
    if j >= 0:
        from gfsem.sim.geometry import poses_from_frame
        Pp = poses_from_frame(np.asarray(recs[j]["path"]), recs[j]["belief"])
        ax.plot(Pp[:, 0], Pp[:, 1], "--", color="tab:red", lw=1.4,
                label="chosen path at key scan" if j == ki else f"last chosen path (t={recs[j]['t']:.1f} s)", zorder=7)
    ax.plot(*pose[:2], "o", color="red", mfc="none", ms=11, mew=1.5, label=f"key scan t={recs[ki]['t']:.1f} s", zorder=8)
    ax.set_aspect("equal")
    lo, hi = tr[:, :2].min(0) - 6, tr[:, :2].max(0) + 6
    ax.set_xlim(lo[0], max(hi[0], gl[:, 0].max() + 3))
    ax.set_ylim(min(lo[1], gl[:, 1].min() - 1), max(hi[1], gl[:, 1].max() + 1))
    ax.set_title(f"GoalFlow (goalflow_l3): {r.outcome} at {r.time:.1f} s, {r.n_brake_plans} brake scans", fontsize=9)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.85)
    return lc


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from diagnose_goalflow_smoke import Rec
    from render_closed_loop import episode, panel
    from gfsem.sim.closed_loop import run_episode
    from gfsem.sim.scene import generate_scene
    fails, succ = pick()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RULE.txt").write_text(
        "Closed-loop stills, baseline (a) | GoalFlow goalflow_l3 (near-observation memory).\n"
        f"Rule: every feasible goalflow_l3 smoke-test failure (seeds 0-39, L0/L4): {fails};\n"
        f"plus the 3 lowest-numbered successful seeds per level: {succ}.\n"
        "Key scan = first brake, else the collision / last scan.\n")
    P = Rec()
    for tag, items in (("fail", fails), ("success", succ)):
        for lvl, seed in items:
            fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
            sc, pa, cap, ra = episode(seed, lvl, "a")
            panel(axs[0], sc, pa, cap, ra, "a", seed)
            r = run_episode(generate_scene(seed), P, lvl, record=True)
            lc = gf_panel(axs[1], sc, P, r)
            fig.colorbar(lc, ax=axs, fraction=0.02, label="speed [m/s]")
            fig.suptitle(f"seed {seed} at {lvl}: baseline (a) vs GoalFlow (goalflow_l3)", fontsize=10)
            fig.savefig(OUT / f"{tag}_seed{seed:02d}_{lvl}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(tag, lvl, seed, "a:", ra.outcome, "gf:", r.outcome, flush=True)


if __name__ == "__main__":
    main()
