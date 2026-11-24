#!/usr/bin/env python
"""Noise comparison L0-L4: preset table + measured effect, per-seed sheets, L4 stall strips/GIFs,
per-level outcome chart.

    python scripts/noise_comparison.py --out ~/Documents/H2politO_L0_viewer/noise
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gfsem.planners.corridor import CorridorConfig, input_from_observation  # noqa: E402
from gfsem.planners.corridor_lattice import LatticeCorridorPlanner  # noqa: E402
from gfsem.sim.closed_loop import CorridorObserved, run_episode  # noqa: E402
from gfsem.sim.config import BOLLARD, NOISE_LEVELS, LidarConfig, SceneConfig, SimConfig  # noqa: E402
from gfsem.sim.geometry import footprint_corners, from_frame, to_frame  # noqa: E402
from gfsem.sim.lidar import scan  # noqa: E402
from gfsem.sim.raster import RasterConfig, rasterize  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#5f5e58", "#e1e0d9"
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 9})


# ---------------------------------------------------------------- 1. table
def measure(seed: int, s_pose: float, n: int = 30) -> dict:
    sc = generate_scene(seed)
    pose = sc.track.pose(s_pose)
    out = {}
    for lvl in LEVELS:
        rows = []
        for k in range(n):
            o = scan(sc, pose, NOISE_LEVELS[lvl], np.random.default_rng([seed, k]))
            P = from_frame(o.points, pose)
            dist = (np.hypot(P[:, None, 0] - sc.centres[None, :, 0], P[:, None, 1] - sc.centres[None, :, 1])
                    - sc.radii[None]).min(1) if len(P) else np.zeros(0)
            clutter = dist > 0.6
            E = from_frame(o.edge_points, pose)
            _, de = sc.track.frenet(E)
            rows.append((len(P), clutter.sum(), np.abs(dist[~clutter]).mean() if (~clutter).any() else 0,
                         len(E), np.abs(np.abs(de) - sc.track.width / 2).mean() if len(E) else 0,
                         np.isnan(o.scan_ranges).sum()))
        r = np.array(rows, float).mean(0)
        out[lvl] = dict(hits=r[0], clutter=r[1], range_err=r[2], edge_pts=r[3], edge_err=r[4], dropped=r[5])
    return out


def table_md(meas: dict, seed: int, s_pose: float) -> str:
    L = ["| Parameter | " + " | ".join(LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    rows = [("Range noise σ [cm]", lambda n: f"{100 * n.range_sigma:.0f}"),
            ("Angular jitter σ [°]", lambda n: f"{n.angle_sigma_deg:.2f}"),
            ("Point dropout [%]", lambda n: f"{100 * n.dropout:.0f}"),
            ("Clutter clusters / scan (mean)", lambda n: f"{n.clutter_rate:.1f}"),
            ("Edge noise σ [cm]", lambda n: f"{100 * n.edge_sigma:.0f}"),
            ("Edge dropouts / side / scan (1–2 m)", lambda n: f"{n.edge_dropout_rate:.1f}"),
            ("Class flip p_mis [%]", lambda n: f"{100 * n.p_mis:.0f}"),
            ("Odom drift / replan, xy σ [cm]", lambda n: f"{100 * n.drift_xy:.1f}"),
            ("Odom drift / replan, yaw σ [°]", lambda n: f"{n.drift_yaw_deg:.2f}")]
    for name, f in rows:
        L.append(f"| {name} | " + " | ".join(f(NOISE_LEVELS[l]) for l in LEVELS) + " |")
    L.append(f"| *Measured, seed {seed} at s = {s_pose:.1f} m (mean of 30 scans):* | | | | | |")
    for name, key, fmt in [("LiDAR hit points", "hits", "{:.0f}"), ("Clutter points", "clutter", "{:.1f}"),
                           ("Mean range error [cm]", "range_err", "{:.1f}"),
                           ("Edge points", "edge_pts", "{:.0f}"), ("Mean edge error [cm]", "edge_err", "{:.1f}"),
                           ("Rays with all returns dropped", "dropped", "{:.0f}")]:
        scale = 100 if "err" in key else 1
        L.append(f"| {name} | " + " | ".join(fmt.format(scale * meas[l][key]) for l in LEVELS) + " |")
    return "\n".join(L)


# ---------------------------------------------------------------- 2. sheets
def find_seeds(max_seed: int = 120):
    """One rows layout, one scatter layout, one with a bollard hidden behind another bollard."""
    rows = scat = hidden = None
    lc = LidarConfig()
    for s in range(max_seed):
        sc = generate_scene(s)
        if not sc.feasible:
            continue
        s_pose = sc.s_field[0] - 3.0
        pose = sc.track.pose(s_pose)
        if hidden is None:
            o = scan(sc, pose, NOISE_LEVELS["L0"], np.random.default_rng(0), lc)
            P = from_frame(o.points, pose)
            loc = to_frame(sc.centres, pose)
            for j in np.nonzero(sc.classes == BOLLARD)[0]:
                x, y = loc[j]
                b = np.degrees(np.arctan2(y, x - lc.mount_x))
                if not (4 < x < 22 and abs(b) < 55):
                    continue
                if len(P) and (np.hypot(*(P - sc.centres[j]).T) < 0.35).any():
                    continue
                # blocked by a nearer bollard on (almost) the same bearing
                for i in np.nonzero(sc.classes == BOLLARD)[0]:
                    xi, yi = loc[i]
                    if i != j and 1.5 < xi < x and abs(np.degrees(np.arctan2(yi, xi - lc.mount_x)) - b) < 2.0:
                        hidden = (s, s_pose)
                        break
                if hidden:
                    break
        if rows is None and sc.layout == "rows" and (hidden is None or hidden[0] != s):
            rows = (s, s_pose)
        if scat is None and sc.layout == "scatter" and (hidden is None or hidden[0] != s):
            scat = (s, s_pose)
        if rows and scat and hidden:
            break
    return [("rows layout", *rows), ("scatter layout", *scat), ("hidden bollard", *hidden)]


def draw_scene_ego(ax, sc, pose, obs, plan=None, lim=(-2, 24, -8, 8)):
    """Ego frame (x forward = up, y left = left)."""
    loc = to_frame(sc.centres, pose)
    for (x, y), r in zip(loc, sc.radii):
        ax.add_patch(plt.Circle((y, x), r, fill=False, ls=":", color="0.55", lw=0.8))
    inp = input_from_observation(obs, CorridorConfig(), SceneConfig.from_yaml())
    for e in (inp.left, inp.right):
        if len(e):
            ax.plot(e[:, 1], e[:, 0], color="#1baf7a", lw=1.0)
    ax.scatter(obs.edge_points[:, 1], obs.edge_points[:, 0], s=2, color="#1baf7a")
    for c, r in zip(inp.centres, inp.radii):
        ax.add_patch(plt.Circle((c[1], c[0]), r, fill=False, color=CRIT, lw=1.2))
    ax.scatter(obs.points[:, 1], obs.points[:, 0], s=3, color=INK, zorder=4)
    if plan is not None and len(plan):
        ax.plot(plan[:, 1], plan[:, 0], color="#a03cc8", lw=1.8, zorder=5)
    ax.add_patch(plt.Polygon(np.array([[0.58, -0.64], [0.58, 2.24], [-0.58, 2.24], [-0.58, -0.64]]),
                             color=BLUE, alpha=0.5))
    ax.set_xlim(lim[3], lim[2])
    ax.set_ylim(lim[0], lim[1])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    return len(inp.centres)


def sheet(label: str, seed: int, s_pose: float, out: Path):
    sc = generate_scene(seed)
    pose = sc.track.pose(s_pose)
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.6), gridspec_kw=dict(height_ratios=[1.35, 1]))
    rc = RasterConfig()
    ext = [rc.y_max, rc.y_min, rc.x_min, rc.x_max]
    for c, lvl in enumerate(LEVELS):
        obs = scan(sc, pose, NOISE_LEVELS[lvl], np.random.default_rng([seed, 99]), speed=3.0)
        pl = LatticeCorridorPlanner(mode="observed")
        res = pl.plan_observation(obs, 3.0, 0.0)
        nfit = draw_scene_ego(axes[0, c], sc, pose, obs, None if res.brake else res.path)
        status = "BRAKE: " + str(res.info.get("reason", ""))[:28] if res.brake else "plan ok"
        axes[0, c].set_title(f"{lvl}: {len(obs.points)} hits, {nfit} fitted (true {sc.n})\n{status}", fontsize=9)
        R = rasterize(obs, rc)
        rgb = np.stack([np.clip(R[0] * 1.0 + R[4] * 0.0, 0, 1),
                        np.clip(0.55 * R[3], 0, 1),
                        np.clip(0.75 * R[2], 0, 1)], -1)
        rgb = np.where(R[0][..., None] > 0, np.array([1.0, 1.0, 1.0]), rgb)
        axes[1, c].imshow(rgb, extent=ext, origin="upper")
        axes[1, c].set_title("BEV: hits (white), drivable (green), unknown (blue)", fontsize=8)
        axes[1, c].tick_params(labelsize=7)
    fig.suptitle(f"Seed {seed} ({label}), same pose s = {s_pose:.1f} m. Top: LiDAR hits (black), fitted "
                 f"obstacles (red), true obstacles (dotted), observed edges (green), observed plan (purple)",
                 fontsize=10)
    fig.tight_layout()
    f = out / f"sheet_seed{seed:03d}_{label.replace(' ', '_')}.png"
    fig.savefig(f, dpi=100)
    plt.close(fig)
    return f


# ---------------------------------------------------------------- 3. L4 stalls
def stall_seeds_verified(results: Path, k: int = 2, max_try: int = 12):
    """Candidates from the evaluation (L3 success, L4 stall), re-run at L4 with the current code;
    returns the first k that still stall."""
    cfg = SimConfig()
    out = []
    for s in stall_seeds(results, k=max_try):
        ep = run_episode(generate_scene(s), CorridorObserved(LatticeCorridorPlanner(mode="observed")), "L4", cfg)
        print(f"  seed {s}: L4 now {ep.outcome}", flush=True)
        if ep.outcome in ("stopped", "timeout"):
            out.append(s)
        if len(out) == k:
            break
    return out


def stall_seeds(results: Path, k: int = 2):
    R = list(csv.DictReader(open(results)))
    obs = {(r["noise"], int(r["seed"])): r for r in R if r["planner"] == "observed" and r["feasible"] == "True"}
    cand = [s for (lvl, s), r in obs.items() if lvl == "L4" and r["outcome"] == "stopped"
            and obs.get(("L3", s), {}).get("success") == "True"]
    cand.sort()
    return cand[:k]


def frame(ax, sc, rec, traj_w, cfg):
    pose = rec["pose"]
    obs = rec["obs"]
    for c, r in zip(sc.centres, sc.radii):
        ax.add_patch(plt.Circle(c, r, color="0.75", zorder=1))
    P = from_frame(obs.points, pose)
    ax.scatter(P[:, 0], P[:, 1], s=3, color=INK, zorder=3)
    inp = input_from_observation(obs, CorridorConfig(), cfg.scene)
    for c, r in zip(from_frame(inp.centres, pose).reshape(-1, 2), inp.radii):
        ax.add_patch(plt.Circle(c, r, fill=False, color=CRIT, lw=1.0, zorder=4))
    E = from_frame(obs.edge_points, pose)
    ax.scatter(E[:, 0], E[:, 1], s=2, color="#1baf7a", zorder=2)
    if len(rec["path"]):
        pw = from_frame(rec["path"][:, :2], pose)
        ax.plot(pw[:, 0], pw[:, 1], color="#a03cc8", lw=1.8, zorder=5)
    ax.plot(traj_w[:, 0], traj_w[:, 1], color=BLUE, lw=1.0, alpha=0.6, zorder=2)
    fp = footprint_corners(pose[None], cfg.vehicle.body_x, cfg.vehicle.width)[0]
    ax.fill(fp[:, 0], fp[:, 1], color=BLUE, alpha=0.6, zorder=6)
    poly = sc.track.polygon()
    ax.plot(*np.vstack([poly, poly[:1]]).T, color="0.3", lw=0.8)
    g = sc.goal_line
    ax.plot(g[:, 0], g[:, 1], "k--", lw=1)
    c = pose[:2] + 8 * np.array([np.cos(pose[2]), np.sin(pose[2])])
    ax.set_xlim(c[0] - 13, c[0] + 13)
    ax.set_ylim(c[1] - 10, c[1] + 10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    info = rec["info"] or {}
    trig = info.get("trigger", "?")
    reason = info.get("reason", "")
    ax.set_title(f"t = {rec['t']:.1f} s  v = {rec.get('v', 0):.1f} m/s\n{trig}"
                 + (f": {reason[:34]}" if rec["brake"] else ""), fontsize=8,
                 color=CRIT if rec["brake"] else INK)


def stall_media(seed: int, out: Path):
    from matplotlib.animation import FuncAnimation, PillowWriter
    cfg = SimConfig()
    sc = generate_scene(seed)
    pl = LatticeCorridorPlanner(mode="observed")
    ep = run_episode(sc, CorridorObserved(pl), "L4", cfg, record=True)
    traj = ep.trajectory
    for r in ep.records:
        k = min(int(round(r["t"] / cfg.control.dt)), len(traj) - 1)
        r["v"] = float(traj[k, 3])
    recs = ep.records
    first_brake = next((i for i, r in enumerate(recs) if r["brake"]), len(recs) - 1)
    lo = max(0, first_brake - 20)
    idx = np.linspace(lo, len(recs) - 1, 8).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, i in zip(axes.ravel(), idx):
        frame(ax, sc, recs[i], traj, cfg)
    fig.suptitle(f"Seed {seed} at L4 (succeeds at L3): {ep.outcome} at {ep.time:.1f} s. Grey: true obstacles; "
                 f"black: LiDAR hits; red: fitted obstacles; purple: plan; blue: car and driven path",
                 fontsize=10)
    fig.tight_layout()
    f1 = out / f"L4_stall_seed{seed:03d}_strip.png"
    fig.savefig(f1, dpi=95)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sub = recs[::3]

    def upd(i):
        ax.clear()
        frame(ax, sc, sub[i], traj, cfg)
        return []
    anim = FuncAnimation(fig, upd, frames=len(sub), blit=False)
    f2 = out / f"L4_stall_seed{seed:03d}.gif"
    anim.save(f2, writer=PillowWriter(fps=5), dpi=70)
    plt.close(fig)
    return f1, f2, ep.outcome, ep.time


# ---------------------------------------------------------------- 4. chart
def outcome_chart(results: Path, out: Path):
    R = list(csv.DictReader(open(results)))
    agg = defaultdict(lambda: defaultdict(list))
    for r in R:
        if r["feasible"] != "True" or r["planner"] not in ("observed", "funnel_obs"):
            continue
        agg[r["planner"]][r["noise"]].append(r)
    names = {"observed": "corridor-lattice, observed", "funnel_obs": "corridor-funnel (original design), observed"}
    cats = [("success", GOOD, lambda r: r["outcome"] == "success"),
            ("stalled", WARN, lambda r: r["outcome"] in ("stopped", "timeout")),
            ("track exit", SERIOUS, lambda r: r["outcome"] == "exit"),
            ("collision", CRIT, lambda r: r["outcome"] == "collision")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    x = np.arange(len(LEVELS))
    for ax, pl in zip(axes[:2], ("observed", "funnel_obs")):
        bottom = np.zeros(len(LEVELS))
        for name, col, f in cats:
            v = np.array([100 * np.mean([f(r) for r in agg[pl][l]]) for l in LEVELS])
            ax.bar(x, v, 0.62, bottom=bottom, color=col, edgecolor=SURF, linewidth=2, label=name)
            if name == "success":
                for xi, vi in zip(x, v):
                    ax.text(xi, vi / 2, f"{vi:.0f}%", ha="center", va="center", color="#ffffff", fontsize=9)
            bottom += v
        ax.set_xticks(x, LEVELS)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of feasible scenes (n = 477)")
        ax.set_title(names[pl], fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(loc="lower left", fontsize=8, frameon=True, facecolor=SURF, edgecolor=GRID)
    ax = axes[2]
    for pl, col, mk in (("observed", BLUE, "o"), ("funnel_obs", ORANGE, "s")):
        v = [np.mean([float(r["flips"]) for r in agg[pl][l]]) for l in LEVELS]
        ax.plot(x, v, color=col, lw=2, marker=mk, ms=7, mec=SURF, mew=2, label=names[pl].split(",")[0])
        ax.text(x[-1] + 0.12, v[-1], f"{v[-1]:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_xticks(x, LEVELS)
    ax.set_ylabel("door flips per run (left/right switches)")
    ax.set_title("Door flips per run", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=True, facecolor=SURF, edgecolor=GRID)
    fig.suptitle("Observed planners by noise level, seeds 0–499 (closed loop, same scenes and controller). "
                 "Evaluation run BEFORE the edge-fill fix; rerun queued", fontsize=10)
    fig.tight_layout()
    f = out / "outcomes_by_level.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / "Documents/H2politO_L0_viewer/noise")
    ap.add_argument("--results", type=Path, default=Path("reports/lidar_demo/l0/results.csv"))
    ap.add_argument("--table-seed", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sc = generate_scene(a.table_seed)
    s_pose = sc.s_field[0] - 3.0
    md = table_md(measure(a.table_seed, s_pose), a.table_seed, s_pose)
    (a.out / "noise_table.md").write_text("# Noise presets L0-L4\n\n" + md + "\n")
    print(md, flush=True)
    print(outcome_chart(a.results, a.out), flush=True)
    for label, seed, sp in find_seeds():
        print(sheet(label, seed, sp, a.out), flush=True)
    for s in stall_seeds_verified(a.results):
        print(stall_media(s, a.out), flush=True)


if __name__ == "__main__":
    main()
