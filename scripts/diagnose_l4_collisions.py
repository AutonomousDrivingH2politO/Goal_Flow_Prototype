#!/usr/bin/env python
"""The full goalflow_l3_circles run's two L4 collisions ("safety-filter miss, obstacle in hit memory"): why did the
remembered obstacle not block? Replays only; diagnose and report, no fix.

    python scripts/diagnose_l4_collisions.py [--out reports/lidar_demo/l3_goalflow/l4_collisions]

Per collision: the hit obstacle; for the last 15 scans: whether a filter circle covers it (centre within 0.6 m), that
circle's source (this scan / memory), centre error and radius vs truth, the chosen executed path's clearance to that
circle as the filter sees it vs its true clearance to the obstacle. Classification: memory position drift (a circle
was there but offset so the filter saw >= 0.12 m while the truth was < 0.12 m), radius (centre fine, radius too small),
filter lookup (circle there and the filter's own distance < 0.12 m, yet the path was taken: a bug), absent (no circle
for the obstacle in the filter). A render of the last scan before contact per collision.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_selection import path_clearance  # noqa: E402
from gfsem.planners.goalflow_planner import GoalFlowPlanner  # noqa: E402
from gfsem.sim.closed_loop import run_episode  # noqa: E402
from gfsem.sim.config import SimConfig  # noqa: E402
from gfsem.sim.geometry import box_circle_clearance, to_frame  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402

FULL = Path("reports/lidar_demo/l3_goalflow/full_goalflow_l3_circles/results.csv")


class Rec(GoalFlowPlanner):
    def reset(self):
        super().reset()
        self.log = []

    @torch.no_grad()
    def plan(self, obs, scene, true_pose, state):
        res = super().plan(obs, scene, true_pose, state)
        self.log.append(dict(k=self.k - 1, true_pose=np.asarray(true_pose, float).copy(),
                             filt=np.asarray(self.filter_points, float).reshape(-1, 3).copy(), n_cur=self.n_cur,
                             brake=bool(res.brake), path=None if res.brake else np.asarray(res.path, float)[:, :2].copy(),
                             hits=np.asarray(obs.points, float).reshape(-1, 2).copy()))
        return res


def path_to_circle(path_xy, c):
    """Clearance of Juno's box along the executed path to one circle (x, y, r), as the filter computes it."""
    veh = SimConfig().vehicle
    O = torch.tensor([[[c[0], c[1], c[2]]]], dtype=torch.float32)
    return float(path_clearance(torch.from_numpy(path_xy[None]).float(), O, torch.tensor([1]), veh)[0])


def render(L, sc, j, f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    veh = SimConfig().vehicle
    T = to_frame(sc.centres, L["true_pose"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (c, r) in enumerate(zip(T, sc.radii)):
        ax.add_patch(plt.Circle(c, r, fill=False, color="black" if i == j else "0.6", lw=2 if i == j else 1))
    for i, c in enumerate(L["filt"]):
        ax.add_patch(plt.Circle(c[:2], c[2], fill=False, ls="--", color="tab:red" if i < L["n_cur"] else "tab:orange"))
    if len(L["hits"]):
        ax.plot(L["hits"][:, 0], L["hits"][:, 1], ".", ms=2, color="tab:blue")
    if L["path"] is not None:
        ax.plot(L["path"][:, 0], L["path"][:, 1], "-", color="tab:green")
    x0, x1 = veh.body_x
    h = veh.width / 2
    ax.plot([x0, x1, x1, x0, x0], [-h, -h, h, h, -h], color="tab:green")
    ax.set_xlim(-2, 10)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title(f"seed {sc.seed} L4, scan {L['k']}: black = hit obstacle (truth), red/orange dashed = filter circles "
                 f"(scan / memory), green = chosen path", fontsize=7)
    fig.savefig(f, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/l3_goalflow/l4_collisions"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(r["seed"]) for r in csv.DictReader(open(FULL)) if r["noise"] == "L4" and r["outcome"] == "collision"]
    veh = SimConfig().vehicle
    P = Rec(circle_filter=True, safety_margin=0.12)
    out = []
    for seed in seeds:
        sc = generate_scene(seed)
        r = run_episode(sc, P, "L4", record=True)
        fp = r.trajectory[-1, :3]
        j = int(box_circle_clearance(fp[None], sc.centres, sc.radii, veh.body_x, veh.width)[0].argmin())
        rows = []
        for L in P.log[-15:]:
            T = to_frame(sc.centres[j:j + 1], L["true_pose"])[0]
            F = L["filt"]
            d = np.hypot(F[:, 0] - T[0], F[:, 1] - T[1]) if len(F) else np.zeros(0)
            row = dict(scan=L["k"], brake=L["brake"], obstacle_ego=[round(float(T[0]), 2), round(float(T[1]), 2)])
            if L["path"] is not None:
                O = torch.tensor([[[T[0], T[1], sc.radii[j]]]], dtype=torch.float32)
                row["path_true_clearance"] = round(float(path_clearance(torch.from_numpy(L["path"][None]).float(), O,
                                                                        torch.tensor([1]), veh)[0]), 3)
            if len(d) and d.min() <= 0.6:
                i = int(d.argmin())
                row.update(circle="memory" if i >= L["n_cur"] else "scan", centre_err=round(float(d[i]), 3),
                           radius=round(float(F[i, 2]), 3), true_radius=round(float(sc.radii[j]), 3))
                if L["path"] is not None:
                    row["path_filter_clearance"] = round(path_to_circle(L["path"], F[i]), 3)
            else:
                row["circle"] = "absent"
            rows.append(row)
        last = [x for x in rows if not x["brake"]][-1] if any(not x["brake"] for x in rows) else rows[-1]
        if last["circle"] == "absent":
            cause = "absent (no filter circle for the obstacle at the last planning scan)"
        elif last.get("path_filter_clearance", 1) < 0.12:
            cause = "filter lookup (the filter's own clearance < 0.12 m, yet the path was taken)"
        elif last["radius"] + 1e-6 < last["true_radius"] and last["centre_err"] < 0.05:
            cause = "radius (circle radius below the true radius)"
        else:
            cause = "memory position drift (circle offset: the filter saw >= 0.12 m, the truth was < 0.12 m)"
        render(P.log[-1], sc, j, a.out / f"seed_{seed}_L4.png")
        out.append(dict(seed=seed, outcome=r.outcome, obstacle=j, cause=cause, last_scans=rows))
        print(json.dumps(dict(seed=seed, outcome=r.outcome, cause=cause, last=last)), flush=True)
    (a.out / "l4_collisions.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
