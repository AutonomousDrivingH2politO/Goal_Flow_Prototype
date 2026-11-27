#!/usr/bin/env python
"""The combined run's filter-miss collisions: what happened to the remembered circle? (replays only; no fix)

    python scripts/diagnose_combined_collisions.py [--out reports/lidar_demo/l3_goalflow/combined_collisions]

Replays (deterministic) every `goalflow_l3_combined` collision with cause "safety-filter miss (obstacle in hit memory)",
tracing the memory fix's events (re-anchor = a remembered circle's centre replaced by a current circle within 0.3 m;
drop = removed by the free-ray rule) for circles near the hit obstacle (within 0.6 m in the true world frame). At the
last planning scan before contact, the cause is classified:
  * re-anchored to a different obstacle / to a noise cluster: a remembered circle at the hit obstacle was moved by a
    re-anchor to a centre whose nearest true obstacle surface is another obstacle / none within 0.3 m, and no filter
    circle covers the hit obstacle any more;
  * dropped by the free-ray rule: a remembered circle at the hit obstacle was dropped, none covers it any more;
  * re-anchored to an offset observation: the covering remembered circle's last re-anchor moved it farther from the
    truth, and the filter saw >= 0.12 m while the truth was < 0.12 m;
  * drifted: the covering circle is offset without such a re-anchor (odometry drift / averaging);
  * other: none of the above (e.g. never remembered, current-scan circle offset, filter lookup).
One render per cause (last planning scan: truth, filter circles, chosen path).
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from diagnose_l4_collisions import path_to_circle, render  # noqa: E402
from eval_selection import path_clearance  # noqa: E402
from gfsem.planners.goalflow_planner import GoalFlowPlanner  # noqa: E402
from gfsem.sim.closed_loop import run_episode  # noqa: E402
from gfsem.sim.config import SimConfig  # noqa: E402
from gfsem.sim.geometry import box_circle_clearance, from_frame, to_frame  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402

RUN = Path("reports/lidar_demo/l3_goalflow/full_goalflow_l3_combined/results.csv")


class Rec(GoalFlowPlanner):
    def reset(self):
        super().reset()
        self.log, self.events = [], []

    def _mem_fix(self, obs, C, R, pose, used):
        before = [list(c) for c in self.circ]
        u0 = used.copy()
        super()._mem_fix(obs, C, R, pose, used)
        tp = self._tp
        w = lambda xy: from_frame(to_frame(np.asarray([xy], float), pose), tp)[0]  # noqa: E731
        after = [list(c) for c in self.circ]
        moved = {tuple(np.round(c[:2], 9)) for c in after}
        # replacements: circles whose centre changed to a matched current circle; drops: circles that vanished
        k_after = 0
        for c in before:
            if k_after < len(after) and after[k_after][3] in (c[3], c[3] + 1) and \
                    np.hypot(after[k_after][0] - c[0], after[k_after][1] - c[1]) <= 0.3 + 1e-9:
                if after[k_after][3] == c[3] + 1:
                    self.events.append(dict(k=self.k, kind="reanchor", old=w(c[:2]), new=w(after[k_after][:2])))
                k_after += 1
            else:
                self.events.append(dict(k=self.k, kind="drop", old=w(c[:2])))
        del moved, u0

    @torch.no_grad()
    def plan(self, obs, scene, true_pose, state):
        self._tp = np.asarray(true_pose, float).copy()
        res = super().plan(obs, scene, true_pose, state)
        self.log.append(dict(k=self.k - 1, true_pose=self._tp, filt=np.asarray(self.filter_points, float).reshape(-1, 3).copy(),
                             n_cur=self.n_cur, brake=bool(res.brake),
                             path=None if res.brake else np.asarray(res.path, float)[:, :2].copy(),
                             hits=np.asarray(obs.points, float).reshape(-1, 2).copy()))
        return res


def nearest_obstacle(sc, xy):
    d = np.hypot(*(sc.centres - xy).T) - sc.radii
    i = int(d.argmin())
    return i, float(d[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/l3_goalflow/combined_collisions"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(r["noise"], int(r["seed"])) for r in csv.DictReader(open(RUN)) if r["outcome"] == "collision"
            and r["collision_cause"].startswith("safety-filter miss (obstacle in hit memory)")]
    veh = SimConfig().vehicle
    P = Rec(circle_filter=True, safety_margin=0.12, mem_fix=True, edge_w=1.0)
    out, rendered = [], set()
    for lvl, seed in jobs:
        sc = generate_scene(seed)
        r = run_episode(sc, P, lvl, record=True)
        fp = r.trajectory[-1, :3]
        j = int(box_circle_clearance(fp[None], sc.centres, sc.radii, veh.body_x, veh.width)[0].argmin())
        cj = sc.centres[j]
        near_ev = [e for e in P.events if np.hypot(*(e["old"] - cj)) <= 0.6]
        plans = [L for L in P.log if not L["brake"]]
        L = plans[-1] if plans else P.log[-1]
        T = to_frame(cj[None], L["true_pose"])[0]
        F = L["filt"]
        d = np.hypot(F[:, 0] - T[0], F[:, 1] - T[1]) if len(F) else np.zeros(0)
        rec = dict(level=lvl, seed=seed, outcome=r.outcome, obstacle=j, last_scan=L["k"],
                   events_near_obstacle=[dict(k=e["k"], kind=e["kind"], old_err=round(float(np.hypot(*(e["old"] - cj))), 3),
                                              **({"new_nearest_obstacle": nearest_obstacle(sc, e["new"])[0],
                                                  "new_surface_dist": round(nearest_obstacle(sc, e["new"])[1], 3),
                                                  "new_err": round(float(np.hypot(*(e["new"] - cj))), 3)}
                                                 if e["kind"] == "reanchor" else {})) for e in near_ev])
        if L["path"] is not None:
            O = torch.tensor([[[T[0], T[1], sc.radii[j]]]], dtype=torch.float32)
            rec["path_true_clearance"] = round(float(path_clearance(torch.from_numpy(L["path"][None]).float(), O,
                                                                    torch.tensor([1]), veh)[0]), 3)
        covered = bool(len(d) and d.min() <= 0.6)
        if covered:
            i = int(d.argmin())
            rec.update(circle="memory" if i >= L["n_cur"] else "scan", centre_err=round(float(d[i]), 3))
            if L["path"] is not None:
                rec["path_filter_clearance"] = round(path_to_circle(L["path"], F[i]), 3)
        last_re = [e for e in rec["events_near_obstacle"] if e["kind"] == "reanchor"]
        drops = [e for e in rec["events_near_obstacle"] if e["kind"] == "drop"]
        if not covered:
            moved_away = [e for e in last_re if e["new_err"] > 0.6]
            if moved_away:
                e = moved_away[-1]
                cause = ("re-anchored to a different obstacle" if e["new_surface_dist"] <= 0.3 and e["new_nearest_obstacle"] != j
                         else "re-anchored to a noise cluster")
            elif drops:
                cause = "dropped by the free-ray rule"
            else:
                cause = "other (no circle, never remembered near the obstacle)"
        elif rec.get("path_filter_clearance", 1.0) < 0.12:
            cause = "other (filter lookup: filter clearance < 0.12 m)"
        elif rec["circle"] == "memory" and last_re and last_re[-1]["new_err"] > last_re[-1]["old_err"]:
            cause = "re-anchored to an offset observation"
        elif rec["circle"] == "memory":
            cause = "drifted"
        else:
            cause = "other (current-scan circle offset)"
        rec["cause"] = cause
        if cause not in rendered:
            render(L, sc, j, a.out / f"{cause.split(' (')[0].replace(' ', '_')}_{lvl}_seed{seed}.png")
            rendered.add(cause)
        out.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "events_near_obstacle"}), flush=True)
    c = Counter(x["cause"] for x in out)
    (a.out / "combined_collisions.json").write_text(json.dumps(dict(counts=c, rows=out), indent=1, default=float))
    print(json.dumps(dict(counts=c), indent=1))


if __name__ == "__main__":
    main()
