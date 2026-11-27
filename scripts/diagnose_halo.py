#!/usr/bin/env python
"""Halo hypothesis: do raw-hit scatter (L4 range noise) plus the memory union close tight gaps? Replays only.

    python scripts/diagnose_halo.py out.json [seeds]

goalflow_l3_exact, L4, distance-proportional drift. At each "no safe candidate" scan with a truly clear candidate
(true clearance >= 0.12 m): every filter point (current hit or remembered point) that alone flags a truly clear
candidate. Per blocking point: current vs remembered; protrusion = signed distance to the nearest true obstacle surface
(positive = outside the disc, i.e. sticking out into free space / the gap; negative = inside the disc); the number of
scans so far with a current hit within 0.3 m of that obstacle's surface; and its rank among all filter points of that
obstacle (outermost decile or not).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_selection import path_clearance  # noqa: E402
from gfsem.models.goalflow import observed_collision_exact  # noqa: E402
from gfsem.planners.goalflow_planner import GoalFlowPlanner  # noqa: E402
from gfsem.sim.closed_loop import run_episode  # noqa: E402
from gfsem.sim.config import SimConfig  # noqa: E402
from gfsem.sim.geometry import from_frame, to_frame  # noqa: E402
from gfsem.sim.raster import rasterize  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402


class Rec(GoalFlowPlanner):
    def reset(self):
        super().reset()
        self.log = []

    @torch.no_grad()
    def plan(self, obs, scene, true_pose, state):
        k = self.k
        res = super().plan(obs, scene, true_pose, state)
        tp = np.asarray(true_pose, float).copy()
        cur = np.asarray(obs.points, float).reshape(-1, 2)
        pose_b = np.asarray(obs.pose_odom, float)
        mem = np.concatenate([to_frame(p, pose_b) for _, p in self.mem if len(p)]) if self.mem else np.zeros((0, 2))
        L = dict(k=k, true_pose=tp, brake=res.brake, cur=cur, mem=mem,
                 cur_world=from_frame(cur, tp) if len(cur) else np.zeros((0, 2)))
        if res.brake:
            torch.manual_seed(self.seed * 1_000_003 + k)
            bev = torch.from_numpy(rasterize(obs, self.rc))[None].to(self.dev)
            ego = torch.from_numpy(np.asarray(obs.ego_state, np.float32))[None].to(self.dev)
            r = self.model.plan(bev, ego, M=self.M, n_steps=1, goal_K=self.goal_K, safety=True,
                                safety_points=[self.filter_points], safety_margin=self.safety_margin)
            L["S"] = r["samples"][0].cpu().numpy()
        self.log.append(L)
        return res


def main():
    seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "1,13,21,29,35").split(",")]
    veh = SimConfig().vehicle
    P = Rec(exact_filter=True, safety_margin=0.12)
    out = []
    for seed in seeds:
        sc = generate_scene(seed)
        r = run_episode(sc, P, "L4", record=True)                 # distance-proportional drift (default)
        rows = []
        for i, L in enumerate(P.log):
            if "S" not in L:
                continue
            T = to_frame(sc.centres, L["true_pose"])
            O = torch.zeros(1, sc.n, 3)
            O[0, :, :2] = torch.from_numpy(T)
            O[0, :, 2] = torch.from_numpy(sc.radii)
            S = torch.from_numpy(L["S"]).float()
            clr = path_clearance(S, O.expand(len(S), -1, -1), torch.full((len(S),), sc.n), veh).numpy()
            ok = clr >= 0.12
            if not ok.any():
                continue
            pts = np.concatenate([L["cur"], L["mem"]]) if len(L["mem"]) else L["cur"]
            src = np.r_[np.zeros(len(L["cur"]), int), np.ones(len(L["mem"]), int)]
            if not len(pts):
                continue
            dsurf = np.hypot(pts[:, None, 0] - T[None, :, 0], pts[:, None, 1] - T[None, :, 1]) - sc.radii[None]
            own, prot = dsurf.argmin(1), dsurf.min(1)
            Sok = S[ok][None]
            for j in range(len(pts)):
                if not observed_collision_exact(Sok, [torch.from_numpy(pts[j:j + 1]).float()], 0.12)[0].any():
                    continue
                o = int(own[j])
                mine = (own == o) & (prot <= 0.5)
                rank = float((prot[mine] < prot[j]).mean()) if mine.sum() > 1 else 1.0
                n_scans = 0
                for M_ in P.log[:i + 1]:
                    W = M_["cur_world"]
                    if len(W) and ((np.hypot(*(W - sc.centres[o]).T) - sc.radii[o]) <= 0.3).any():
                        n_scans += 1
                rows.append(dict(t=L["k"] * 0.1, source="remembered" if src[j] else "current",
                                 protrusion=float(prot[j]), obstacle=o, scans_contributing=n_scans,
                                 rank_in_obstacle_scatter=rank, outermost_decile=rank >= 0.9,
                                 obstacle_points=int(mine.sum()),
                                 obstacle_median_protrusion=float(np.median(prot[mine])) if mine.any() else None))
        rr = rows
        summ = dict(seed=seed, outcome=r.outcome, blocking_points=len(rr))
        if rr:
            pr = np.array([x["protrusion"] for x in rr])
            summ.update(share_remembered=float(np.mean([x["source"] == "remembered" for x in rr])),
                        protrusion_median=float(np.median(pr)), protrusion_p10=float(np.percentile(pr, 10)),
                        protrusion_p90=float(np.percentile(pr, 90)),
                        share_outermost_decile=float(np.mean([x["outermost_decile"] for x in rr])),
                        scans_contributing_median=float(np.median([x["scans_contributing"] for x in rr])),
                        obstacle_points_median=float(np.median([x["obstacle_points"] for x in rr])),
                        obstacle_median_protrusion=float(np.median([x["obstacle_median_protrusion"] for x in rr
                                                                    if x["obstacle_median_protrusion"] is not None])))
        out.append(dict(summary=summ, rows=rr))
        print(json.dumps(summ), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
