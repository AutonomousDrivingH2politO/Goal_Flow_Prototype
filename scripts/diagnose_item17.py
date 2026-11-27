#!/usr/bin/env python
"""DEVIATIONS item 17: route missing vs precision at goalflow_l3's smoke-test stalls (replays only).

    python scripts/diagnose_item17.py out.json

For each stall (L0: 4, 36; L4: 1, 18, 28, 32), at every "no safe candidate" scan: GoalFlow's 3 NMS goals and
their candidates (executed, ego frame); baseline (a)'s route = the side of each true obstacle that (a)'s driven
path passes (side_signs, near = 3 m) over the next 8 m from its point nearest the car; whether some candidate of each
goal follows that route; the best true clearance (Juno's box vs true discs) per goal.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_selection import path_clearance, side_signs  # noqa: E402
from gfsem.planners.goalflow_planner import GoalFlowPlanner  # noqa: E402
from gfsem.sim.closed_loop import run_episode  # noqa: E402
from gfsem.sim.config import SimConfig  # noqa: E402
from gfsem.sim.geometry import poses_to_frame, to_frame  # noqa: E402
from gfsem.sim.raster import rasterize  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402

STALLS = [(4, "L0"), (36, "L0"), (1, "L4"), (18, "L4"), (28, "L4"), (32, "L4")]


class Rec(GoalFlowPlanner):
    def reset(self):
        super().reset()
        self.log = []

    @torch.no_grad()
    def plan(self, obs, scene, true_pose, state):
        k = self.k
        hits = None
        res = super().plan(obs, scene, true_pose, state)
        if res.brake:                              # re-run the identical call (same seed, same memory mask)
            pose_b = np.asarray(obs.pose_odom, float)
            mem = [to_frame(p, pose_b) for _, p in self.mem if len(p)]
            P = np.concatenate([np.asarray(obs.points, float).reshape(-1, 2)] + mem)
            hits = self._mask(P)
            torch.manual_seed(self.seed * 1_000_003 + k)
            bev = torch.from_numpy(rasterize(obs, self.rc))[None].to(self.dev)
            ego = torch.from_numpy(np.asarray(obs.ego_state, np.float32))[None].to(self.dev)
            r = self.model.plan(bev, ego, M=self.M, n_steps=1, goal_K=self.goal_K, safety=True, safety_slack=0.0,
                                safety_hits=hits)
            self.log.append(dict(t=k * self.scan_dt, true_pose=np.asarray(true_pose, float).copy(),
                                 S=r["samples"][0].cpu().numpy(), goal_of=r["goal_of"][0].cpu().numpy(),
                                 goals=r["goals"][0].cpu().numpy(), no_safe=bool(r["no_safe"][0])))
        return res

    def _mask(self, P):
        rc = self.rc
        m = np.zeros((rc.size, rc.size), np.float32)
        if len(P):
            i = np.floor((rc.x_max - P[:, 0]) / rc.res).astype(int)
            j = np.floor((rc.y_max - P[:, 1]) / rc.res).astype(int)
            ok = (i >= 0) & (i < rc.size) & (j >= 0) & (j < rc.size)
            m[i[ok], j[ok]] = 1.0
        return torch.from_numpy(m)[None].to(self.dev)


def main():
    from render_closed_loop import episode
    veh = SimConfig().vehicle
    G = Rec()
    out = []
    for seed, lvl in STALLS:
        sc = generate_scene(seed)
        ra = episode(seed, lvl, "a")[3]
        A = ra.trajectory[:, :3]
        rg = run_episode(sc, G, lvl, record=True)
        O = torch.zeros(1, sc.n, 3)
        scans = []
        for L in G.log:
            if not L["no_safe"]:
                continue
            Ae = poses_to_frame(A, L["true_pose"])
            i0 = int(np.argmin(np.hypot(Ae[:, 0], Ae[:, 1])))
            seg = Ae[i0:, :2]
            s = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
            seg = seg[s <= 8.0]
            Oe = to_frame(sc.centres, L["true_pose"])
            O[0, :, :2] = torch.from_numpy(Oe)
            O[0, :, 2] = torch.from_numpy(sc.radii)
            n = torch.tensor([sc.n])
            a_sig = side_signs(torch.from_numpy(seg[None, None]).float(), O, n)[0, 0]          # (O,)
            S = torch.from_numpy(L["S"]).float()
            c_sig = side_signs(S[None], O, n)[0]                                                 # (M, O)
            need = a_sig != 0
            follows = ((~need[None]) | (c_sig == a_sig[None])).all(-1).numpy()                   # (M,)
            clr = path_clearance(S, O.expand(len(S), -1, -1), n.expand(len(S)), veh).numpy()
            per_goal = []
            for g in range(int(L["goal_of"].max()) + 1):
                m = L["goal_of"] == g
                per_goal.append(dict(goal=[round(float(x), 2) for x in L["goals"][g][:2]],
                                     follows_a_route=bool(follows[m].any()),
                                     best_true_clearance=float(clr[m].max())))
            scans.append(dict(t=L["t"], a_route_obstacles=int(need.sum()),
                              a_route_among_goals=any(p["follows_a_route"] for p in per_goal), per_goal=per_goal,
                              best_clearance_on_a_route=float(clr[follows].max()) if follows.any() else None))
        k = len(scans)
        row = dict(seed=seed, level=lvl, a=ra.outcome, goalflow=rg.outcome, stalled_scans=k,
                   share_scans_a_route_among_goals=float(np.mean([x["a_route_among_goals"] for x in scans])) if k else None,
                   median_best_clearance_any_goal=float(np.median([max(p["best_true_clearance"] for p in x["per_goal"])
                                                                   for x in scans])) if k else None,
                   median_best_clearance_on_a_route=float(np.median([x["best_clearance_on_a_route"] for x in scans
                                                                     if x["best_clearance_on_a_route"] is not None]))
                   if any(x["best_clearance_on_a_route"] is not None for x in scans) else None,
                   scans=scans)
        out.append(row)
        print(json.dumps({k_: v for k_, v in row.items() if k_ != "scans"}), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
