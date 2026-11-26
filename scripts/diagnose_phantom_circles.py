#!/usr/bin/env python
"""Phantom circles in goalflow_l3_circles (replays only).

    python scripts/diagnose_phantom_circles.py out.json seed,seed,...   (L4, distance-proportional drift)

At each "no safe candidate" scan: the filter's circles (current scan vs remembered), phantoms = circles with no true
obstacle centre within 0.5 m, sightings of remembered circles, and the circles that alone flag a truly clear
candidate (true clearance >= 0.12 m), split real vs phantom and by sightings (1 vs >= 2).
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
from gfsem.sim.geometry import to_frame  # noqa: E402
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
        if res.brake:
            circ = np.asarray(self.filter_points, float).reshape(-1, 3)
            pose_b = np.asarray(obs.pose_odom, float)
            # remembered circles in the ego frame with their sightings (to tell them apart from this scan's circles)
            mem = [(to_frame(np.array([[c[0], c[1]]]), pose_b)[0], c[3]) for c in self.circ]
            torch.manual_seed(self.seed * 1_000_003 + k)
            bev = torch.from_numpy(rasterize(obs, self.rc))[None].to(self.dev)
            ego = torch.from_numpy(np.asarray(obs.ego_state, np.float32))[None].to(self.dev)
            r = self.model.plan(bev, ego, M=self.M, n_steps=1, goal_K=self.goal_K, safety=True,
                                safety_points=[circ], safety_margin=self.safety_margin)
            self.log.append(dict(k=k, true_pose=np.asarray(true_pose, float).copy(), circ=circ, mem=mem,
                                 S=r["samples"][0].cpu().numpy()))
        return res


def main():
    seeds = [int(s) for s in sys.argv[2].split(",")]
    veh = SimConfig().vehicle
    P = Rec(circle_filter=True, safety_margin=0.12)
    out = []
    for seed in seeds:
        sc = generate_scene(seed)
        r = run_episode(sc, P, "L4", record=True)
        agg = dict(scans=0, circles=0, phantoms=0, blocking=0, blocking_phantom=0, blocking_phantom_1sight=0,
                   blocking_real=0)
        for L in P.log:
            T = to_frame(sc.centres, L["true_pose"])
            C = L["circ"]
            if not len(C):
                continue
            agg["scans"] += 1
            dT = np.hypot(C[:, None, 0] - T[None, :, 0], C[:, None, 1] - T[None, :, 1]).min(1)
            phantom = dT > 0.5
            agg["circles"] += len(C)
            agg["phantoms"] += int(phantom.sum())
            O = torch.zeros(1, sc.n, 3)
            O[0, :, :2] = torch.from_numpy(T)
            O[0, :, 2] = torch.from_numpy(sc.radii)
            S = torch.from_numpy(L["S"]).float()
            ok = path_clearance(S, O.expand(len(S), -1, -1), torch.full((len(S),), sc.n), veh).numpy() >= 0.12
            if not ok.any():
                continue
            for j in range(len(C)):
                if not observed_collision_exact(S[ok][None], [torch.from_numpy(C[j:j + 1]).float()], 0.12)[0].any():
                    continue
                agg["blocking"] += 1
                if phantom[j]:
                    agg["blocking_phantom"] += 1
                    sights = [n for (p, n) in L["mem"] if np.hypot(*(p - C[j, :2])) < 0.05]
                    if sights and sights[0] <= 1:
                        agg["blocking_phantom_1sight"] += 1
                else:
                    agg["blocking_real"] += 1
        out.append(dict(seed=seed, outcome=r.outcome, **agg))
        print(json.dumps(out[-1]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
