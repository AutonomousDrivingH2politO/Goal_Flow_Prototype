#!/usr/bin/env python
"""Replays of the 4 L4 stalls (1, 18, 28, 32) under goalflow_l3_exact (exact box-to-point filter, 0.12 m).

    python scripts/diagnose_exact_filter.py out.json

A. At each "no safe candidate" scan: candidates that truly clear the true obstacles by >= 0.12 m; the hit points
   within 0.12 m of their box (what blocks them), split current vs remembered, and by distance to the nearest true
   obstacle surface: on the surface (<= 0.05 m), displaced (0.05-0.3 m, range noise / drift), clutter (> 0.3 m).
B. Every scan: for each true obstacle within 4 m of the car that has >= 1 current hit, its number of current hits
   and whether it would pass the proposed rule (another hit within 0.3 m in the same scan, or seen in >= 2 scans);
   and how many near-field clutter hits would pass it.
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

SEEDS = (1, 18, 28, 32)


class Rec(GoalFlowPlanner):
    def reset(self):
        super().reset()
        self.log = []

    @torch.no_grad()
    def plan(self, obs, scene, true_pose, state):
        k = self.k
        res = super().plan(obs, scene, true_pose, state)
        cur = np.asarray(obs.points, float).reshape(-1, 2)
        pose_b = np.asarray(obs.pose_odom, float)
        mem = np.concatenate([to_frame(p, pose_b) for _, p in self.mem if len(p)]) if self.mem else np.zeros((0, 2))
        L = dict(t=k * self.scan_dt, cur=cur, mem=mem, true_pose=np.asarray(true_pose, float).copy(), brake=res.brake,
                 cur_world=from_frame(cur, np.asarray(true_pose, float)) if len(cur) else np.zeros((0, 2)))
        if res.brake:
            torch.manual_seed(self.seed * 1_000_003 + k)
            bev = torch.from_numpy(rasterize(obs, self.rc))[None].to(self.dev)
            ego = torch.from_numpy(np.asarray(obs.ego_state, np.float32))[None].to(self.dev)
            r = self.model.plan(bev, ego, M=self.M, n_steps=1, goal_K=self.goal_K, safety=True,
                                safety_points=[self.filter_points], safety_margin=self.safety_margin)
            L["S"] = r["samples"][0].cpu().numpy()
        self.log.append(L)
        return res


def surf_dist(P, C, R):
    if not len(P):
        return np.zeros(0)
    return (np.hypot(P[:, None, 0] - C[None, :, 0], P[:, None, 1] - C[None, :, 1]) - R[None]).min(1)


def main():
    veh = SimConfig().vehicle
    P = Rec(exact_filter=True, safety_margin=0.12)
    out = []
    for seed in SEEDS:
        sc = generate_scene(seed)
        r = run_episode(sc, P, "L4", record=True)
        A = dict(current=dict(on=0, displaced=0, clutter=0), remembered=dict(on=0, displaced=0, clutter=0))
        n_brake = n_clear = 0
        for L in P.log:
            if not L["brake"] or "S" not in L:
                continue
            n_brake += 1
            T = to_frame(sc.centres, L["true_pose"])
            O = torch.zeros(1, sc.n, 3)
            O[0, :, :2] = torch.from_numpy(T)
            O[0, :, 2] = torch.from_numpy(sc.radii)
            S = torch.from_numpy(L["S"]).float()
            clr = path_clearance(S, O.expand(len(S), -1, -1), torch.full((len(S),), sc.n), veh).numpy()
            ok = clr >= 0.12
            if not ok.any():
                continue
            n_clear += 1
            for name, H in (("current", L["cur"]), ("remembered", L["mem"])):
                if not len(H):
                    continue
                Hb = torch.from_numpy(H).float()
                # blocking points: those that alone flag at least one truly clear candidate
                blk = np.zeros(len(H), bool)
                for i in range(len(H)):
                    blk[i] = bool(observed_collision_exact(S[ok][None], [Hb[i:i + 1]], margin=0.12)[0].any())
                d = surf_dist(H[blk], T, sc.radii)
                A[name]["on"] += int((d <= 0.05).sum())
                A[name]["displaced"] += int(((d > 0.05) & (d <= 0.3)).sum())
                A[name]["clutter"] += int((d > 0.3).sum())
        # B: hits per real obstacle within 4 m, and the proposed rule
        prev_world = None
        per_obs, fail_obs, clutter_near, clutter_pass = [], 0, 0, 0
        for L in P.log:
            cur, Wc = L["cur"], L["cur_world"]
            if len(cur):
                T = to_frame(sc.centres, L["true_pose"])
                near_obs = np.flatnonzero(np.hypot(T[:, 0], T[:, 1]) <= 4.0)
                d_hit = np.hypot(cur[:, None, 0] - T[None, :, 0], cur[:, None, 1] - T[None, :, 1]) - sc.radii[None]
                own = d_hit.argmin(1)
                on_obs = d_hit.min(1) <= 0.3
                nb = np.hypot(cur[:, None, 0] - cur[None, :, 0], cur[:, None, 1] - cur[None, :, 1])
                np.fill_diagonal(nb, np.inf)
                has_nb = (nb <= 0.3).any(1)
                seen2 = np.zeros(len(cur), bool)
                if prev_world is not None and len(prev_world):
                    seen2 = (np.hypot(Wc[:, None, 0] - prev_world[None, :, 0],
                                      Wc[:, None, 1] - prev_world[None, :, 1]) <= 0.3).any(1)
                passes = has_nb | seen2
                for j in near_obs:
                    m = on_obs & (own == j)
                    if m.any():
                        per_obs.append(int(m.sum()))
                        fail_obs += int(not passes[m].any())
                rng = np.hypot(cur[:, 0] - 1.6, cur[:, 1])
                cn = (~on_obs) & (rng <= 4.0)
                clutter_near += int(cn.sum())
                clutter_pass += int((cn & passes).sum())
            prev_world = Wc
        row = dict(seed=seed, outcome=r.outcome, brake_scans=n_brake, brake_scans_with_truly_clear=n_clear,
                   blocking_hits=A, near_obstacle_scans=len(per_obs),
                   min_hits_per_near_obstacle=int(min(per_obs)) if per_obs else None,
                   median_hits_per_near_obstacle=float(np.median(per_obs)) if per_obs else None,
                   near_obstacle_scans_failing_rule=fail_obs, near_clutter_hits=clutter_near,
                   near_clutter_hits_passing_rule=clutter_pass)
        out.append(row)
        print(json.dumps(row), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
