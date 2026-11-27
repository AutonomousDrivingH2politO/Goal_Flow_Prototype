#!/usr/bin/env python
"""Where did the blocking remembered points' drift accrue: while stopped or while moving? (replays only)

    python scripts/diagnose_drift_exposure.py out.json

goalflow_l3_exact on the L4 stalls (1, 18, 28, 32). Each remembered cell's storage scan is recorded. At each brake
scan with a truly clear candidate (>= 0.12 m), for every remembered point that alone flags a truly clear candidate:
the scans of drift exposure since storage, split into stopped (ego speed < 0.05 m/s) and moving; drift increments are
i.i.d. per scan in the noise model, so the displacement variance splits in proportion. Seed 1: where the hit
obstacle's remembered points were at the last scan before contact.
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
from gfsem.sim.geometry import box_circle_clearance, to_frame  # noqa: E402
from gfsem.sim.raster import rasterize  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402


class Rec(GoalFlowPlanner):
    def reset(self):
        super().reset()
        self.log, self.stored, self.speeds = [], {}, []

    @torch.no_grad()
    def plan(self, obs, scene, true_pose, state):
        k = self.k
        before = set(self.cells)
        res = super().plan(obs, scene, true_pose, state)
        for key in set(self.cells) - before:
            self.stored[key] = k
        self.speeds.append(float(obs.ego_state[0]))
        pose_b = np.asarray(obs.pose_odom, float)
        used = [(key, c) for key, c in self.cells.items()]
        L = dict(k=k, true_pose=np.asarray(true_pose, float).copy(), brake=res.brake, pose_b=pose_b,
                 mem_keys=[key for key, _ in used], mem_odom=np.array([c[1:3] for _, c in used]) if used else np.zeros((0, 2)))
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
    veh = SimConfig().vehicle
    P = Rec(exact_filter=True, safety_margin=0.12)
    out = []
    for seed in (1, 18, 28, 32):
        sc = generate_scene(seed)
        r = run_episode(sc, P, "L4", record=True)
        stopped = moving = 0
        n_pts = 0
        for L in P.log:
            if "S" not in L or not len(L["mem_odom"]):
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
            ego_pts = to_frame(L["mem_odom"], L["pose_b"])
            for i, key in enumerate(L["mem_keys"]):
                if not observed_collision_exact(S[ok][None], [torch.from_numpy(ego_pts[i:i + 1]).float()], 0.12)[0].any():
                    continue
                k0 = P.stored.get(key, 0)
                sp = np.asarray(P.speeds[k0 + 1:L["k"] + 1])
                stopped += int((sp < 0.05).sum())
                moving += int((sp >= 0.05).sum())
                n_pts += 1
        row = dict(seed=seed, outcome=r.outcome, blocking_remembered_points=n_pts, exposure_scans_stopped=stopped,
                   exposure_scans_moving=moving,
                   share_stopped=stopped / max(stopped + moving, 1))
        if r.outcome == "collision":
            tr = r.trajectory
            j = int(box_circle_clearance(tr[-1:, :3], sc.centres, sc.radii, veh.body_x, veh.width)[0].argmin())
            L = P.log[-1]
            c = to_frame(sc.centres[j:j + 1], L["true_pose"])[0]
            ego_pts = to_frame(L["mem_odom"], L["pose_b"]) if len(L["mem_odom"]) else np.zeros((0, 2))
            d = np.hypot(*(ego_pts - c).T) - sc.radii[j] if len(ego_pts) else np.zeros(0)
            near = d[d < 1.0]
            row["collision"] = dict(obstacle=j, cls=int(sc.classes[j]), remembered_points_within_1m=int(len(near)),
                                    their_surface_distance_median=float(np.median(near)) if len(near) else None,
                                    min=float(near.min()) if len(near) else None)
        out.append(row)
        print(json.dumps(row), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
