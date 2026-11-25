#!/usr/bin/env python
"""Replay L4 stall seeds with the position-based hit memory: blocking remembered cells inside vs outside the
region the current scan can see (unseen = within 2.1 m of the sensor, outside the 120 deg FOV, or behind the
ray's first return). Replays only.  python scripts/diagnose_memory_visibility.py 1,4,9 out.json"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from gfsem.planners.goalflow_planner import GoalFlowPlanner, unseen_mask  # noqa: E402
from gfsem.sim.closed_loop import run_episode  # noqa: E402
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
            pose = np.asarray(obs.pose_odom, float)
            rem = np.concatenate([to_frame(p, pose) for _, p in self.mem]) if self.mem else np.zeros((0, 2))
            torch.manual_seed(self.seed * 1_000_003 + k)
            bev = torch.from_numpy(rasterize(obs, self.rc))[None].to(self.dev)
            ego = torch.from_numpy(np.asarray(obs.ego_state, np.float32))[None].to(self.dev)
            S = self.model.plan(bev, ego, M=self.M, n_steps=1, goal_K=self.goal_K)["samples"][0].cpu().numpy()
            self.log.append(dict(rem=rem, S=S.reshape(-1, 2), obs=obs))
        return res


def main():
    P = Rec(visibility_gate=False)
    out = []
    for seed in map(int, sys.argv[1].split(",")):
        r = run_episode(generate_scene(seed), P, "L4", record=True)
        blk_un = blk_vis = 0
        for L in P.log:
            R = L["rem"]
            if not len(R):
                continue
            d = np.hypot(R[:, None, 0] - L["S"][None, :, 0], R[:, None, 1] - L["S"][None, :, 1]).min(1)
            B = R[d <= 0.65]
            if len(B):
                u = unseen_mask(B, L["obs"])
                blk_un += int(u.sum())
                blk_vis += int((~u).sum())
        n = max(len(P.log), 1)
        row = dict(seed=seed, outcome=r.outcome, brake_scans=len(P.log),
                   blocking_remembered_unseen_per_scan=blk_un / n, blocking_remembered_visible_per_scan=blk_vis / n)
        out.append(row)
        print(json.dumps(row), flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
