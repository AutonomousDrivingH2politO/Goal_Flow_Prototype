#!/usr/bin/env python
"""Replay diagnostics for the closed-loop GoalFlow smoke test (hit memory 1.5 s). No new runs: replays only.

    python scripts/diagnose_goalflow_smoke.py out.json

1. Cone collisions (4 L0, 12 L4, 23 L4): last scan with a hit on the cone, time / speed / distance since,
   bearing and range at the last sighting, why it vanished at the next scan.
2. "No safe candidate" stalls (17 L0; 17, 28, 32, 36 L4): in the stalled scans, the hits that block candidates
   (within 0.65 m of a candidate's executed path), split into current vs remembered, and clutter (no true
   obstacle within 0.3 m, true-pose ego frame).
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from gfsem.planners.goalflow_planner import GoalFlowPlanner  # noqa: E402
from gfsem.sim.closed_loop import run_episode  # noqa: E402
from gfsem.sim.config import SimConfig  # noqa: E402
from gfsem.sim.geometry import box_circle_clearance, from_frame, to_frame  # noqa: E402
from gfsem.sim.raster import rasterize  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402


class Rec(GoalFlowPlanner):
    """Same planner; records per scan the candidates and the hits (current / remembered) the filter used."""

    def reset(self):
        super().reset()
        self.log = []

    @torch.no_grad()
    def plan(self, obs, scene, true_pose, state):
        pose_b = np.asarray(obs.pose_odom, float)
        mem_ego = [to_frame(p, pose_b) for t, p in self.mem
                   if self.k * self.scan_dt - t <= self.hit_memory_s + 1e-9 and len(p)]
        mem_ego = np.concatenate(mem_ego) if mem_ego else np.zeros((0, 2))
        k = self.k
        torch.manual_seed(self.seed * 1_000_003 + k)
        res = super().plan(obs, scene, true_pose, state)
        # candidates: re-run the identical model call (same seed) to get the samples
        torch.manual_seed(self.seed * 1_000_003 + k)
        bev = torch.from_numpy(rasterize(obs, self.rc))[None].to(self.dev)
        ego = torch.from_numpy(np.asarray(obs.ego_state, np.float32))[None].to(self.dev)
        hits = self._mask(np.concatenate([np.asarray(obs.points, float).reshape(-1, 2), mem_ego]))
        r = self.model.plan(bev, ego, M=self.M, n_steps=1, goal_K=self.goal_K, safety=True, safety_slack=0.0,
                            safety_hits=hits)
        self.log.append(dict(t=k * self.scan_dt, cur=np.asarray(obs.points, float).reshape(-1, 2), mem=mem_ego,
                             samples=r["samples"][0].cpu().numpy(), no_safe=bool(r["no_safe"][0]),
                             true_pose=np.asarray(true_pose, float).copy(), v=float(state.v)))
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


def cone_case(P, sc, r, lid, veh):
    tr = r.trajectory
    j = int(box_circle_clearance(tr[-1:, :3], sc.centres, sc.radii, veh.body_x, veh.width)[0].argmin())
    sx = np.array([lid.mount_x, lid.mount_y])
    last = None
    for i, L in enumerate(P.log):
        c = to_frame(sc.centres[j:j + 1], L["true_pose"])[0]
        if len(L["cur"]) and (np.hypot(*(L["cur"] - c).T) <= sc.radii[j] + 0.15).any():
            last = i
    out = dict(cone=j, height=float(sc.heights[j]), radius=float(sc.radii[j]), contact_t=float(r.time))
    if last is None:
        out["last_seen"] = None
        return out
    L = P.log[last]
    c = to_frame(sc.centres[j:j + 1], L["true_pose"])[0]
    k0 = int(round(L["t"] / 0.05))
    dist = float(np.hypot(*np.diff(tr[k0:, :2], axis=0).T).sum())
    out.update(last_seen_t=L["t"], time_since=float(r.time - L["t"]), speed_at_last=L["v"],
               mean_speed_since=dist / max(r.time - L["t"], 1e-6), distance_since=dist,
               bearing_deg=float(math.degrees(math.atan2(c[1] - sx[1], c[0] - sx[0]))),
               range_surface=float(np.hypot(*(c - sx)) - sc.radii[j]))
    if last + 1 < len(P.log):
        n = P.log[last + 1]
        cn = to_frame(sc.centres[j:j + 1], n["true_pose"])[0]
        b = math.degrees(math.atan2(cn[1] - sx[1], cn[0] - sx[0]))
        rng = float(np.hypot(*(cn - sx)) - sc.radii[j])
        reach = (lid.mount_height - sc.heights[j]) / math.tan(math.radians(abs(lid.v_min_deg - lid.pitch_deg)))
        why = ("out of the 120 deg FOV" if abs(b) > lid.hfov_deg / 2 else "blind zone (< 1.5 m)" if rng < lid.blind_zone
               else "below the lowest beam" if sc.heights[j] < lid.mount_height and rng < reach
               else "in view but no return (occlusion / dropout)")
        out.update(next_bearing_deg=b, next_range_surface=rng, vanished_because=why, lowest_beam_reach=float(reach))
    out["in_memory_window_at_contact"] = out["time_since"] <= P.hit_memory_s
    return out


def stall_case(P, sc):
    rows = []
    for L in P.log:
        if not L["no_safe"]:
            continue
        S = L["samples"].reshape(-1, 2)                         # all executed candidate points
        T = to_frame(sc.centres, L["true_pose"]) if sc.n else np.zeros((0, 2))
        res = {}
        for name, H in (("current", L["cur"]), ("remembered", L["mem"])):
            if not len(H):
                res[name] = dict(blocking=0, clutter=0)
                continue
            d = np.hypot(H[:, None, 0] - S[None, :, 0], H[:, None, 1] - S[None, :, 1]).min(1)
            blk = H[d <= 0.65]
            if len(T) and len(blk):
                dt = np.hypot(blk[:, None, 0] - T[None, :, 0], blk[:, None, 1] - T[None, :, 1]) - sc.radii[None]
                cl = int((dt.min(1) > 0.3).sum())
            else:
                cl = int(len(blk))
            res[name] = dict(blocking=int(len(blk)), clutter=cl, total=int(len(H)))
        rows.append(dict(t=L["t"], **{f"{k}_{kk}": v for k, d in res.items() for kk, v in d.items()}))
    agg = {}
    for key in ("current_blocking", "current_clutter", "remembered_blocking", "remembered_clutter", "remembered_total",
                "current_total"):
        vals = [x.get(key, 0) for x in rows]
        agg[key] = float(np.mean(vals)) if vals else None
    return dict(stalled_scans=len(rows), mean_per_scan=agg)


def main():
    cfg = SimConfig()
    P = Rec()
    out = {"cones": [], "stalls": []}
    for seed, lvl in ((4, "L0"), (12, "L4"), (23, "L4")):
        sc = generate_scene(seed)
        r = run_episode(sc, P, lvl, record=True)
        row = dict(seed=seed, level=lvl, outcome=r.outcome, **cone_case(P, sc, r, cfg.lidar, cfg.vehicle))
        out["cones"].append(row)
        print(json.dumps(row, default=float), flush=True)
    for seed, lvl in ((17, "L0"), (17, "L4"), (28, "L4"), (32, "L4"), (36, "L4")):
        sc = generate_scene(seed)
        r = run_episode(sc, P, lvl, record=True)
        row = dict(seed=seed, level=lvl, outcome=r.outcome, **stall_case(P, sc))
        out["stalls"].append(row)
        print(json.dumps(row, default=float), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
