"""Closed-loop GoalFlow planner (L3 configuration, decided 2026-11-25).

Input: the single-scan BEV raster it was trained on (no tracked obstacles). Every scan: NMS-K3 goals, M = 128
samples, 1 Euler step, candidates rolled out through pure pursuit + bicycle and scored, the safety filter on the
current scan's observed hits (cell_slack 0) drops flagged candidates, shadow path off. If no candidate survives
the filter, brake (the geometric check stays authoritative). The chosen executed path is handed to the same
pure-pursuit controller as the corridor baselines.

Hit memory (safety filter only; ``memory_range`` > 0): observed LiDAR hits kept in a map of 0.2 m cells in the odom
frame (believed, noisy, drifting pose ``obs.pose_odom``; never the true pose). A cell is used once it was hit in
>= ``memory_min_scans`` scans, and only while it lies ahead of the rear bumper and within ``memory_range`` m of the
car (position-based, no time limit); cells behind the bumper are dropped. Current-scan hits always count. Reason: a 0.40 m cone falls below the lowest beam within ~2.03 m of the
sensor, so a single scan loses it right where it matters; the corridor baseline (a) has implicit memory
through its kept plan. The BEV input stays single-scan.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..sim.raster import RasterConfig, rasterize


@dataclass
class GFResult:
    path: np.ndarray
    brake: bool
    speed: float = 3.0
    info: dict = field(default_factory=dict)


class GoalFlowPlanner:
    name = "goalflow_l3"

    def __init__(self, ckpt: Path = Path("runs/l2_full_seed0/goalflow_B_best.pt"), device: str = "auto",
                 M: int = 128, goal_K: int = 3, speed: float = 3.0, seed: int = 0, memory_range: float = 4.0,
                 memory_min_scans: int = 2, memory_cell: float = 0.2, scan_dt: float = 0.1):
        from ..models.common import pick_device
        from ..train import load
        self.dev = pick_device(device)
        self.model = load(Path(ckpt), self.dev).eval()
        self.M, self.goal_K, self.speed, self.seed = M, goal_K, speed, seed
        self.memory_range, self.memory_min_scans, self.memory_cell = memory_range, memory_min_scans, memory_cell
        self.scan_dt = scan_dt
        from ..sim.config import VehicleConfig
        self.rear_x = VehicleConfig.from_yaml().body_x[0]
        self.rc = RasterConfig()
        self.reset()

    def reset(self):
        self.stats: list[tuple[str, float]] = []
        self.k = 0
        self.last = {}
        self.cells: dict[tuple[int, int], list] = {}      # odom cell -> [scans hit, x, y] (believed odom frame)
        self.mem: list[tuple[float, np.ndarray]] = []     # remembered points used at the last scan (diagnostics)

    def _memory_hits(self, obs) -> torch.Tensor:
        """(1, H, W) hit mask: this scan's hits plus remembered cells (hit in >= min scans, ahead of the rear
        bumper, within range), mapped from the believed odom frame into the current ego frame."""
        from ..sim.geometry import from_frame, to_frame
        pose = np.asarray(obs.pose_odom, float)
        cur = np.asarray(obs.points, float).reshape(-1, 2)
        if len(cur):
            W = from_frame(cur, pose)
            keys = set()
            for (x, y) in W:
                k = (int(np.floor(x / self.memory_cell)), int(np.floor(y / self.memory_cell)))
                if k in keys:
                    continue
                keys.add(k)
                c = self.cells.get(k)
                if c is None:
                    self.cells[k] = [1, x, y]
                else:
                    c[0] += 1
        rem = np.zeros((0, 2))
        if self.cells:
            K = list(self.cells)
            A = np.array([self.cells[k] for k in K], float)
            loc = to_frame(A[:, 1:3], pose)
            behind = loc[:, 0] < self.rear_x
            for k in [K[i] for i in np.flatnonzero(behind)]:
                del self.cells[k]
            use = ~behind & (A[:, 0] >= self.memory_min_scans) & (np.hypot(loc[:, 0], loc[:, 1]) <= self.memory_range)
            rem = loc[use]
        self.mem = [(self.k * self.scan_dt, from_frame(rem, pose))] if len(rem) else []
        P = np.concatenate([cur, rem]) if len(rem) else cur
        rc = self.rc
        m = np.zeros((rc.size, rc.size), np.float32)
        if len(P):
            i = np.floor((rc.x_max - P[:, 0]) / rc.res).astype(int)
            j = np.floor((rc.y_max - P[:, 1]) / rc.res).astype(int)
            ok = (i >= 0) & (i < rc.size) & (j >= 0) & (j < rc.size)
            m[i[ok], j[ok]] = 1.0
        return torch.from_numpy(m)[None].to(self.dev)

    @torch.no_grad()
    def plan(self, obs, scene, true_pose, state):
        t0 = time.perf_counter()
        torch.manual_seed(self.seed * 1_000_003 + self.k)
        bev = torch.from_numpy(rasterize(obs, self.rc))[None].to(self.dev)
        ego = torch.from_numpy(np.asarray(obs.ego_state, np.float32))[None].to(self.dev)
        hits = self._memory_hits(obs) if self.memory_range > 0 else None
        r = self.model.plan(bev, ego, M=self.M, n_steps=1, goal_K=self.goal_K, safety=True, safety_slack=0.0,
                            safety_hits=hits)
        no_safe = bool(r["no_safe"][0])
        self.k += 1
        if self.dev.type == "mps":
            torch.mps.synchronize()
        self.last = dict(obs=obs, true_pose=np.asarray(true_pose, float).copy(), no_safe=no_safe)
        if no_safe:
            self.stats.append(("brake", 1000 * (time.perf_counter() - t0)))
            return GFResult(np.zeros((0, 3)), True, 0.0, {"trigger": "brake", "reason": "no safe candidate"})
        xy = r["traj_main"][0].cpu().numpy().astype(np.float64)                  # executed path, ego frame
        P = np.concatenate([np.zeros((1, 2)), xy])
        d = np.diff(P, axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        path = np.concatenate([P, np.r_[yaw[:1], yaw][:, None]], 1)
        self.stats.append(("plan", 1000 * (time.perf_counter() - t0)))
        return GFResult(path, False, self.speed, {"trigger": "plan"})


def collision_cause(planner: GoalFlowPlanner, scene, final_pose, veh, lidar) -> str:
    """Why the episode's collision happened, from the last scan before it: 'no safe candidate' (the planner
    was braking), 'near-field cone' (the hit cone was inside the lowest beam's blind range at the last scan),
    'obstacle unseen this scan' (no hit on it), else 'safety-filter miss' (seen, yet the chosen path hit it)."""
    import math
    from ..sim.geometry import box_circle_clearance, to_frame
    j = int(box_circle_clearance(np.asarray(final_pose, float)[None], scene.centres, scene.radii,
                                 veh.body_x, veh.width)[0].argmin())
    L = planner.last
    if not L:
        return "unknown"
    if L["no_safe"]:
        return "no safe candidate"
    c = to_frame(scene.centres[j:j + 1], L["true_pose"])[0]
    pts = np.asarray(L["obs"].points, float).reshape(-1, 2)
    n_hits = int((np.hypot(*(pts - c).T) <= scene.radii[j] + 0.15).sum()) if len(pts) else 0
    if n_hits == 0 and getattr(planner, "mem", None):
        pose_b = np.asarray(L["obs"].pose_odom, float)
        mp = [to_frame(p, pose_b) for _, p in planner.mem if len(p)]
        if mp and (np.hypot(*(np.concatenate(mp) - c).T) <= scene.radii[j] + 0.25).any():
            return "safety-filter miss (obstacle in hit memory)"
    if n_hits == 0:
        h = float(scene.heights[j])
        reach = (lidar.mount_height - h) / math.tan(abs(math.radians(lidar.v_min_deg - lidar.pitch_deg)))
        rng = math.hypot(c[0] - lidar.mount_x, c[1] - lidar.mount_y) - scene.radii[j]
        if h < lidar.mount_height and rng < reach:
            return "near-field cone"
        return "obstacle unseen this scan"
    return "safety-filter miss"
