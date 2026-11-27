"""Closed-loop GoalFlow planner (L3 configuration, decided 2026-11-25).

Input: the single-scan BEV raster it was trained on (no tracked obstacles). Every scan: NMS-K3 goals, M = 128
samples, 1 Euler step, candidates rolled out through pure pursuit + bicycle and scored, the safety filter on the
current scan's observed hits (cell_slack 0) drops flagged candidates, shadow path off. If no candidate survives
the filter, brake (the geometric check stays authoritative). The chosen executed path is handed to the same
pure-pursuit controller as the corridor baselines.

Hit memory (safety filter only; ``memory_range`` > 0): observed LiDAR hits kept in a map of 0.2 m cells in the odom
frame (believed, noisy, drifting pose ``obs.pose_odom``; never the true pose). A cell is used once it was hit in
>= ``memory_min_scans`` scans, and only while it lies ahead of the rear bumper and within ``memory_range`` m of the
car (position-based, no time limit); cells behind the bumper are dropped. With ``visibility_gate`` remembered cells
are used only where the current scan cannot see (``unseen_mask``: within 2.1 m of the sensor, outside the FOV, or
behind the first return along their ray); elsewhere the current scan wins. With ``memory_obs_range`` > 0 only hits
observed within that range of the sensor are remembered. Defaults = the variant picked by the pre-set rule
(2026-11-25): near-observation memory (4 m), no visibility gate. Current-scan hits always count. Reason: a 0.40 m cone falls below the lowest beam within ~2.03 m of the
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


def unseen_mask(pts: np.ndarray, obs, near: float = 2.1) -> np.ndarray:
    """(N,) bool: points (ego frame) the current scan cannot see: within ``near`` m of the sensor (blind zone and
    below-beam zone of a 0.40 m cone), outside the horizontal FOV, or behind the first return along their ray
    (occluded; a ray whose returns were all dropped counts as not seeing)."""
    sx, sy = obs.sensor_xy
    dx, dy = pts[:, 0] - sx, pts[:, 1] - sy
    rng = np.hypot(dx, dy)
    b = np.arctan2(dy, dx)
    a0, st = obs.ray_angle0, obs.ray_step
    n = len(obs.scan_ranges)
    idx = np.round((b - a0) / st).astype(int)
    out_fov = (idx < 0) | (idx >= n)
    sr = np.asarray(obs.scan_ranges, float)[np.clip(idx, 0, n - 1)]
    occluded = ~out_fov & (np.isnan(sr) | (rng > sr + 0.3))
    return (rng < near) | out_fov | occluded


class GoalFlowPlanner:
    name = "goalflow_l3"

    def __init__(self, ckpt: Path = Path("runs/l2_full_seed0/goalflow_B_best.pt"), device: str = "auto",
                 M: int = 128, goal_K: int = 3, speed: float = 3.0, seed: int = 0, memory_range: float = 4.0,
                 memory_min_scans: int = 2, memory_cell: float = 0.2, scan_dt: float = 0.1,
                 visibility_gate: bool = False, memory_obs_range: float = 4.0, exact_filter: bool = False,
                 safety_margin: float = 0.12, goal_jitter: float = 0.0, refine_steps: int = 0,
                 circle_filter: bool = False, circle_gate: float = 0.6, filter_off: bool = False,
                 log_scans: bool = False, entry_scans: int = 1, lambda3: float = 0.0, edge_w: float = 0.0,
                 mem_fix: bool = False):
        from ..models.common import pick_device
        from ..train import load
        self.dev = pick_device(device)
        self.model = load(Path(ckpt), self.dev).eval()
        self.M, self.goal_K, self.speed, self.seed = M, goal_K, speed, seed
        self.memory_range, self.memory_min_scans, self.memory_cell = memory_range, memory_min_scans, memory_cell
        self.scan_dt = scan_dt
        self.visibility_gate = visibility_gate
        self.memory_obs_range = memory_obs_range      # > 0: only hits observed within this range of the sensor
        self.exact_filter, self.safety_margin = exact_filter, safety_margin
        self.goal_jitter, self.refine_steps = goal_jitter, refine_steps
        self.circle_filter, self.circle_gate = circle_filter, circle_gate
        # filter_off (ablation, circle filter only): the filter's verdict is computed and logged but not applied;
        # the planner executes the top-ranked candidate and never brakes. log_scans: per-scan record (top-ranked
        # candidate, its filter verdict, the executed path) for true-clearance scoring after the episode.
        self.filter_off, self.log_scans = filter_off, log_scans
        # pre-registered changes (DEVIATIONS 2026-11-27), all off by default: entry_scans = 2: a circle enters the
        # filter only if a circle was also seen within circle_gate in the previous scan (odom frame), near-field
        # circles exempt; lambda3: consistency with the previous chosen path; edge_w: track-edge hinge critic;
        # refine_steps with the circle filter: refinement against the circles.
        self.entry_scans, self.lambda3, self.edge_w = entry_scans, lambda3, edge_w
        # (a2) memory fix: a remembered circle's centre is replaced by the matching current circle (gate 0.3 m), and a
        # remembered circle is dropped when this scan's free rays cover its footprint, only where the lowest beam
        # would hit an obstacle of its class (else the free channel passes over a near-field cone it cannot see)
        self.mem_fix = mem_fix
        from ..sim.config import LidarConfig
        self.lidar = LidarConfig()                       # mount for the visibility guard (eval sets the run's mount)
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
        self.circ: list[list] = []                        # circle memory: [x, y, r, n] in the believed odom frame
        self.last_brake: dict = {}
        self.scan_log: list[dict] = []
        self.prev_C = np.zeros((0, 2))                    # previous scan's cluster centres (odom frame)
        self.prev_path = None                             # previous chosen path (odom frame) for lambda3
        self.n_cur = 0                                    # circles from the current scan (first rows of the filter)

    def _memory_hits(self, obs) -> torch.Tensor:
        """(1, H, W) hit mask: this scan's hits plus remembered cells (hit in >= min scans, ahead of the rear
        bumper, within range), mapped from the believed odom frame into the current ego frame."""
        from ..sim.geometry import from_frame, to_frame
        pose = np.asarray(obs.pose_odom, float)
        cur = np.asarray(obs.points, float).reshape(-1, 2)
        store = cur
        if self.memory_obs_range > 0 and len(cur):       # remember only near-field observations (small drift lever)
            sx, sy = obs.sensor_xy
            store = cur[np.hypot(cur[:, 0] - sx, cur[:, 1] - sy) <= self.memory_obs_range]
        if len(store):
            W = from_frame(store, pose)
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
            if self.visibility_gate and len(rem):          # the current scan wins wherever it can see
                rem = rem[unseen_mask(rem, obs)]
        self.mem = [(self.k * self.scan_dt, from_frame(rem, pose))] if len(rem) else []
        P = np.concatenate([cur, rem]) if len(rem) else cur
        self.filter_points = P                     # exact filter: current hits + remembered stored points
        rc = self.rc
        m = np.zeros((rc.size, rc.size), np.float32)
        if len(P):
            i = np.floor((rc.x_max - P[:, 0]) / rc.res).astype(int)
            j = np.floor((rc.y_max - P[:, 1]) / rc.res).astype(int)
            ok = (i >= 0) & (i < rc.size) & (j >= 0) & (j < rc.size)
            m[i[ok], j[ok]] = 1.0
        return torch.from_numpy(m)[None].to(self.dev)

    def _circles(self, obs) -> np.ndarray:
        """(N, 3) circles (ego frame): this scan's circles (cluster_obstacles, as baseline (a)) plus remembered
        near-field circles (observed within memory_obs_range of the sensor, centre averaged over sightings within
        circle_gate, kept until behind the rear bumper or > memory_range away; believed odom frame)."""
        from ..planners.corridor import CorridorConfig, cluster_obstacles
        from ..sim.config import SceneConfig
        from ..sim.geometry import from_frame, to_frame
        if not hasattr(self, "_ccfg"):
            self._ccfg, self._scfg = CorridorConfig(), SceneConfig.from_yaml()
        pose = np.asarray(obs.pose_odom, float)
        C, R, _ = cluster_obstacles(obs, self._ccfg, self._scfg)
        C = np.asarray(C, float).reshape(-1, 2)
        sx, sy = obs.sensor_xy
        used = np.zeros(len(C), bool)
        if self.mem_fix and self.circ:
            self._mem_fix(obs, C, R, pose, used)
        if len(C):
            near = np.hypot(C[:, 0] - sx, C[:, 1] - sy) <= (self.memory_obs_range or 1e9)
            W = from_frame(C, pose)
            for (x, y), r, nr, u in zip(W, R, near, used):
                if not nr or u:
                    continue
                best, bd = None, self.circle_gate
                for m in self.circ:
                    d = np.hypot(m[0] - x, m[1] - y)
                    if d < bd:
                        best, bd = m, d
                if best is None:
                    self.circ.append([x, y, float(r), 1])
                else:                                   # running mean of the centre over sightings
                    n = best[3]
                    best[0] = (best[0] * n + x) / (n + 1)
                    best[1] = (best[1] * n + y) / (n + 1)
                    best[2] = max(best[2], float(r))
                    best[3] = n + 1
        rem = np.zeros((0, 3))
        if self.circ:
            A = np.array(self.circ, float)
            loc = to_frame(A[:, :2], pose)
            keep = loc[:, 0] >= self.rear_x
            self.circ = [c for c, k_ in zip(self.circ, keep) if k_]
            use = keep & (np.hypot(loc[:, 0], loc[:, 1]) <= self.memory_range)
            rem = np.c_[loc[use], A[use, 2]]
        cur = np.c_[C, np.asarray(R, float)] if len(C) else np.zeros((0, 3))
        if len(C):
            Wc = from_frame(C, pose)
            if self.entry_scans >= 2:                   # 2-scan entry rule (near-field circles exempt)
                seen = (np.hypot(Wc[:, None, 0] - self.prev_C[None, :, 0], Wc[:, None, 1] - self.prev_C[None, :, 1])
                        .min(1) <= self.circle_gate) if len(self.prev_C) else np.zeros(len(C), bool)
                cur = cur[seen | near]
            self.prev_C = Wc
        else:
            self.prev_C = np.zeros((0, 2))
        self.n_cur = len(cur)
        return np.concatenate([cur, rem]) if len(rem) else cur

    def _mem_fix(self, obs, C, R, pose, used):
        """(a2): re-observed remembered circles take the current centre (gate 0.3 m); unseen ones are dropped when the
        free rays of this scan pass beyond them across their whole angular span, where they would be visible."""
        import math
        from ..sim.geometry import from_frame, to_frame
        A = np.array(self.circ, float)
        loc = to_frame(A[:, :2], pose)
        W = from_frame(C, pose) if len(C) else np.zeros((0, 2))
        sx, sy = obs.sensor_xy
        ang, rng = obs.ray_angles, obs.scan_ranges
        lo = -self.lidar.elev_range[0]                   # lowest beam below the horizontal (rad)
        keep = np.ones(len(A), bool)
        for i in range(len(A)):
            if len(C):
                d = np.hypot(W[:, 0] - A[i, 0], W[:, 1] - A[i, 1])
                j = int(d.argmin())
                if d[j] <= 0.3:
                    self.circ[i][0], self.circ[i][1] = float(W[j, 0]), float(W[j, 1])
                    self.circ[i][2] = float(R[j])
                    self.circ[i][3] += 1
                    used[j] = True
                    continue
            r = A[i, 2]
            dx, dy = loc[i, 0] - sx, loc[i, 1] - sy
            dist = math.hypot(dx, dy)
            h_obj = self._scfg.bollard_height if abs(r - self._scfg.bollard_radius) < 1e-6 else self._scfg.cone_height
            reach = (self.lidar.mount_height - h_obj) / math.tan(lo) if lo > 0 else 0.0
            if dist - r <= obs.blind_zone or dist - r < reach:
                continue                                  # near surface inside the blind zone or below the lowest beam
            half = math.asin(min(1.0, r / dist))
            b = math.atan2(dy, dx)
            m = np.abs(ang - b) <= half
            if not m.any() or b - half < ang[0] or b + half > ang[-1]:
                continue                                  # (partly) outside the field of view: keep
            rr = rng[m]
            if np.all(np.isfinite(rr) & (rr > dist + r)):
                keep[i] = False                           # free rays pass beyond its whole footprint
        self.circ = [c for c, k in zip(self.circ, keep) if k]

    @torch.no_grad()
    def plan(self, obs, scene, true_pose, state):
        t0 = time.perf_counter()
        torch.manual_seed(self.seed * 1_000_003 + self.k)
        bev = torch.from_numpy(rasterize(obs, self.rc))[None].to(self.dev)
        ego = torch.from_numpy(np.asarray(obs.ego_state, np.float32))[None].to(self.dev)
        hits = self._memory_hits(obs) if self.memory_range > 0 else None
        if self.circle_filter:
            self.filter_points = self._circles(obs)
            kw = {}
            pose_b = np.asarray(obs.pose_odom, float)
            if self.lambda3 > 0:
                from ..sim.geometry import to_frame
                kw.update(lambdas=(1.0, 0.5, self.lambda3),
                          prev_path=[None if self.prev_path is None else to_frame(self.prev_path, pose_b)])
            if self.edge_w > 0:
                kw.update(edge_w=self.edge_w, edge_points=[_densify_edges(obs)])
            if self.refine_steps > 0:
                kw.update(refine_steps=self.refine_steps)
            r = self.model.plan(bev, ego, M=self.M, n_steps=1, goal_K=self.goal_K, safety=not self.filter_off,
                                safety_points=[self.filter_points], safety_margin=self.safety_margin, **kw)
            if self.filter_off:
                from ..models.goalflow import observed_collision_exact
                r["safe"] = ~observed_collision_exact(r["samples"], [self.filter_points], margin=self.safety_margin)
                r["no_safe"] = torch.zeros(1, dtype=torch.bool)
        elif self.exact_filter:
            if hits is None:
                self.filter_points = np.asarray(obs.points, float).reshape(-1, 2)
            r = self.model.plan(bev, ego, M=self.M, n_steps=1, goal_K=self.goal_K, safety=True,
                                safety_points=[self.filter_points], safety_margin=self.safety_margin,
                                goal_jitter=self.goal_jitter, refine_steps=self.refine_steps)
        else:
            r = self.model.plan(bev, ego, M=self.M, n_steps=1, goal_K=self.goal_K, safety=True, safety_slack=0.0,
                                safety_hits=hits)
        no_safe = bool(r["no_safe"][0])
        if self.log_scans and "safe" in r:
            top = int(r["scores_raw"][0].argmax())
            fr, go = r["scores_raw"][0], r["goal_of"][0]
            gb = torch.stack([fr[go == g].max() for g in torch.unique(go)]).sort(descending=True).values
            spread = float(gb[0] - gb[min(2, len(gb) - 1)])     # best minus 3rd-best goal group (pre-filter score)
            self.scan_log.append(dict(k=self.k, true_pose=np.asarray(true_pose, float).copy(), no_safe=no_safe,
                                      top_rejected=not bool(r["safe"][0, top]), spread=spread,
                                      top=r["samples"][0, top].cpu().numpy(),
                                      exe=None if no_safe else r["traj_main"][0].cpu().numpy()))
        self.k += 1
        if self.dev.type == "mps":
            torch.mps.synchronize()
        self.last = dict(obs=obs, true_pose=np.asarray(true_pose, float).copy(), no_safe=no_safe)
        if no_safe:                                  # for stall diagnostics (logged at episode end)
            self.last_brake = dict(S=r["samples"][0].cpu().numpy(), true_pose=self.last["true_pose"],
                                   filt=np.asarray(getattr(self, "filter_points", np.zeros((0, 2))), float))
        if no_safe:
            self.prev_path = None
            self.stats.append(("brake", 1000 * (time.perf_counter() - t0)))
            return GFResult(np.zeros((0, 3)), True, 0.0, {"trigger": "brake", "reason": "no safe candidate"})
        xy = r["traj_main"][0].cpu().numpy().astype(np.float64)                  # executed path, ego frame
        if self.lambda3 > 0:
            from ..sim.geometry import from_frame
            self.prev_path = from_frame(xy, np.asarray(obs.pose_odom, float))
        P = np.concatenate([np.zeros((1, 2)), xy])
        d = np.diff(P, axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        path = np.concatenate([P, np.r_[yaw[:1], yaw][:, None]], 1)
        self.stats.append(("plan", 1000 * (time.perf_counter() - t0)))
        return GFResult(path, False, self.speed, {"trigger": "plan"})


def _densify_edges(obs, step: float = 0.1, max_gap: float = 2.0) -> np.ndarray:
    """Observed road-edge points (ego frame) per side, sorted along x and linearly densified to ``step`` across gaps up
    to ``max_gap`` m, so the box cannot slip between sparse edge samples."""
    P = np.asarray(obs.edge_points, float).reshape(-1, 2)
    S_ = np.asarray(obs.edge_side).reshape(-1)
    out = []
    for sd in np.unique(S_):
        Q = P[S_ == sd]
        Q = Q[np.argsort(Q[:, 0])]
        out.append(Q)
        for a, b in zip(Q[:-1], Q[1:]):
            g = float(np.hypot(*(b - a)))
            if step < g <= max_gap:
                t = np.arange(step, g, step)[:, None] / g
                out.append(a + t * (b - a))
    return np.concatenate(out) if out else np.zeros((0, 2))


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
