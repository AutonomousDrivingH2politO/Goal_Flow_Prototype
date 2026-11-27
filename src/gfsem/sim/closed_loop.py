"""Closed-loop episodes: LiDAR -> planner -> pure pursuit -> kinematic bicycle, with metrics.

Every planner gets the same scene seed, the same noise RNG streams and the same controller.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .config import NOISE_LEVELS, NoiseLevel, SimConfig
from .geometry import box_circle_clearance, footprint_corners, poses_from_frame, wrap
from .lidar import Observation, privileged_view, scan
from .scene import Scene
from .vehicle import VehicleState, pure_pursuit, step


class Planner:
    """Interface. ``plan`` returns a PlanResult-like object with .path (ego frame, (K,3)),
    .brake and .speed."""
    name = "planner"

    def reset(self) -> None: ...

    def plan(self, obs: Observation, scene: Scene, true_pose: np.ndarray, state: VehicleState): ...


class CorridorPrivileged(Planner):
    name = "corridor_privileged"

    def __init__(self, planner, name: str | None = None):
        self.p = planner
        if name:
            self.name = name

    def reset(self):
        self.p.reset()

    def plan(self, obs, scene, true_pose, state):
        from ..planners.corridor import input_from_privileged
        inp = input_from_privileged(privileged_view(scene, true_pose))
        inp.pose_odom = true_pose
        return self.p.plan(inp, state.v, state.steer)


class CorridorObserved(Planner):
    name = "corridor_observed"

    def __init__(self, planner, name: str | None = None):
        self.p = planner
        if name:
            self.name = name

    def reset(self):
        self.p.reset()

    def plan(self, obs, scene, true_pose, state):
        return self.p.plan_observation(obs, state.v, state.steer)


@dataclass
class EpisodeResult:
    seed: int
    planner: str
    noise: str
    feasible: bool
    outcome: str                 # success | collision | exit | stopped | timeout
    time: float
    min_clearance: float
    flips: int
    peak_lat_acc: float
    peak_lat_jerk: float
    n_replans: int
    n_brake_plans: int
    plan_ms: float
    trajectory: np.ndarray | None = None
    records: list = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.outcome == "success"

    @property
    def correct_brake(self) -> bool:
        return (not self.feasible) and self.outcome in ("stopped", "timeout")

    def row(self) -> dict:
        d = {k: getattr(self, k) for k in ("seed", "planner", "noise", "feasible", "outcome", "time",
                                            "min_clearance", "flips", "peak_lat_acc", "peak_lat_jerk",
                                            "n_replans", "n_brake_plans", "plan_ms")}
        d["success"] = self.success
        d["correct_brake"] = self.correct_brake
        return d


def side_signature(path_w: np.ndarray, centres: np.ndarray, s_obs: np.ndarray, s_ego: float,
                   look: float = 15.0, near: float = 3.0) -> dict[int, int]:
    """Which side the planned path passes each obstacle ahead: +1 obstacle on the path's left."""
    out: dict[int, int] = {}
    if len(path_w) < 3 or len(centres) == 0:
        return out
    P = path_w[:, :2]
    cand = np.nonzero((s_obs > s_ego) & (s_obs < s_ego + look))[0]
    for j in cand:
        d = np.hypot(*(P - centres[j]).T)
        i = int(np.argmin(d))
        if d[i] > near or i == 0 or i >= len(P) - 1:
            continue
        t = P[i + 1] - P[i - 1]
        v = centres[j] - P[i]
        out[int(j)] = 1 if t[0] * v[1] - t[1] * v[0] > 0 else -1
    return out


def run_episode(scene: Scene, planner: Planner, noise: NoiseLevel | str = "L0", cfg: SimConfig | None = None,
                noise_seed: int | None = None, record: bool = False, start=(0.0, 0.0, 0.0),
                drift_per_distance: bool = True, drift_ref_ds: float = 0.30) -> EpisodeResult:
    """``drift_per_distance`` (default, 2026-11-26): each per-replan odometry-drift increment is scaled by
    sqrt(ds / ``drift_ref_ds``) with ds the distance driven since the previous replan, so drift variance grows with
    distance; 0.30 m = one replan at 3 m/s, where the drift equals the old time-based value. False = the old
    time-based random walk (same draws, unscaled)."""
    cfg = cfg or SimConfig()
    noise = NOISE_LEVELS[noise] if isinstance(noise, str) else noise
    veh, ctl, track = cfg.vehicle, cfg.control, scene.track
    seed = scene.seed if noise_seed is None else noise_seed
    rng_scan = np.random.default_rng([seed, 1])
    rng_drift = np.random.default_rng([seed, 2])
    planner.reset()

    p0 = track.pose(*start)
    juno = None
    if ctl.speed_mode == "juno_bangbang":
        from .vehicle import JunoSpeed
        jc = ctl.juno
        assert jc.a_open <= veh.max_accel and jc.d_brake <= veh.max_decel, "Juno placeholders exceed vehicle limits"
        juno = JunoSpeed(jc, ctl.dt)
    else:
        assert ctl.speed_mode == "constant", ctl.speed_mode
    st = VehicleState(p0[0], p0[1], p0[2], cfg.scene.spawn_speed if juno is None else ctl.juno.v_hi, 0.0)
    drift = np.zeros(3)
    last_replan_xy = None
    path_odom = np.zeros((0, 3))
    idx = 0
    v_cmd = cfg.scene.target_speed
    brake = False
    s_obs = track.frenet(scene.centres)[0] if scene.n else np.zeros(0)
    x_rng = veh.body_x

    outcome, t = "timeout", 0.0
    min_clr = np.inf
    flips = 0
    last_side: dict[int, int] = {}
    lat_prev = None
    peak_lat, peak_jerk = 0.0, 0.0
    n_replans = n_brake = 0
    plan_time = 0.0
    stopped_for = 0.0
    traj = []
    records = []
    n_steps = int(ctl.timeout / ctl.dt)
    for k in range(n_steps):
        t = k * ctl.dt
        true_pose = st.pose
        if k % ctl.replan_every == 0:
            if noise.drift_xy > 0 or noise.drift_yaw_deg > 0:
                inc = np.array([rng_drift.normal(0, noise.drift_xy), rng_drift.normal(0, noise.drift_xy),
                                rng_drift.normal(0, math.radians(noise.drift_yaw_deg))])
                if drift_per_distance:
                    ds = float(np.hypot(*(true_pose[:2] - last_replan_xy))) if last_replan_xy is not None else 0.0
                    inc = inc * math.sqrt(ds / drift_ref_ds)
                drift = drift + inc
            last_replan_xy = true_pose[:2].copy()
            belief = true_pose + drift
            yaw_rate = st.v * math.tan(st.steer) / veh.wheelbase
            obs = scan(scene, true_pose, noise, rng_scan, cfg.lidar, belief=belief,
                       speed=st.v, steer=st.steer, yaw_rate=yaw_rate)
            t0 = time.perf_counter()
            res = planner.plan(obs, scene, true_pose, st)
            plan_time += time.perf_counter() - t0
            n_replans += 1
            brake = bool(res.brake)
            n_brake += int(brake)
            if not brake and len(res.path):
                path_odom = poses_from_frame(res.path, belief)
                idx = 0
                v_cmd = res.speed
                # door flips: compare which side of each obstacle the new plan passes
                path_w = poses_from_frame(res.path, true_pose)
                s_ego = float(track.frenet(true_pose[:2])[0])
                sig = side_signature(path_w, scene.centres, s_obs, s_ego)
                for j, sd in sig.items():
                    if j in last_side and last_side[j] != sd:
                        flips += 1
                    last_side[j] = sd
            else:
                v_cmd = 0.0
            if record:
                records.append(dict(t=t, pose=true_pose.copy(), belief=belief.copy(), obs=obs,
                                    path=np.array(res.path), brake=brake, info=getattr(res, "info", {})))
        belief_now = st.pose + drift
        if len(path_odom):
            delta, idx = pure_pursuit(belief_now[0], belief_now[1], belief_now[2], path_odom,
                                      ctl.lookahead, veh.wheelbase, idx)
        else:
            delta = 0.0
        v_target = (0.0 if brake else v_cmd) if juno is None else juno.target(st.v, brake)
        st = step(st, v_target, delta, ctl.dt, veh)
        traj.append((st.x, st.y, st.yaw, st.v, st.steer))

        # --- checks on the true state
        pose = st.pose
        if scene.n:
            clr = float(box_circle_clearance(pose[None], scene.centres, scene.radii, x_rng, veh.width).min())
            min_clr = min(min_clr, clr)
            if clr < 0:
                outcome = "collision"
                break
        cs = footprint_corners(pose[None], x_rng, veh.width)[0]
        _, dc = track.frenet(cs)
        if np.abs(dc).max() > track.width / 2:
            outcome = "exit"
            break
        lat = st.v ** 2 * math.tan(st.steer) / veh.wheelbase
        peak_lat = max(peak_lat, abs(lat))
        if lat_prev is not None:
            peak_jerk = max(peak_jerk, abs(lat - lat_prev) / ctl.dt)
        lat_prev = lat
        s_now = float(track.frenet(pose[:2])[0])
        if s_now >= scene.s_goal:
            outcome = "success"
            t = (k + 1) * ctl.dt
            break
        if st.v < ctl.stop_speed:
            stopped_for += ctl.dt
            if stopped_for >= ctl.stop_hold:
                outcome = "stopped"
                t = (k + 1) * ctl.dt
                break
        else:
            stopped_for = 0.0
    return EpisodeResult(scene.seed, planner.name, noise.name, scene.feasible, outcome, t,
                         float(min_clr), flips, peak_lat, peak_jerk, n_replans, n_brake,
                         1000 * plan_time / max(n_replans, 1),
                         np.array(traj) if record else None, records)
