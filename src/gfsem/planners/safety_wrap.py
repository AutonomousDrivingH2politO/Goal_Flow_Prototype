"""GoalFlow's safety filter around a classical planner (DEVIATIONS 2026-11-27, "(a)+filter").

``FilteredObserved`` wraps an observed corridor planner (baseline (a)). Every scan it updates the same circle set and
near-field circle memory as ``goalflow_l3_combined`` (this scan's ``cluster_obstacles`` circles + remembered near-field
circles with the memory fix), simulates the executed path (pure pursuit + the vehicle model, from the current speed and
steering, along the planner's path for GoalFlow's horizon of 12 m) and checks it with the exact 0.12 m box-to-circle
filter. If the check fails, the scan brakes. The planner itself is unchanged (it keeps its own plan and checks).
"""
from __future__ import annotations

import numpy as np

from .goalflow_planner import GFResult, GoalFlowPlanner


class _CircleMemory(GoalFlowPlanner):
    """Only the circle filter and its near-field memory from GoalFlowPlanner (no model)."""

    def __init__(self, lidar=None):          # noqa: D107 (deliberately no super().__init__: no model is loaded)
        from ..sim.config import LidarConfig, VehicleConfig
        self.memory_range, self.memory_obs_range, self.circle_gate = 4.0, 4.0, 0.6
        self.rear_x = VehicleConfig.from_yaml().body_x[0]
        self.entry_scans, self.mem_fix = 1, True
        self.lidar = lidar or LidarConfig()
        self.reset_memory()

    def reset_memory(self):
        self.circ, self.prev_C, self.n_cur = [], np.zeros((0, 2)), 0


class FilteredObserved:
    def __init__(self, planner, name: str, horizon: float = 12.0, margin: float = 0.12):
        from ..sim.config import ControllerConfig, VehicleConfig
        self.p, self.name, self.horizon, self.margin = planner, name, horizon, margin
        self.veh, self.ctl = VehicleConfig.from_yaml(), ControllerConfig()
        self.mem = _CircleMemory()
        self.filter_brakes = 0

    @property
    def lidar(self):
        return self.mem.lidar

    @lidar.setter
    def lidar(self, cfg):
        self.mem.lidar = cfg

    def reset(self):
        self.p.reset()
        self.mem.reset_memory()
        self.filter_brakes = 0

    def executed_path(self, path: np.ndarray, v: float, steer: float, v_cmd: float) -> np.ndarray:
        """Pure-pursuit tracking of ``path`` (ego frame) from the origin with the closed loop's controller and vehicle
        model, up to ``horizon`` m driven or the path's end; (K, 2) positions."""
        from ..sim.vehicle import VehicleState, pure_pursuit, step
        st, idx, out, s = VehicleState(0.0, 0.0, 0.0, max(v, 0.5), steer), 0, [], 0.0
        end = path[-1, :2]
        for _ in range(int(2 * self.horizon / (max(v_cmd, 0.5) * self.ctl.dt)) + 1):
            delta, idx = pure_pursuit(st.x, st.y, st.yaw, path, self.ctl.lookahead, self.veh.wheelbase, idx)
            nxt = step(st, v_cmd, delta, self.ctl.dt, self.veh)
            s += float(np.hypot(nxt.x - st.x, nxt.y - st.y))
            st = nxt
            out.append((st.x, st.y))
            if s >= self.horizon or float(np.hypot(st.x - end[0], st.y - end[1])) < 0.3:
                break
        return np.asarray(out)

    def plan(self, obs, scene, true_pose, state):
        import torch
        from ..models.goalflow import observed_collision_exact
        res = self.p.plan_observation(obs, state.v, state.steer)
        circles = self.mem._circles(obs)                 # memory is updated every scan
        if res.brake or not len(res.path) or not len(circles):
            return res
        ex = self.executed_path(np.asarray(res.path, float), state.v, state.steer, float(res.speed))
        if not len(ex):
            return res
        ex = ex[np.r_[np.arange(0, len(ex), 3), len(ex) - 1]]          # <= ~0.15 m spacing; densified inside
        unsafe = bool(observed_collision_exact(torch.from_numpy(ex[None, None]).float(),
                                               [circles], margin=self.margin)[0, 0])
        if not unsafe:
            return res
        self.filter_brakes += 1
        ev = getattr(self.p, "events", None)
        if isinstance(ev, dict):
            ev["filter_brake"] = ev.get("filter_brake", 0) + 1
        return GFResult(np.zeros((0, 3)), True, 0.0, {"trigger": "brake", "reason": "safety filter"})
