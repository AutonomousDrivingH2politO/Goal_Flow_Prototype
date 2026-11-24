"""Kinematic bicycle model (rear-axle reference) and pure-pursuit path tracking."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import VehicleConfig


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    v: float
    steer: float = 0.0

    @property
    def pose(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw])

    def copy(self) -> "VehicleState":
        return VehicleState(self.x, self.y, self.yaw, self.v, self.steer)


def step(st: VehicleState, v_cmd: float, steer_cmd: float, dt: float, veh: VehicleConfig) -> VehicleState:
    """Rate-limited steering, accel-limited speed, then exact-enough Euler integration."""
    ds_max = math.radians(veh.max_steer_rate_deg) * dt
    steer = st.steer + max(-ds_max, min(ds_max, steer_cmd - st.steer))
    steer = max(-veh.max_steer, min(veh.max_steer, steer))
    dv = v_cmd - st.v
    dv = max(-veh.max_decel * dt, min(veh.max_accel * dt, dv))
    v = max(0.0, st.v + dv)
    vm = 0.5 * (st.v + v)
    yaw_rate = vm * math.tan(steer) / veh.wheelbase
    yaw_mid = st.yaw + 0.5 * yaw_rate * dt
    return VehicleState(st.x + vm * math.cos(yaw_mid) * dt, st.y + vm * math.sin(yaw_mid) * dt,
                        st.yaw + yaw_rate * dt, v, steer)


def pure_pursuit(x: float, y: float, yaw: float, path: np.ndarray, lookahead: float,
                 wheelbase: float, idx: int = 0) -> tuple[float, int]:
    """Steering angle toward the path point ``lookahead`` ahead. Returns (steer, nearest index).

    ``path`` is (K,2+) in the same frame as the pose, ordered, roughly evenly spaced.
    ``idx`` is a monotone progress hint so the tracker never jumps back along the path.
    """
    K = len(path)
    if K == 0:
        return 0.0, 0
    px = path[:, 0]
    py = path[:, 1]
    # nearest point in a forward window from the hint
    hi = min(K, idx + 40)
    d2 = (px[idx:hi] - x) ** 2 + (py[idx:hi] - y) ** 2
    near = idx + int(np.argmin(d2))
    # first point at least `lookahead` away beyond the nearest
    d = np.hypot(px[near:] - x, py[near:] - y)
    j = np.nonzero(d >= lookahead)[0]
    if len(j):
        tx, ty = px[near + j[0]], py[near + j[0]]
    else:  # extend the last segment
        if K >= 2:
            hx, hy = px[-1] - px[-2], py[-1] - py[-2]
            n = math.hypot(hx, hy) or 1.0
            hx, hy = hx / n, hy / n
        else:
            hx, hy = math.cos(yaw), math.sin(yaw)
        # solve |p_end + s h - p| = lookahead for s >= 0
        ex, ey = px[-1] - x, py[-1] - y
        b = ex * hx + ey * hy
        c = ex * ex + ey * ey - lookahead * lookahead
        s = -b + math.sqrt(max(b * b - c, 0.0))
        tx, ty = px[-1] + s * hx, py[-1] + s * hy
    dx, dy = tx - x, ty - y
    c, s_ = math.cos(yaw), math.sin(yaw)
    lx, ly = c * dx + s_ * dy, -s_ * dx + c * dy
    ld2 = lx * lx + ly * ly
    if ld2 < 1e-9:
        return 0.0, near
    kappa = 2.0 * ly / ld2
    return math.atan(wheelbase * kappa), near


def rollout_path(path: np.ndarray, start: VehicleState, veh: VehicleConfig, lookahead: float,
                 speed: float, ds: float = 0.25, max_len: float = 40.0) -> np.ndarray:
    """Simulate pure pursuit + bicycle along ``path`` (frame of ``start``); returns (K,3) poses
    spaced ``ds`` apart in distance, until the path end is passed or ``max_len`` is driven."""
    st = start.copy()
    st.v = max(speed, 0.5)
    dt = ds / st.v
    out = [(st.x, st.y, st.yaw)]
    idx = 0
    end = path[-1, :2]
    n_steps = int(max_len / ds)
    if len(path) >= 2:
        hx, hy = path[-1, 0] - path[-2, 0], path[-1, 1] - path[-2, 1]
    else:
        hx, hy = 1.0, 0.0
    for _ in range(n_steps):
        delta, idx = pure_pursuit(st.x, st.y, st.yaw, path, lookahead, veh.wheelbase, idx)
        st = step(st, st.v, delta, dt, veh)
        out.append((st.x, st.y, st.yaw))
        if (st.x - end[0]) * hx + (st.y - end[1]) * hy > 0:
            break
    return np.array(out)


def resample_polyline(pts: np.ndarray, ds: float) -> np.ndarray:
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts.copy()
    seg = np.hypot(*np.diff(pts, axis=0).T)
    keep = np.concatenate([[True], seg > 1e-6])
    pts = pts[keep]
    if len(pts) < 2:
        return pts
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    s = np.arange(0.0, cum[-1] + 1e-9, ds)
    if s[-1] < cum[-1] - 1e-6:
        s = np.append(s, cum[-1])
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], -1)
