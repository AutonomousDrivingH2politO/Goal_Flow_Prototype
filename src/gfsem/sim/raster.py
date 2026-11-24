"""BEV raster of an ``Observation``: the learned planner's input, mirroring the live Nav2 costmap.

Channels (float32 in [0, 1]):
  0 hits        LiDAR hits, binned, no inflation
  1 free        cells crossed by a ray before its return (obstacle-layer raytrace clearing)
  2 unknown     cells no ray reached, incl. occlusion shadows (track_unknown_space)
  3 drivable    area between the observed road edges
  4 class       1 bollard, 0.5 cone, on hit cells
  5 goal        goal line, or the raster's far edge if the goal is beyond it
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import BOLLARD, CONE
from .edges import observed_edges
from .lidar import Observation

CHANNELS = ("hits", "free", "unknown", "drivable", "class", "goal")


@dataclass(frozen=True)
class RasterConfig:
    size: int = 128
    res: float = 0.2
    x_min: float = -3.2
    y_min: float = -12.8
    ray_sample: float = 0.1
    track_width: float = 12.0     # nominal width for the edge fill (same as the observed planners)
    edge_bin: float = 1.0

    @property
    def x_max(self) -> float:
        return self.x_min + self.size * self.res

    @property
    def y_max(self) -> float:
        return self.y_min + self.size * self.res


def world_to_cell(xy: np.ndarray, cfg: RasterConfig) -> tuple[np.ndarray, np.ndarray]:
    """Ego (x, y) -> (row, col). Row 0 is the far edge (max x); col 0 is max y (left)."""
    xy = np.asarray(xy)
    row = np.floor((cfg.x_max - xy[..., 0]) / cfg.res).astype(np.int64)
    col = np.floor((cfg.y_max - xy[..., 1]) / cfg.res).astype(np.int64)
    return row, col


def cell_centres(cfg: RasterConfig) -> np.ndarray:
    """(size, size, 2) ego coordinates of cell centres."""
    i = np.arange(cfg.size)
    x = cfg.x_max - (i + 0.5) * cfg.res
    y = cfg.y_max - (i + 0.5) * cfg.res
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.stack([X, Y], -1)


def _inb(r, c, n):
    return (r >= 0) & (r < n) & (c >= 0) & (c < n)


def rasterize(obs: Observation, cfg: RasterConfig | None = None) -> np.ndarray:
    cfg = cfg or RasterConfig()
    n = cfg.size
    out = np.zeros((6, n, n), np.float32)

    # 0, 4: hits and class hint
    if len(obs.points):
        r, c = world_to_cell(obs.points, cfg)
        m = _inb(r, c, n)
        out[0, r[m], c[m]] = 1.0
        val = np.where(obs.labels[m] == BOLLARD, 1.0, np.where(obs.labels[m] == CONE, 0.5, 0.0))
        np.maximum.at(out[4], (r[m], c[m]), val.astype(np.float32))

    # 1, 2: raytrace clearing from the sensor; everything not reached is unknown
    ang = obs.ray_angles
    rng = obs.scan_ranges
    ok = np.isfinite(rng)
    max_r = np.hypot(cfg.x_max - obs.sensor_xy[0], max(abs(cfg.y_min), cfg.y_max))
    rr = np.minimum(rng[ok], max_r)
    a = ang[ok]
    ts = np.arange(obs.blind_zone, max_r, cfg.ray_sample)
    T = ts[None, :]
    valid = T < rr[:, None] - cfg.res * 0.5          # stop just short of the hit cell
    X = obs.sensor_xy[0] + T * np.cos(a)[:, None]
    Y = obs.sensor_xy[1] + T * np.sin(a)[:, None]
    r, c = world_to_cell(np.stack([X[valid], Y[valid]], -1), cfg)
    m = _inb(r, c, n)
    out[1, r[m], c[m]] = 1.0
    out[1][out[0] > 0] = 0.0
    out[2] = ((out[1] == 0) & (out[0] == 0)).astype(np.float32)

    # 3: drivable polygon from the observed edges
    out[3] = drivable_mask(obs, cfg)

    # 5: goal line, clipped to the raster; if no part of it is inside, the far edge (rows 0-1,
    # the max-x end) inside the drivable corridor. Always 2 px thick.
    out[5] = goal_channel(obs.goal_line.astype(np.float64), out[3], cfg)
    return out


def clip_segment(p0: np.ndarray, p1: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Liang-Barsky: clip segment p0-p1 to the box [lo, hi]; None if it misses the box."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for k in range(2):
        for p, q in ((-d[k], p0[k] - lo[k]), (d[k], hi[k] - p0[k])):
            if abs(p) < 1e-12:
                if q < 0:
                    return None
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return None
    return p0 + t0 * d, p0 + t1 * d


def goal_channel(goal: np.ndarray, drivable: np.ndarray, cfg: RasterConfig, thickness: int = 2) -> np.ndarray:
    n = cfg.size
    img = np.zeros((n, n), np.uint8)
    eps = 1e-6
    lo = np.array([cfg.x_min + eps, cfg.y_min + eps])
    hi = np.array([cfg.x_max - eps, cfg.y_max - eps])
    seg = clip_segment(goal[0], goal[1], lo, hi)
    if seg is not None:
        r, c = world_to_cell(np.stack(seg), cfg)
        cv2.line(img, (int(c[0]), int(r[0])), (int(c[1]), int(r[1])), 1, thickness)
    else:
        far = drivable[:thickness] > 0
        if not far.any():
            far = np.ones((thickness, n), bool)
        img[:thickness] = far
    return img.astype(np.float32)


def drivable_mask(obs: Observation, cfg: RasterConfig) -> np.ndarray:
    """Polygon between the observed edges, with the same binning and 12 m edge fill as the
    observed corridor planners (``sim.edges.observed_edges``)."""
    n = cfg.size
    left, right = observed_edges(obs.edge_points, obs.edge_side, cfg.track_width, cfg.edge_bin)
    if len(left) < 2 or len(right) < 2:
        return np.zeros((n, n), np.float32)

    def extend(e):
        # extrapolate back to the raster's rear edge along the first edge segment direction
        k = min(len(e) - 1, 4)
        d = e[k] - e[0]
        d = d / max(np.linalg.norm(d), 1e-6)
        if d[0] <= 0.2:
            d = np.array([1.0, 0.0])
        back = e[0] - d * (e[0, 0] - cfg.x_min + cfg.res) / d[0]
        return np.concatenate([back[None], e])

    poly = np.concatenate([extend(left), extend(right)[::-1]])
    r, c = world_to_cell(poly, cfg)
    img = np.zeros((n, n), np.uint8)
    cv2.fillPoly(img, [np.stack([c, r], -1).astype(np.int32)], 1)
    return img.astype(np.float32)
