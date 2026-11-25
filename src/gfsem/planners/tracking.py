"""Short-term perception memory for the observed planners (DEVIATIONS Part C, L3 items 1-2, 5).

Both pieces work in the odom frame (the believed pose ``obs.pose_odom``), so they carry odometry drift;
memories are kept short for that reason.

* ``EdgeMemory``: road-edge points actually observed in earlier scans. The forward 120 deg LiDAR cannot
  see the edge beside and behind the car, where the planners otherwise fall back to the 12 m fill (the
  opposite edge offset by the track width). On a curved track that fill can sit up to ~0.4 m inside
  the true edge next to the car, which made the car's own pose fail the 0.12 m check and caused the
  L0 stalls (A6). Remembered points are used only where the current scan has none for that side
  (behind its first observed point), and for at most ``keep_s`` seconds.
* ``ObstacleTracker``: clusters from each scan are associated with tracks (nearest within a gate);
  tracks smooth position and radius. A detection is always used in the scan it appears in (no
  confirmation delay for safety); a track seen in >= ``confirm`` scans persists through up to
  ``persist`` missed scans while in the field of view (LiDAR dropouts), and longer outside it (beside
  and behind the car, where the sensor cannot see it). Unconfirmed clutter does not persist.

The GoalFlow safety filter in the closed loop takes the same tracked obstacles (``ego_obstacles``), so
the comparison with the corridor baseline uses identical perception.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..sim.geometry import from_frame, to_frame


@dataclass(frozen=True)
class TrackingConfig:
    edge_keep_s: float = 2.0
    edge_gap: float = 0.5           # memory points must lie this far behind the first current point
    gate: float = 0.6               # m association gate (centre distance)
    alpha: float = 0.5              # smoothing weight of a new measurement
    confirm: int = 2                # scans seen before a track may persist
    persist: int = 5                # missed scans (in view) before a confirmed track is dropped
    persist_out_of_view: int = 30   # missed scans while outside the field of view
    fov_deg: float = 120.0
    view_range: float = 25.0
    scan_dt: float = 0.1


class EdgeMemory:
    def __init__(self, cfg: TrackingConfig = TrackingConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.pts = np.zeros((0, 2))
        self.side = np.zeros(0, np.int8)
        self.age = np.zeros(0)

    def update(self, edge_points: np.ndarray, edge_side: np.ndarray, pose_odom: np.ndarray):
        """Add this scan's observed points; returns (points, side) in the ego frame: the current scan
        plus remembered points behind the current coverage of each side."""
        cfg = self.cfg
        self.age = self.age + cfg.scan_dt
        keep = self.age <= cfg.edge_keep_s
        self.pts, self.side, self.age = self.pts[keep], self.side[keep], self.age[keep]
        cur = np.asarray(edge_points, float).reshape(-1, 2)
        cs = np.asarray(edge_side, np.int8).reshape(-1)
        out_p, out_s = [cur], [cs]
        if len(self.pts):
            loc = to_frame(self.pts, pose_odom)
            for sd in (1, -1):
                m_cur = cs == sd
                x0 = cur[m_cur, 0].min() if m_cur.any() else np.inf
                m = (self.side == sd) & (loc[:, 0] < x0 - cfg.edge_gap)
                if m.any():
                    out_p.append(loc[m])
                    out_s.append(np.full(m.sum(), sd, np.int8))
        if len(cur):
            self.pts = np.concatenate([self.pts, from_frame(cur, pose_odom)])
            self.side = np.concatenate([self.side, cs])
            self.age = np.concatenate([self.age, np.zeros(len(cur))])
        return np.concatenate(out_p), np.concatenate(out_s)


class ObstacleTracker:
    def __init__(self, cfg: TrackingConfig = TrackingConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.C = np.zeros((0, 2))     # odom frame
        self.R = np.zeros(0)
        self.seen = np.zeros(0, int)
        self.missed = np.zeros(0, int)

    def in_view(self, loc: np.ndarray, sensor_x: float = 1.6) -> np.ndarray:
        ang = np.arctan2(loc[:, 1], loc[:, 0] - sensor_x)
        rng = np.hypot(loc[:, 0] - sensor_x, loc[:, 1])
        return (np.abs(ang) <= math.radians(self.cfg.fov_deg / 2)) & (rng <= self.cfg.view_range) & (loc[:, 0] > sensor_x)

    def update(self, centres: np.ndarray, radii: np.ndarray, pose_odom: np.ndarray):
        """Associate this scan's clusters (ego frame); returns tracked (centres, radii) in the ego frame:
        every current detection (smoothed if it matches a track) plus persisting confirmed tracks."""
        cfg = self.cfg
        Cm = from_frame(np.asarray(centres, float).reshape(-1, 2), pose_odom) if len(centres) else np.zeros((0, 2))
        Rm = np.asarray(radii, float).reshape(-1)
        nT = len(self.C)
        matched_t = np.zeros(nT, bool)
        newC, newR, newS = [], [], []
        if nT and len(Cm):
            d = np.hypot(Cm[:, None, 0] - self.C[None, :, 0], Cm[:, None, 1] - self.C[None, :, 1])
            pairs = sorted(((d[i, j], i, j) for i in range(len(Cm)) for j in range(nT) if d[i, j] <= cfg.gate))
            used_m = set()
            for _, i, j in pairs:
                if i in used_m or matched_t[j]:
                    continue
                used_m.add(i)
                matched_t[j] = True
                a = cfg.alpha
                self.C[j] = (1 - a) * self.C[j] + a * Cm[i]
                self.R[j] = (1 - a) * self.R[j] + a * Rm[i]
                self.seen[j] += 1
                self.missed[j] = 0
            for i in range(len(Cm)):
                if i not in used_m:
                    newC.append(Cm[i])
                    newR.append(Rm[i])
        else:
            newC, newR = list(Cm), list(Rm)
        # unmatched tracks: persist if confirmed, longer outside the field of view
        if nT:
            loc = to_frame(self.C, pose_odom)
            view = self.in_view(loc)
            self.missed[~matched_t] += 1
            lim = np.where(view, cfg.persist, cfg.persist_out_of_view)
            alive = matched_t | ((self.seen >= cfg.confirm) & (self.missed <= lim) & (loc[:, 0] > -8.0))
            self.C, self.R, self.seen, self.missed = self.C[alive], self.R[alive], self.seen[alive], self.missed[alive]
        if newC:
            self.C = np.concatenate([self.C, np.array(newC).reshape(-1, 2)])
            self.R = np.concatenate([self.R, np.array(newR)])
            self.seen = np.concatenate([self.seen, np.ones(len(newC), int)])
            self.missed = np.concatenate([self.missed, np.zeros(len(newC), int)])
        return self.ego_obstacles(pose_odom)

    def ego_obstacles(self, pose_odom: np.ndarray):
        """Live tracks in the ego frame (what both the corridor baseline and GoalFlow's safety filter use)."""
        if not len(self.C):
            return np.zeros((0, 2)), np.zeros(0)
        return to_frame(self.C, pose_odom), self.R.copy()

    def signature(self, res: float = 0.1) -> tuple:
        """Coarse fingerprint of the tracked world (for failed-search caching)."""
        return tuple(sorted(map(tuple, np.round(self.C / res).astype(int).tolist())))
