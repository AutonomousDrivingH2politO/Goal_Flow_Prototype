"""Track geometry, SE(2) helpers and Juno footprint collision tests (vectorised numpy)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def to_frame(pts: np.ndarray, pose) -> np.ndarray:
    """World points (N,2) -> frame of ``pose`` = (x, y, yaw)."""
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    d = np.asarray(pts, float) - (x, y)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def from_frame(pts: np.ndarray, pose) -> np.ndarray:
    """Points in the frame of ``pose`` -> world."""
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    p = np.asarray(pts, float)
    return np.stack([x + c * p[..., 0] - s * p[..., 1], y + s * p[..., 0] + c * p[..., 1]], -1)


def poses_to_frame(poses: np.ndarray, pose) -> np.ndarray:
    out = np.empty_like(np.asarray(poses, float))
    out[..., :2] = to_frame(poses[..., :2], pose)
    out[..., 2] = wrap(poses[..., 2] - pose[2])
    return out


def poses_from_frame(poses: np.ndarray, pose) -> np.ndarray:
    out = np.empty_like(np.asarray(poses, float))
    out[..., :2] = from_frame(poses[..., :2], pose)
    out[..., 2] = wrap(poses[..., 2] + pose[2])
    return out


def compose(a, b):
    """Pose a ∘ b."""
    x, y = from_frame(np.array([b[0], b[1]]), a)
    return np.array([x, y, float(wrap(a[2] + b[2]))])


@dataclass(frozen=True)
class Track:
    """Straight (kappa = 0) or constant-curvature centreline starting at the origin, heading +x.

    s is arc length along the centreline, d the lateral offset (positive left).
    """
    kappa: float
    width: float
    s_min: float
    s_max: float

    def centre(self, s):
        s = np.asarray(s, float)
        k = self.kappa
        if abs(k) < 1e-9:
            return np.stack([s, np.zeros_like(s)], -1)
        return np.stack([np.sin(k * s) / k, (1 - np.cos(k * s)) / k], -1)

    def heading(self, s):
        return self.kappa * np.asarray(s, float)

    def point(self, s, d):
        s = np.asarray(s, float)
        d = np.asarray(d, float)
        c = self.centre(s)
        th = self.heading(s)
        return c + np.stack([-np.sin(th) * d, np.cos(th) * d], -1)

    def pose(self, s, d=0.0, dpsi=0.0):
        p = self.point(s, d)
        return np.array([p[0], p[1], float(wrap(self.heading(s) + dpsi))])

    def frenet(self, pts):
        """World points (...,2) -> (s, d)."""
        p = np.asarray(pts, float)
        k = self.kappa
        if abs(k) < 1e-9:
            return p[..., 0], p[..., 1]
        R = 1.0 / k
        dx, dy = p[..., 0], p[..., 1] - R
        phi = np.arctan2(dx / R, -dy / R)
        rho = np.hypot(dx, dy)
        return phi * R, R - np.sign(R) * rho

    def edges(self, s0: float, s1: float, spacing: float):
        s = np.arange(s0, s1 + 1e-9, spacing)
        h = self.width / 2
        return s, self.point(s, np.full_like(s, h)), self.point(s, np.full_like(s, -h))

    def polygon(self, spacing: float = 0.5) -> np.ndarray:
        _, left, right = self.edges(self.s_min, self.s_max, spacing)
        return np.concatenate([right, left[::-1]], 0)


def footprint_corners(poses: np.ndarray, x_rng: tuple[float, float], width: float,
                      inflate: float = 0.0) -> np.ndarray:
    """(N,3) poses -> (N,4,2) body corners in world."""
    poses = np.atleast_2d(poses)
    x0, x1 = x_rng[0] - inflate, x_rng[1] + inflate
    h = width / 2 + inflate
    loc = np.array([[x0, -h], [x1, -h], [x1, h], [x0, h]])
    c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    X = poses[:, 0:1] + c * loc[:, 0] - s * loc[:, 1]
    Y = poses[:, 1:2] + s * loc[:, 0] + c * loc[:, 1]
    return np.stack([X, Y], -1)


def box_circle_clearance(poses: np.ndarray, centres: np.ndarray, radii: np.ndarray,
                         x_rng: tuple[float, float], width: float) -> np.ndarray:
    """Signed clearance (N, M) between the Juno box at each pose and each circle.

    Negative means overlap. Uses the exact box-circle distance.
    """
    poses = np.atleast_2d(poses)
    if len(centres) == 0:
        return np.full((len(poses), 0), np.inf)
    c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    dx = centres[None, :, 0] - poses[:, 0:1]
    dy = centres[None, :, 1] - poses[:, 1:2]
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    h = width / 2
    qx = np.clip(lx, x_rng[0], x_rng[1])
    qy = np.clip(ly, -h, h)
    dist = np.hypot(lx - qx, ly - qy)
    inside = (lx > x_rng[0]) & (lx < x_rng[1]) & (np.abs(ly) < h)
    dist = np.where(inside, -np.minimum.reduce([lx - x_rng[0], x_rng[1] - lx, h - np.abs(ly)]), dist)
    return dist - radii[None, :]


def boxes_hit_points(poses: np.ndarray, pts: np.ndarray, x_rng: tuple[float, float],
                     width: float, inflate: float = 0.0) -> np.ndarray:
    """(N,) bool: does the (inflated) box at each pose contain any of the points?"""
    poses = np.atleast_2d(poses)
    if len(pts) == 0:
        return np.zeros(len(poses), bool)
    c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    dx = pts[None, :, 0] - poses[:, 0:1]
    dy = pts[None, :, 1] - poses[:, 1:2]
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    h = width / 2 + inflate
    m = (lx > x_rng[0] - inflate) & (lx < x_rng[1] + inflate) & (np.abs(ly) < h)
    return m.any(1)
