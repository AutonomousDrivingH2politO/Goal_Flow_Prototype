"""2D LiDAR + road-edge observation model (RoboSense EMX at 1.10 m, pitched 7 deg down)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import BOLLARD, CONE, LidarConfig, NoiseLevel
from .geometry import compose, from_frame, to_frame, wrap
from .scene import Scene


@dataclass
class Observation:
    """What a planner may see. Everything in the ego frame (rear axle, x forward, y left).

    Kept small on purpose: datasets store this and rasterise on the fly. A CARLA LiDAR or a
    real costmap can produce the same structure (see ``carla_adapter``).
    """
    points: np.ndarray            # (N,2) float32 LiDAR hits
    labels: np.ndarray            # (N,) uint8 camera class per hit: 1 bollard, 2 cone
    scan_ranges: np.ndarray       # (R,) float32 per-ray clearing range from the sensor, NaN = none
    edge_points: np.ndarray       # (E,2) float32 observed road-edge points
    edge_side: np.ndarray         # (E,) int8 +1 left edge, -1 right edge
    goal_line: np.ndarray         # (2,2) float32 goal line endpoints (right, left)
    ego_state: np.ndarray         # (5,) float32 speed, steer, yaw rate, heading rel. track, lateral offset
    sensor_xy: tuple[float, float] = (1.60, 0.0)
    ray_angle0: float = -math.radians(60.0)
    ray_step: float = math.radians(0.2)
    blind_zone: float = 1.5
    pose_odom: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))  # believed pose

    @property
    def ray_angles(self) -> np.ndarray:
        return self.ray_angle0 + self.ray_step * np.arange(len(self.scan_ranges))

    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.points, self.labels, self.scan_ranges, self.edge_points,
                                      self.edge_side, self.goal_line, self.ego_state, self.pose_odom))


def _ray_circle(o: np.ndarray, u: np.ndarray, c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Distance along each ray (R,) to each circle (M,), inf when missed. o: (2,), u: (R,2)."""
    if len(c) == 0:
        return np.full((len(u), 0), np.inf)
    oc = c - o                                   # (M,2)
    b = u @ oc.T                                 # (R,M)
    q = (oc ** 2).sum(1) - r ** 2                # (M,)
    disc = b ** 2 - q[None, :]
    t = b - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def visible_hits(t: np.ndarray, heights: np.ndarray, cfg: LidarConfig) -> np.ndarray:
    """Which (ray, obstacle) intersections produce a return, accounting for vertical beam span.

    Walk obstacles front to back along each ray, tracking the lowest beam elevation not yet
    blocked. An obstacle at range t is hit if that beam still meets it below its top. Ground
    blocks beams below atan(-h_mount / t). Bollards (top level with the sensor) then occlude
    everything behind them; cones only block the beams that strike them.
    """
    R, M = t.shape
    hit = np.zeros((R, M), bool)
    if M == 0:
        return hit
    e_min, e_max = cfg.elev_range
    hm = cfg.mount_height
    order = np.argsort(t, axis=1)
    e_low = np.full(R, e_min)
    rows = np.arange(R)
    for k in range(M):
        j = order[:, k]
        tk = t[rows, j]
        valid = np.isfinite(tk) & (tk >= cfg.blind_zone) & (tk <= cfg.max_range) & (e_low < e_max)
        if not valid.any():
            break
        tk_s = np.where(valid, tk, 1.0)
        hk = heights[j]
        e_eff = np.maximum(e_low, np.arctan2(-hm, tk_s))
        e_top = np.arctan2(hk - hm, tk_s)
        h_ok = valid & (e_eff < e_top - 1e-9)   # beams level with a top do not count
        hit[rows[h_ok], j[h_ok]] = True
        e_low = np.where(h_ok, np.maximum(e_low, e_top), e_low)
    return hit


def scan(scene: Scene, pose: np.ndarray, noise: NoiseLevel, rng: np.random.Generator,
         cfg: LidarConfig | None = None, belief: np.ndarray | None = None,
         speed: float = 0.0, steer: float = 0.0, yaw_rate: float = 0.0) -> Observation:
    """Simulate one LiDAR scan + road edges + camera labels from the true ``pose``.

    ``belief`` is the localisation estimate (odom frame drifts from truth); it only affects
    quantities that come from the map: the goal line and the track-relative ego status.
    """
    cfg = cfg or LidarConfig()
    belief = pose if belief is None else belief
    track = scene.track
    sens = compose(pose, np.array([cfg.mount_x, cfg.mount_y, 0.0]))
    o = sens[:2]
    ang = cfg.ray_angles
    u = np.stack([np.cos(sens[2] + ang), np.sin(sens[2] + ang)], -1)

    # --- obstacles ---
    near = np.hypot(*(scene.centres - o).T) < cfg.max_range + 1.0 if scene.n else np.zeros(0, bool)
    idx = np.nonzero(near)[0]
    t = _ray_circle(o, u, scene.centres[idx], scene.radii[idx])
    hit = visible_hits(t, scene.heights[idx], cfg)
    ri, oi = np.nonzero(hit)
    rng_t = t[ri, oi]
    bear = ang[ri].copy()
    obj = idx[oi]
    n = len(rng_t)
    if noise.range_sigma > 0:
        rng_t = rng_t + rng.normal(0, noise.range_sigma, n)
    if noise.angle_sigma_deg > 0:
        bear = bear + rng.normal(0, math.radians(noise.angle_sigma_deg), n)
    keep = rng.random(n) >= noise.dropout
    # Per-object camera class, wrong with p_mis (detection never depends on it).
    true_cls = scene.classes.astype(np.uint8)
    flip = rng.random(scene.n) < noise.p_mis
    obs_cls = np.where(flip, np.where(true_cls == BOLLARD, CONE, BOLLARD), true_cls).astype(np.uint8)

    # clearing range per ray
    scan_r = np.full(cfg.n_rays, cfg.max_range, np.float32)
    has_ret = np.zeros(cfg.n_rays, bool)
    has_ret[ri] = True
    kept_max = np.full(cfg.n_rays, -np.inf)
    np.maximum.at(kept_max, ri[keep], rng_t[keep])
    scan_r[has_ret] = np.where(np.isfinite(kept_max[has_ret]), kept_max[has_ret], np.nan)

    pts_s = np.stack([rng_t * np.cos(bear), rng_t * np.sin(bear)], -1)[keep]
    labels = obs_cls[obj[keep]]

    # --- clutter near the track edges ---
    s_ego, _ = track.frenet(pose[:2])
    n_cl = rng.poisson(noise.clutter_rate) if noise.clutter_rate > 0 else 0
    cl_pts, cl_lab = [], []
    h = track.width / 2
    for _ in range(n_cl):
        side = rng.choice([-1.0, 1.0])
        s = s_ego + rng.uniform(2.0, cfg.edge_range)
        d = side * (h + rng.uniform(*cfg.clutter_band))
        c = track.point(s, d)
        k = int(rng.integers(1, 5))
        p = c + rng.normal(0, 0.12, (k, 2))
        ps = to_frame(p, sens)
        r = np.hypot(ps[:, 0], ps[:, 1])
        b = np.arctan2(ps[:, 1], ps[:, 0])
        ok = (r > cfg.blind_zone) & (r < cfg.max_range) & (np.abs(b) <= ang[-1])
        if ok.any():
            cl_pts.append(ps[ok])
            cl_lab.append(np.full(ok.sum(), rng.choice([BOLLARD, CONE]), np.uint8))
    if cl_pts:
        pts_s = np.concatenate([pts_s] + cl_pts)
        labels = np.concatenate([labels] + cl_lab)

    # sensor frame -> ego frame
    pts_e = pts_s + np.array([cfg.mount_x, cfg.mount_y])  # sensor has zero yaw offset

    # --- road edges (segmentation + ground intensity fusion stand-in) ---
    bollard_t = np.where(scene.classes[idx][None, :] == BOLLARD, t, np.inf).min(1) \
        if len(idx) else np.full(cfg.n_rays, np.inf)
    e_pts, e_side = [], []
    s_rng = np.arange(s_ego - 2.0, s_ego + cfg.edge_range + 2.0, cfg.edge_spacing)
    for side in (1, -1):
        p = track.point(s_rng, np.full_like(s_rng, side * h))
        ps = to_frame(p, sens)
        r = np.hypot(ps[:, 0], ps[:, 1])
        b = np.arctan2(ps[:, 1], ps[:, 0])
        ok = (r > cfg.blind_zone) & (r <= cfg.edge_range) & (np.abs(b) <= ang[-1])
        ray = np.clip(np.round((b - ang[0]) / (ang[1] - ang[0])).astype(int), 0, cfg.n_rays - 1)
        ok &= r < bollard_t[ray]
        n_drop = rng.poisson(noise.edge_dropout_rate) if noise.edge_dropout_rate > 0 else 0
        for _ in range(n_drop):
            a = rng.uniform(s_rng[0], s_rng[-1])
            ok &= ~((s_rng >= a) & (s_rng <= a + rng.uniform(1.0, 2.0)))
        ps = ps[ok]
        if noise.edge_sigma > 0:
            ps = ps + rng.normal(0, noise.edge_sigma, ps.shape)
        e_pts.append(ps + np.array([cfg.mount_x, cfg.mount_y]))
        e_side.append(np.full(len(ps), side, np.int8))

    # --- map-derived quantities through the believed pose ---
    goal = to_frame(scene.goal_line, belief)
    s_b, d_b = track.frenet(belief[:2])
    ego = np.array([speed, steer, yaw_rate, float(wrap(belief[2] - track.heading(s_b))), d_b],
                   np.float32)

    return Observation(
        points=pts_e.astype(np.float32).reshape(-1, 2),
        labels=labels.astype(np.uint8),
        scan_ranges=scan_r,
        edge_points=np.concatenate(e_pts).astype(np.float32).reshape(-1, 2),
        edge_side=np.concatenate(e_side),
        goal_line=goal.astype(np.float32),
        ego_state=ego,
        sensor_xy=(cfg.mount_x, cfg.mount_y),
        ray_angle0=float(ang[0]),
        ray_step=float(ang[1] - ang[0]),
        blind_zone=cfg.blind_zone,
        pose_odom=np.asarray(belief, np.float32),
    )


def privileged_view(scene: Scene, pose: np.ndarray, edge_spacing: float = 1.0):
    """True obstacles and track edges in the ego frame (for the privileged teacher)."""
    track = scene.track
    s_ego, _ = track.frenet(pose[:2])
    s, left, right = track.edges(s_ego - 3.0, scene.s_goal + 3.0, edge_spacing)
    return dict(
        centres=to_frame(scene.centres, pose).reshape(-1, 2),
        radii=scene.radii.copy(),
        classes=scene.classes.copy(),
        left=to_frame(left, pose),
        right=to_frame(right, pose),
        goal_line=to_frame(scene.goal_line, pose),
    )


__all__ = ["Observation", "scan", "privileged_view", "visible_hits", "from_frame"]
