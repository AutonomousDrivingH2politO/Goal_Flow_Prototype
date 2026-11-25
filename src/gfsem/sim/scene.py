"""Scene generator for the SEM obstacle course and the kinematic feasibility oracle."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BOLLARD, CONE, SceneConfig, VehicleConfig
from .geometry import Track, box_circle_clearance, footprint_corners


@dataclass
class Scene:
    seed: int
    track: Track
    centres: np.ndarray      # (M,2) world
    radii: np.ndarray        # (M,)
    heights: np.ndarray      # (M,)
    classes: np.ndarray      # (M,) BOLLARD / CONE
    s_field: tuple[float, float]
    s_goal: float
    feasible: bool
    layout: str              # "rows", "scatter", "wall"
    oracle_path: np.ndarray | None = None  # (K,3) poses of one feasible lattice path

    @property
    def goal_line(self) -> np.ndarray:
        """(2,2) world endpoints (right, left) of the goal line."""
        h = self.track.width / 2
        return self.track.point(np.array([self.s_goal, self.s_goal]), np.array([-h, h]))

    @property
    def n(self) -> int:
        return len(self.centres)


def _d_limit(cfg: SceneConfig, r: float) -> float:
    return cfg.track_width / 2 - cfg.edge_keepout - r


class _Placer:
    def __init__(self, cfg: SceneConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.sd: list[tuple[float, float]] = []
        self.cls: list[int] = []

    def ok(self, s: float, d: float, r: float) -> bool:
        if abs(d) > _d_limit(self.cfg, r):
            return False
        for (s2, d2) in self.sd:
            if (s - s2) ** 2 + (d - d2) ** 2 < self.cfg.min_spacing ** 2:
                return False
        return True

    def add(self, s, d, c):
        self.sd.append((float(s), float(d)))
        self.cls.append(c)

    def scatter(self, n: int, c: int, s_rng: tuple[float, float], r: float):
        placed, tries = 0, 0
        while placed < n and tries < 200 * max(n, 1):
            tries += 1
            s = self.rng.uniform(*s_rng)
            lim = _d_limit(self.cfg, r)
            d = self.rng.uniform(-lim, lim)
            if self.ok(s, d, r):
                self.add(s, d, c)
                placed += 1
        return placed

    def row(self, s_row: float, budget: int, n_gaps: int, r: float) -> int:
        """A lateral wall of bollards with ``n_gaps`` door-sized openings. Returns count placed."""
        cfg, rng = self.cfg, self.rng
        lim = _d_limit(cfg, r)
        gaps = []
        for _ in range(n_gaps):
            w = rng.uniform(2.6, 4.0)       # centre-to-centre width of the opening
            gc = rng.uniform(-lim + w / 2, lim - w / 2)
            gaps.append((gc - w / 2, gc + w / 2))
        d = -lim + rng.uniform(0, 0.4)
        if rng.random() < 0.5:              # build from the other edge half the time
            sign = -1
        else:
            sign = 1
        placed = 0
        while d <= lim and placed < budget:
            in_gap = [g for g in gaps if g[0] < d < g[1]]
            if in_gap:
                d = in_gap[0][1]
                if self.ok(s_row, sign * d, r):
                    self.add(s_row, sign * d, BOLLARD)
                    placed += 1
                d += rng.uniform(*cfg.row_spacing)
                continue
            s = s_row + rng.uniform(-0.25, 0.25)
            if self.ok(s, sign * d, r):
                self.add(s, sign * d, BOLLARD)
                placed += 1
            d += rng.uniform(*cfg.row_spacing)
        return placed


def _sample_layout(rng: np.random.Generator, cfg: SceneConfig, s0: float, wall: bool):
    s1 = s0 + cfg.field_length
    p = _Placer(cfg, rng)
    nb = int(rng.integers(cfg.n_bollards[0], cfg.n_bollards[1] + 1))
    nc = int(rng.integers(cfg.n_cones[0], cfg.n_cones[1] + 1))
    rb = cfg.bollard_radius
    if wall:
        layout = "wall"
        s_w = rng.uniform(s0 + 6.0, s1 - 2.0)
        nb = max(nb, 10)
        used = p.row(s_w, nb, 0, rb)
        p.scatter(nb - used, BOLLARD, (s0, s1), rb)
    elif rng.random() < cfg.p_rows:
        layout = "rows"
        n_rows = 1 if nb < 10 else int(rng.integers(1, 3))
        if n_rows == 2:
            a = rng.uniform(s0 + 2.0, s1 - cfg.row_min_gap_s - 2.0)
            rows = [a, rng.uniform(a + cfg.row_min_gap_s, s1 - 1.0)]
        else:
            rows = [rng.uniform(s0 + 3.0, s1 - 3.0)]
        left = nb
        for i, sr in enumerate(rows):
            budget = left if i == len(rows) - 1 else min(left, nb // len(rows) + 1)
            left -= p.row(sr, budget, int(rng.integers(1, 3)), rb)
        p.scatter(left, BOLLARD, (s0, s1), rb)
    else:
        layout = "scatter"
        p.scatter(nb, BOLLARD, (s0, s1), rb)
    p.scatter(nc, CONE, (s0, s1), cfg.cone_radius)
    sd = np.array(p.sd, float).reshape(-1, 2)
    cls = np.array(p.cls, np.int8)
    return sd, cls, layout


def oracle_feasible(track: Track, centres: np.ndarray, radii: np.ndarray, s_goal: float,
                    cfg: SceneConfig, veh: VehicleConfig, start=(0.0, 0.0, 0.0),
                    return_path: bool = False):
    """Lattice search: can Juno reach the goal line with |kappa| <= frac * kappa_max?

    Forward-only arcs of length ``oracle_step`` with ``oracle_n_kappa`` curvatures, states
    de-duplicated on a (s, d, heading-error) grid, Juno box checked every 0.25 m along each arc
    against obstacle discs inflated by 0.25 m and the track edges.
    """
    kmax = veh.kappa_max * cfg.oracle_kappa_frac
    ks = np.linspace(-kmax, kmax, cfg.oracle_n_kappa)
    L = cfg.oracle_step
    rad = radii + cfg.oracle_inflation
    h = track.width / 2
    psi_res = np.radians(cfg.oracle_psi_res_deg)
    x_rng = veh.body_x

    front = np.array([track.pose(start[0], start[1], start[2])])
    parents: list[np.ndarray] = []
    states: list[np.ndarray] = [front]
    visited: set[int] = set()
    n_iter = int((s_goal - start[0]) / L * 1.6) + 5

    def collide(P):
        bad = np.zeros(len(P), bool)
        if len(centres):
            near = np.abs(track.frenet(centres)[0][None, :] - track.frenet(P[:, :2])[0][:, None]) < 4.0
            if near.any():
                clr = box_circle_clearance(P, centres, rad, x_rng, veh.width)
                bad |= ((clr < 0) & near).any(1)
        cs = footprint_corners(P, x_rng, veh.width)
        _, dc = track.frenet(cs.reshape(-1, 2))
        bad |= (np.abs(dc.reshape(-1, 4)) > h).any(1)
        return bad

    for _ in range(n_iter):
        if len(front) == 0:
            return (False, None) if return_path else False
        F = len(front)
        th = front[:, 2][:, None]
        k = ks[None, :]
        out = []
        for frac in (0.25, 0.5, 0.75, 1.0):   # every 0.25 m, same as the planner's lattice
            l = L * frac
            dth = k * l
            small = np.abs(k) < 1e-9
            ksafe = np.where(small, 1.0, k)
            dx = np.where(small, l * np.cos(th), (np.sin(th + dth) - np.sin(th)) / ksafe)
            dy = np.where(small, l * np.sin(th), (np.cos(th) - np.cos(th + dth)) / ksafe)
            out.append(np.stack([front[:, 0:1] + dx, front[:, 1:2] + dy,
                                 np.broadcast_to(th + dth, dx.shape)], -1).reshape(-1, 3))
        end = out[-1]
        par = np.repeat(np.arange(F), len(ks))
        ok = np.ones(len(end), bool)
        for smp in out:
            ok[ok] = ~collide(smp[ok])
        end, par = end[ok], par[ok]
        if len(end) == 0:
            front = end
            continue
        s, d = track.frenet(end[:, :2])
        psi = (end[:, 2] - track.heading(s) + np.pi) % (2 * np.pi) - np.pi
        keep = np.abs(psi) < np.radians(60)
        end, par, s, d, psi = end[keep], par[keep], s[keep], d[keep], psi[keep]
        if (s >= s_goal).any():
            if not return_path:
                return True
            i = int(np.argmax(s >= s_goal))
            path = [end[i]]
            j = par[i]
            for lvl in range(len(parents) - 1, -1, -1):
                path.append(states[lvl + 1][j])
                j = parents[lvl][j]
            path.append(states[0][0])
            return True, np.array(path[::-1])
        key = (np.round(s / (L / 2)).astype(np.int64) * 1_000_003
               + np.round(d / cfg.oracle_d_res).astype(np.int64) * 1009
               + np.round(psi / psi_res).astype(np.int64))
        _, first = np.unique(key, return_index=True)
        first = np.array([i for i in first if int(key[i]) not in visited], dtype=np.int64)
        visited.update(int(x) for x in key[first])
        front = end[first]
        parents.append(par[first])
        states.append(front)
    return (False, None) if return_path else False


def generate_scene(seed: int, cfg: SceneConfig | None = None, veh: VehicleConfig | None = None,
                   want_no_route: bool | None = None) -> Scene:
    """Deterministic in ``seed``. About ``p_no_route`` of scenes have no feasible route."""
    cfg = cfg or SceneConfig.from_yaml()
    veh = veh or VehicleConfig.from_yaml()
    rng = np.random.default_rng(seed)
    if want_no_route is None:
        want_no_route = bool(rng.random() < cfg.p_no_route)
    if rng.random() < cfg.p_straight:
        kappa = 0.0
    else:
        kappa = float(rng.choice([-1, 1]) * rng.uniform(0.0, cfg.kappa_max_track))
    s0 = float(rng.uniform(*cfg.field_start))
    s_goal = s0 + cfg.field_length + cfg.goal_margin
    track = Track(kappa, cfg.track_width, -cfg.track_back, s_goal + cfg.track_after_goal)
    use_wall = want_no_route and rng.random() < 0.5

    last = None
    for attempt in range(cfg.max_resample):
        if want_no_route and attempt >= 15:   # random layouts rarely block; fall back to a wall
            use_wall = True
        sd, cls, layout = _sample_layout(rng, cfg, s0, wall=use_wall)
        centres = track.point(sd[:, 0], sd[:, 1]).reshape(-1, 2)
        radii = np.where(cls == BOLLARD, cfg.bollard_radius, cfg.cone_radius)
        heights = np.where(cls == BOLLARD, cfg.bollard_height, cfg.cone_height)
        feas, path = oracle_feasible(track, centres, radii, s_goal, cfg, veh, return_path=True)
        last = Scene(seed, track, centres, radii, heights, cls, (s0, s0 + cfg.field_length),
                     s_goal, bool(feas), layout, path)
        if feas != want_no_route:
            return last
    return last  # label is still the oracle's verdict
