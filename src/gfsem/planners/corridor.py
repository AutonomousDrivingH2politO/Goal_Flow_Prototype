"""Delaunay corridor planner (the team's agreed design), in privileged and observed modes.

1. Delaunay triangulation over obstacle centres + track-edge points every 1 m.
2. Each edge is a door (gap - r_a - r_b >= width + 2 * clearance) or a wall. In observed mode
   doors whose midpoint lies in an occlusion shadow / unknown space are tentative (penalised).
3. A* over triangles to the goal line, cost grows as doors narrow.
4. Commit only to the next door (hysteresis across replans, kept in the odom frame).
5. Line-of-sight smoothing (funnel algorithm through shrunk portals), then a curvature check:
   the path is rolled out with the real controller and steering limit; a door the rollout
   cannot pass without collision is blocked and A* reruns.

This module is the "corridor-funnel (original design)" baseline. Finding (docs/DEVIATIONS.md):
funnel smoothing + pure pursuit cannot reliably produce paths that a 10 deg steering limit can
follow. ``corridor_lattice.py`` keeps steps 1-4 and 6 and replaces step 5.
6. Brake when there is no route.

Everything happens in the ego frame (rear axle, x forward, y left).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, cKDTree

from ..sim.config import BOLLARD, CONE, SceneConfig, VehicleConfig
from ..sim.edges import observed_edges
from ..sim.geometry import box_circle_clearance, boxes_hit_points, from_frame, to_frame
from ..sim.lidar import Observation
from ..sim.vehicle import VehicleState, resample_polyline, rollout_path

OBST, LEFT, RIGHT, GOAL, BACK = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class CorridorConfig:
    edge_spacing: float = 1.0
    door_clearance: float = 0.25          # per side, the door rule
    portal_margin: float = 0.25           # extra clearance beyond half-width when string pulling
    centre_margin_extra: float = 0.35     # second attempt: aim closer to door centres
    gap_pref: float = 3.5                 # doors narrower than this get a penalty
    w_narrow: float = 1.5
    tentative_penalty: float = 4.0
    commit_bonus: float = 4.0
    commit_match: float = 0.8
    w_turn: float = 3.0                   # penalty per radian of heading change between portals
    max_iters: int = 8
    rollout_horizon: float = 45.0
    rollout_ds: float = 0.25
    check_inflation: float = 0.25         # one margin: = door clearance = oracle inflation
    back_x: float = -3.0
    # observed-mode perception
    cluster_tol: float = 0.35
    min_cluster_pts: int = 1
    edge_bin: float = 1.0


@dataclass
class PlannerInput:
    centres: np.ndarray
    radii: np.ndarray
    left: np.ndarray                     # (L,2) ordered from back to front
    right: np.ndarray
    goal_line: np.ndarray                # (2,2) right, left
    unknown_fn: object = None            # callable (N,2) -> bool mask, or None (everything known)
    pose_odom: np.ndarray | None = None  # for commit memory


@dataclass
class PlanResult:
    path: np.ndarray                     # (K,3) ego-frame poses; empty if braking
    brake: bool
    doors: list = field(default_factory=list)   # [(pa, pb)] ego-frame door endpoints in order
    exit_point: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    @property
    def speed(self) -> float:
        return 0.0 if self.brake else float(self.info.get("speed", 3.0))


# ---------------------------------------------------------------- helpers

def _cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def string_pull(start: np.ndarray, portals: list[tuple[np.ndarray, np.ndarray]], end: np.ndarray) -> np.ndarray:
    """Simple stupid funnel algorithm. ``portals`` are (left, right) as seen when travelling."""
    P = [(start, start)] + list(portals) + [(end, end)]
    apex, left, right = start, start, start
    ai = li = ri = 0
    out = [start]
    i = 1
    guard = 0
    while i < len(P) and guard < 10 * len(P) + 10:
        guard += 1
        pl, pr = P[i]
        if _cross(apex, right, pr) >= 0:            # right side tightens
            if np.allclose(apex, right) or _cross(apex, left, pr) < 0:
                right, ri = pr, i
            else:
                out.append(left)
                apex, ai = left, li
                left = right = apex
                li = ri = ai
                i = ai + 1
                continue
        if _cross(apex, left, pl) <= 0:             # left side tightens
            if np.allclose(apex, left) or _cross(apex, right, pl) > 0:
                left, li = pl, i
            else:
                out.append(right)
                apex, ai = right, ri
                left = right = apex
                li = ri = ai
                i = ai + 1
                continue
        i += 1
    if not np.allclose(out[-1], end):
        out.append(end)
    return np.array(out)


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    cond = (y1 > y) != (y2 > y)
    xint = (x2 - x1) * (y - y1) / np.where(y2 == y1, 1e-12, y2 - y1) + x1
    return ((cond & (x < xint)).sum(1) % 2) == 1


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(a + t * ab - p))


def _polyline_dist(p: np.ndarray, line: np.ndarray) -> float:
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    q = a + t[:, None] * ab
    return float(np.hypot(*(q - p).T).min())


def _arclen(e: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(e, axis=0).T))])


def _cut(e: np.ndarray, s_cut: float) -> np.ndarray:
    s = _arclen(e)
    keep = s < s_cut
    keep[0] = True                      # never cut an edge down to a single point
    p = np.array([np.interp(s_cut, s, e[:, 0]), np.interp(s_cut, s, e[:, 1])])
    return np.concatenate([e[keep], p[None]])


def _extend_back(e: np.ndarray, x_back: float) -> np.ndarray:
    if e[0, 0] <= x_back + 0.5:
        return e
    k = min(len(e) - 1, 4)
    d = e[k] - e[0]
    n = np.linalg.norm(d)
    d = d / n if n > 1e-6 else np.array([1.0, 0.0])
    if d[0] < 0.3:
        d = np.array([1.0, 0.0])
    back = e[0] - d * (e[0, 0] - x_back) / d[0]
    return np.concatenate([back[None], e])


# ---------------------------------------------------------------- inputs

def input_from_privileged(view: dict) -> PlannerInput:
    return PlannerInput(view["centres"], view["radii"], view["left"], view["right"], view["goal_line"])


def fit_circle_fixed_r(pts: np.ndarray, r: float, sensor: np.ndarray, iters: int = 8) -> np.ndarray:
    c0 = pts.mean(0)
    d = c0 - sensor
    c = c0 + r * d / max(np.linalg.norm(d), 1e-6)
    if len(pts) < 3:
        return c
    for _ in range(iters):
        v = c - pts
        n = np.linalg.norm(v, axis=1)
        n = np.maximum(n, 1e-6)
        res = n - r
        J = v / n[:, None]
        try:
            step = np.linalg.lstsq(J, -res, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        c = c + step
        if np.linalg.norm(step) < 1e-4:
            break
    if np.linalg.norm(c - c0) > 2 * r:            # degenerate fit -> fallback
        c = c0 + r * d / max(np.linalg.norm(d), 1e-6)
    return c


def cluster_obstacles(obs: Observation, cfg: CorridorConfig, scfg: SceneConfig):
    """Euclidean clustering + fixed-radius circle fit per cluster (radius from its camera class)."""
    P = obs.points.astype(np.float64)
    if len(P) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int8)
    tree = cKDTree(P)
    pairs = tree.query_pairs(cfg.cluster_tol, output_type="ndarray")
    n = len(P)
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) \
        else coo_matrix((n, n))
    k, lab = connected_components(A, directed=False)
    sensor = np.asarray(obs.sensor_xy, float)
    C, R, K = [], [], []
    for i in range(k):
        m = lab == i
        if m.sum() < cfg.min_cluster_pts:
            continue
        pts = P[m]
        cls = BOLLARD if (obs.labels[m] == BOLLARD).sum() * 2 >= m.sum() else CONE
        r = scfg.bollard_radius if cls == BOLLARD else scfg.cone_radius
        C.append(fit_circle_fixed_r(pts, r, sensor))
        R.append(r)
        K.append(cls)
    return np.array(C).reshape(-1, 2), np.array(R), np.array(K, np.int8)


def make_unknown_fn(obs: Observation):
    ang0, step = obs.ray_angle0, obs.ray_step
    n = len(obs.scan_ranges)
    sens = np.asarray(obs.sensor_xy, float)
    rng = np.nan_to_num(obs.scan_ranges, nan=0.0)

    def fn(pts: np.ndarray) -> np.ndarray:
        d = np.asarray(pts, float) - sens
        r = np.hypot(d[:, 0], d[:, 1])
        b = np.arctan2(d[:, 1], d[:, 0])
        i = np.round((b - ang0) / step).astype(int)
        out_fov = (i < 0) | (i >= n)
        ic = np.clip(i, 0, n - 1)
        return out_fov | (r > rng[ic] + 0.3) | (r < obs.blind_zone)
    return fn


def input_from_observation(obs: Observation, cfg: CorridorConfig, scfg: SceneConfig) -> PlannerInput:
    C, R, _ = cluster_obstacles(obs, cfg, scfg)
    left, right = observed_edges(obs.edge_points, obs.edge_side, scfg.track_width, cfg.edge_bin)
    return PlannerInput(C, R, left, right, obs.goal_line.astype(float), make_unknown_fn(obs),
                        np.asarray(obs.pose_odom, float))


# ---------------------------------------------------------------- planner

class CorridorPlanner:
    def __init__(self, veh: VehicleConfig | None = None, scfg: SceneConfig | None = None,
                 cfg: CorridorConfig | None = None, lookahead: float = 2.5, mode: str = "privileged",
                 cost_noise: float = 0.0, lr_bias: float | None = 0.0,
                 rng: np.random.Generator | None = None):
        """``cost_noise``: +-fraction of random multiplicative edge-cost noise per plan.
        ``lr_bias``: cost per metre of door lateral offset (negative favours left);
        None draws U(-0.5, 0.5) per plan. Both are for multimodal teacher replans."""
        self.veh = veh or VehicleConfig.from_yaml()
        self.scfg = scfg or SceneConfig.from_yaml()
        self.cfg = cfg or CorridorConfig()
        self.lookahead = lookahead
        self.mode = mode
        self.cost_noise = cost_noise
        self.lr_bias = lr_bias
        self.rng = rng or np.random.default_rng(0)
        self.commit: np.ndarray | None = None    # (2,2) committed door endpoints, odom frame
        self._roll_veh = self.veh
        self.reset()

    def reset(self):
        self.commit = None
        self._last_chain: list[int] = []
        self._last_cost: float = float("inf")

    # -- public
    def plan_observation(self, obs: Observation, speed: float, steer: float) -> PlanResult:
        inp = input_from_observation(obs, self.cfg, self.scfg)
        return self.plan(inp, speed, steer)

    def _match_doors(self, G, pts_local: np.ndarray, tol: float) -> list[tuple[int, int]]:
        """Graph edges whose endpoints match the door endpoints ``pts_local`` (2,2) within tol."""
        V = G["V"]
        da = np.hypot(*(V - pts_local[0]).T)
        db = np.hypot(*(V - pts_local[1]).T)
        ia = np.nonzero(da < tol)[0]
        ib = np.nonzero(db < tol)[0]
        return [tuple(sorted((int(i), int(j)))) for i in ia for j in ib if i != j]

    def _commit_local(self, odom):
        if self.commit is None:
            return None
        loc = to_frame(self.commit, odom) if odom is not None else self.commit
        if loc.mean(0)[0] < 0.5:                    # passed it
            self.commit = None
            return None
        return loc

    def _set_commit(self, G, doors, odom):
        """Commit to the next door that involves an obstacle."""
        nxt = [(a, b) for a, b in doors if G["T"][a] == OBST or G["T"][b] == OBST]
        if nxt:
            pts = np.stack([G["V"][nxt[0][0]], G["V"][nxt[0][1]]])
            self.commit = from_frame(pts, odom) if odom is not None else pts

    def plan(self, inp: PlannerInput, speed: float, steer: float) -> PlanResult:
        """Original design: funnel smoothing, rollout curvature check, block failing door, brake."""
        cfg = self.cfg
        G = self._build(inp)
        if G is None:
            return PlanResult(np.zeros((0, 3)), True, info={"reason": "no edges"})
        odom = inp.pose_odom
        commit_local = self._commit_local(odom)
        start = VehicleState(0.0, 0.0, 0.0, speed, steer)
        reason = "no route"
        blocked: set[tuple[int, int]] = set()
        for it in range(cfg.max_iters):
            seq = self._astar(G, blocked, commit_local)
            if seq is None:
                reason = "no route" if it == 0 else "all doors blocked by curvature check"
                break
            doors, exit_edge = seq
            res = None
            for extra in (0.0, cfg.centre_margin_extra):
                res = self._smooth_and_check(G, doors, exit_edge, start, extra)
                if res["ok"]:
                    break
            if res["ok"]:
                self._set_commit(G, doors, odom)
                door_pts = [(G["V"][a], G["V"][b]) for a, b in doors]
                return PlanResult(res["path"], False, door_pts, res["exit"],
                                  info={"iters": it + 1, "speed": self.scfg.target_speed,
                                        "polyline": res["polyline"], "tentative": res["tentative"]})
            if res["fail_door"] is None:
                reason = "rollout failed with no doors"
                break
            blocked.add(tuple(sorted(doors[res["fail_door"]])))
        return PlanResult(np.zeros((0, 3)), True, info={"reason": reason, "graph": G})

    # -- graph construction
    def _build(self, inp: PlannerInput):
        cfg, veh = self.cfg, self.veh
        if len(inp.left) < 2 or len(inp.right) < 2:
            return None
        left = _extend_back(inp.left[np.argsort(inp.left[:, 0])], cfg.back_x)
        right = _extend_back(inp.right[np.argsort(inp.right[:, 0])], cfg.back_x)
        # planning horizon: goal line if it lies within the observed edges, else the edge frontier
        gr, gl = inp.goal_line[0], inp.goal_line[1]
        sL, sR = _arclen(left), _arclen(right)
        iL = int(np.argmin(np.hypot(*(left - gl).T)))
        iR = int(np.argmin(np.hypot(*(right - gr).T)))
        goal_inside = iL < len(left) - 1 and iR < len(right) - 1
        if goal_inside:
            gL, gR = gl, gr
            # cut edges where they meet the goal line
            left = _cut(left, sL[iL] + float(np.dot(gl - left[iL], left[min(iL + 1, len(left) - 1)] - left[iL])
                                              / max(np.linalg.norm(left[min(iL + 1, len(left) - 1)] - left[iL]), 1e-6)))
            right = _cut(right, sR[iR] + float(np.dot(gr - right[iR], right[min(iR + 1, len(right) - 1)] - right[iR])
                                                / max(np.linalg.norm(right[min(iR + 1, len(right) - 1)] - right[iR]), 1e-6)))
            left[-1], right[-1] = gL, gR
        else:
            # frontier: shorter side's end (1 m back), closest point on the other side
            if sL[-1] - 1.0 <= sR[-1] - 1.0:
                left = _cut(left, max(sL[-1] - 1.0, 0.5))
                j = int(np.argmin(np.hypot(*(right - left[-1]).T)))
                right = right[: j + 1]
            else:
                right = _cut(right, max(sR[-1] - 1.0, 0.5))
                j = int(np.argmin(np.hypot(*(left - right[-1]).T)))
                left = left[: j + 1]
            gL, gR = left[-1], right[-1]
        if len(left) < 2 or len(right) < 2:
            return None
        Ls = resample_polyline(left, cfg.edge_spacing)
        Rs = resample_polyline(right, cfg.edge_spacing)
        Gs = resample_polyline(np.stack([gL, gR]), cfg.edge_spacing)[1:-1]
        Bs = resample_polyline(np.stack([Rs[0], Ls[0]]), cfg.edge_spacing)[1:-1]
        poly = np.concatenate([Ls, Gs, Rs[::-1], Bs])
        # obstacles inside the planning polygon (centre within r of it counts)
        C, Rr = inp.centres, inp.radii
        if len(C):
            inside = points_in_polygon(C, poly)
            C, Rr = C[inside], Rr[inside]
        V = np.concatenate([C, Ls, Rs, Gs, Bs]) if len(C) else np.concatenate([Ls, Rs, Gs, Bs])
        T = np.concatenate([np.full(len(C), OBST), np.full(len(Ls), LEFT), np.full(len(Rs), RIGHT),
                            np.full(len(Gs), GOAL), np.full(len(Bs), BACK)])
        rad = np.concatenate([Rr, np.zeros(len(V) - len(C))])
        # the two goal-line endpoints count as goal vertices too (for exit edges)
        is_goal = T == GOAL
        is_goal[len(C) + len(Ls) - 1] = True
        is_goal[len(C) + len(Ls) + len(Rs) - 1] = True
        # position of each boundary vertex along the polygon (Ls, Gs, Rs reversed, Bs); -1 = obstacle
        nC, nL, nR, nG, nB = len(C), len(Ls), len(Rs), len(Gs), len(Bs)
        bpos = np.full(len(V), -1)
        bpos[nC:nC + nL] = np.arange(nL)
        bpos[nC + nL + nR:nC + nL + nR + nG] = nL + np.arange(nG)
        bpos[nC + nL:nC + nL + nR] = nL + nG + np.arange(nR)[::-1]
        bpos[nC + nL + nR + nG:] = nL + nG + nR + np.arange(nB)
        try:
            tri = Delaunay(V)
        except Exception:
            tri = Delaunay(V + np.random.default_rng(0).normal(0, 1e-6, V.shape))
        S = tri.simplices
        cen = V[S].mean(1)
        valid = points_in_polygon(cen, poly)
        thr = veh.width + 2 * cfg.door_clearance
        thr_edge = veh.width + cfg.door_clearance
        unknown = inp.unknown_fn
        return dict(V=V, T=T, rad=rad, is_goal=is_goal, bpos=bpos, nB=len(poly), tri=tri, S=S, cen=cen, valid=valid,
                    poly=poly, thr=thr, thr_edge=thr_edge, unknown=unknown, nC=len(C),
                    left=Ls, right=Rs, goal=(gL, gR), goal_inside=goal_inside)

    def _edge_kind(self, G, a: int, b: int):
        """Returns ('wall'|'door'|'exit', gap). Walls: consecutive boundary vertices, or an
        obstacle gap below the door rule. Exits: consecutive vertices on the goal line."""
        V, rad, bpos = G["V"], G["rad"], G["bpos"]
        pa, pb = bpos[a], bpos[b]
        if pa >= 0 and pb >= 0:
            dp = abs(int(pa) - int(pb))
            if dp == 1 or dp == G["nB"] - 1:
                if G["is_goal"][a] and G["is_goal"][b]:
                    return "exit", float(np.linalg.norm(V[a] - V[b]))
                return "wall", 0.0
        T = G["T"]
        if (T[a] == OBST) != (T[b] == OBST) and {T[a], T[b]} & {LEFT, RIGHT}:
            # obstacle <-> track edge: gap to the edge *line*, clearance on the obstacle side only
            # (the oracle keeps 0.25 m from obstacles and 0 m from the track edge)
            o, e = (a, b) if T[a] == OBST else (b, a)
            line = G["left"] if T[e] == LEFT else G["right"]
            d = _polyline_dist(V[o], line)
            gap = float(d - rad[o])
            return ("door" if gap >= G["thr_edge"] else "wall"), gap
        gap = float(np.linalg.norm(V[a] - V[b]) - rad[a] - rad[b])
        return ("door" if gap >= G["thr"] else "wall"), gap

    # -- search
    def _astar(self, G, blocked, commit_local):
        cfg = self.cfg
        tri, S, valid, V, cen = G["tri"], G["S"], G["valid"], G["V"], G["cen"]
        start_pt = np.array([0.8, 0.0])
        t0 = int(tri.find_simplex(start_pt))
        if t0 < 0 or not valid[t0]:
            vi = np.nonzero(valid)[0]
            if len(vi) == 0:
                return None
            t0 = int(vi[np.argmin(np.hypot(*(cen[vi] - start_pt).T))])
        gL, gR = G["goal"]
        cache: dict = G.setdefault("edge_cache", {})   # shared across A* calls on this graph
        bias = self.lr_bias if self.lr_bias is not None else float(self.rng.uniform(-0.5, 0.5))
        noise_cache: dict = {}

        def edge_info(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in cache:
                kind, gap = self._edge_kind(G, *key)
                tent = False
                if kind == "door" and G["unknown"] is not None:
                    tent = bool(G["unknown"]((V[a] + V[b])[None] / 2)[0])
                cache[key] = (kind, gap, tent)
            return cache[key]

        def h(p):
            return _seg_dist(p, gL, gR)

        start_pos = start_pt
        openq = [(h(start_pos), 0.0, t0, -1)]
        best = {t0: 0.0}
        parent: dict[int, tuple[int, tuple[int, int]] | None] = {t0: None}
        pos = {t0: start_pos}
        head = {t0: 0.0}
        goal_tri, exit_edge = None, None
        while openq:
            f, g, t, _ = heapq.heappop(openq)
            if g > best.get(t, np.inf) + 1e-9:
                continue
            # exit?
            simp = S[t]
            found = False
            for k in range(3):
                a, b = int(simp[(k + 1) % 3]), int(simp[(k + 2) % 3])
                kind, _, _ = edge_info(a, b)
                if kind == "exit":
                    goal_tri, exit_edge, found = t, (a, b), True
                    break
            if found:
                break
            for k in range(3):
                nb = int(tri.neighbors[t, k])
                if nb < 0 or not valid[nb]:
                    continue
                a, b = int(simp[(k + 1) % 3]), int(simp[(k + 2) % 3])
                kind, gap, tent = edge_info(a, b)
                if kind != "door":
                    continue
                key = (a, b) if a < b else (b, a)
                if key in blocked:
                    continue
                mid = (V[a] + V[b]) / 2
                step = mid - pos[t]
                dist = float(np.hypot(*step))
                hd = math.atan2(step[1], step[0]) if dist > 1e-6 else head[t]
                dturn = abs((hd - head[t] + math.pi) % (2 * math.pi) - math.pi)
                c = dist + cfg.w_narrow * max(0.0, cfg.gap_pref - gap) ** 2 + cfg.w_turn * dturn
                if tent:
                    c += cfg.tentative_penalty
                if self.cost_noise > 0:
                    if key not in noise_cache:
                        noise_cache[key] = 1.0 + float(self.rng.uniform(-self.cost_noise, self.cost_noise))
                    c *= noise_cache[key]
                if bias:
                    c = max(0.05, c + bias * float(mid[1]))
                if commit_local is not None:
                    if np.linalg.norm(mid - commit_local.mean(0)) < cfg.commit_match:
                        c = max(0.05, c - cfg.commit_bonus)
                ng = g + c
                if ng < best.get(nb, np.inf) - 1e-9:
                    best[nb] = ng
                    parent[nb] = (t, (a, b))
                    pos[nb] = mid
                    head[nb] = hd
                    heapq.heappush(openq, (ng + h(mid), ng, nb, t))
        if goal_tri is None:
            return None
        self._last_cost = float(best[goal_tri])
        doors, chain = [], [goal_tri]
        t = goal_tri
        while parent[t] is not None:
            pt, e = parent[t]
            doors.append(e)
            chain.append(pt)
            t = pt
        self._last_chain = chain[::-1]
        return doors[::-1], exit_edge

    # -- smoothing + curvature check
    def _portal(self, G, a, b, margin):
        V, rad = G["V"], G["rad"]
        pa, pb = V[a], V[b]
        d = pb - pa
        L = float(np.linalg.norm(d))
        u = d / max(L, 1e-9)
        lo = rad[a] + margin
        hi = L - rad[b] - margin
        if hi < lo:
            m = pa + u * (rad[a] + (L - rad[a] - rad[b]) / 2)
            return m, m
        return pa + u * lo, pa + u * hi

    def _goal_normal(self, G, toward: np.ndarray):
        gL, gR = G["goal"]
        gdir = gL - gR
        nrm = np.array([gdir[1], -gdir[0]])
        nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
        mid = (gL + gR) / 2
        if np.dot(nrm, mid - toward) < 0:
            nrm = -nrm
        return nrm

    def _smooth_and_check(self, G, doors, exit_edge, start: VehicleState, extra: float):
        """Funnel path through shrunk portals, then the rollout check."""
        cfg, veh = self.cfg, self.veh
        V = G["V"]
        margin = veh.width / 2 + cfg.portal_margin + extra
        portals = []
        prev_c = np.array([0.8, 0.0])
        seq = list(doors) + [exit_edge]
        for i, (a, b) in enumerate(seq):
            if i < len(doors):
                p1, p2 = self._portal(G, a, b, margin)
            else:  # exit edge: plain segment on the goal line
                p1, p2 = V[a], V[b]
            mid = (p1 + p2) / 2
            if i + 1 < len(seq):
                nxt = (V[seq[i + 1][0]] + V[seq[i + 1][1]]) / 2
            else:
                nxt = mid + (mid - prev_c)
            tdir = nxt - prev_c
            if _cross(prev_c, prev_c + tdir, p1) >= 0:
                portals.append((p1, p2))
            else:
                portals.append((p2, p1))
            prev_c = mid
        exitL, exitR = portals[-1]
        end = (exitL + exitR) / 2
        poly = string_pull(np.array([0.0, 0.0]), portals[:-1], end)
        return self._rollout_check(G, poly, start, doors)

    def _rollout_check(self, G, poly: np.ndarray, start: VehicleState, doors) -> dict:
        """Track ``poly`` with pure pursuit under the steering limit; reject on collision."""
        cfg, veh = self.cfg, self.veh
        V = G["V"]
        gL, gR = G["goal"]
        nrm = self._goal_normal(G, np.zeros(2))
        end = poly[-1]
        poly_ext = np.concatenate([poly, (end + 6.0 * nrm)[None]])
        dense = resample_polyline(poly_ext, cfg.rollout_ds)
        total = float(_arclen(poly).max())
        horizon = min(total + 1.0, cfg.rollout_horizon)
        roll = rollout_path(dense, start, veh, self.lookahead, max(start.v, 1.0),
                            ds=cfg.rollout_ds, max_len=horizon)
        before = (roll[:, :2] - gR) @ nrm < 0
        chk = roll[before] if before.any() else roll[:1]
        bad = self._collides(G, chk, cfg.check_inflation)
        tentative = []
        if G["unknown"] is not None and doors:
            mids = np.stack([(V[a] + V[b]) / 2 for a, b in doors])
            tentative = list(G["unknown"](mids))
        if not bad.any():
            return dict(ok=True, path=roll, exit=end, polyline=poly, tentative=tentative)
        k = int(np.argmax(bad))
        # blame the door nearest the collision (body centre); narrower doors win ties
        fail_door = None
        if doors:
            x_rng = veh.body_x
            pc = chk[k]
            c = pc[:2] + (x_rng[0] + x_rng[1]) / 2 * np.array([np.cos(pc[2]), np.sin(pc[2])])
            score = []
            for a, b in doors:
                gap = np.linalg.norm(V[a] - V[b]) - G["rad"][a] - G["rad"][b]
                score.append(_seg_dist(c, V[a], V[b]) + 0.2 * gap)
            fail_door = int(np.argmin(score))
        return dict(ok=False, fail_door=fail_door, path=roll, exit=end, polyline=poly, tentative=tentative)

    def _edge_pts(self, G):
        if "edge_pts" not in G:
            G["edge_pts"] = np.concatenate([resample_polyline(G["left"], 0.25),
                                            resample_polyline(G["right"], 0.25)])
        return G["edge_pts"]

    def _collides(self, G, poses: np.ndarray, inflation: float) -> np.ndarray:
        veh = self.veh
        x_rng = veh.body_x
        nC = G["nC"]
        bad = np.zeros(len(poses), bool)
        if nC:
            clr = box_circle_clearance(poses, G["V"][:nC], G["rad"][:nC] + inflation, x_rng, veh.width)
            bad |= (clr < 0).any(1)
        bad |= boxes_hit_points(poses, self._edge_pts(G), x_rng, veh.width, inflate=0.0)
        return bad


def _crossed(traj: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """Has the polyline ``traj`` crossed the line through door (a, b), within the door span (+1 m)?"""
    if len(traj) < 2:
        return False
    d = b - a
    s = (traj[:, 0] - a[0]) * d[1] - (traj[:, 1] - a[1]) * d[0]
    ch = np.nonzero(np.sign(s[:-1]) != np.sign(s[1:]))[0]
    if len(ch) == 0:
        return False
    L2 = float(np.dot(d, d))
    for i in ch:
        t = float(np.dot(traj[i] - a, d)) / max(L2, 1e-9)
        if -1.0 / math.sqrt(L2) <= t <= 1 + 1.0 / math.sqrt(L2):
            return True
    return False
