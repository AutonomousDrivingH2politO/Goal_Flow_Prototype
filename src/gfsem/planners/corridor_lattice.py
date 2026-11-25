"""Corridor planner with a curvature-feasible smoother: A* picks the route, a lattice search
inside that route's corridor makes it drivable.

Steps 1-4 and 6 of the agreed design are unchanged (Delaunay doors/walls, A* with narrow-door
and tentative-door costs, brake when no route). Step 5 (funnel + pure-pursuit check) is replaced:

* For each of the top routes from A* (distinct door sequences), search forward arcs with
  |kappa| <= 0.8 kappa_max inside the route's corridor (its triangles and every triangle that
  shares a vertex with them), with obstacles inflated by 0.25 m (= door rule = oracle).
* Validate by simulating the closed loop (pure pursuit + bicycle, same dt) on the lattice path;
  the tracked footprint must keep the same 0.25 m. First route that passes wins.

Modes
* privileged (the teacher): the world is static and fully known, so it plans once per episode
  and replans only if the car drifts off the plan.
* observed (the baseline): event-triggered. Each replan checks the kept plan against the new
  observation; it replans only if the plan collides (0.25 m), runs short of the horizon, or the
  car deviates. Replans warm-start from the still-valid prefix of the old plan, with a smaller
  beam, and fall back to a full plan from the current pose.

Observed-mode additions (DEVIATIONS Part C, L3 items 1-4, 8, 9; each behind a config flag, all on by
default in observed mode, privileged mode unchanged):
* perception memory (``tracking.py``): tracked obstacles and remembered edge points beside the car;
* start relief: a violation the car is already in at its current pose (e.g. a fill edge drawn inside
  the true edge next to it, A6) does not reject every plan, as long as the tracked path leaves it
  within ``start_relief_m``;
* kept-plan hysteresis: a kept plan is dropped at once on a hard violation (footprint overlap, no
  inflation), but on a soft one (< ``keep_inflation``) only after ``keep_soft_scans`` scans in a row;
  new plans still need the 0.12 m acceptance margin;
* fallback and hold: if replanning fails while the kept plan has no hard violation, keep driving it;
  otherwise brake and hold until a plan is found on ``hold_confirm`` scans in a row (no brake/plan
  alternation, the L4 crawl);
* per-replan time budget ``budget_s``: past it the search stops; keep the kept plan if it has no hard
  violation, else brake;
* failed-search cache: after a failed replan, brake without searching while the car has moved less
  than ``fail_cache_move`` and the tracked obstacles are unchanged, for up to ``fail_cache_scans``.
"""
from __future__ import annotations

import heapq
import time
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from ..sim.config import SceneConfig, VehicleConfig
from ..sim.geometry import poses_from_frame, poses_to_frame
from ..sim.lattice import EXPANSIONS, CollisionWorld, LatticeConfig, lattice_search
from ..sim.lidar import Observation
from ..sim.vehicle import VehicleState, rollout_path
from ..sim.edges import observed_edges
from .corridor import (OBST, CorridorConfig, CorridorPlanner, PlannerInput, PlanResult, _arclen,
                       cluster_obstacles, input_from_observation, make_unknown_fn)
from .tracking import EdgeMemory, ObstacleTracker, TrackingConfig


@dataclass(frozen=True)
class LatticeCorridorConfig:
    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    n_routes: int = 6                 # routes tried per full plan
    obs_beam: int = 300               # observed mode: smaller beam than the teacher (800)
    obs_n_routes: int = 3
    validate_inflation: float = 0.12  # acceptance tolerance for the tracked rollout (planning uses 0.25)
    kappa_ladder: tuple = (0.8, 0.9, 0.95)   # fractions of kappa_max tried in order (0.95 = oracle)
    last_resort_beam: int = 0                # >0: privileged final rung, 0.95 kappa_max with this beam (off:
                                             # tested at 3000, rescued no L0 seed, ~5 s per failing scene)
    dt: float = 0.05                  # closed-loop validation step (= controller dt)
    deviation_replan: float = 0.5     # m off the plan -> full replan
    extend_short: float = 4.0         # observed: replan when the plan ends this far short of the horizon
    warm_backoff: float = 4.0         # m of valid prefix dropped before an invalid pose
    warm_min_prefix: float = 2.0      # shorter prefixes -> plan from the current pose
    # retries inside the same route when the tracked path fails validation: stronger clearance
    # preference (cost shaping only; the margin stays 0.25 m). (w_clear, clear_pref)
    shaping: tuple = ((5.0, 1.5), (20.0, 2.0))
    # observed-mode L3 additions (see module docstring). Every default reproduces the pre-L3 planner
    # (50e9ad9); ``l3_config()`` turns the approved set on.
    tracking: bool = False            # obstacle tracking (tracking.py)
    edge_memory: bool = False         # remembered edge points beside the car (tracking.py)
    tracking_cfg: TrackingConfig = field(default_factory=TrackingConfig)
    start_relief_m: float = 0.0       # 0 = off (not part of the approved L3 set)
    keep_hysteresis: bool = False     # kept plan: hard / soft check with persistence (off = original 0.12 m check)
    keep_inflation: float = 0.06      # soft threshold for a kept plan when keep_hysteresis (hard = overlap)
    keep_soft_scans: int = 2
    fallback: bool = False            # keep driving a kept plan with no hard violation when replanning fails
    hold_confirm: int = 1             # >1: after a brake, require this many consecutive successful plans
    budget_s: float = 0.0             # wall-clock per-replan cap; 0 = off
    fail_cache_scans: int = 0         # >0: skip searching after a failed replan while nothing changed
    fail_cache_move: float = 0.3


def l3_config(**kw) -> LatticeCorridorConfig:
    """The approved L3 observed set (2026-11-25): tracking, edge memory, kept-plan hysteresis with
    fallback, hold, failed-search cache; no time budget (set after the Orin calibration), no start relief."""
    base = dict(tracking=True, edge_memory=True, keep_hysteresis=True, fallback=True, hold_confirm=2,
                fail_cache_scans=10, budget_s=0.0, start_relief_m=0.0)
    base.update(kw)
    return LatticeCorridorConfig(**base)


class LatticeCorridorPlanner(CorridorPlanner):
    def __init__(self, veh: VehicleConfig | None = None, scfg: SceneConfig | None = None,
                 cfg: CorridorConfig | None = None, lcfg: LatticeCorridorConfig | None = None,
                 lookahead: float = 2.5, mode: str = "privileged"):
        self.lcfg = lcfg or LatticeCorridorConfig()
        super().__init__(veh, scfg, cfg, lookahead, mode)

    def reset(self):
        super().reset()
        self.plan_w: np.ndarray | None = None     # kept plan (K,3), odom frame
        self.no_route = False                     # privileged: static world, remember failure
        self.stats: list[tuple[str, float]] = []  # (trigger, ms) per call
        self.expansions: list[int] = []           # lattice successor evaluations per call
        self.fail_causes: Counter = Counter()
        self.kappa_log: list[float] = []          # curvature level each full plan needed
        tc = self.lcfg.tracking_cfg
        self.edge_mem, self.tracker = EdgeMemory(tc), ObstacleTracker(tc)
        self.soft_count = 0
        self.held = False
        self.confirm_count = 0
        self.scan = 0
        self.fail_cache = None                    # (scan, odom pose, tracker signature)
        self._deadline = None
        self._skip_search = False
        self.events: Counter = Counter()          # budget hits, cache hits, fallbacks, holds, reliefs

    # ---------------------------------------------------------------- routes
    def routes(self, G, commit_local=None, max_routes: int = 6):
        """Distinct A* routes, cheapest first (Yen-style: block one obstacle door at a time)."""
        first = self._astar(G, set(), commit_local)
        if first is None:
            return
        heap = [(self._last_cost, 0, first, list(self._last_chain), frozenset())]
        tried = {frozenset()}
        seen = set()
        tick = 1
        n = 0
        while heap and n < max_routes:
            cost, _, (doors, exit_edge), chain, blocked = heapq.heappop(heap)
            sig = tuple(sorted(tuple(sorted(d)) for d in doors if OBST in (G["T"][d[0]], G["T"][d[1]])))
            if sig in seen:
                continue
            seen.add(sig)
            n += 1
            yield doors, exit_edge, chain
            for d in doors:
                if OBST not in (G["T"][d[0]], G["T"][d[1]]):
                    continue
                b = blocked | {tuple(sorted(d))}
                if b in tried:
                    continue
                tried.add(b)
                r = self._astar(G, set(b), commit_local)
                if r is not None:
                    heapq.heappush(heap, (self._last_cost, tick, r, list(self._last_chain), b))
                    tick += 1

    def corridor_mask(self, G, chain, doors) -> np.ndarray:
        """Triangles in the route's homotopy class: flood fill from the start triangle, crossing
        only edges without an obstacle endpoint or edges equivalent to one of the route's doors
        (track-edge vertices collapse to one 'L' / 'R' vertex per side)."""
        T, S, tri, valid = G["T"], G["S"], G["tri"], G["valid"]

        def ident(v):
            return int(v) if T[v] == OBST else -int(T[v])     # boundary vertices by type

        allowed_ids = {frozenset((ident(a), ident(b))) for a, b in doors}
        mask = np.zeros(len(S), bool)
        stack = [chain[0]]
        mask[chain[0]] = True
        while stack:
            t = stack.pop()
            simp = S[t]
            for k in range(3):
                nb = int(tri.neighbors[t, k])
                if nb < 0 or mask[nb] or not valid[nb]:
                    continue
                a, b = int(simp[(k + 1) % 3]), int(simp[(k + 2) % 3])
                if T[a] == OBST or T[b] == OBST:
                    if frozenset((ident(a), ident(b))) not in allowed_ids:
                        continue
                mask[nb] = True
                stack.append(nb)
        mask[chain] = True
        return mask

    def corridor_fn(self, G, chain, doors=None):
        if doors is None:   # fallback: chain + vertex ring (not homotopy-safe)
            S = G["S"]
            allowed = G["valid"] & np.isin(S, np.unique(S[chain])).any(1)
        else:
            allowed = self.corridor_mask(G, chain, doors)
        tri = G["tri"]

        def fn(cen):
            simp = tri.find_simplex(cen)
            ok = simp >= 0
            ok[ok] = allowed[simp[ok]]
            return ok | (self._goal_side(G, cen) >= -0.5)   # past the goal line is fine
        return fn

    # ---------------------------------------------------------------- helpers
    def world(self, G) -> CollisionWorld:
        nC = G["nC"]
        return CollisionWorld(G["V"][:nC], G["rad"][:nC], G["left"], G["right"], self.veh)

    def _goal_side(self, G, pts):
        gL, gR = G["goal"]
        nrm = self._goal_normal(G, np.zeros(2))
        return (pts[:, :2] - gR) @ nrm

    def validate(self, G, world, path: np.ndarray, state: VehicleState) -> bool:
        """Closed-loop check: track ``path`` with pure pursuit; the tracked footprint must keep
        ``validate_inflation`` (0.12 m) -- the plan itself keeps 0.25 m."""
        speed = max(state.v, self.scfg.target_speed)
        L = float(_arclen(path[:, :2])[-1])
        ext = np.concatenate([path[:, :2], path[-1:, :2] + 3.0 * np.array(
            [np.cos(path[-1, 2]), np.sin(path[-1, 2])])[None]])
        tr = rollout_path(ext, state, self.veh, self.lookahead, speed, ds=speed * self.lcfg.dt,
                          max_len=L + 1.0)
        tr = tr[self._goal_side(G, tr) < 0]
        if len(tr) == 0:
            return True
        ok = world.clearance(tr, self.lcfg.validate_inflation) >= 0
        return bool(self._relieved(world, tr, ok, self.lcfg.validate_inflation).all())

    def _relieved(self, world, poses, ok, inflation):
        """Start relief (observed mode): if the car's current pose already violates ``inflation``,
        ignore the leading violating poses, provided they end within ``start_relief_m`` of arc."""
        if self.mode != "observed" or self.lcfg.start_relief_m <= 0 or ok.all() or ok[0]:
            return ok
        if world.clearance(np.zeros(3), inflation)[0] >= 0:
            return ok
        first_ok = int(np.argmax(ok)) if ok.any() else len(ok)
        s = _arclen(poses[:, :2])
        if first_ok < len(ok) and s[first_ok] <= self.lcfg.start_relief_m:
            self.events["relief"] += 1
            ok = ok.copy()
            ok[:first_ok] = True
        return ok

    def _lat_cfg(self, beam: int | None):
        lc = self.lcfg.lattice
        return lc if beam is None else replace(lc, beam=beam)

    def plan_full(self, G, world, state: VehicleState, beam=None, n_routes=None, record_fail=True):
        """Try the top routes at each curvature level; returns (lattice path, route) or (None, cause).
        The curvature level a plan needed is appended to ``self.kappa_log``."""
        base = self._lat_cfg(beam)
        k0 = np.tan(state.steer) / self.veh.wheelbase
        start = np.zeros(3)
        n_routes = n_routes or self.lcfg.n_routes
        cause = "no route (A*)"
        routes = _LazyList(self.routes(G, None, n_routes))
        ladder = [(kf, base.beam) for kf in self.lcfg.kappa_ladder]
        if self.mode == "privileged" and self.lcfg.last_resort_beam > base.beam:
            ladder.append((self.lcfg.kappa_ladder[-1], self.lcfg.last_resort_beam))
        for kf, bm in ladder:
            cfg = replace(base, kappa_frac=kf, beam=bm)
            for doors, exit_edge, chain in routes:
                if self._over_budget():
                    cause = "per-replan time budget exceeded"
                    break
                fn = self.corridor_fn(G, chain, doors)
                lat = lattice_search(world, start, G["goal"], cfg, fn, k0, deadline=self._deadline)
                if lat is None:
                    cause = "lattice found no curvature-feasible path in any route corridor"
                    continue
                if self.validate(G, world, lat, state):
                    self.kappa_log.append(kf)
                    return lat, (doors, exit_edge, chain)
                cause = "tracked path broke the 0.12 m acceptance tolerance in every route"
                for wc, cp in self.lcfg.shaping:
                    if self._over_budget():
                        break
                    lat = lattice_search(world, start, G["goal"], replace(cfg, w_clear=wc, clear_pref=cp), fn, k0,
                                         deadline=self._deadline)
                    if lat is not None and self.validate(G, world, lat, state):
                        self.kappa_log.append(kf)
                        return lat, (doors, exit_edge, chain)
            if self._over_budget():
                cause = "per-replan time budget exceeded"
                break
        if record_fail:
            self.fail_causes[cause] += 1
        return None, cause

    def _over_budget(self) -> bool:
        return self._deadline is not None and time.perf_counter() > self._deadline

    def teacher_paths(self, inp: PlannerInput, K: int = 8, state: VehicleState | None = None,
                      max_routes: int | None = None):
        """Multimodal labels: the lattice inside each of the top-K distinct routes."""
        G = self._build(inp)
        if G is None:
            return []
        world = self.world(G)
        state = state or VehicleState(0, 0, 0, self.scfg.target_speed, 0)
        out, sigs = [], set()
        for doors, exit_edge, chain in self.routes(G, None, max_routes or 2 * K):
            lat = lattice_search(world, np.zeros(3), G["goal"], self.lcfg.lattice, self.corridor_fn(G, chain, doors))
            if lat is None or not self.validate(G, world, lat, state):
                continue
            sig = side_signature_local(lat, G["V"][:G["nC"]])
            if sig in sigs:
                continue
            sigs.add(sig)
            out.append(lat)
            if len(out) >= K:
                break
        return out

    # ---------------------------------------------------------------- main
    def plan_observation(self, obs: Observation, speed: float, steer: float) -> PlanResult:
        if self.mode == "observed" and (self.lcfg.tracking or self.lcfg.edge_memory):
            inp = self.tracked_input(obs)
        else:
            inp = input_from_observation(obs, self.cfg, self.scfg)
        return self.plan(inp, speed, steer)

    def tracked_input(self, obs: Observation) -> PlannerInput:
        """Planner input from tracked obstacles and edges with memory (same structure as the raw one)."""
        pose = np.asarray(obs.pose_odom, float)
        C, R, _ = cluster_obstacles(obs, self.cfg, self.scfg)
        if self.lcfg.tracking:
            C, R = self.tracker.update(C, R, pose)
        if self.lcfg.edge_memory:
            ep, es = self.edge_mem.update(obs.edge_points, obs.edge_side, pose)
        else:
            ep, es = obs.edge_points, obs.edge_side
        left, right = observed_edges(ep, es, self.scfg.track_width, self.cfg.edge_bin)
        return PlannerInput(C, R, left, right, obs.goal_line.astype(float), make_unknown_fn(obs), pose)

    def _result(self, path, trigger, t0, **info):
        ms = 1000 * (time.perf_counter() - t0)
        self.stats.append((trigger, ms))
        self.expansions.append(EXPANSIONS[0] - self._exp0)
        if path is None:
            return PlanResult(np.zeros((0, 3)), True, info={"trigger": trigger, **info})
        return PlanResult(path, False, [], path[-1, :2],
                          info={"trigger": trigger, "speed": self.scfg.target_speed, **info})

    def plan(self, inp: PlannerInput, speed: float, steer: float) -> PlanResult:
        t0 = time.perf_counter()
        self._exp0 = EXPANSIONS[0]
        if self.mode == "observed":
            return self._plan_observed(inp, speed, steer, t0)
        return self._plan_core(inp, speed, steer, t0)

    def _plan_observed(self, inp: PlannerInput, speed: float, steer: float, t0: float) -> PlanResult:
        """Observed mode: budget, failed-search cache, kept-plan fallback and hold around the core."""
        L = self.lcfg
        self.scan += 1
        self._deadline = t0 + L.budget_s if L.budget_s > 0 else None
        odom = inp.pose_odom if inp.pose_odom is not None else np.zeros(3)
        self._skip_search = False
        if L.fail_cache_scans and self.fail_cache is not None:
            sc, p0, sig = self.fail_cache
            if (self.scan - sc <= L.fail_cache_scans and float(np.hypot(*(odom[:2] - p0[:2]))) < L.fail_cache_move
                    and sig == self.tracker.signature()):
                self.events["cache_hit"] += 1
                self._skip_search = True             # the kept plan (if any) may still be used
            else:
                self.fail_cache = None
        kept_before = self.plan_w
        res = self._plan_core(inp, speed, steer, t0)
        if res.brake:
            fallback = res.info.pop("fallback", None)
            if fallback is not None and L.fallback:  # kept plan still has no hard violation
                self.events["fallback"] += 1
                self.plan_w = kept_before
                self.stats[-1] = ("keep_fallback", self.stats[-1][1])
                return PlanResult(fallback, False, [], fallback[-1, :2],
                                  info={"trigger": "keep_fallback", "speed": self.scfg.target_speed})
            if res.info.get("reason") not in ("no edges", "cached failure") and L.fail_cache_scans:
                self.fail_cache = (self.scan, odom.copy(), self.tracker.signature())
            if "budget" in str(res.info.get("reason", "")):
                self.events["budget"] += 1
            self.held = True
            self.confirm_count = 0
            return res
        if self.held and L.hold_confirm > 1 and res.info.get("trigger") in ("full", "warm"):
            self.confirm_count += 1
            if self.confirm_count < L.hold_confirm:
                self.events["hold"] += 1
                self.plan_w = None                   # plan again next scan; release on confirmation
                self.stats[-1] = ("hold", self.stats[-1][1])
                return PlanResult(np.zeros((0, 3)), True, info={"trigger": "hold"})
        self.held = False
        self.confirm_count = 0
        return res

    def _plan_core(self, inp: PlannerInput, speed: float, steer: float, t0: float) -> PlanResult:
        odom = inp.pose_odom if inp.pose_odom is not None else np.zeros(3)
        state = VehicleState(0.0, 0.0, 0.0, speed, steer)
        if self.mode == "privileged" and self.no_route:
            return self._result(None, "brake", t0, reason="no route")

        rest = None
        if self.plan_w is not None:
            loc = poses_to_frame(self.plan_w, odom)
            i0 = int(np.argmin(np.hypot(loc[:, 0], loc[:, 1])))
            dev = float(np.hypot(*loc[i0, :2]))
            rest = loc[i0:]
            if dev > self.lcfg.deviation_replan or len(rest) < 2:
                rest = None
        if rest is not None and self.mode == "privileged":
            return self._result(rest, "keep", t0)

        G = self._build(inp)
        if G is None:
            self.plan_w = None
            return self._result(None, "brake", t0, reason="no edges")
        world = self.world(G)

        fallback = None
        if rest is not None and not self.lcfg.keep_hysteresis:   # observed, original (pre-L3) check
            side = self._goal_side(G, rest)
            before = side < 0
            clr = world.clearance(rest[before], self.lcfg.validate_inflation) if before.any() else np.zeros(0)
            bad = np.nonzero(clr < 0)[0]
            short = side[-1] < -self.lcfg.extend_short
            if len(bad) == 0 and not short:
                return self._result(rest, "keep", t0)
        elif rest is not None:   # observed, L3: hard / soft check with persistence
            side = self._goal_side(G, rest)
            before = side < 0
            P = rest[before]
            if len(P):
                hard_ok = self._relieved(world, P, world.clearance(P, 0.0) >= 0, 0.0)
                soft_ok = self._relieved(world, P, world.clearance(P, self.lcfg.keep_inflation) >= 0,
                                         self.lcfg.keep_inflation)
            else:
                hard_ok = soft_ok = np.ones(0, bool)
            short = side[-1] < -self.lcfg.extend_short
            if hard_ok.all():
                fallback = rest                      # usable if replanning fails
                self.soft_count = 0 if soft_ok.all() else self.soft_count + 1
            if hard_ok.all() and self.soft_count < self.lcfg.keep_soft_scans and not short:
                return self._result(rest, "keep", t0)
            bad = np.nonzero(~soft_ok)[0] if hard_ok.all() else np.nonzero(~hard_ok)[0]
        if self._skip_search:                        # failed-search cache hit (observed)
            self.plan_w = None
            return self._result(None, "brake", t0, reason="cached failure", fallback=fallback)
        if rest is not None:
            # warm start from the valid prefix
            s = _arclen(rest[:, :2])
            if len(bad):
                cut_s = s[bad[0]] - self.lcfg.warm_backoff
            else:
                cut_s = s[-1]
            if cut_s >= self.lcfg.warm_min_prefix:
                c = int(np.searchsorted(s, cut_s, side="right") - 1)
                prefix = rest[: c + 1]
                k0 = 0.0
                if c >= 1:
                    dth = (prefix[c, 2] - prefix[c - 1, 2] + np.pi) % (2 * np.pi) - np.pi
                    k0 = dth / max(s[c] - s[c - 1], 1e-6)
                lat = lattice_search(world, prefix[c], G["goal"], self._lat_cfg(self.lcfg.obs_beam), None, k0)
                if lat is not None:
                    new = np.concatenate([prefix, lat[1:]])
                    if self.validate(G, world, new, state):
                        self.plan_w = poses_from_frame(new, odom)
                        return self._result(new, "warm", t0)

        beam = None if self.mode == "privileged" else self.lcfg.obs_beam
        n_routes = None if self.mode == "privileged" else self.lcfg.obs_n_routes
        lat, route = self.plan_full(G, world, state, beam, n_routes)
        if lat is None:
            self.plan_w = None
            if self.mode == "privileged":
                self.no_route = True
            return self._result(None, "brake", t0, reason=route, fallback=fallback)
        self.plan_w = poses_from_frame(lat, odom)
        self.soft_count = 0
        return self._result(lat, "full", t0)


def side_signature_local(path: np.ndarray, centres: np.ndarray, near: float = 3.0) -> tuple:
    """Which side of each nearby obstacle a path passes (+1 obstacle left of the path)."""
    sig = []
    P = path[:, :2]
    for j, c in enumerate(centres):
        d = np.hypot(*(P - c).T)
        i = int(np.argmin(d))
        if d[i] > near or i == 0 or i >= len(P) - 1:
            continue
        t = P[i + 1] - P[i - 1]
        v = c - P[i]
        sig.append((j, 1 if t[0] * v[1] - t[1] * v[0] > 0 else -1))
    return tuple(sig)


class _LazyList:
    """Iterate a generator more than once, computing items only when first needed."""
    def __init__(self, gen):
        self.gen, self.items, self.done = gen, [], False

    def __iter__(self):
        i = 0
        while True:
            if i < len(self.items):
                yield self.items[i]
            elif self.done:
                return
            else:
                try:
                    self.items.append(next(self.gen))
                except StopIteration:
                    self.done = True
                    return
                yield self.items[i]
            i += 1
