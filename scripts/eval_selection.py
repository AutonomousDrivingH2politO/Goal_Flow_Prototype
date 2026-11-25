#!/usr/bin/env python
"""Open-loop comparison of selection variants on the validation split (frozen seed-0 checkpoints).

    python scripts/eval_selection.py --run runs/l2_full_seed0

PRE-REGISTRATION. Every metric, variant, rule and threshold below was fixed on 2026-11-25, before any
result of this script existed (only 2-batch smoke tests were run, for crashes). Validation only: never
run this on the test split (the L1 test split is spent and frozen; see DEVIATIONS).

Checkpoints: goalflow_B_best.pt (its encoder and goal scorer are bit-identical to goalflow_A_best.pt:
Stage B started from the Stage A best checkpoint and kept them frozen), regression_best.pt.

Variants (M = 128 samples everywhere; the same frames and the same sample noise per batch):
  M0, M1, M2            paper Table 2 (mean outputs)
  M3                    single g* (the trained model as reported), n = 1 Euler step
  M3 n2, M3 n4          2 and 4 Euler steps
  M3+safety             observed-hit safety filter (distance transform, see goalflow.observed_collision)
  M3 NMS-K2/K3(+safety) K goals by greedy NMS on the final goal score (1.5 m, 3.0 log-unit floor); M/K
                        samples toward each; candidates scored across all with l4 * Phi(goal score)
  M3 route-K2/K3(+safety)  DEVIATIONS Part C item 7, no privileged input: the observed corridor-lattice
                        planner is run on the frame's noisy LiDAR observation (single scan, no tracking);
                        its top-K routes (A* cost order, distinct door sequences); for route k the goal is
                        the best-scoring vocabulary goal (body centre) inside route k's corridor and in no
                        other selected route's corridor; if there is none, the best inside route k's
                        corridor; if the corridor holds no goal or the planner finds fewer than K routes,
                        g*. M/K samples toward each; critics as for NMS-K.
  regression            baseline
  corridor-lattice observed (cold)   the observed corridor-lattice planner's own path on the same frame,
                        planned cold from the single scan (no tracking, no warm start, none of the L3
                        additions), time-parametrised from the ego speed toward 3 m/s at 1 m/s^2 and rolled
                        out like every candidate; a brake counts as standing still (planner_brake logged).
                        This is the route source of the route-K variants, so it shows GoalFlow's added value.

Teacher routes. "all routes" = every stored mode; stored modes are already deduplicated by homotopy
signature (sim_dataset.py:152), so these are the homotopy-distinct routes. "distinct" = the stricter
endpoint rule: walk routes in stored order (driven first); a route is distinct if its endpoint is >= 1.5 m
from every distinct route kept so far.

Coverage of the chosen candidates (multi-route frames; top-k by trajectory score after the safety filter):
  cover@k / cover@k_distinct      some candidate endpoint within 1.0 m of every route endpoint
  cover@k_1to1(_distinct)         the same with one-to-one assignment (each candidate covers at most one
                                  route; Hungarian on endpoint distance <= 1.0 m), so one endpoint halfway
                                  between two routes cannot cover both
  cover@k_homotopy                candidate covers route j if side_signature_local(candidate, true
                                  obstacles, near = 3 m) contains every (obstacle, side) pair of route j's
                                  signature (recomputed from the stored 16-waypoint target and the batch's
                                  true obstacles, origin prepended)
  k = 1, 5, 10, all; cover_route_best* = the best-scored candidate of each goal (one per goal)
  on_mode / on_mode_distinct / on_mode_homotopy   the selected executed path matches some route
  k_goals_diff_homotopy  share of frames whose goals' best candidates differ in homotopy (some obstacle
                         within 3 m of both, passed on opposite sides); n_distinct_goals
Safety: g* collision (Juno box at the selected goal vs true discs); executed-path collision (Juno's
footprint along the selected executed path, 2x densified, overlaps a true disc), clearance < 0.12 m,
track exit. Safety-filter quality on the M3 candidates: recall = share of candidates with true clearance
< 0 that the filter flags; false-alarm rate on collision-free candidates and on the teacher routes.
Shadow: main and shadow paths both evaluated per frame (sweep 0.5-3 m and "off" computed from one run).
Latency: wall time per frame at batch 32 on the Mac's MPS GPU (throughput, not the per-frame n = 1 Orin
latency of the 50 ms criterion; do not compare them).
Saved per frame for post-hoc analysis: top-10 candidate endpoints and side signs, the K goals, the
selected index, true obstacles, route endpoints and route signatures.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Subset  # noqa: E402

from gfsem.data.scene_index import SceneIndex  # noqa: E402
from gfsem.data.sim_torch import MemmapSplit, SimFrames  # noqa: E402
from gfsem.models.common import pick_device  # noqa: E402
from gfsem.models.goalflow import hits_distance, observed_collision  # noqa: E402
from gfsem.models.rollout import RolloutConfig, kappa_feasible, rollout  # noqa: E402
from gfsem.sim.config import VehicleConfig  # noqa: E402
from gfsem.sim.geometry import from_frame  # noqa: E402
from gfsem.train import goal_collides, load, obstacles_near_path  # noqa: E402

VARIANTS = {
    "M0": dict(use_goal=False),
    "M1": dict(use_dac=False, use_scorer=False),
    "M2": dict(use_scorer=False),
    "M3": dict(),
    "M3 n2": dict(n_steps=2),
    "M3 n4": dict(n_steps=4),
    "M3+safety": dict(safety=True),
    "M3 NMS-K2": dict(goal_K=2),
    "M3 NMS-K2+safety": dict(goal_K=2, safety=True),
    "M3 NMS-K3": dict(goal_K=3),
    "M3 NMS-K3+safety": dict(goal_K=3, safety=True),
    "M3 route-K2": dict(route_K=2),
    "M3 route-K2+safety": dict(route_K=2, safety=True),
    "M3 route-K3": dict(route_K=3),
    "M3 route-K3+safety": dict(route_K=3, safety=True),
}
DISTINCT_SEP = 1.5
COVER_R = 1.0
NEAR = 3.0
SHADOW_SWEEP = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, float("inf"))
ROUTE_KMAX = 3
CELL_SLACK = 0.05     # from scripts/calibrate_safety.py (rule: largest slack with driven false alarm <= 1%)


class RouteFrames(Dataset):
    """SimFrames item + the observed corridor planner's top-K route corridors as vocabulary masks."""

    def __init__(self, base: SimFrames, vocab: np.ndarray, kmax: int = ROUTE_KMAX):
        self.base, self.vocab, self.kmax = base, vocab, kmax
        self.pl = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        from gfsem.data.sim_dataset import observation_from_shard
        from gfsem.planners.corridor import input_from_observation
        from gfsem.planners.corridor_lattice import LatticeCorridorConfig, LatticeCorridorPlanner
        out = self.base[i]
        if self.pl is None:
            self.pl = LatticeCorridorPlanner(mode="observed", lcfg=LatticeCorridorConfig(tracking=False))
            self.pp = LatticeCorridorPlanner(mode="observed", lcfg=LatticeCorridorConfig(
                tracking=False, start_relief_m=0.0, budget_s=0.0, hold_confirm=1, fail_cache_scans=0))
            xc = sum(self.pl.veh.body_x) / 2
            V = self.vocab
            self.cen = V[:, :2] + xc * np.stack([np.cos(V[:, 2]), np.sin(V[:, 2])], -1)
        k, j = self.base.index[i]
        obs = observation_from_shard(self.base.shards[k], j, self.base.cfg.lidar)
        masks = np.zeros((self.kmax, len(self.vocab)), bool)
        n = 0
        try:
            inp = input_from_observation(obs, self.pl.cfg, self.pl.scfg)
            G = self.pl._build(inp)
            if G is not None:
                for doors, _, chain in self.pl.routes(G, None, self.kmax):
                    masks[n] = self.pl.corridor_fn(G, chain, doors)(self.cen)
                    n += 1
        except Exception:          # degenerate triangulation: no routes for this frame
            n = 0
        out["route_masks"] = torch.from_numpy(masks)
        out["n_routes"] = torch.tensor(n)
        # the observed corridor-lattice planner's own path, planned cold from this single frame with the
        # original behaviour (no tracking, start relief, budget, hold or cache)
        wp, ok = np.zeros((16, 2), np.float32), False
        try:
            self.pp.reset()
            res = self.pp.plan_observation(obs, float(obs.ego_state[0]), float(obs.ego_state[1]))
            if not res.brake and len(res.path) >= 2:
                wp, ok = time_parametrize(res.path[:, :2], float(obs.ego_state[0])), True
        except Exception:
            pass
        out["planner_path"] = torch.from_numpy(wp)
        out["planner_ok"] = torch.tensor(ok)
        return out


def time_parametrize(path_xy: np.ndarray, v0: float, v_target: float = 3.0, a_max: float = 1.0,
                     dt: float = 0.25, T: int = 16) -> np.ndarray:
    """Geometric path (ego frame, from the car) -> T waypoints at dt: speed from v0 toward v_target at
    a_max (the closed loop's speed command), clamped at the path's end."""
    seg = np.hypot(*np.diff(path_xy, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = dt * np.arange(1, T + 1)
    s_t = []
    v, s, tt = v0, 0.0, 0.0
    for tk in t:
        while tt < tk - 1e-9:
            h = min(0.01, tk - tt)
            vn = min(v + a_max * h, v_target) if v < v_target else max(v - 3.0 * h, v_target)
            s += 0.5 * (v + vn) * h
            v, tt = vn, tt + h
        s_t.append(s)
    s_t = np.minimum(np.array(s_t), cum[-1])
    return np.stack([np.interp(s_t, cum, path_xy[:, 0]), np.interp(s_t, cum, path_xy[:, 1])], -1).astype(np.float32)


def route_goal_indices(score, masks, n_routes, K):
    """(B, K) vocabulary indices per the pre-registered rule; also fallback counts."""
    g_star = score.argmax(-1)
    idx = []
    fb = {"exclusive": 0, "corridor": 0, "g*": 0}
    rr = torch.arange(K, device=score.device)
    valid = rr[None] < n_routes[:, None]                                        # (B, K)
    Mk = masks[:, :K] & valid[..., None]
    for k in range(K):
        others = (Mk & (rr != k)[None, :, None]).any(1)
        excl = Mk[:, k] & ~others
        s_ex = score.masked_fill(~excl, -float("inf"))
        s_in = score.masked_fill(~Mk[:, k], -float("inf"))
        has_ex, has_in = excl.any(1), Mk[:, k].any(1)
        i = torch.where(has_ex, s_ex.argmax(1), torch.where(has_in, s_in.argmax(1), g_star))
        fb["exclusive"] += int(has_ex.sum())
        fb["corridor"] += int((~has_ex & has_in).sum())
        fb["g*"] += int((~has_in).sum())
        idx.append(i)
    return torch.stack(idx, 1), fb


def densify(paths: torch.Tensor):
    """(B, T, 2) from the ego origin -> poses (B, 2T+1, 3) with heading from finite differences."""
    P = torch.cat([torch.zeros_like(paths[:, :1]), paths], 1)
    mid = 0.5 * (P[:, 1:] + P[:, :-1])
    Q = torch.cat([P[:, :1], torch.stack([mid, P[:, 1:]], 2).flatten(1, 2)], 1)
    d = torch.diff(Q, dim=1)
    yaw = torch.atan2(d[..., 1], d[..., 0])
    still = d.norm(dim=-1) < 1e-3
    yaw = torch.cat([torch.zeros_like(yaw[:, :1]), yaw], 1)
    for k in range(1, yaw.shape[1]):                        # hold the heading while stopped
        yaw[:, k] = torch.where(still[:, k - 1], yaw[:, k - 1], yaw[:, k])
    return torch.cat([Q, yaw[..., None]], -1)


def path_clearance(paths, obstacles, n_obs, veh: VehicleConfig) -> torch.Tensor:
    """(B,) minimum surface distance between Juno's box along the path and any true obstacle disc."""
    X = densify(paths)
    x0, x1 = veh.body_x
    h = veh.width / 2
    c, s = torch.cos(X[..., 2])[..., None], torch.sin(X[..., 2])[..., None]
    dx = obstacles[:, None, :, 0] - X[..., 0:1]
    dy = obstacles[:, None, :, 1] - X[..., 1:2]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    d = torch.hypot(lx - lx.clamp(x0, x1), ly - ly.clamp(-h, h)) - obstacles[:, None, :, 2]
    valid = torch.arange(obstacles.shape[1], device=paths.device)[None, None] < n_obs[:, None, None]
    return d.masked_fill(~valid, 1e9).flatten(1).min(1).values


def track_exit(paths: np.ndarray, poses: np.ndarray, seeds: np.ndarray, si: SceneIndex, veh) -> np.ndarray:
    from gfsem.sim.geometry import footprint_corners
    X = densify(torch.from_numpy(paths)).numpy()
    out = np.zeros(len(paths), bool)
    for b in range(len(paths)):
        tr = si.get(int(seeds[b])).track
        W = np.empty_like(X[b], dtype=np.float64)
        W[:, :2] = from_frame(X[b, :, :2].astype(np.float64), poses[b].astype(np.float64))
        W[:, 2] = X[b, :, 2] + poses[b, 2]
        cs = footprint_corners(W, veh.body_x, veh.width).reshape(-1, 2)
        _, d = tr.frenet(cs)
        out[b] = bool((np.abs(d) > tr.width / 2).any())
    return out


def side_signs(paths, obstacles, n_obs, near: float = NEAR):
    """side_signature_local, vectorised: paths (B, K, T, 2) from the origin, obstacles (B, O, 3).
    Returns int8 (B, K, O): +1 obstacle left of the path, -1 right, 0 not in the signature (farther than
    ``near`` from the path's closest point, closest point at an end, or padding)."""
    P = torch.cat([torch.zeros_like(paths[:, :, :1]), paths], 2)                 # (B, K, T+1, 2)
    C = obstacles[:, None, :, :2]                                                 # (B, 1, O, 2)
    d = (P[:, :, :, None] - C[:, :, None]).norm(dim=-1)                          # (B, K, T+1, O)
    dmin, i = d.min(2)                                                            # (B, K, O)
    T1 = P.shape[2]
    ok = (i > 0) & (i < T1 - 1) & (dmin <= near)
    ok &= torch.arange(obstacles.shape[1], device=paths.device)[None, None] < n_obs[:, None, None]
    ip, im = (i + 1).clamp(max=T1 - 1), (i - 1).clamp(min=0)

    def g(ix):
        return torch.gather(P, 2, ix[..., None].expand(-1, -1, -1, 2))
    t = g(ip) - g(im)
    v = C.expand(-1, P.shape[1], -1, -1) - g(i)
    cross = t[..., 0] * v[..., 1] - t[..., 1] * v[..., 0]
    sgn = torch.where(cross > 0, 1, -1).to(torch.int8)
    return torch.where(ok, sgn, torch.zeros_like(sgn))


def distinct_routes(modes, n_modes, sep: float = DISTINCT_SEP):
    E = modes[:, :, -1]
    keep = torch.zeros(E.shape[0], 8, dtype=torch.bool, device=E.device)
    keep[:, 0] = n_modes > 0
    for j in range(1, 8):
        far = ((E[:, :j] - E[:, j:j + 1]).norm(dim=-1) >= sep) | ~keep[:, :j]
        keep[:, j] = (j < n_modes) & far.all(1)
    return keep


def cover_end(cands, modes, valid, radius: float = COVER_R):
    d = torch.cdist(modes[:, :, -1], cands[:, :, -1])
    return ((d <= radius).any(-1) | ~valid).all(1)


def cover_end_1to1(cands, modes, valid, radius: float = COVER_R) -> torch.Tensor:
    """Each candidate covers at most one route: maximum matching on distance <= radius (Hungarian)."""
    d = torch.cdist(modes[:, :, -1], cands[:, :, -1]).cpu().numpy()             # (B, 8, K)
    v = valid.cpu().numpy()
    out = np.zeros(len(d), bool)
    for b in range(len(d)):
        D = d[b][v[b]]
        if len(D) == 0:
            out[b] = True
            continue
        if D.shape[1] < D.shape[0]:
            continue
        ok = D <= radius
        if not ok.any(1).all():
            continue
        r, c = linear_sum_assignment(np.where(ok, D, 1e6))
        out[b] = bool((D[r, c] <= radius).all())
    return torch.from_numpy(out)


def cover_homotopy(cand_signs, mode_signs, valid):
    """cand_signs (B, K, O), mode_signs (B, 8, O): candidate covers route j if it holds every non-zero
    (obstacle, side) pair of route j. Returns (B,) bool over the valid routes, and (B, 8, K) matches."""
    need = mode_signs != 0                                                        # (B, 8, O)
    match = (~need[:, :, None] | (cand_signs[:, None] == mode_signs[:, :, None])).all(-1)   # (B, 8, K)
    return (match.any(-1) | ~valid).all(1), match


def on_route_homotopy(sg, mode_signs, valid):
    """(B,) the single path (sg: (B, 1, O)) holds the full signature of at least one valid route."""
    _, match = cover_homotopy(sg, mode_signs, valid)
    return (match[..., 0] & valid).any(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/l2_full_seed0"))
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/l2_selection"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    split = "val"                                   # validation only, by design
    dev = pick_device("auto")
    root = Path("data/sim")
    mm = MemmapSplit(root / f"mm_{split}")
    si = SceneIndex([root / f"scenes_{split}.npz"])
    gf = load(a.run / "goalflow_B_best.pt", dev).eval()
    rg = load(a.run / "regression_best.pt", dev).eval()
    vocab_np = gf.scorer.vocab.cpu().numpy()
    base = SimFrames([mm], vocab=vocab_np, scene_index=si, driven_only=True)
    ds = RouteFrames(base, vocab_np)
    n_modes = np.diff(np.asarray(mm["targets_off"]))
    subset = set(np.random.default_rng(1234).choice(len(ds), min(4000, len(ds)), replace=False).tolist())
    multi = set(np.flatnonzero(n_modes > 1).tolist())
    idx = sorted(subset | multi)
    in_subset = np.array([i in subset for i in idx])
    dl = DataLoader(Subset(ds, idx), batch_size=a.bs, shuffle=False, num_workers=a.workers,
                    multiprocessing_context="spawn" if a.workers else None, persistent_workers=False)
    veh = VehicleConfig.from_yaml()
    names = [v for v in a.variants.split(",") if v] + ["regression", "corridor-lattice observed (cold)"]
    n_batches = min(len(dl), a.max_batches or len(dl))
    print(f"{split}: {len(idx)} frames ({len(subset)} subset, {len(multi)} multi-mode), {n_batches} batches, "
          f"M={a.M}, device {dev}", flush=True)
    rec: dict[str, dict[str, list]] = {v: {} for v in names}
    save_v: dict[str, dict[str, list]] = {v: {} for v in names}
    save_c: dict[str, list] = {}
    common: dict[str, list] = {}
    route_fb = {k: {"exclusive": 0, "corridor": 0, "g*": 0} for k in (2, 3)}
    sq = {"cand_true_col": 0, "cand_true_col_flagged": 0, "cand_free": 0, "cand_free_flagged": 0,
          "teacher_routes": 0, "teacher_flagged": 0, "driven_routes": 0, "driven_flagged": 0}
    t0 = time.time()
    tlat = {v: 0.0 for v in names}

    def put(R, k, x):
        R.setdefault(k, []).append(x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(x))

    for bi, b in enumerate(dl):
        if bi >= n_batches:
            break
        b = {k: v.to(dev) for k, v in b.items()}
        modes, nm = b["modes"], b["n_modes"]
        obst, nob = b["obstacles"], b["n_obstacles"]
        valid_all = torch.arange(8, device=dev)[None] < nm[:, None]
        valid_d = distinct_routes(modes, nm)
        mode_signs = side_signs(modes, obst, nob)                                 # (B, 8, O)
        mode_signs = torch.where(valid_all[..., None], mode_signs, torch.zeros_like(mode_signs))
        dist = hits_distance(b["bev"][:, 0])
        tflag = observed_collision(modes, b["bev"][:, 0], dist=dist, cell_slack=CELL_SLACK)             # teacher false alarms
        sq["teacher_routes"] += int(valid_all.sum())
        sq["teacher_flagged"] += int((tflag & valid_all).sum())
        sq["driven_routes"] += int(valid_all[:, 0].sum())
        sq["driven_flagged"] += int((tflag[:, 0] & valid_all[:, 0]).sum())
        put(common, "n_modes", nm)
        put(common, "n_distinct", valid_d.sum(1))
        put(common, "near_obst", obstacles_near_path(modes[:, 0], obst, nob))
        put(save_c, "obstacles", obst.half())
        put(save_c, "mode_ends", modes[:, :, -1].half())
        put(save_c, "mode_signs", mode_signs)
        put(save_c, "n_routes_planner", b["n_routes"])
        dis, dac = gf.scorer(*gf.perceive(b["bev"], b["ego"])[:2])
        score = F.log_softmax(dis, -1) + F.logsigmoid(dac)
        rg_idx = {}
        for K in (2, 3):
            rg_idx[K], fb = route_goal_indices(score, b["route_masks"], b["n_routes"], K)
            for k_, v_ in fb.items():
                route_fb[K][k_] += v_
        for v in names:
            torch.manual_seed(a.seed * 100_003 + bi)
            ts = time.time()
            if v == "regression":
                pr = rg.norm.inverse(rg.reg(*rg.perceive(b["bev"], b["ego"])[:2]))
                r = dict(traj_raw=pr, traj=rollout(pr, b["ego"][:, 0], b["ego"][:, 1], RolloutConfig()))
            elif v.startswith("corridor-lattice"):
                pp, okp = b["planner_path"], b["planner_ok"]
                ex = rollout(pp, b["ego"][:, 0], b["ego"][:, 1], RolloutConfig())
                ex = torch.where(okp[:, None, None], ex, torch.zeros_like(ex))       # brake: stays put
                r = dict(traj_raw=pp, traj=ex)
                put(rec[v], "planner_brake", ~okp)
            else:
                kw = dict(VARIANTS[v])
                n_steps = kw.pop("n_steps", 1)
                rK = kw.pop("route_K", None)
                if rK:
                    kw["route_goals"] = rg_idx[rK]
                if kw.get("safety"):
                    kw["safety_slack"] = CELL_SLACK
                r = gf.plan(b["bev"], b["ego"], M=a.M, n_steps=n_steps, **kw)
            if dev.type == "mps":
                torch.mps.synchronize()
            tlat[v] += time.time() - ts
            R, S_ = rec[v], save_v[v]

            def path_metrics(tag, tr):
                d = (tr[:, None] - modes).norm(dim=-1)
                put(R, f"ADE{tag}", d[:, 0].mean(-1))
                put(R, f"FDE{tag}", d[:, 0, -1])
                de = (tr[:, None, -1] - modes[:, :, -1]).norm(dim=-1)
                for w, val in (("", valid_all), ("_distinct", valid_d)):
                    put(R, f"on_mode{w}{tag}", de.masked_fill(~val, 1e9).min(1).values <= COVER_R)
                put(R, f"on_mode_homotopy{tag}", on_route_homotopy(side_signs(tr[:, None], obst, nob), mode_signs, valid_all))
                put(R, f"clearance{tag}", path_clearance(tr, obst, nob, veh))
                put(R, f"exit{tag}", torch.from_numpy(track_exit(
                    tr.cpu().numpy(), b["pose"].cpu().numpy(), b["seed"].cpu().numpy(), si, veh)))

            path_metrics("", r["traj"])
            d = (r["traj_raw"][:, None] - modes).norm(dim=-1)
            put(R, "ADE_raw", d[:, 0].mean(-1))
            put(R, "FDE_raw", d[:, 0, -1])
            if "traj_main" in r:
                path_metrics("_main", r["traj_main"])
                path_metrics("_shadow", r["traj_shadow"])
                put(R, "shadow_dev", r["shadow_dev"])
            if "samples_raw" in r:
                put(R, "kappa_ok_raw", kappa_feasible(r["samples_raw"]).float().mean(1))
                put(R, "kappa_ok_raw_1.1", kappa_feasible(r["samples_raw"], tol=1.1).float().mean(1))
                put(R, "cover_all_raw", cover_end(r["samples_raw"], modes, valid_all))
                put(R, "cover_all_raw_distinct", cover_end(r["samples_raw"], modes, valid_d))
            g = r.get("goal_sel", r.get("goal"))
            if g is not None:
                put(R, "gstar_col", goal_collides(g, obst, nob))
            if "scores" in r:
                S = r["samples"]
                Bn = S.shape[0]
                ar = torch.arange(Bn, device=dev)[:, None]
                order = r["scores"].argsort(1, descending=True)
                go = r["goal_of"]
                K = int(go.max()) + 1
                fbest = torch.stack([r["scores"].masked_fill(go != k, -1e18).argmax(1) for k in range(K)], 1)
                sets = {"@1": order[:, :1], "@5": order[:, :5], "@10": order[:, :10], "_all": order,
                        "_route_best": fbest}
                all_signs = side_signs(S, obst, nob)                              # (B, M, O)
                for key, sel in sets.items():
                    C = S[ar, sel]
                    put(R, f"cover{key}", cover_end(C, modes, valid_all))
                    put(R, f"cover{key}_distinct", cover_end(C, modes, valid_d))
                    if key != "_all":
                        put(R, f"cover{key}_1to1", cover_end_1to1(C, modes, valid_all))
                        put(R, f"cover{key}_1to1_distinct", cover_end_1to1(C, modes, valid_d))
                    put(R, f"cover{key}_homotopy", cover_homotopy(all_signs[ar, sel], mode_signs, valid_all)[0])
                G = r["goals"]
                put(R, "n_distinct_goals", (torch.cdist(G[..., :2], G[..., :2]) > 1e-6).any(-1).sum(-1).clamp_min(1))
                bs = all_signs[ar, fbest].to(torch.int16)                         # (B, K, O)
                opp = ((bs[:, :, None] * bs[:, None, :]) < 0).flatten(1).any(1)
                put(R, "k_goals_diff_homotopy", opp)
                if K > 1:
                    anyc = torch.stack([goal_collides(G[:, k], obst, nob) for k in range(K)], 1).any(1)
                    put(R, "any_goal_col", anyc)
                put(R, "used_shadow", r["used_shadow"])
                if "no_safe" in r:
                    put(R, "no_safe", r["no_safe"])
                    put(R, "safe_share", r["safe"].float().mean(1))
                if v == "M3":                                                     # safety-filter quality
                    flag = observed_collision(S, b["bev"][:, 0], dist=dist, cell_slack=CELL_SLACK)
                    clr = path_clearance(S.reshape(-1, S.shape[2], 2), obst.repeat_interleave(S.shape[1], 0),
                                         nob.repeat_interleave(S.shape[1], 0), veh).view(Bn, -1)
                    col = clr < 0
                    sq["cand_true_col"] += int(col.sum())
                    sq["cand_true_col_flagged"] += int((col & flag).sum())
                    sq["cand_free"] += int((~col).sum())
                    sq["cand_free_flagged"] += int((~col & flag).sum())
                put(S_, "top10_ends", S[ar, order[:, :10]][:, :, -1].half())
                put(S_, "top10_signs", all_signs[ar, order[:, :10]])
                put(S_, "goals", G.half())
                put(S_, "selected", r["best"])
        if bi % 10 == 0 or bi == n_batches - 1:
            el = time.time() - t0
            print(f"  batch {bi + 1}/{n_batches}  {el / 60:.1f} min  eta {el / (bi + 1) * (n_batches - bi - 1) / 60:.1f} min",
                  flush=True)

    nm_all = torch.cat(common["n_modes"]).numpy()
    nd_all = torch.cat(common["n_distinct"]).numpy()
    near = torch.cat(common["near_obst"]).numpy()
    Fn = len(nm_all)
    sub = in_subset[:Fn]
    groups = {"subset4000": sub, "multi_all_routes": nm_all > 1, "multi_distinct_routes": nd_all > 1,
              "subset_near_obstacle": sub & near}
    out = {"preregistered": "metrics, variants and rules fixed 2026-11-25 before any result (see docstring)",
           "frames": {k: int(m.sum()) for k, m in groups.items()}, "M": a.M, "seed": a.seed,
           "rule_distinct": f"endpoints >= {DISTINCT_SEP} m apart (greedy, driven first)",
           "checkpoints": {"goalflow": str(a.run / "goalflow_B_best.pt"),
                           "goal_scorer_and_encoder_from": "goalflow_A_best.pt (bit-identical weights)",
                           "regression": str(a.run / "regression_best.pt")},
           "latency_ms_per_frame_batch32_mps_throughput": {v: 1000 * tlat[v] / Fn for v in names},
           "route_goal_fallbacks_slots": route_fb,
           "safety_filter_M3_candidates": {
               "recall_on_true_collisions": sq["cand_true_col_flagged"] / max(sq["cand_true_col"], 1),
               "false_alarm_on_collision_free_candidates": sq["cand_free_flagged"] / max(sq["cand_free"], 1),
               "false_alarm_on_all_teacher_routes": sq["teacher_flagged"] / max(sq["teacher_routes"], 1),
               "false_alarm_on_driven_routes": sq["driven_flagged"] / max(sq["driven_routes"], 1),
               "cell_slack": CELL_SLACK, **sq},
           "variants": {}, "shadow_sweep": {}}
    for v in names:
        R = {k: torch.cat(x).numpy() for k, x in rec[v].items()}
        for tag in ("", "_main", "_shadow"):
            if f"clearance{tag}" in R:
                R[f"collision{tag}"] = R[f"clearance{tag}"] < 0
                R[f"clear_lt_0.12{tag}"] = R[f"clearance{tag}"] < 0.12
        res = {}
        for gname, m in groups.items():
            dct = {}
            for k, x in R.items():
                if k.startswith("clearance") or k == "shadow_dev":
                    continue
                if k.startswith("cover"):
                    if not gname.startswith("multi"):
                        continue
                    if k.endswith("_distinct") != (gname == "multi_distinct_routes"):
                        continue
                dct[k] = float(np.mean(x[m])) if m.any() else None
            res[gname] = dct
        out["variants"][v] = res
        if "shadow_dev" in R:
            sw = {}
            for thr in SHADOW_SWEEP:
                use = R["shadow_dev"] > thr
                row = {"shadow_rate": {g: float(use[m].mean()) for g, m in groups.items()}}
                for k in ("ADE", "FDE", "collision", "clear_lt_0.12", "exit", "on_mode", "on_mode_distinct",
                          "on_mode_homotopy"):
                    x = np.where(use, R[f"{k}_shadow"], R[f"{k}_main"])
                    row[k] = {g: float(x[m].mean()) for g, m in groups.items()}
                sw["off" if thr == float("inf") else f"{thr:g}"] = row
            out["shadow_sweep"][v] = sw
        np.savez_compressed(a.out / f"frames_{v.replace(' ', '_').replace('+', 'p')}.npz",
                            **R, **{k: torch.cat(x).numpy() for k, x in save_v[v].items()})
    np.savez_compressed(a.out / "frames_common.npz", idx=np.array(idx[:Fn]), n_modes=nm_all, n_distinct=nd_all,
                        near=near, in_subset=sub, **{k: torch.cat(x).numpy() for k, x in save_c.items()})
    (a.out / "selection_val.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    with torch.no_grad():
        main()
