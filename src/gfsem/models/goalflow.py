"""BEV GoalFlow: perception + goal scorer + rectified-flow generator, with inference.

Inference (PROMPT_LIDAR_DEMO.md section 3):
  1. pick g* from the goal scorer,
  2. draw M noise samples, integrate n Euler steps (batched),
  3. score: f = -l1 Phi(endpoint dist to g*) + l2 Phi(progress) - l3 Phi(dist to previous path),
  4. shadow: 16 samples with the goal masked; if the best one differs from the chosen path by more
     than ``shadow_thresh`` metres (mean waypoint distance), output it instead.

Before scoring, every candidate is rolled out through pure pursuit + the bicycle model with the
10 deg steering limit (``models/rollout.py``), as in the teacher validation, and the rolled-out path
is what gets scored and returned. The share of raw samples within kappa_max is reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .common import Normalizer
from .encoders import AuxBEVHead, BEVEncoder, EgoEncoder
from .flow import euler_sample, rf_training_pair
from .flow_generator import FlowGenerator, RegressionHead
from .goal_scorer import GoalScorer, select_goal
from .rollout import RolloutConfig, kappa_feasible, rollout


@dataclass
class ModelConfig:
    D: int = 256
    T: int = 16
    heads: int = 8
    scorer_layers: int = 3
    scorer_self_attn: bool = False   # True = standard decoder with N x N goal self-attention (ablation)
    gen_layers: int = 4
    env_queries: int = 16
    sigma: float = 0.1
    t_dist: str = "uniform"          # or logit_normal
    goal_dropout: float = 0.1
    aux_head: bool = False
    regression: bool = False         # also build the regression baseline head


def minmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    lo, hi = x.min(dim, keepdim=True).values, x.max(dim, keepdim=True).values
    return (x - lo) / (hi - lo).clamp_min(1e-6)


def diverse_goals(score: torch.Tensor, vocab: torch.Tensor, K: int, sep: float, floor: float):
    """Greedy NMS over the final goal score: best goal, then the best goal at least ``sep`` m (x, y)
    from every goal taken so far, up to K. A slot whose best remaining score is more than ``floor`` below
    the top score repeats the top goal (its samples then go to g* too). Returns (B, K, 3), (B, K)."""
    B, N = score.shape
    s = score.clone()
    top = score.max(-1).values
    idx = []
    for _ in range(K):
        i = s.argmax(-1)
        ok = s.gather(1, i[:, None])[:, 0] >= top - floor
        i = torch.where(ok, i, idx[0] if idx else i)
        idx.append(i)
        d = (vocab[None, :, :2] - vocab[i][:, None, :2]).norm(dim=-1)        # (B, N)
        s = s.masked_fill(d < sep, -1e9)
    I = torch.stack(idx, 1)
    return vocab[I], score.gather(1, I)


def hits_distance(hits: torch.Tensor) -> torch.Tensor:
    """(B, H, W) hit mask -> (B, H, W) distance [m] from each cell centre to the nearest hit-cell centre
    (cv2 exact Euclidean distance transform; 1e3 where a frame has no hits)."""
    import cv2
    from ..sim.raster import RasterConfig
    res = RasterConfig().res
    h = (hits > 0.5).cpu().numpy()
    out = np.full(h.shape, 1e3, np.float32)
    for i in range(len(h)):
        if h[i].any():
            out[i] = cv2.distanceTransform((~h[i]).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * res
    return torch.from_numpy(out).to(hits.device)


def observed_collision(paths: torch.Tensor, hits: torch.Tensor, nx: int = 8, ny: int = 4,
                       max_step: float = 0.2, cell_slack: float = 0.1, dist: torch.Tensor | None = None) -> torch.Tensor:
    """(B, M) bool: does Juno's footprint along an executed path (B, M, T, 2; ego frame) come within reach of
    an observed LiDAR hit cell?

    The footprint (2.875 x 1.165 m) is split into nx x ny sub-rectangles, each covered by its circumscribed
    circle (8 x 4: r = 0.231 m; overshoot beyond the box ~0.085 m laterally, ~0.05 m longitudinally); poses
    are densified to <= ``max_step`` m, headings forward-filled through stationary steps. A circle collides
    when the distance transform of the hits channel at its centre, minus ``cell_slack`` (a hit cell spans
    +-0.1 m), is below its radius. Sees only what the LiDAR saw (front faces; occluded parts are missed)."""
    from ..sim.config import VehicleConfig
    from ..sim.raster import RasterConfig
    rc, veh = RasterConfig(), VehicleConfig.from_yaml()
    B, M, T, _ = paths.shape
    if dist is None:
        dist = hits_distance(hits)
    P = torch.cat([torch.zeros_like(paths[:, :, :1]), paths], 2)                 # (B, M, T+1, 2)
    seg = (P[:, :, 1:] - P[:, :, :-1]).norm(dim=-1)
    f = int(min(max(math.ceil(float(seg.max()) / max_step), 1), 12))
    w = torch.arange(f + 1, device=paths.device, dtype=paths.dtype)[:-1] / f      # (f,)
    Q = (P[:, :, :-1, None] * (1 - w[:, None]) + P[:, :, 1:, None] * w[:, None]).flatten(2, 3)
    Q = torch.cat([Q, P[:, :, -1:]], 2)                                            # (B, M, f*T+1, 2)
    d = torch.diff(Q, dim=2)
    yaw = torch.atan2(d[..., 1], d[..., 0])
    mov = d.norm(dim=-1) > 1e-4
    # forward-fill the heading through stationary steps (stopped / braking candidates keep their last
    # heading; the start pose has heading 0 in the ego frame)
    yaw = torch.where(mov, yaw, torch.full_like(yaw, float("nan")))
    yaw = torch.cat([torch.zeros_like(yaw[..., :1]), yaw], 2)
    for k in range(1, yaw.shape[2]):
        yaw[..., k] = torch.where(torch.isnan(yaw[..., k]), yaw[..., k - 1], yaw[..., k])
    x0, x1 = veh.body_x
    h = veh.width / 2
    sx, sy = (x1 - x0) / nx, 2 * h / ny
    r = 0.5 * math.hypot(sx, sy)
    cx = x0 + sx * (torch.arange(nx, device=paths.device) + 0.5)
    cy = -h + sy * (torch.arange(ny, device=paths.device) + 0.5)
    gx, gy = torch.meshgrid(cx, cy, indexing="ij")
    gx, gy = gx.reshape(-1).to(paths.dtype), gy.reshape(-1).to(paths.dtype)
    c, s_ = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    wx = Q[..., 0:1] + c * gx - s_ * gy
    wy = Q[..., 1:2] + s_ * gx + c * gy                                            # (B, M, K, nx*ny)
    # grid_sample coordinates (align_corners=False): column <-> y, row <-> x
    u = ((rc.y_max - wy) / rc.res) / rc.size * 2 - 1
    v = ((rc.x_max - wx) / rc.res) / rc.size * 2 - 1
    grid = torch.stack([u, v], -1).reshape(B, M, -1, 2)
    D = torch.nn.functional.grid_sample(dist[:, None].to(paths.dtype), grid, mode="bilinear",
                                        padding_mode="border", align_corners=False)[:, 0]   # (B, M, K*n)
    inside = ((u.abs() <= 1) & (v.abs() <= 1)).reshape(B, M, -1)
    return ((D - cell_slack < r) & inside).any(-1)


class GoalFlow(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab: np.ndarray, normalizer: Normalizer):
        super().__init__()
        self.cfg = cfg
        self.norm = normalizer
        self.bev = BEVEncoder(6, cfg.D)
        self.ego = EgoEncoder(cfg.D)
        self.scorer = GoalScorer(vocab, cfg.D, cfg.scorer_layers, cfg.heads, self_attn=cfg.scorer_self_attn)
        self.gen = FlowGenerator(cfg.T, cfg.D, cfg.heads, cfg.gen_layers, cfg.env_queries)
        self.aux = AuxBEVHead(cfg.D) if cfg.aux_head else None
        self.reg = RegressionHead(cfg.T, cfg.D, cfg.heads) if cfg.regression else None

    # ---------------------------------------------------------------- pieces
    def perceive(self, bev, ego):
        F_bev, fmap = self.bev(bev)
        return F_bev, self.ego(ego), fmap

    def goal_norm(self, g: torch.Tensor) -> torch.Tensor:
        """(B, 3) metric goal -> normalised (x, y) + heading."""
        return torch.cat([self.norm(g[..., :2]), g[..., 2:3]], -1)

    def velocity(self, x_t, t, env, goal_tok):
        return self.gen(x_t, t, env, goal_tok)

    # ---------------------------------------------------------------- training losses
    def flow_loss(self, F_bev, F_ego, traj_xy, goal):
        """L1 rectified-flow loss; the goal is the ground-truth endpoint, dropped with p = goal_dropout."""
        x1 = self.norm(traj_xy)
        x_t, t, v = rf_training_pair(x1, self.cfg.sigma, self.cfg.t_dist)
        env = self.gen.env_tokens(F_bev, F_ego)
        drop = torch.rand(x1.shape[0], device=x1.device) < self.cfg.goal_dropout
        g_tok = self.gen.goal_token(self.goal_norm(goal), drop, x1.shape[0], x1.device)
        return (self.velocity(x_t, t, env, g_tok) - v).abs().mean()

    def regression_loss(self, F_bev, F_ego, traj_xy):
        return (self.reg(F_bev, F_ego) - self.norm(traj_xy)).abs().mean()

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def sample(self, F_bev, F_ego, goal: torch.Tensor | None, M: int = 128, n_steps: int = 1,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """M trajectories per scene: (B, M, T, 2) metric. goal None -> goal masked (shadow)."""
        B = F_bev.shape[0]
        env = self.gen.env_tokens(F_bev, F_ego).repeat_interleave(M, 0)
        if goal is None:
            g_tok = self.gen.goal_token(None, None, B * M, F_bev.device)
        else:
            g_tok = self.gen.goal_token(self.goal_norm(goal), None, B, F_bev.device).repeat_interleave(M, 0)
        x0 = self.cfg.sigma * torch.randn(B * M, self.cfg.T, 2, device=F_bev.device, generator=generator)
        x = euler_sample(lambda x, t: self.velocity(x, t, env, g_tok), x0, n_steps)
        return self.norm.inverse(x).view(B, M, self.cfg.T, 2)

    @torch.no_grad()
    def plan(self, bev, ego, M: int = 128, n_steps: int = 1, lambdas=(1.0, 0.5, 0.0), w=(1.0, 1.0),
             prev_path: torch.Tensor | None = None, shadow_M: int = 16, shadow_thresh: float = 1.5,
             use_scorer: bool = True, use_dac: bool = True, use_goal: bool = True,
             roll: RolloutConfig | None = RolloutConfig(), goal_K: int = 1, goal_sep: float = 1.5,
             goal_floor: float = 3.0, l4: float = 1.0, safety: bool = False,
             route_goals: torch.Tensor | None = None, safety_slack: float = 0.1):
        """Full GoalFlow (M3). Flags give the paper's ablations:
        M0: use_goal=False -> mean of goal-free samples; M1: use_dac=False, use_scorer=False (mean of
        goal-conditioned samples); M2: use_scorer=False; M3: all on.
        ``roll``: roll candidates out through pure pursuit + bicycle before scoring (None = off).
        ``goal_K`` > 1 (our extension, DEVIATIONS Part C item 7): up to K goals that are pairwise at least
        ``goal_sep`` m apart and within ``goal_floor`` of the best final score (log units), M/K samples
        toward each; candidates are scored across all of them, with ``l4`` * Phi(own goal's score).
        ``route_goals`` (B, K) vocabulary indices: use these K goals instead of NMS (route-based selection,
        computed outside, e.g. from the observed corridor planner's routes).
        ``safety``: drop candidates whose executed footprint touches an observed hit cell (keep all if
        none survive; ``no_safe`` reports it)."""
        F_bev, F_ego, _ = self.perceive(bev, ego)
        dis, dac = self.scorer(F_bev, F_ego)
        vocab = self.scorer.vocab
        B = bev.shape[0]
        v0, st0 = ego[:, 0], ego[:, 1]

        def drive(paths):           # (B, K, T, 2) -> rolled out, same shape
            if roll is None:
                return paths
            K = paths.shape[1]
            r = rollout(paths.reshape(B * K, self.cfg.T, 2), v0.repeat_interleave(K), st0.repeat_interleave(K), roll)
            return r.view(B, K, self.cfg.T, 2)

        out = {}
        if not use_goal:
            S = self.sample(F_bev, F_ego, None, M, n_steps)
            out.update(samples_raw=S, kappa_ok_raw=kappa_feasible(S).float().mean(1),
                       traj_raw=S.mean(1), traj=drive(S.mean(1, keepdim=True))[:, 0], goal=None)
            return out
        g, score = select_goal(dis, dac if use_dac else torch.zeros_like(dac), vocab, w[0], w[1] if use_dac else 0.0)
        ar = torch.arange(B, device=bev.device)
        if route_goals is not None and use_scorer:
            goal_K = route_goals.shape[1]
            G, G_score = vocab[route_goals], score.gather(1, route_goals)
        elif goal_K > 1 and use_scorer:
            G, G_score = diverse_goals(score, vocab, goal_K, goal_sep, goal_floor)      # (B, K, 3), (B, K)
        if goal_K > 1 and use_scorer:
            Mk = M // goal_K
            S_raw = torch.cat([self.sample(F_bev, F_ego, G[:, k], Mk, n_steps) for k in range(goal_K)], 1)
            goal_of = torch.arange(goal_K, device=bev.device).repeat_interleave(Mk)[None].expand(B, -1)
        else:
            G, G_score = g[:, None], score.max(-1).values[:, None]
            S_raw = self.sample(F_bev, F_ego, g, M, n_steps)
            goal_of = torch.zeros(B, M, dtype=torch.long, device=bev.device)
        out.update(samples_raw=S_raw, kappa_ok_raw=kappa_feasible(S_raw).float().mean(1), goal=g,
                   goal_score=score, dis_logits=dis, dac_logits=dac, goals=G, goal_of=goal_of)
        if not use_scorer:
            out.update(traj_raw=S_raw.mean(1), traj=drive(S_raw.mean(1, keepdim=True))[:, 0], samples=S_raw)
            return out
        S = drive(S_raw)
        l1, l2, l3 = lambdas
        g_own = G[ar[:, None], goal_of]                           # (B, M, 3) each candidate's own goal
        f_dis = (S[:, :, -1] - g_own[..., :2]).norm(dim=-1)
        f_pg = S[:, :, -1, 0]                                     # progress along ego x (track-aligned at t)
        f = -l1 * minmax(f_dis) + l2 * minmax(f_pg)
        if goal_K > 1:                                            # prefer the better-scored route
            f = f + l4 * minmax(G_score[ar[:, None], goal_of])
        if l3 > 0 and prev_path is not None:
            f = f - l3 * minmax((S - prev_path[:, None]).norm(dim=-1).mean(-1))
        if safety:                                                # observed hits only (no privileged input)
            safe = ~observed_collision(S, bev[:, 0], cell_slack=safety_slack)
            any_safe = safe.any(1, keepdim=True)
            f = torch.where(safe | ~any_safe, f, torch.full_like(f, -1e9))
            out.update(safe=safe, no_safe=~any_safe[:, 0])
        best = f.argmax(1)
        traj = S[ar, best]
        out["goal_sel"] = G[ar, goal_of[ar, best]]
        # shadow: goal masked; its best sample by the goal-free term (progress), never the mean
        # (averaging the shadow samples would bring back mode averaging)
        shadow = drive(self.sample(F_bev, F_ego, None, shadow_M, n_steps))
        sh_best = shadow[ar, shadow[:, :, -1, 0].argmax(1)]
        dev = (sh_best - traj).norm(dim=-1).mean(-1)
        use_shadow = dev > shadow_thresh
        traj_main = traj
        traj = torch.where(use_shadow[:, None, None], sh_best, traj)
        out.update(samples=S, traj=traj, traj_raw=S_raw[ar, best], scores=f, best=best, shadow=shadow,
                   used_shadow=use_shadow, traj_main=traj_main, traj_shadow=sh_best, shadow_dev=dev)
        return out
