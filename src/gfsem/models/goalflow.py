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
             roll: RolloutConfig | None = RolloutConfig()):
        """Full GoalFlow (M3). Flags give the paper's ablations:
        M0: use_goal=False -> mean of goal-free samples; M1: use_dac=False, use_scorer=False (mean of
        goal-conditioned samples); M2: use_scorer=False; M3: all on.
        ``roll``: roll candidates out through pure pursuit + bicycle before scoring (None = off)."""
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
        S_raw = self.sample(F_bev, F_ego, g, M, n_steps)
        out.update(samples_raw=S_raw, kappa_ok_raw=kappa_feasible(S_raw).float().mean(1), goal=g,
                   goal_score=score, dis_logits=dis, dac_logits=dac)
        if not use_scorer:
            out.update(traj_raw=S_raw.mean(1), traj=drive(S_raw.mean(1, keepdim=True))[:, 0], samples=S_raw)
            return out
        S = drive(S_raw)
        l1, l2, l3 = lambdas
        f_dis = (S[:, :, -1] - g[:, None, :2]).norm(dim=-1)
        f_pg = S[:, :, -1, 0]                                     # progress along ego x (track-aligned at t)
        f = -l1 * minmax(f_dis) + l2 * minmax(f_pg)
        if l3 > 0 and prev_path is not None:
            f = f - l3 * minmax((S - prev_path[:, None]).norm(dim=-1).mean(-1))
        best = f.argmax(1)
        ar = torch.arange(B, device=bev.device)
        traj = S[ar, best]
        # shadow: goal masked; its best sample by the goal-free term (progress), never the mean
        # (averaging the shadow samples would bring back mode averaging)
        shadow = drive(self.sample(F_bev, F_ego, None, shadow_M, n_steps))
        sh_best = shadow[ar, shadow[:, :, -1, 0].argmax(1)]
        dev = (sh_best - traj).norm(dim=-1).mean(-1)
        use_shadow = dev > shadow_thresh
        traj = torch.where(use_shadow[:, None, None], sh_best, traj)
        out.update(samples=S, traj=traj, traj_raw=S_raw[ar, best], scores=f, best=best, shadow=shadow,
                   used_shadow=use_shadow)
        return out
