"""Rectified-flow trajectory generator (GoalFlow Section 3.2.4, Fig. 4) and the regression baseline."""
from __future__ import annotations

import torch
from torch import nn

from .common import mlp, sinusoidal


class FlowGenerator(nn.Module):
    """Tokens: 16 waypoint tokens (linear(x_t) + learned position), 1 time token, 1 goal token
    (or a learned null token), 16 environment queries that cross-attend to F_bev + F_ego.
    4 transformer-encoder layers; the velocity is read off the waypoint tokens."""

    def __init__(self, T: int = 16, D: int = 256, heads: int = 8, layers: int = 4, env_queries: int = 16,
                 env_layers: int = 2, enc_dim: int = 32):
        super().__init__()
        self.T, self.enc_dim = T, enc_dim
        self.wp_in = nn.Linear(2, D)
        self.wp_pos = nn.Parameter(torch.randn(T, D) * 0.02)
        self.t_mlp = mlp(enc_dim, D, D, layers=2)
        self.g_mlp = mlp(3 * enc_dim, D, D, layers=2)
        self.null_goal = nn.Parameter(torch.zeros(D))
        self.env_q = nn.Parameter(torch.randn(env_queries, D) * 0.02)
        dec = nn.TransformerDecoderLayer(D, heads, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.env = nn.TransformerDecoder(dec, env_layers)
        enc = nn.TransformerEncoderLayer(D, heads, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.type_emb = nn.Parameter(torch.randn(4, D) * 0.02)    # env, goal, traj, time
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2))

    def env_tokens(self, F_bev: torch.Tensor, F_ego: torch.Tensor) -> torch.Tensor:
        mem = F_bev + F_ego[:, None, :]
        q = self.env_q[None].expand(F_bev.shape[0], -1, -1)
        return self.env(q, mem)

    def goal_token(self, goal_n: torch.Tensor | None, drop: torch.Tensor | None, B: int, device):
        """goal_n: (B, 3) goal in normalised (x, y) + raw heading; drop: (B,) bool -> null token."""
        if goal_n is None:
            return self.null_goal[None].expand(B, -1)
        g = self.g_mlp(sinusoidal(goal_n, self.enc_dim, max_period=8.0))
        if drop is not None:
            g = torch.where(drop[:, None], self.null_goal[None].expand(B, -1), g)
        return g

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, env: torch.Tensor, goal_tok: torch.Tensor) -> torch.Tensor:
        """x_t (B, T, 2) normalised, t (B,), env (B, Q, D), goal_tok (B, D) -> velocity (B, T, 2)."""
        B = x_t.shape[0]
        wp = self.wp_in(x_t) + self.wp_pos + self.type_emb[2]
        tt = self.t_mlp(sinusoidal(t[:, None], self.enc_dim, max_period=2.0))[:, None] + self.type_emb[3]
        gt = goal_tok[:, None] + self.type_emb[1]
        h = torch.cat([env + self.type_emb[0], gt, wp, tt], 1)
        h = self.body(h)
        Q = env.shape[1]
        return self.out(h[:, Q + 1:Q + 1 + self.T])


class RegressionHead(nn.Module):
    """Deterministic baseline on the same features: 16 queries cross-attend F_bev + F_ego -> (x, y)."""

    def __init__(self, T: int = 16, D: int = 256, heads: int = 8, layers: int = 3):
        super().__init__()
        self.q = nn.Parameter(torch.randn(T, D) * 0.02)
        dec = nn.TransformerDecoderLayer(D, heads, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec, layers)
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2))

    def forward(self, F_bev, F_ego):
        mem = F_bev + F_ego[:, None, :]
        return self.out(self.dec(self.q[None].expand(F_bev.shape[0], -1, -1), mem))   # (B, T, 2) normalised
