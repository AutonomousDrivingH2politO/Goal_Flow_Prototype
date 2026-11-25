"""Goal vocabulary and goal scorer (GoalFlow Section 3.2.3)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .common import mlp, sinusoidal

THETA_SCALE = 2.0      # 1 rad of heading counts as 2 m in the goal metric


def goal_metric(g: np.ndarray | torch.Tensor):
    """(..., 3) (x, y, theta) -> (..., 3) with theta scaled, so Euclidean = the goal distance."""
    if isinstance(g, torch.Tensor):
        return torch.cat([g[..., :2], THETA_SCALE * g[..., 2:3]], -1)
    return np.concatenate([g[..., :2], THETA_SCALE * g[..., 2:3]], -1)


def kmeans_vocabulary(endpoints: np.ndarray, n: int, iters: int = 30, seed: int = 0,
                      max_points: int = 200_000) -> np.ndarray:
    """k-means (k-means++ init, Lloyd iterations) on (x, y, theta) endpoints in the goal metric."""
    rng = np.random.default_rng(seed)
    X = goal_metric(endpoints.astype(np.float64))
    if len(X) > max_points:
        X = X[rng.choice(len(X), max_points, replace=False)]
    n = min(n, len(X))
    C = [X[rng.integers(len(X))]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, n):
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    C = np.array(C)
    for _ in range(iters):
        lab = np.empty(len(X), np.int64)
        for a in range(0, len(X), 20000):                  # chunked assignment
            d = ((X[a:a + 20000, None, :] - C[None]) ** 2).sum(-1)
            lab[a:a + 20000] = d.argmin(1)
        for k in range(n):
            m = lab == k
            if m.any():
                C[k] = X[m].mean(0)
            else:
                C[k] = X[rng.integers(len(X))]
    C[:, 2] /= THETA_SCALE
    return C.astype(np.float32)


class CrossAttnLayer(nn.Module):
    """Pre-norm cross-attention + FFN, no self-attention among the queries (each goal is scored
    independently given the BEV). Avoids the N x N attention map over the vocabulary."""

    def __init__(self, D: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(D)
        self.att = nn.MultiheadAttention(D, heads, batch_first=True)
        self.n2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, q, mem):
        q = q + self.att(self.n1(q), mem, mem, need_weights=False)[0]
        return q + self.ff(self.n2(q))


class GoalScorer(nn.Module):
    """Queries = MLP(sinusoidal(vocab)) + F_ego; transformer decoder over F_bev; dis and dac heads."""

    def __init__(self, vocab: np.ndarray, D: int = 256, layers: int = 3, heads: int = 8,
                 enc_dim: int = 32, self_attn: bool = False):
        super().__init__()
        self.register_buffer("vocab", torch.as_tensor(vocab, dtype=torch.float32))   # (N, 3) ego frame
        self.enc_dim = enc_dim
        self.q_mlp = mlp(3 * enc_dim, D, D, layers=2)
        self.self_attn = self_attn
        if self_attn:   # standard decoder (self-attention among all N goal queries): N x N memory
            dec = nn.TransformerDecoderLayer(D, heads, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
            self.decoder = nn.TransformerDecoder(dec, layers)
        else:
            self.decoder = nn.ModuleList([CrossAttnLayer(D, heads) for _ in range(layers)])
        self.head_dis = mlp(D, D, 1, layers=2)
        self.head_dac = mlp(D, D, 1, layers=2)

    def encode_points(self, g: torch.Tensor) -> torch.Tensor:
        """(..., 3) goal points in the ego frame -> (..., D)."""
        z = torch.stack([g[..., 0] / 15.0, g[..., 1] / 10.0, g[..., 2]], -1)
        return self.q_mlp(sinusoidal(z, self.enc_dim, max_period=8.0))

    def forward(self, F_bev: torch.Tensor, F_ego: torch.Tensor, points: torch.Tensor | None = None):
        """Returns logits (B, N) for distance and DAC; ``points`` (B, N, 3) overrides the vocabulary
        (dense goal-score maps)."""
        B = F_bev.shape[0]
        P = self.vocab[None].expand(B, -1, -1) if points is None else points
        q = self.encode_points(P) + F_ego[:, None, :]
        if self.self_attn:
            h = self.decoder(q, F_bev)
        else:
            h = q
            for layer in self.decoder:
                h = layer(h, F_bev)
        return self.head_dis(h).squeeze(-1), self.head_dac(h).squeeze(-1)


def goal_loss(dis_logits, dac_logits, vocab, g_gt, dac_target, T: float = 0.5,
              w4: float = 1.0, w5: float = 0.005):
    """L_goal = w4 L_dis + w5 L_dac (Eqs. 14-16); distance target softmax(-||g_i - g_gt|| / T)."""
    d = torch.cdist(goal_metric(g_gt)[:, None, :], goal_metric(vocab)[None].expand(g_gt.shape[0], -1, -1))[:, 0]
    target = torch.softmax(-d / T, -1)
    l_dis = -(target * F.log_softmax(dis_logits, -1)).sum(-1).mean()
    l_dac = F.binary_cross_entropy_with_logits(dac_logits, dac_target)
    return w4 * l_dis + w5 * l_dac, dict(l_dis=l_dis.detach(), l_dac=l_dac.detach())


def select_goal(dis_logits, dac_logits, vocab, w1: float = 1.0, w2: float = 1.0):
    """g* = argmax_i w1 log softmax(dis) + w2 log sigmoid(dac). Returns (g* (B,3), score (B,N))."""
    score = w1 * F.log_softmax(dis_logits, -1) + w2 * F.logsigmoid(dac_logits)
    idx = score.argmax(-1)
    return vocab[idx], score
