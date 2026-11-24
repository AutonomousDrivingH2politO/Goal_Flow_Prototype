"""Sanity toys (PROMPT.md section 5): the project's rectified-flow code on 1D and 2D bimodal data.

1D: Roelants' bimodal toy, reconstructed by flow matching.
2D: a disc in front of the start; expert paths pass it left or right 50/50.
    - regression averages the modes and drives through the disc,
    - the unconditioned flow produces both modes,
    - goal conditioning (endpoint) separates them cleanly.
"""
import numpy as np
import pytest
import torch
from torch import nn

from gfsem.models.common import mlp
from gfsem.models.flow import euler_sample, rf_training_pair


def _train(model, batch_fn, steps, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        loss = batch_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


class VField(nn.Module):
    def __init__(self, d, cond=0, h=128):
        super().__init__()
        self.net = mlp(d + 1 + cond, h, d, layers=4)

    def forward(self, x, t, c=None):
        z = [x, t[:, None]] + ([c] if c is not None else [])
        return self.net(torch.cat(z, -1))


# ---------------------------------------------------------------- 1D
def mixture_1d(n, rng):
    m = rng.random(n) < 0.6
    return np.where(m, rng.normal(-0.8, 0.5, n), rng.normal(1.5, 0.2, n)).astype(np.float32)


def test_1d_bimodal_flow_matching():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    f = VField(1, h=64)

    def batch():
        x1 = torch.from_numpy(mixture_1d(256, rng))[:, None]
        xt, t, v = rf_training_pair(x1, sigma=1.0)
        return ((f(xt, t) - v) ** 2).mean()

    _train(f, batch, 2500, lr=1e-3)
    x = euler_sample(f, torch.randn(4000, 1), n_steps=20).squeeze(1).numpy()
    near_right = np.mean(np.abs(x - 1.5) < 0.4)
    near_left = np.mean(np.abs(x + 0.8) < 0.8)
    assert 0.25 < near_right < 0.55          # true weight 0.4
    assert near_left > 0.4                   # true weight 0.6
    assert np.mean(np.abs(x - 0.6) < 0.15) < 0.08   # little mass in the valley between the modes


# ---------------------------------------------------------------- 2D
T = 16
DISC, R = np.array([5.0, 0.0]), 1.0


def expert_paths(n, rng):
    """(n, T, 2) paths from (0,0) to x = 10, passing the disc at y = +-2 (left/right 50/50)."""
    side = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    x = np.linspace(10 / T, 10, T)
    y = side[:, None] * 2.0 * np.clip(np.sin(np.pi * np.clip(x / 7.0, 0, 1) / 2) ** 2, 0, 1)
    y = y + rng.normal(0, 0.05, (n, 1))
    return np.stack([np.broadcast_to(x, y.shape), y], -1).astype(np.float32), side


def norm(p):
    return p / np.array([5.0, 2.5], np.float32) - np.array([1.0, 0.0], np.float32)


def denorm(z):
    return (z + np.array([1.0, 0.0], np.float32)) * np.array([5.0, 2.5], np.float32)


def through_disc(p):
    return (np.hypot(p[..., 0] - DISC[0], p[..., 1] - DISC[1]) < R).any(-1)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(1)


def test_2d_regression_averages_through_the_disc(rng):
    torch.manual_seed(0)
    reg = mlp(1, 64, T * 2, layers=3)

    def batch():
        p, _ = expert_paths(256, rng)
        x1 = torch.from_numpy(norm(p)).reshape(256, -1)
        return (reg(torch.zeros(256, 1)) - x1).abs().mean()

    _train(reg, batch, 800)
    pred = denorm(reg(torch.zeros(1, 1)).detach().reshape(T, 2).numpy())
    assert through_disc(pred[None])[0], "regression should average the two modes into the disc"


def test_2d_unconditioned_flow_has_both_modes(rng):
    torch.manual_seed(0)
    f = VField(T * 2, h=256)

    def batch():
        p, _ = expert_paths(256, rng)
        x1 = torch.from_numpy(norm(p)).reshape(256, -1)
        xt, t, v = rf_training_pair(x1, sigma=1.0)
        return (f(xt, t) - v).abs().mean()

    _train(f, batch, 2500, lr=1e-3)
    s = denorm(euler_sample(f, torch.randn(1000, T * 2), n_steps=10).reshape(-1, T, 2).numpy())
    left = np.mean(s[:, T // 2, 1] > 0)
    assert 0.3 < left < 0.7
    assert through_disc(s).mean() < 0.15


def test_2d_goal_conditioning_separates_modes(rng):
    """sigma = 0.1 as in GoalFlow; the goal is the endpoint (x, y)."""
    torch.manual_seed(0)
    f = VField(T * 2, cond=2, h=256)

    def batch():
        p, _ = expert_paths(256, rng)
        z = norm(p)
        x1 = torch.from_numpy(z).reshape(256, -1)
        g = torch.from_numpy(z[:, -1])
        xt, t, v = rf_training_pair(x1, sigma=0.1)
        return (f(xt, t, g) - v).abs().mean()

    _train(f, batch, 2000, lr=1e-3)
    for side in (1.0, -1.0):
        g = torch.tensor(norm(np.array([[10.0, 2.0 * side]], np.float32))).repeat(500, 1)
        s = denorm(euler_sample(lambda x, t: f(x, t, g), 0.1 * torch.randn(500, T * 2), n_steps=1)
                   .reshape(-1, T, 2).numpy())
        assert np.mean(np.sign(s[:, T // 2, 1]) == side) > 0.97
        assert through_disc(s).mean() < 0.03
