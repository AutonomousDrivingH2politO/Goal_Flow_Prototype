"""Rectified flow utilities (GoalFlow Eqs. 4-5, 10-11; Liu et al. 2022).

t = 0 is noise, t = 1 is data: x_t = (1 - t) x0 + t x1, target velocity v = x1 - x0.
"""
from __future__ import annotations

from typing import Callable

import torch


def sample_t(n: int, device, dist: str = "uniform") -> torch.Tensor:
    if dist == "uniform":
        return torch.rand(n, device=device)
    if dist == "logit_normal":
        return torch.sigmoid(torch.randn(n, device=device))
    raise ValueError(dist)


def rf_training_pair(x1: torch.Tensor, sigma: float = 0.1, t_dist: str = "uniform"):
    """Returns (x_t, t, v_target) for a batch of normalised targets x1 (B, ...)."""
    x0 = sigma * torch.randn_like(x1)
    t = sample_t(x1.shape[0], x1.device, t_dist)
    tb = t.view(-1, *([1] * (x1.dim() - 1)))
    return (1 - tb) * x0 + tb * x1, t, x1 - x0


@torch.no_grad()
def euler_sample(vfn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x0: torch.Tensor,
                 n_steps: int = 1, return_path: bool = False):
    """Integrate dx/dt = v(x, t) from t = 0 to 1 with uniform Euler steps."""
    x = x0
    path = [x0]
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i / n_steps, device=x.device, dtype=x.dtype)
        x = x + vfn(x, t) / n_steps
        if return_path:
            path.append(x)
    return (x, path) if return_path else x
