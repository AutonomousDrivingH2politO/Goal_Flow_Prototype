import math

import numpy as np
import torch

from gfsem.models.common import Normalizer
from gfsem.models.goalflow import GoalFlow, ModelConfig
from gfsem.models.rollout import RolloutConfig, discrete_curvature, kappa_feasible, rollout

T = 16


def arc_path(kappa, v=3.0):
    s = v * 0.25 * torch.arange(1, T + 1, dtype=torch.float32)
    if abs(kappa) < 1e-9:
        return torch.stack([s, torch.zeros_like(s)], -1)
    return torch.stack([torch.sin(kappa * s) / kappa, (1 - torch.cos(kappa * s)) / kappa], -1)


def test_curvature_estimate_and_feasibility():
    cfg = RolloutConfig()
    k = discrete_curvature(arc_path(0.08))
    assert torch.allclose(k[2:], torch.full_like(k[2:], 0.08), atol=0.01)
    assert kappa_feasible(arc_path(0.9 * cfg.kappa_max))
    assert not kappa_feasible(arc_path(2.0 * cfg.kappa_max))


def test_rollout_tracks_feasible_paths_and_bounds_infeasible_ones():
    cfg = RolloutConfig()
    paths = torch.stack([arc_path(0.0), arc_path(0.7 * cfg.kappa_max), arc_path(3.0 * cfg.kappa_max)])
    r = rollout(paths, torch.full((3,), 3.0), torch.zeros(3), cfg)
    assert r.shape == paths.shape
    assert (r[0] - paths[0]).norm(dim=-1).max() < 0.05                  # straight: exact
    assert (r[1] - paths[1]).norm(dim=-1).max() < 0.6                   # feasible arc: close
    assert (r[2] - paths[2]).norm(dim=-1).max() > 1.0                   # too sharp: cannot follow
    assert (discrete_curvature(r).abs() <= cfg.kappa_max * 1.05).all()  # executed paths obey 10 deg


def test_goalflow_plan_rolls_out_candidates():
    torch.manual_seed(0)
    V = np.random.default_rng(0).normal(size=(64, 3)).astype(np.float32) * [3, 2, 0.2] + [11, 0, 0]
    m = GoalFlow(ModelConfig(D=64, heads=4, scorer_layers=1, gen_layers=1), V, Normalizer((0.7, -3.6), (12, 3.6))).eval()
    bev, ego = torch.rand(2, 6, 128, 128), torch.tensor([[3.0, 0, 0, 0, 0]] * 2)
    r = m.plan(bev, ego, M=8, n_steps=1)
    assert r["traj"].shape == (2, T, 2) and r["samples"].shape == (2, 8, T, 2)
    assert r["kappa_ok_raw"].shape == (2,)
    assert (discrete_curvature(r["samples"]).abs() <= RolloutConfig().kappa_max * 1.05).all()
    r0 = m.plan(bev, ego, M=8, use_goal=False)
    assert r0["traj"].shape == (2, T, 2)
