"""Curvature feasibility of trajectory candidates and their closed-loop rollout.

Each candidate (16 waypoints at 0.25 s, ego frame) is driven through the same pure pursuit
(2.5 m lookahead) and kinematic bicycle (steering limited to 10 deg, rate- and accel-limited) as the
teacher validation, so the scorer and critics judge the path Juno would actually drive.
Vectorised over candidates in torch (runs on the model's device).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutConfig:
    wheelbase: float = 1.60
    max_steer_deg: float = 10.0
    max_steer_rate_deg: float = 40.0
    max_accel: float = 1.0
    max_decel: float = 3.0
    lookahead: float = 2.5
    dt: float = 0.05
    wp_dt: float = 0.25

    @property
    def kappa_max(self) -> float:
        return math.tan(math.radians(self.max_steer_deg)) / self.wheelbase


def discrete_curvature(xy: torch.Tensor) -> torch.Tensor:
    """(..., T, 2) waypoints after the ego origin -> (..., T-1) curvature at interior points."""
    p = torch.cat([torch.zeros_like(xy[..., :1, :]), xy], -2)
    d = p[..., 1:, :] - p[..., :-1, :]
    th = torch.atan2(d[..., 1], d[..., 0])
    ds = d.norm(dim=-1)
    dth = torch.remainder(th[..., 1:] - th[..., :-1] + math.pi, 2 * math.pi) - math.pi
    seg = 0.5 * (ds[..., 1:] + ds[..., :-1])
    k = dth / seg.clamp_min(1e-3)
    return torch.where(seg > 0.05, k, torch.zeros_like(k))     # ignore near-stationary steps


def kappa_feasible(xy: torch.Tensor, cfg: RolloutConfig = RolloutConfig(), tol: float = 1.0) -> torch.Tensor:
    """(...,) bool: every interior curvature within tol * kappa_max."""
    return (discrete_curvature(xy).abs() <= tol * cfg.kappa_max).all(-1)


@torch.no_grad()
def rollout(cands: torch.Tensor, v0: torch.Tensor, steer0: torch.Tensor,
            cfg: RolloutConfig = RolloutConfig()) -> torch.Tensor:
    """Drive each candidate path with pure pursuit + bicycle from (0, 0, 0, v0, steer0).

    cands (N, T, 2) metric ego frame; v0, steer0 (N,). Returns executed positions (N, T, 2) at the
    candidates' own timestamps (k * wp_dt). The speed command follows each candidate's speed profile.
    """
    N, T, _ = cands.shape
    dev, dt = cands.device, cfg.dt
    P = torch.cat([torch.zeros(N, 1, 2, device=dev, dtype=cands.dtype), cands], 1)      # (N, T+1, 2)
    seg = P[:, 1:] - P[:, :-1]
    v_ref = seg.norm(dim=-1) / cfg.wp_dt                                                 # (N, T)
    last_dir = seg[:, -1] / seg[:, -1].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x = torch.zeros(N, device=dev, dtype=cands.dtype)
    y = torch.zeros_like(x)
    yaw = torch.zeros_like(x)
    v = v0.to(cands.dtype).clone()
    st = steer0.to(cands.dtype).clone()
    near = torch.zeros(N, dtype=torch.long, device=dev)
    idx_ar = torch.arange(T + 1, device=dev)[None]
    smax, srate = math.radians(cfg.max_steer_deg), math.radians(cfg.max_steer_rate_deg) * dt
    per = int(round(cfg.wp_dt / dt))
    out = []
    for k in range(T * per):
        pos = torch.stack([x, y], -1)
        d = (P - pos[:, None]).norm(dim=-1)                                             # (N, T+1)
        # monotone progress: nearest point at or after the previous nearest, within a window
        win = (idx_ar >= near[:, None]) & (idx_ar <= near[:, None] + 4)
        near = torch.where(win, d, torch.full_like(d, 1e9)).argmin(1)
        ahead = (idx_ar >= near[:, None]) & (d >= cfg.lookahead)
        has = ahead.any(1)
        j = torch.where(ahead, idx_ar.expand(N, -1), torch.full_like(idx_ar.expand(N, -1), T + 1)).min(1).values
        tgt = P[torch.arange(N, device=dev), j.clamp(max=T)]
        ext = P[:, -1] + last_dir * (cfg.lookahead - d[:, -1]).clamp_min(0)[:, None]
        tgt = torch.where(has[:, None], tgt, ext)
        dx, dy = tgt[:, 0] - x, tgt[:, 1] - y
        c, s = torch.cos(yaw), torch.sin(yaw)
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        ld2 = (lx * lx + ly * ly).clamp_min(1e-6)
        st_cmd = torch.atan(cfg.wheelbase * 2 * ly / ld2)
        st = (st + (st_cmd - st).clamp(-srate, srate)).clamp(-smax, smax)
        v_cmd = v_ref[:, min(k // per, T - 1)]
        v_new = (v + (v_cmd - v).clamp(-cfg.max_decel * dt, cfg.max_accel * dt)).clamp_min(0)
        vm = 0.5 * (v + v_new)
        yr = vm * torch.tan(st) / cfg.wheelbase
        ym = yaw + 0.5 * yr * dt
        x, y, yaw, v = x + vm * torch.cos(ym) * dt, y + vm * torch.sin(ym) * dt, yaw + yr * dt, v_new
        if (k + 1) % per == 0:
            out.append(torch.stack([x, y], -1))
    return torch.stack(out, 1)
