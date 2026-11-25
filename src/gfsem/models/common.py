"""Device choice, sinusoidal encodings and the trajectory normalisation H."""
from __future__ import annotations

import math
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402


def pick_device(pref: str = "auto") -> torch.device:
    """cuda, then mps, then cpu (fp32 on MPS)."""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sinusoidal(x: torch.Tensor, dim: int, max_period: float = 100.0) -> torch.Tensor:
    """Encode the last axis (each scalar) with ``dim`` sin/cos features; output (..., k * dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=x.device, dtype=x.dtype) / half)
    a = x[..., None] * freqs * 2 * math.pi
    return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)


def pos2d(h: int, w: int, dim: int) -> torch.Tensor:
    """2D sin/cos position encoding (h*w, dim)."""
    ys, xs = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    return sinusoidal(torch.stack([ys, xs], -1).reshape(-1, 2), dim // 2, max_period=10.0)


def mlp(i: int, h: int, o: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.GELU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class Normalizer(nn.Module):
    """Per-axis affine map of trajectory (x, y) to about [-1, 1], from the 1st/99th percentiles."""

    def __init__(self, lo=(0.0, -1.0), hi=(1.0, 1.0)):
        super().__init__()
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))

    @classmethod
    def fit(cls, xy: np.ndarray) -> "Normalizer":
        p = xy.reshape(-1, 2)
        return cls(tuple(np.percentile(p, 1, axis=0)), tuple(np.percentile(p, 99, axis=0)))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return 2 * (xy - self.lo) / (self.hi - self.lo) - 1

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return (z + 1) / 2 * (self.hi - self.lo) + self.lo


def heading_from_xy(xy: torch.Tensor) -> torch.Tensor:
    """(..., T, 2) waypoints from the ego origin -> (..., T) headings by finite differences."""
    prev = torch.cat([torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], -2)
    d = xy - prev
    return torch.atan2(d[..., 1], d[..., 0])


def ego_features(ego: torch.Tensor) -> torch.Tensor:
    """Scale the ego status (speed, steer, yaw rate, heading rel. track, lateral offset)."""
    scale = torch.tensor([3.0, 0.175, 0.5, 0.5, 6.0], device=ego.device, dtype=ego.dtype)
    return ego / scale
