"""Perception: BEV raster encoder (6 x 128 x 128 -> 256 tokens of size D) and ego encoder."""
from __future__ import annotations

import torch
from torch import nn

from .common import ego_features, mlp, pos2d


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                     if stride != 1 or cin != cout else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.b1(self.c1(x)))
        return self.act(self.b2(self.c2(y)) + self.skip(x))


class BEVEncoder(nn.Module):
    """Small ResNet: 128 -> 64 -> 32 -> 16, then 1x1 to D; + 2D sin/cos position; 256 tokens."""

    def __init__(self, in_ch: int = 6, D: int = 256, widths=(32, 64, 128, 192)):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.stem = nn.Sequential(nn.Conv2d(in_ch, w0, 3, 2, 1, bias=False), nn.BatchNorm2d(w0),
                                  nn.ReLU(inplace=True))                       # 64
        self.layers = nn.Sequential(ResBlock(w0, w1), ResBlock(w1, w1),
                                    ResBlock(w1, w2, 2), ResBlock(w2, w2),     # 32
                                    ResBlock(w2, w3, 2), ResBlock(w3, w3))     # 16
        self.proj = nn.Conv2d(w3, D, 1)
        self.register_buffer("pos", pos2d(16, 16, D), persistent=False)
        self.out_hw = (16, 16)

    def forward(self, bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (tokens (B, 256, D), feature map (B, D, 16, 16))."""
        f = self.proj(self.layers(self.stem(bev)))
        tok = f.flatten(2).transpose(1, 2) + self.pos
        return tok, f


class EgoEncoder(nn.Module):
    def __init__(self, D: int = 256, n_in: int = 5):
        super().__init__()
        self.net = mlp(n_in, D, D, layers=3)

    def forward(self, ego: torch.Tensor) -> torch.Tensor:
        return self.net(ego_features(ego))           # (B, D)


class AuxBEVHead(nn.Module):
    """Optional: predict the *true* obstacle discs and drivable area (2 x 64 x 64 logits)."""

    def __init__(self, D: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(D, 128, 4, 2, 1), nn.ReLU(inplace=True),     # 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),    # 64
            nn.Conv2d(64, 2, 3, 1, 1))

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.net(fmap)
