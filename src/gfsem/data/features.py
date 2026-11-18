"""Frozen image backbones for the planner's image tokens (PROMPT.md §2 "Image features").

Default: DINOv2 ViT-S/14 at 518×294 → 37×21 patch tokens of size 384. Alternative: ImageNet
ResNet-34 (layer3, stride 16). Also wraps the team's TwinLiteNetPlus drivable-area model,
used offline for pseudo-labels only (never as model input).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class BackboneSpec:
    name: str = "dinov2_vits14"
    width: int = 518
    height: int = 294
    patch: int = 14

    @property
    def grid(self) -> tuple[int, int]:
        """(rows, cols) of the token grid."""
        return self.height // self.patch, self.width // self.patch


def preprocess(bgr_frames: list[np.ndarray], width: int, height: int, device: str, flip: bool = False) -> torch.Tensor:
    """BGR uint8 frames → normalised RGB tensor (B, 3, H, W). ``flip`` mirrors horizontally."""
    arr = np.stack([cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)[:, :, ::-1] for f in bgr_frames])
    if flip:
        arr = arr[:, :, ::-1]
    x = torch.from_numpy(np.ascontiguousarray(arr)).to(device).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


class DinoV2Tokens(nn.Module):
    """Frozen DINOv2; returns patch tokens (B, rows, cols, C)."""

    def __init__(self, spec: BackboneSpec) -> None:
        super().__init__()
        self.spec = spec
        self.net = torch.hub.load("facebookresearch/dinov2", spec.name, verbose=False)
        self.net.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net.forward_features(x)["x_norm_patchtokens"]
        r, c = self.spec.grid
        return out.reshape(x.shape[0], r, c, -1)


class ResNet34Tokens(nn.Module):
    """Frozen ImageNet ResNet-34 up to layer3 (stride 16, 256 channels); returns (B, rows, cols, C)."""

    def __init__(self) -> None:
        super().__init__()
        import torchvision

        m = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)
        self.net.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).permute(0, 2, 3, 1)


class TwinLiteDrivable:
    """Team TwinLiteNetPlus, used read-only for drivable-area pseudo-labels.

    Input frames are resized to 640×360 and padded to 640×384 (stride 32), as the repo's letterbox does.
    """

    def __init__(self, repo: Path, config: str = "large", device: str = "cuda") -> None:
        sys.path.insert(0, str(repo))
        from model.model import TwinLiteNetPlus  # type: ignore[import-not-found]

        self.model = TwinLiteNetPlus(SimpleNamespace(config=config)).to(device)
        ckpt = next((repo / "pretrained").rglob(f"{config}.pth"))
        state = torch.load(ckpt, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval().half()
        self.device = device

    @torch.no_grad()
    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """Return a drivable-area probability map at the input resolution (H, W), float32 in [0, 1]."""
        H, W = bgr.shape[:2]
        img = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).to(self.device)
        x = x.half().unsqueeze(0) / 255.0
        da, _ = self.model(x)
        p = torch.softmax(da.float(), 1)[0, 1, 12:372]
        p = F.interpolate(p[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        return p.cpu().numpy()


def mask_to_grid(mask: np.ndarray, spec: BackboneSpec) -> np.ndarray:
    """Average a full-resolution mask over each backbone patch → (rows, cols)."""
    m = cv2.resize(mask.astype(np.float32), (spec.width, spec.height), interpolation=cv2.INTER_AREA)
    r, c = spec.grid
    return m[: r * spec.patch, : c * spec.patch].reshape(r, spec.patch, c, spec.patch).mean(axis=(1, 3))
