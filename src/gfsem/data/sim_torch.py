"""PyTorch dataset over the L1 simulator shards. Rasters are built at load time.

Each item: BEV raster (6,128,128), ego status (5,), one target mode (16 waypoints, sampled
uniformly among the frame's distinct homotopy modes), its endpoint as the goal, and optionally
DAC targets over the goal vocabulary and auxiliary true-scene masks. True scenes are
regenerated from the seed (deterministic) and cached per worker.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..sim.config import SimConfig
from ..sim.geometry import box_circle_clearance, footprint_corners, from_frame
from ..sim.raster import RasterConfig, rasterize
from ..sim.scene import generate_scene
from .sim_dataset import observation_from_shard, targets_from_shard


class MemmapSplit:
    """A split repacked by scripts/prepare_training.py (mm_<split>/<array>.npy). Dict-like; arrays
    are memory-mapped on first access in each process and the object pickles as its path only."""

    def __init__(self, path):
        self.path = Path(path)
        self._a: dict = {}

    def __getstate__(self):
        return {"path": self.path, "_a": {}}

    def __getitem__(self, k):
        if k not in self._a:
            self._a[k] = np.load(self.path / f"{k}.npy", mmap_mode="r")
        return self._a[k]

    def __contains__(self, k):
        return (self.path / f"{k}.npy").exists()


def load_shards(pattern_or_paths) -> list[dict]:
    paths = sorted(glob.glob(pattern_or_paths)) if isinstance(pattern_or_paths, str) else list(pattern_or_paths)
    return [dict(np.load(p)) for p in paths]


class SimFrames(Dataset):
    def __init__(self, shards: list[dict], vocab: np.ndarray | None = None, aux: bool = False,
                 cfg: SimConfig | None = None, raster: RasterConfig | None = None, aux_res: int = 64,
                 driven_only: bool = False, scene_index=None):
        self.shards = shards
        self.index = [(k, j) for k, a in enumerate(shards) for j in range(len(a["seed"]))]
        self.vocab = vocab
        self.aux = aux
        self.cfg = cfg or SimConfig()
        self.rc = raster or RasterConfig()
        self.aux_res = aux_res
        self.driven_only = driven_only
        self._scenes: dict[int, object] = {}
        self.scene_index = scene_index
        self.max_obs = 24

    def __len__(self) -> int:
        return len(self.index)

    def all_targets(self) -> np.ndarray:
        """(K, 16, 3) every stored mode (for the vocabulary and the normaliser)."""
        return np.concatenate([np.asarray(a["targets"]) for a in self.shards])

    def scene(self, seed: int):
        if self.scene_index is not None and seed in self.scene_index:
            return self.scene_index.get(seed)
        s = self._scenes.get(seed)
        if s is None:
            s = generate_scene(int(seed), self.cfg.scene, self.cfg.vehicle)
            if len(self._scenes) > 256:
                self._scenes.clear()
            self._scenes[seed] = s
        return s

    def dac_targets(self, seed: int, pose: np.ndarray, vocab: np.ndarray) -> np.ndarray:
        """1 if a Juno box at each vocabulary pose (ego frame) lies inside the track and clear of
        every true obstacle disc (full polygon test, no inflation)."""
        sc = self.scene(seed)
        veh = self.cfg.vehicle
        P = np.empty((len(vocab), 3))
        P[:, :2] = from_frame(vocab[:, :2], pose)
        P[:, 2] = vocab[:, 2] + pose[2]
        ok = np.ones(len(vocab), bool)
        if sc.n:
            ok &= box_circle_clearance(P, sc.centres, sc.radii, veh.body_x, veh.width).min(1) >= 0
        cs = footprint_corners(P, veh.body_x, veh.width).reshape(-1, 2)
        _, d = sc.track.frenet(cs)
        ok &= (np.abs(d).reshape(-1, 4) <= sc.track.width / 2).all(1)
        return ok.astype(np.float32)

    def aux_targets(self, seed: int, pose: np.ndarray) -> np.ndarray:
        """(2, R, R): true obstacle discs and true drivable area on the raster extent."""
        sc = self.scene(seed)
        R = self.aux_res
        res = self.rc.size * self.rc.res / R
        i = np.arange(R)
        x = self.rc.x_max - (i + 0.5) * res
        y = self.rc.y_max - (i + 0.5) * res
        X, Y = np.meshgrid(x, y, indexing="ij")
        W = from_frame(np.stack([X, Y], -1).reshape(-1, 2), pose)
        _, d = sc.track.frenet(W)
        drv = (np.abs(d) <= sc.track.width / 2).reshape(R, R)
        obs = np.zeros(len(W), bool)
        for c, r in zip(sc.centres, sc.radii):
            obs |= np.hypot(W[:, 0] - c[0], W[:, 1] - c[1]) <= r + res / 2
        return np.stack([obs.reshape(R, R), drv]).astype(np.float32)

    def __getitem__(self, i: int) -> dict:
        k, j = self.index[i]
        a = self.shards[k]
        obs = observation_from_shard(a, j, self.cfg.lidar)
        bev = rasterize(obs, self.rc)
        T, sigs = targets_from_shard(a, j)
        m = 0 if self.driven_only else int(torch.randint(len(T), (1,)).item())
        traj = T[m]
        seed, pose = int(a["seed"][j]), a["pose"][j].astype(np.float64)
        out = dict(bev=torch.from_numpy(bev), ego=torch.from_numpy(a["ego"][j].astype(np.float32)),
                   traj=torch.from_numpy(traj[:, :2].copy()), goal=torch.from_numpy(traj[-1].copy()),
                   modes=torch.from_numpy(np.pad(T[:, :, :2], ((0, 8 - min(len(T), 8)), (0, 0), (0, 0)))[:8]),
                   n_modes=torch.tensor(min(len(T), 8)), seed=torch.tensor(seed), idx=torch.tensor(i),
                   recovery=torch.tensor(bool(a["recovery"][j])))
        sc = self.scene(seed)
        obs = np.zeros((self.max_obs, 3), np.float32)        # true obstacles, ego frame: x, y, r
        if sc.n:
            from ..sim.geometry import to_frame
            loc = to_frame(sc.centres, pose)
            k = min(sc.n, self.max_obs)
            obs[:k, :2] = loc[:k]
            obs[:k, 2] = sc.radii[:k]
        out["obstacles"] = torch.from_numpy(obs)
        out["n_obstacles"] = torch.tensor(min(sc.n, self.max_obs))
        out["pose"] = torch.from_numpy(pose.astype(np.float32))
        if self.vocab is not None:
            out["dac"] = torch.from_numpy(self.dac_targets(seed, pose, self.vocab))
        if self.aux:
            out["aux"] = torch.from_numpy(self.aux_targets(seed, pose))
        return out
