"""Precomputed true-scene geometry per seed (obstacles, track, goal), so training targets (DAC, aux)
do not regenerate scenes (~0.5 s each, oracle included) on every sample."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sim.geometry import Track


@dataclass
class SceneLite:
    seed: int
    track: Track
    centres: np.ndarray
    radii: np.ndarray
    classes: np.ndarray
    s_goal: float
    feasible: bool

    @property
    def n(self) -> int:
        return len(self.centres)


def scene_record(seed: int) -> dict:
    from ..sim.scene import generate_scene
    sc = generate_scene(seed)
    t = sc.track
    return dict(seed=seed, kappa=t.kappa, width=t.width, s_min=t.s_min, s_max=t.s_max, s_goal=sc.s_goal,
                feasible=sc.feasible, centres=sc.centres.astype(np.float32), radii=sc.radii.astype(np.float32),
                classes=sc.classes.astype(np.int8))


def save_index(records: list[dict], path: Path):
    records = sorted(records, key=lambda r: r["seed"])
    off = np.concatenate([[0], np.cumsum([len(r["centres"]) for r in records])]).astype(np.int64)
    np.savez(path,
             seed=np.array([r["seed"] for r in records], np.int64),
             track=np.array([[r["kappa"], r["width"], r["s_min"], r["s_max"], r["s_goal"]] for r in records], np.float64),
             feasible=np.array([r["feasible"] for r in records], bool),
             centres=np.concatenate([r["centres"] for r in records]).reshape(-1, 2),
             radii=np.concatenate([r["radii"] for r in records]),
             classes=np.concatenate([r["classes"] for r in records]),
             offsets=off)


class SceneIndex:
    def __init__(self, paths):
        self.paths = [str(p) for p in paths]
        self._d = None

    def _load(self):
        parts = [dict(np.load(p)) for p in self.paths]
        seeds, rows = [], {}
        for k, a in enumerate(parts):
            for i, s in enumerate(a["seed"]):
                rows[int(s)] = (k, i)
        self._d = (parts, rows)

    def __getstate__(self):
        return {"paths": self.paths, "_d": None}

    def __contains__(self, seed: int) -> bool:
        if self._d is None:
            self._load()
        return int(seed) in self._d[1]

    def get(self, seed: int) -> SceneLite:
        if self._d is None:
            self._load()
        parts, rows = self._d
        k, i = rows[int(seed)]
        a = parts[k]
        kappa, width, s_min, s_max, s_goal = a["track"][i]
        o0, o1 = a["offsets"][i], a["offsets"][i + 1]
        return SceneLite(int(seed), Track(float(kappa), float(width), float(s_min), float(s_max)),
                         a["centres"][o0:o1].astype(np.float64), a["radii"][o0:o1].astype(np.float64),
                         a["classes"][o0:o1], float(s_goal), bool(a["feasible"][i]))
