#!/usr/bin/env python
"""Hits per obstacle by noise level (unit check, no planner): scenes 0..N-1, the start pose, one scan each; per
obstacle in the field of view at 2-5 / 8-12 / 18-22 m from the sensor, the number of returns within r + 0.15 m of its
centre (mean over obstacles; occluded ones count as 0).

    python scripts/hits_per_obstacle.py L2,L4,EMX,emx_L2art,emx_L4art [N]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from gfsem.sim.config import BOLLARD, NOISE_LEVELS, LidarConfig  # noqa: E402
from gfsem.sim.geometry import to_frame  # noqa: E402
from gfsem.sim.lidar import scan  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402

levels = sys.argv[1].split(",")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
cfg = LidarConfig()
bins = ((2, 5), (8, 12), (18, 22))
print("| level | " + " | ".join(f"bollard {a}-{b} m" for a, b in bins) + " | " + " | ".join(f"cone {a}-{b} m" for a, b in bins) + " |")
print("|---" * (1 + 2 * len(bins)) + "|")
for lvl in levels:
    acc = {(c, b): [] for c in (0, 1) for b in bins}
    for seed in range(N):
        sc = generate_scene(seed)
        rng = np.random.default_rng([seed, 1])
        for s0 in (0.0, 10.0, 20.0):                       # three poses along the start of the track
            pose = sc.track.pose(s0, 0.0)
            obs = scan(sc, pose, NOISE_LEVELS[lvl], rng, cfg)
            pts = np.asarray(obs.points, float).reshape(-1, 2)
            C = to_frame(sc.centres, pose)
            d = np.hypot(C[:, 0] - cfg.mount_x, C[:, 1])
            ang = np.degrees(np.arctan2(C[:, 1], C[:, 0] - cfg.mount_x))
            for j in range(sc.n):
                if abs(ang[j]) > 55:
                    continue
                for b in bins:
                    if b[0] <= d[j] <= b[1]:
                        n = int((np.hypot(*(pts - C[j]).T) <= sc.radii[j] + 0.15).sum()) if len(pts) else 0
                        acc[(int(sc.classes[j] == BOLLARD), b)].append(n)
    cell = lambda v: f"{np.mean(v):.1f} (n={len(v)})" if v else "-"  # noqa: E731
    print(f"| {lvl} | " + " | ".join(cell(acc[(1, b)]) for b in bins) + " | " + " | ".join(cell(acc[(0, b)]) for b in bins) + " |")
