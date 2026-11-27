#!/usr/bin/env python
"""EMX pitch check at the roof mount (queue item 6; scans only, no planner).

    python scripts/pitch_check.py [--scenes 60] [--out reports/lidar_demo/mounting]

EMX noise preset, `roof` mount (1.05 m, 1.20 m, placeholder body profile) at pitch 5-9 deg. Dev scenes 0..N-1, three
poses near the start of each track, one scan each; per obstacle in the field of view at 10 / 15 / 20 / 25 / 30 m
(+-1 m) from the sensor: returns within r + 0.15 m (occluded = 0) and whether `cluster_obstacles` returns a circle
within 0.5 m of its true centre. Cones and bollards separately.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from gfsem.planners.corridor import CorridorConfig, cluster_obstacles  # noqa: E402
from gfsem.sim.config import BOLLARD, NOISE_LEVELS, LidarConfig, SceneConfig  # noqa: E402
from gfsem.sim.geometry import to_frame  # noqa: E402
from gfsem.sim.lidar import scan  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402

DISTS = (10, 15, 20, 25, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/mounting"))
    a = ap.parse_args()
    ccfg, scfg = CorridorConfig(), SceneConfig.from_yaml()
    scenes = [generate_scene(s) for s in range(a.scenes)]
    res = {}
    lines = ["| pitch | class | " + " | ".join(f"{d} m: hits / circle %" for d in DISTS) + " |", "|---" * (2 + len(DISTS)) + "|"]
    for p in (5, 6, 7, 8, 9):
        cfg = LidarConfig.with_mount("roof", pitch_deg=float(p))
        acc = {(c, d): [] for c in (0, 1) for d in DISTS}
        for sc in scenes:
            rng = np.random.default_rng([sc.seed, 7, p])
            for s0 in (0.0, 10.0, 20.0):
                pose = sc.track.pose(s0, 0.0)
                obs = scan(sc, pose, NOISE_LEVELS["EMX"], rng, cfg)
                pts = np.asarray(obs.points, float).reshape(-1, 2)
                C, _, _ = cluster_obstacles(obs, ccfg, scfg)
                C = np.asarray(C, float).reshape(-1, 2)
                T = to_frame(sc.centres, pose)
                rng_s = np.hypot(T[:, 0] - cfg.mount_x, T[:, 1] - cfg.mount_y)
                ang = np.degrees(np.arctan2(T[:, 1] - cfg.mount_y, T[:, 0] - cfg.mount_x))
                for j in range(sc.n):
                    if abs(ang[j]) > 55:
                        continue
                    for d in DISTS:
                        if abs(rng_s[j] - d) <= 1.0:
                            n = int((np.hypot(*(pts - T[j]).T) <= sc.radii[j] + 0.15).sum()) if len(pts) else 0
                            circ = bool(len(C) and np.hypot(*(C - T[j]).T).min() <= 0.5)
                            acc[(int(sc.classes[j] == BOLLARD), d)].append((n, circ))
        for c, name in ((0, "cone"), (1, "bollard")):
            cells = []
            for d in DISTS:
                v = acc[(c, d)]
                res[f"pitch{p}_{name}_{d}m"] = dict(n=len(v), hits=float(np.mean([x[0] for x in v])) if v else None,
                                                    circle=float(np.mean([x[1] for x in v])) if v else None)
                cells.append(f"{np.mean([x[0] for x in v]):.1f} / {100 * np.mean([x[1] for x in v]):.0f}% (n={len(v)})" if v else "-")
            lines.append(f"| {p} | {name} | " + " | ".join(cells) + " |")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "pitch_check.json").write_text(json.dumps(res, indent=1))
    (a.out / "pitch_check.md").write_text("EMX preset, roof mount, dev scenes 0-%d x 3 poses; returns per obstacle / share with a "
                                         "fitted circle within 0.5 m.\n\n" % (a.scenes - 1) + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
