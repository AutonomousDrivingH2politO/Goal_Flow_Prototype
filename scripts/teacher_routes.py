#!/usr/bin/env python
"""Distinct teacher routes per scene: the lattice inside each of the top-K distinct A* routes,
deduplicated by homotopy signature, evaluated at the spawn pose.

    python scripts/teacher_routes.py --seeds 0-499 --K 8 --workers 2
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def count(job):
    seed, K = job
    from gfsem.planners.corridor import input_from_privileged
    from gfsem.planners.corridor_lattice import LatticeCorridorPlanner
    from gfsem.sim.lidar import privileged_view
    from gfsem.sim.scene import generate_scene
    sc = generate_scene(seed)
    p0 = sc.track.pose(0.0)
    n = len(LatticeCorridorPlanner().teacher_paths(input_from_privileged(privileged_view(sc, p0)), K))
    return dict(seed=seed, feasible=sc.feasible, layout=sc.layout, routes=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-499")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/l0/teacher_routes.json"))
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(count, [(s, a.K) for s in range(lo, hi + 1)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows))
    feas = [r for r in rows if r["feasible"]]
    for name, sub in [("all feasible", feas)] + [(l, [r for r in feas if r["layout"] == l]) for l in ("rows", "scatter")]:
        c = Counter(min(r["routes"], 3) for r in sub)
        n = len(sub)
        print(f"{name} ({n}): 0 routes {100*c[0]/n:.1f}%, 1 route {100*c[1]/n:.1f}%, "
              f"2 routes {100*c[2]/n:.1f}%, 3+ routes {100*c[3]/n:.1f}%, mean {sum(r['routes'] for r in sub)/n:.2f}")


if __name__ == "__main__":
    main()
