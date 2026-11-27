#!/usr/bin/env python
"""Single-worker per-component planning time of a GoalFlow closed-loop config (queue item 4).

    python scripts/profile_latency.py [--planner goalflow_l3_combined] [--episodes 12] [--out reports/lidar_demo/l3_goalflow/latency]

One process, nothing else running. Episodes: seeds 0.. alternating L0 / L2 / L4. Each component is wrapped with a
timer and an MPS synchronise (so GPU time lands on the component that queued it; the synchronise adds a little
overhead). Per replan: rasterise, circle filter + memory, perceive (encoder), goal scorer, goal selection (NMS), flow
sampling, rollout, safety filter, edge critic (incl. edge densification), other (scoring, shadow, bookkeeping).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_corridor as E  # noqa: E402
import gfsem.models.goalflow as G  # noqa: E402
import gfsem.planners.goalflow_planner as GP  # noqa: E402
from gfsem.sim.closed_loop import run_episode  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402

T = defaultdict(float)
SYNC = [lambda: None]


def timed(name, fn):
    def w(*a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        SYNC[0]()
        T[name] += time.perf_counter() - t0
        return r
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="goalflow_l3_combined")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/l3_goalflow/latency"))
    a = ap.parse_args()
    P = E.make_planner(a.planner)
    if P.dev.type == "mps":
        SYNC[0] = torch.mps.synchronize
    m = P.model
    m.perceive, m.sample = timed("perceive", m.perceive), timed("sample", m.sample)
    m.scorer.forward = timed("scorer", m.scorer.forward)
    G.diverse_goals = timed("goal_select", G.diverse_goals)
    G.rollout = timed("rollout", G.rollout)
    G.observed_collision_exact = timed("safety_filter", G.observed_collision_exact)
    G.observed_min_distance = timed("edge_critic", G.observed_min_distance)
    GP.rasterize = timed("rasterize", GP.rasterize)
    GP._densify_edges = timed("edge_critic", GP._densify_edges)
    P._circles = timed("circles_memory", P._circles)
    run_episode(generate_scene(999), P, "L2")                    # warm-up (not counted)
    T.clear()
    total, n = 0.0, 0
    lv = ("L0", "L2", "L4")
    for i in range(a.episodes):
        P.stats = []
        run_episode(generate_scene(i), P, lv[i % 3])
        total += sum(ms for _, ms in P.stats) / 1000
        n += len(P.stats)
    comp = {k: 1000 * v / n for k, v in T.items()}
    comp["other"] = 1000 * total / n - sum(comp.values())
    res = dict(planner=a.planner, replans=n, episodes=a.episodes, device=str(P.dev), total_ms=1000 * total / n,
               components_ms=dict(sorted(comp.items(), key=lambda kv: -kv[1])))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.planner}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
