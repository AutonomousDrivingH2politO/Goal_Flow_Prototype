#!/usr/bin/env python
"""False-alarm stalls of the full goalflow_l3_circles run: what blocks a truly clear candidate? (replays only)

    python scripts/diagnose_false_alarms.py [--workers 5] [--out reports/lidar_demo/l3_goalflow/false_alarms]

Replays (deterministic) every full-run stall whose best candidate was truly clear by >= 0.12 m at the last brake scan.
At that scan, every filter circle that alone flags a truly clear candidate is a blocking circle, classified in this
order: phantom / clutter (no true obstacle surface within 0.3 m of its centre); remembered near-field circle (from the
circle memory, not this scan); class-flipped (its radius is the other class's); real obstacle. For real and remembered
circles: centre error vs the true centre, single scan (the circle as used) and the 3-scan mean (mean of the associated
cluster centres over the last 3 scans in the odom frame, gate 0.3 m). Counts per level.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from collections import Counter, deque  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

FULL = Path("reports/lidar_demo/l3_goalflow/full_goalflow_l3_circles/results.csv")
_P = None


def _planner():
    global _P
    if _P is None:
        import torch
        from gfsem.planners.corridor import cluster_obstacles
        from gfsem.planners.goalflow_planner import GoalFlowPlanner
        from gfsem.sim.geometry import from_frame

        class Rec(GoalFlowPlanner):
            def reset(self):
                super().reset()
                self.hist = deque(maxlen=3)            # last 3 scans' cluster centres (odom frame)
                self.snap = None

            @torch.no_grad()
            def plan(self, obs, scene, true_pose, state):
                res = super().plan(obs, scene, true_pose, state)
                C, _, _ = cluster_obstacles(obs, self._ccfg, self._scfg)
                C = np.asarray(C, float).reshape(-1, 2)
                pose_b = np.asarray(obs.pose_odom, float)
                self.hist.append(from_frame(C, pose_b) if len(C) else np.zeros((0, 2)))
                if res.brake:
                    self.snap = dict(lb=dict(self.last_brake), n_cur=self.n_cur, pose_b=pose_b,
                                     hist=[h.copy() for h in self.hist])
                return res

        _P = Rec(circle_filter=True, safety_margin=0.12)
    return _P


def one(job):
    import torch
    from eval_selection import path_clearance
    from gfsem.models.goalflow import observed_collision_exact
    from gfsem.sim.closed_loop import run_episode
    from gfsem.sim.config import BOLLARD, SceneConfig, SimConfig
    from gfsem.sim.geometry import from_frame, to_frame
    from gfsem.sim.scene import generate_scene
    seed, level, outcome_ref = job
    P = _planner()
    sc = generate_scene(seed)
    r = run_episode(sc, P, level, record=True)
    out = dict(seed=seed, level=level, outcome=r.outcome, reproduced=r.outcome == outcome_ref, blocking=[])
    s = P.snap
    if s is None:
        return out
    veh, scfg = SimConfig().vehicle, SceneConfig.from_yaml()
    lb = s["lb"]
    T = to_frame(sc.centres, lb["true_pose"])
    O = torch.zeros(1, sc.n, 3)
    O[0, :, :2] = torch.from_numpy(T)
    O[0, :, 2] = torch.from_numpy(sc.radii)
    S = torch.from_numpy(lb["S"]).float()
    clr = path_clearance(S, O.expand(len(S), -1, -1), torch.full((len(S),), sc.n), veh).numpy()
    ok = clr >= 0.12
    out["best_true_clearance"] = float(clr.max())
    if not ok.any():
        return out
    F = lb["filt"].reshape(-1, 3) if lb["filt"].size else np.zeros((0, 3))
    for j in range(len(F)):
        if not observed_collision_exact(S[ok][None], [torch.from_numpy(F[j:j + 1]).float()], 0.12)[0].any():
            continue
        dsurf = np.hypot(*(T - F[j, :2]).T) - sc.radii
        o = int(dsurf.argmin())
        if dsurf[o] > 0.3:
            cat = "phantom"
        elif j >= s["n_cur"]:
            cat = "remembered"
        elif (abs(F[j, 2] - scfg.bollard_radius) < 1e-6) != (sc.classes[o] == BOLLARD):
            cat = "class_flipped"
        else:
            cat = "real"
        b = dict(category=cat, surface_dist=round(float(dsurf[o]), 3))
        if cat in ("real", "remembered"):
            b["err_single"] = round(float(np.hypot(*(F[j, :2] - T[o]))), 3)
            cw = from_frame(F[j:j + 1, :2], s["pose_b"])[0]
            assoc = [h[np.hypot(*(h - cw).T).argmin()] for h in s["hist"]
                     if len(h) and np.hypot(*(h - cw).T).min() <= 0.3]
            if assoc:
                m = to_frame(np.mean(assoc, 0)[None], s["pose_b"])[0]
                b["err_3scan"] = round(float(np.hypot(*(m - T[o]))), 3)
                b["n_assoc"] = len(assoc)
        out["blocking"].append(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/l3_goalflow/false_alarms"))
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(FULL)) if r["feasible"] == "True" and r["outcome"] in ("stopped", "timeout")
            and r["stall_best_true_clearance"] and float(r["stall_best_true_clearance"]) >= 0.12]
    jobs = [(int(r["seed"]), r["noise"], r["outcome"]) for r in rows]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} false-alarm stall episodes", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = list(pool.imap_unordered(one, jobs))
    res.sort(key=lambda x: (x["level"], x["seed"]))
    (a.out / "false_alarms.json").write_text(json.dumps(res, indent=1))
    L = ["| level | episodes (reproduced) | blocking circles | phantom | remembered | class-flipped | real | "
         "real: centre error single / 3-scan (median m) | remembered: single / 3-scan |", "|---" * 9 + "|"]
    for lvl in sorted({x["level"] for x in res}):
        R = [x for x in res if x["level"] == lvl]
        B = [b for x in R for b in x["blocking"]]
        c = Counter(b["category"] for b in B)

        def med(cat, k):
            v = [b[k] for b in B if b["category"] == cat and k in b]
            return f"{np.median(v):.3f}" if v else "-"
        L.append(f"| {lvl} | {len(R)} ({sum(x['reproduced'] for x in R)}) | {len(B)} | {c['phantom']} | {c['remembered']} | "
                 f"{c['class_flipped']} | {c['real']} | {med('real', 'err_single')} / {med('real', 'err_3scan')} | "
                 f"{med('remembered', 'err_single')} / {med('remembered', 'err_3scan')} |")
    (a.out / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
