#!/usr/bin/env python
"""Phase L0 evaluation: corridor planners closed-loop on fixed seeds and noise levels.

    python scripts/eval_corridor.py --seeds 0-499 --levels L0,L1,L2,L3,L4 --workers 8

Planners
  teacher         corridor-lattice, privileged (true scene, plan once)       -- L0 only
  observed        corridor-lattice, observed (noisy LiDAR clusters, event-triggered)
  funnel_priv     corridor-funnel (original design), privileged              -- L0 only
  funnel_obs      corridor-funnel (original design), observed

Writes <out>/results.csv, <out>/plan_times.csv and <out>/summary.md.
"""
from __future__ import annotations

import os

# One BLAS/OpenMP thread per process: workers are parallel already (set before numpy loads;
# spawned workers inherit the environment).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

PRIVILEGED = {"teacher", "funnel_priv"}
LABELS = {
    "teacher": "corridor-lattice, privileged (teacher)",
    "observed": "corridor-lattice, observed, pre-L3 (50e9ad9)",
    "observed_stubfix": "corridor-lattice, observed, pre-L3 + goal-line stub fix (a)",
    "observed_l3": "corridor-lattice, observed, stub fix + L3 (b)",
    "observed_l3_noedge": "corridor-lattice, observed, stub fix + L3 without edge memory (c; vs single-scan GoalFlow)",
    "observed_l3_notrack": "corridor-lattice, observed, stub fix + L3 without obstacle tracking (d)",
    "goalflow_l3": "GoalFlow, closed loop (NMS-K3 + safety, shadow off, 1 step, single-scan BEV; near-observation memory)",
    "goalflow_l3_nearobs": "GoalFlow, closed loop (= goalflow_l3; near-observation memory)",
    "goalflow_l3_visgate": "GoalFlow, closed loop (visibility-gate memory; lost the pre-set rule)",
    "goalflow_l3_exact": "GoalFlow, closed loop (goalflow_l3 + exact box-to-point safety filter, 0.12 m)",
    "goalflow_l3_exact_jitter": "GoalFlow, closed loop (exact filter + goal jitter +-0.3 m)",
    "goalflow_l3_exact_refine": "GoalFlow, closed loop (exact filter + 8 gradient steps when no candidate is safe)",
    "goalflow_l3_circles": "GoalFlow, closed loop (exact 0.12 m filter against fitted obstacle circles + circle memory)",
    "observed_l3_nocache": "(b) without the failed-search cache",
    "observed_l3_nohold": "(b) without hold",
    "observed_l3_nocache_nohold": "(b) without cache and hold",
    "observed_l3_nohyst": "(b) without kept-plan hysteresis and fallback",
    "funnel_priv": "corridor-funnel (original design), privileged",
    "funnel_obs": "corridor-funnel (original design), observed",
}


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


@lru_cache(maxsize=4)
def _scene(seed: int):
    from gfsem.sim.scene import generate_scene
    return generate_scene(seed)


_GF = None


def make_planner(name: str):
    global _GF
    gf_variants = {"goalflow_l3": dict(),                     # picked (pre-set rule): near-observation memory
                   "goalflow_l3_nearobs": dict(),             # alias of goalflow_l3 (name used in the smoke test)
                   "goalflow_l3_visgate": dict(visibility_gate=True, memory_obs_range=0.0),   # lost the rule
                   "goalflow_l3_exact": dict(exact_filter=True, safety_margin=0.12),        # exact box-to-point filter
                   "goalflow_l3_exact_jitter": dict(exact_filter=True, safety_margin=0.12, goal_jitter=0.3),
                   "goalflow_l3_exact_refine": dict(exact_filter=True, safety_margin=0.12, refine_steps=8),
                   "goalflow_l3_circles": dict(circle_filter=True, safety_margin=0.12)}   # circles, as baseline (a)
    if name in gf_variants:           # closed-loop GoalFlow, L3 configuration (one model per worker)
        if _GF is None:
            from gfsem.planners.goalflow_planner import GoalFlowPlanner
            _GF = GoalFlowPlanner(**gf_variants[name])
            _GF.name = name
        return _GF
    from gfsem.planners.corridor import CorridorConfig, CorridorPlanner
    from gfsem.planners.corridor_lattice import LatticeCorridorPlanner, l3_config
    from gfsem.sim.closed_loop import CorridorObserved, CorridorPrivileged
    if name == "teacher":
        return CorridorPrivileged(LatticeCorridorPlanner(mode="privileged"), name)
    if name == "observed":            # pre-L3 planner (reproduces 50e9ad9)
        return CorridorObserved(LatticeCorridorPlanner(mode="observed"), name)
    if name == "observed_stubfix":    # config (a): pre-L3 + goal-line stub fix (A6)
        return CorridorObserved(LatticeCorridorPlanner(mode="observed", cfg=CorridorConfig(goal_cut_guard=1.0)), name)
    if name == "observed_l3":         # config (b): stub fix + approved L3 set, no time budget
        return CorridorObserved(LatticeCorridorPlanner(mode="observed", cfg=CorridorConfig(goal_cut_guard=1.0),
                                                       lcfg=l3_config()), name)
    if name == "observed_l3_noedge":  # config (c): (b) without edge memory (fair vs single-scan GoalFlow BEV)
        return CorridorObserved(LatticeCorridorPlanner(mode="observed", cfg=CorridorConfig(goal_cut_guard=1.0),
                                                       lcfg=l3_config(edge_memory=False)), name)
    abl = {"observed_l3_nocache": dict(fail_cache_scans=0), "observed_l3_nohold": dict(hold_confirm=1),
           "observed_l3_nocache_nohold": dict(fail_cache_scans=0, hold_confirm=1),
           "observed_l3_nohyst": dict(keep_hysteresis=False, fallback=False)}
    if name in abl:                   # stall ablations of (b)
        return CorridorObserved(LatticeCorridorPlanner(mode="observed", cfg=CorridorConfig(goal_cut_guard=1.0),
                                                       lcfg=l3_config(**abl[name])), name)
    if name == "observed_l3_notrack": # config (d): (b) without obstacle tracking
        return CorridorObserved(LatticeCorridorPlanner(mode="observed", cfg=CorridorConfig(goal_cut_guard=1.0),
                                                       lcfg=l3_config(tracking=False)), name)
    if name == "funnel_priv":
        return CorridorPrivileged(CorridorPlanner(mode="privileged"), name)
    if name == "funnel_obs":
        return CorridorObserved(CorridorPlanner(mode="observed"), name)
    raise ValueError(name)


def _stall_diag(lb: dict, scene, veh) -> dict:
    """At the last "no safe candidate" scan: best true clearance among the candidates (Juno's box vs the true
    discs); the filter element (circle / point) that flags the truly best candidate: its offset from the nearest true
    obstacle (centre distance for circles) and that candidate's true clearance."""
    import torch
    from gfsem.models.goalflow import observed_collision_exact
    from gfsem.sim.geometry import to_frame
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_selection import path_clearance
    T = to_frame(scene.centres, lb["true_pose"])
    O = torch.zeros(1, scene.n, 3)
    O[0, :, :2] = torch.from_numpy(T)
    O[0, :, 2] = torch.from_numpy(scene.radii)
    S = torch.from_numpy(lb["S"]).float()
    clr = path_clearance(S, O.expand(len(S), -1, -1), torch.full((len(S),), scene.n), veh).numpy()
    i = int(clr.argmax())
    out = {"stall_best_true_clearance": round(float(clr[i]), 3)}
    F = lb["filt"]
    for j in range(len(F)):
        if observed_collision_exact(S[i:i + 1][None], [torch.from_numpy(F[j:j + 1]).float()], 0.12)[0, 0]:
            off = float(np.hypot(*(T - F[j, :2]).T).min()) if scene.n else float("nan")
            out.update(stall_blocker_offset=round(off, 3), stall_blocker_true_clearance=round(float(clr[i]), 3))
            break
    return out


def run_one(job: tuple) -> dict:
    from gfsem.sim.closed_loop import run_episode
    from gfsem.sim.config import LidarConfig, SimConfig
    name, seed, level = job[:3]
    mount = job[3] if len(job) > 3 else "front_axle"
    scene = _scene(seed)
    P = make_planner(name)
    gf = name.startswith("goalflow")
    simcfg = SimConfig(lidar=LidarConfig.with_mount(mount))
    r = run_episode(scene, P, level, simcfg, record=gf)
    row = r.row()
    row["mount"] = mount
    inner = getattr(P, "p", P)
    stats = getattr(inner, "stats", [])
    if gf:
        from gfsem.planners.goalflow_planner import collision_cause
        from gfsem.sim.config import SimConfig
        cfg = simcfg
        row["collision_cause"] = (collision_cause(inner, scene, r.trajectory[-1, :3], cfg.vehicle, cfg.lidar)
                                  if r.outcome == "collision" else "")
        row["stall_best_true_clearance"] = ""
        row["stall_blocker_offset"] = ""
        row["stall_blocker_true_clearance"] = ""
        if r.outcome in ("stopped", "timeout") and getattr(inner, "last_brake", None):
            row.update(_stall_diag(inner.last_brake, scene, cfg.vehicle))
    row["layout"] = scene.layout
    row["fail_cause"] = ";".join(f"{k}" for k in getattr(inner, "fail_causes", {}) or [])
    row["triggers"] = dict(Counter(t for t, _ in stats))
    row["plan_times"] = stats
    row["plan_expansions"] = list(getattr(inner, "expansions", []))
    ev = getattr(inner, "events", {}) or {}
    row["guard_refusals"] = int(ev.get("guard_refusal", 0))       # goal-cut guard: brake instead of a bad line
    row["no_edges_brakes"] = int(ev.get("no_edges", 0))
    row["cache_skipped_scans"] = int(ev.get("cache_hit", 0))      # replans skipped by the failed-search cache
    row["hold_scans"] = int(ev.get("hold", 0))                    # plans found but held (waiting for confirmation)
    row["held_scans"] = int(ev.get("held_scans", 0))              # scans spent braked / held since a brake
    row["events"] = " ".join(f"{k}:{v}" for k, v in sorted(ev.items()))
    row["kappa_levels"] = " ".join(f"{k:.2f}" for k in getattr(inner, "kappa_log", []))
    return row


def pct(x) -> str:
    return f"{100 * np.mean(x):.1f}" if len(x) else "-"


def summarise(rows: list[dict], out: Path, seeds: list[int], levels: list[str], elapsed: float):
    by = defaultdict(list)
    for r in rows:
        by[(r["planner"], r["noise"])].append(r)
    L = []
    L.append("# Phase L0: corridor planners in closed loop\n")
    L.append(f"Seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} scenes), generated by `generate_scene(seed)`; "
             f"wall time {elapsed / 60:.1f} min. Produced by `scripts/eval_corridor.py`.\n")
    n_feas = sum(_scene(s).feasible for s in seeds) if len(seeds) <= 2000 else None
    L.append(f"Oracle-feasible scenes: {n_feas} of {len(seeds)}; the rest have no route "
             "(correct behaviour: stop without collision).\n")
    L.append("Feasible-scene outcomes (% of feasible scenes) and no-route braking (% of no-route scenes).\n")
    L.append("| planner | noise | success | collision | track exit | stalled | correct brake (no-route) | "
             "door flips / run | min clearance p5 [m] | peak lat. acc. p95 [m/s²] | time to goal [s] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in LABELS:
        for lvl in levels:
            rs = by.get((name, lvl))
            if not rs:
                continue
            f = [r for r in rs if r["feasible"]]
            nf = [r for r in rs if not r["feasible"]]
            succ = [r for r in f if r["success"]]
            L.append(
                f"| {LABELS[name]} | {lvl} | {pct([r['success'] for r in f])} "
                f"| {pct([r['outcome'] == 'collision' for r in f])} | {pct([r['outcome'] == 'exit' for r in f])} "
                f"| {pct([r['outcome'] in ('stopped', 'timeout') for r in f])} "
                f"| {pct([r['correct_brake'] for r in nf])} ({len(nf)}) "
                f"| {np.mean([r['flips'] for r in rs]):.2f} "
                f"| {np.percentile([r['min_clearance'] for r in rs if np.isfinite(r['min_clearance'])], 5):.2f} "
                f"| {np.percentile([r['peak_lat_acc'] for r in rs], 95):.2f} "
                f"| {np.mean([r['time'] for r in succ]) if succ else float('nan'):.1f} |")
    # planning time
    L.append("\n## Planning time per call (this machine, single core, Python; not Orin numbers)\n")
    L.append("| planner | noise | calls / run | mean [ms] | p95 [ms] | max [ms] | non-keep calls / run | "
             "non-keep mean [ms] | non-keep p95 [ms] | triggers |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in LABELS:
        for lvl in levels:
            rs = by.get((name, lvl))
            if not rs:
                continue
            allt = [m for r in rs for _, m in r["plan_times"]]
            if not allt:   # funnel: no per-call stats, use the episode mean
                allt = [r["plan_ms"] for r in rs]
                L.append(f"| {LABELS[name]} | {lvl} | {np.mean([r['n_replans'] for r in rs]):.0f} "
                         f"| {np.mean(allt):.1f} (episode means) | - | - | - | - | - | every call plans |")
                continue
            nk = [m for r in rs for t, m in r["plan_times"] if t != "keep"]
            trig = Counter()
            for r in rs:
                trig.update(r["triggers"])
            tr = ", ".join(f"{k} {v / len(rs):.1f}" for k, v in sorted(trig.items()))
            L.append(f"| {LABELS[name]} | {lvl} | {len(allt) / len(rs):.0f} | {np.mean(allt):.1f} "
                     f"| {np.percentile(allt, 95):.1f} | {np.max(allt):.0f} | {len(nk) / len(rs):.1f} "
                     f"| {np.mean(nk) if nk else 0:.1f} | {np.percentile(nk, 95) if nk else 0:.1f} | {tr} /run |")
    # goal-cut guard refusals (observed configs): brakes because no observed edge reaches past the car's front
    L.append("\n## Goal-cut guard refusals and no-edge brakes (observed configs)\n")
    L.append("| planner | noise | episodes with a refusal | refusal brakes / run | no-edge brakes / run |")
    L.append("|---|---|---|---|---|")
    for name in LABELS:
        for lvl in levels:
            rs = by.get((name, lvl))
            if not rs or "guard_refusals" not in rs[0]:
                continue
            L.append(f"| {LABELS[name]} | {lvl} | {sum(r['guard_refusals'] > 0 for r in rs)} / {len(rs)} "
                     f"| {np.mean([r['guard_refusals'] for r in rs]):.2f} | {np.mean([r['no_edges_brakes'] for r in rs]):.2f} |")
    # teacher failures
    for name in ("teacher", "funnel_priv"):
        rs = by.get((name, "L0"), [])
        fails = [r for r in rs if r["feasible"] and not r["success"]]
        if rs:
            f = [r for r in rs if r["feasible"]]
            L.append(f"\n## {LABELS[name]}: {len(f) - len(fails)}/{len(f)} feasible scenes succeed "
                     f"({pct([r['success'] for r in f])}%)\n")
            lv = Counter(r["kappa_levels"].split()[0] for r in rs if r.get("kappa_levels"))
            if lv:
                L.append("Curvature level the first plan needed (fraction of kappa_max): "
                         + ", ".join(f"{k}: {v}" for k, v in sorted(lv.items())) + "\n")
            for r in fails:
                L.append(f"- seed {r['seed']} ({r['layout']}): {r['outcome']} at {r['time']:.1f} s"
                         + (f"; {r['fail_cause']}" if r["fail_cause"] else ""))
    (out / "summary.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-499")
    ap.add_argument("--levels", default="L0,L1,L2,L3,L4")
    ap.add_argument("--planners", default="teacher,observed,funnel_priv,funnel_obs")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/l0"))
    ap.add_argument("--l3", action="store_true", help="run 'observed' as config (b): stub fix + approved L3 set")
    ap.add_argument("--mount", default="front_axle", help="LiDAR mount preset (gfsem.sim.config.LIDAR_MOUNTS)")
    a = ap.parse_args()
    if a.l3:
        a.planners = ",".join("observed_l3" if p == "observed" else p for p in a.planners.split(","))
    seeds = parse_seeds(a.seeds)
    levels = a.levels.split(",")
    jobs = []
    for name in a.planners.split(","):
        for lvl in (["L0"] if name in PRIVILEGED else levels):
            jobs += [(name, s, lvl, a.mount) for s in seeds]
    a.out.mkdir(parents=True, exist_ok=True)
    # one run per output folder (house rule): refuse if a live process holds the lock; replace a stale one
    import os
    lock = a.out / "RUNNING.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().split()[0])
            os.kill(pid, 0)
            raise SystemExit(f"{a.out} is in use by pid {pid} (RUNNING.lock); not starting a second run")
        except (ValueError, ProcessLookupError, PermissionError, IndexError):
            pass                                   # stale lock
    lock.write_text(f"{os.getpid()} {time.ctime()} {' '.join(sys.argv)}\n")
    try:
        _main_body(a, jobs, seeds, levels)
    finally:
        lock.unlink(missing_ok=True)


def _main_body(a, jobs, seeds, levels):
    t0 = time.time()
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} episodes, {time.time() - t0:.0f}s", flush=True)
    elapsed = time.time() - t0
    rows.sort(key=lambda r: (r["planner"], r["noise"], r["seed"]))
    keys = [k for k in rows[0] if k not in ("plan_times", "plan_expansions", "triggers")] + ["triggers"]
    with open(a.out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "triggers": " ".join(f"{k}:{v}" for k, v in r["triggers"].items())})
    with open(a.out / "plan_times.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["planner", "noise", "seed", "trigger", "ms", "expansions"])
        for r in rows:
            ex = r.get("plan_expansions") or [""] * len(r["plan_times"])
            for (t, m), e in zip(r["plan_times"], ex):
                w.writerow([r["planner"], r["noise"], r["seed"], t, f"{m:.2f}", e])
    summarise(rows, a.out, seeds, levels, elapsed)
    print((a.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
