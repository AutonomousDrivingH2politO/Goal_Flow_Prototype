#!/usr/bin/env python
"""FINAL_COMPARISON.md: the frozen configs on the fresh test seeds 400000-400499, L0-L4 (DEVIATIONS 2026-11-27).

    python scripts/build_final_comparison.py [reports/lidar_demo/final_test]

Per level: goalflow_final, (a)+filter, (a), (b), (c): success / collision / stalled % of feasible scenes, exits, brake
rate, door flips per run, correct brakes on infeasible scenes. The per-level opponent is the frozen one ((b) at L0, (a)
at L1-L4; (c) alongside); nothing is re-picked. Paired exact McNemar on success per level: GoalFlow vs the opponent,
vs (c), vs (a)+filter; (a)+filter vs (a) and vs (b).
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/lidar_demo/final_test")
RUNS = {"GoalFlow final": "goalflow_final", "(a)+filter": "observed_stubfix_filter", "(a)": "a", "(b)": "b", "(c)": "c"}
R = {k: list(csv.DictReader(open(D / v / "results.csv"))) for k, v in RUNS.items()}
LV = ("L0", "L1", "L2", "L3", "L4")
OPP = {"L0": "(b)", "L1": "(a)", "L2": "(a)", "L3": "(a)", "L4": "(a)"}


def row(rows, l):
    y = [r for r in rows if r["noise"] == l]
    x = [r for r in y if r["feasible"] == "True"]
    nf = [r for r in y if r["feasible"] != "True"]
    n = len(x)
    if not n:
        return "- | - | - | - | - | - | -"
    nb, nr = sum(int(r["n_brake_plans"]) for r in y), sum(int(r["n_replans"]) for r in y)
    return (f"{100 * sum(r['success'] == 'True' for r in x) / n:.1f} | {100 * sum(r['outcome'] == 'collision' for r in x) / n:.1f} | "
            f"{100 * sum(r['outcome'] in ('stopped', 'timeout') for r in x) / n:.1f} | {sum(r['outcome'] == 'exit' for r in x)} | "
            f"{100 * nb / nr:.1f}% | {np.mean([float(r['flips']) for r in y]):.2f} | "
            f"{sum(r['correct_brake'] == 'True' for r in nf)}/{len(nf)}")


def mcnemar(A, B, l):
    a = {r["seed"]: r["success"] == "True" for r in A if r["noise"] == l and r["feasible"] == "True"}
    b = {r["seed"]: r["success"] == "True" for r in B if r["noise"] == l and r["feasible"] == "True"}
    ks = a.keys() & b.keys()
    n01, n10 = sum(a[k] and not b[k] for k in ks), sum(b[k] and not a[k] for k in ks)
    n = n01 + n10
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return f"{n01} vs {n10}, p = {p:.3g}"


L = ["# Final comparison on the fresh test seeds 400000-400499 (frozen configs; DEVIATIONS 2026-11-27)", "",
     "Front-axle mount, constant 3 m/s, distance-proportional drift. Per-level opponent as frozen: (b) at L0, (a) at "
     "L1-L4; (c) alongside. No change of any kind after these numbers.", ""]
for l in LV:
    L += [f"## {l} (opponent {OPP[l]})", "", "| planner | success % | collision % | stalled % | exits | brake rate | "
          "door flips / run | correct brakes |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {row(v, l)} |" for k, v in R.items()]
    G, F = R["GoalFlow final"], R["(a)+filter"]
    L += ["", "McNemar on success (only-first vs only-second successes, exact two-sided p):",
          f"- GoalFlow final vs opponent {OPP[l]}: {mcnemar(G, R[OPP[l]], l)}",
          f"- GoalFlow final vs (c): {mcnemar(G, R['(c)'], l)}",
          f"- GoalFlow final vs (a)+filter: {mcnemar(G, F, l)}",
          f"- (a)+filter vs (a): {mcnemar(F, R['(a)'], l)}",
          f"- (a)+filter vs (b): {mcnemar(F, R['(b)'], l)}", ""]
(D / "FINAL_COMPARISON.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
