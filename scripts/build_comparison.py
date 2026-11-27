#!/usr/bin/env python
"""Per-level comparison of a closed-loop GoalFlow run with the frozen baselines ((b) at L0, (a) at L1-L4, (c) alongside).

    python scripts/build_comparison.py reports/lidar_demo/l3_goalflow/full_goalflow_l3_circles
"""
import csv
import sys
from pathlib import Path

import numpy as np

D = Path("reports/lidar_demo")
run = Path(sys.argv[1])
G = list(csv.DictReader(open(run / "results.csv")))
load = lambda p: list(csv.DictReader(open(D / p)))  # noqa: E731
base = {"L0": ("(b)", load("l0_rerun/b/results.csv")),
        **{l: ("(a)", load("l0_rerun_distdrift/a/results.csv")) for l in ("L1", "L2", "L3", "L4")}}
cc = {"L0": load("l0_rerun/c/results.csv"), **{l: load("l0_rerun_distdrift/c/results.csv") for l in ("L1", "L2", "L3", "L4")}}


def row(R, l):
    x = [r for r in R if r["noise"] == l and r["feasible"] == "True"]
    y = [r for r in R if r["noise"] == l]
    nf = [r for r in y if r["feasible"] != "True"]
    n = len(x)
    nb = sum(int(r["n_brake_plans"]) for r in y)
    nr = sum(int(r["n_replans"]) for r in y)
    return (f"{100 * sum(r['success'] == 'True' for r in x) / n:.1f} / {100 * sum(r['outcome'] == 'collision' for r in x) / n:.1f} / "
            f"{100 * sum(r['outcome'] in ('stopped', 'timeout') for r in x) / n:.1f} / {sum(r['outcome'] == 'exit' for r in x)}"
            f" | brakes {100 * nb / nr:.1f}% | correct brake {sum(r['correct_brake'] == 'True' for r in nf)}/{len(nf)}"
            f" | flips {np.mean([float(r['flips']) for r in y]):.2f} | ms {sum(float(r['plan_ms']) * int(r['n_replans']) for r in y) / nr:.0f}")


L = [f"# Closed-loop comparison: {run.name} vs the frozen baselines (seeds 0-499; success / collision / stalled % of feasible / exits)\n"]
for l in ("L0", "L1", "L2", "L3", "L4"):
    b, R = base[l]
    L.append(f"## {l}\n- GoalFlow ({run.name}): {row(G, l)}\n- baseline {b}: {row(R, l)}\n- (c) same-information: {row(cc[l], l)}")
    st = [r for r in G if r["noise"] == l and r["feasible"] == "True" and r["outcome"] in ("stopped", "timeout")]
    bt = [float(r["stall_best_true_clearance"]) for r in st if r.get("stall_best_true_clearance")]
    if bt:
        L.append(f"- GoalFlow stalls {len(st)}: best true clearance at the last brake scan: median {np.median(bt):.3f} m; "
                 f"truly blocked (<0) {sum(v < 0 for v in bt)}, clear by >= 0.12 m {sum(v >= 0.12 for v in bt)}")
    cz = [r["collision_cause"] for r in G if r["noise"] == l and r["outcome"] == "collision"]
    if cz:
        L.append(f"- GoalFlow collision causes: {dict((c, cz.count(c)) for c in set(cz))}")
(run / "COMPARISON.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
