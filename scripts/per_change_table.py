#!/usr/bin/env python
"""Per-change table (DEVIATIONS 2026-11-27): each change alone vs the full goalflow_l3_circles rows for the same seeds
(0-99, L0/L2/L4). Pooled acceptance: collisions not above the reference AND stalls below it (or success above it).
Flag: an accepted change that lowers success by > 2 points or adds a collision at any single level."""
import csv
from pathlib import Path

import numpy as np

D = Path("reports/lidar_demo/l3_goalflow/per_change")
REF = Path("reports/lidar_demo/l3_goalflow/full_goalflow_l3_circles/results.csv")
LV = ("L0", "L2", "L4")
CH = [("(a) 2-scan entry", "goalflow_l3_circles_entry2"), ("(b) lambda3 = 1.29", "goalflow_l3_circles_lambda3"),
      ("(c) edge critic 1.0", "goalflow_l3_circles_edge"), ("(d) refinement vs circles", "goalflow_l3_circles_refine"),
      ("(a2) memory fix, first run (guard bug)", "goalflow_l3_circles_memfix_guardbug"),
      ("(a2) memory fix", "goalflow_l3_circles_memfix")]


def stats(rows):
    out = {}
    for l in LV + ("all",):
        y = [r for r in rows if l == "all" or r["noise"] == l]
        x = [r for r in y if r["feasible"] == "True"]
        out[l] = dict(n=len(x), succ=sum(r["success"] == "True" for r in x), coll=sum(r["outcome"] == "collision" for r in x),
                      stall=sum(r["outcome"] in ("stopped", "timeout") for r in x), exits=sum(r["outcome"] == "exit" for r in x),
                      flips=float(np.mean([float(r["flips"]) for r in y])))
    return out


ref = stats([r for r in csv.DictReader(open(REF)) if r["noise"] in LV and int(r["seed"]) < 100])
cell = lambda s: f"{100 * s['succ'] / s['n']:.1f} / {s['coll']} / {s['stall']} / {s['exits']} / {s['flips']:.2f}"  # noqa: E731
L = ["Cells: success % / collisions / stalls / exits / door flips per run (seeds 0-99, feasible scenes).", "",
     "| change | L0 | L2 | L4 | pooled | accepted | flag |", "|---|---|---|---|---|---|---|",
     f"| reference (full run, same seeds) | {cell(ref['L0'])} | {cell(ref['L2'])} | {cell(ref['L4'])} | {cell(ref['all'])} | - | - |"]
for lab, name in CH:
    f = D / name / "results.csv"
    if not f.exists():
        continue
    s = stats(list(csv.DictReader(open(f))))
    a, r0 = s["all"], ref["all"]
    acc = a["coll"] <= r0["coll"] and (a["stall"] < r0["stall"] or a["succ"] > r0["succ"])
    flags = [f"{l}: success {100 * (s[l]['succ'] / s[l]['n'] - ref[l]['succ'] / ref[l]['n']):+.1f}" for l in LV
             if 100 * (s[l]["succ"] / s[l]["n"] - ref[l]["succ"] / ref[l]["n"]) < -2]
    flags += [f"{l}: collisions {ref[l]['coll']} -> {s[l]['coll']}" for l in LV if s[l]["coll"] > ref[l]["coll"]]
    L.append(f"| {lab} | {cell(s['L0'])} | {cell(s['L2'])} | {cell(s['L4'])} | {cell(a)} | {'yes' if acc else 'no'} | "
             f"{'; '.join(flags) if (acc and flags) else '-'} |")
(D / "TABLE.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
