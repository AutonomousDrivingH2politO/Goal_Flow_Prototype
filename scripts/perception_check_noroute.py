#!/usr/bin/env python
"""Perception check on 'no route (A*)' stalls: tracked vs true obstacles at the failing scans.

    python scripts/perception_check_noroute.py <planner name> <results.csv> <level> <out.json> [workers]

Replays every stalled episode whose fail causes include 'no route (A*)' with the named eval_corridor config.
At each scan whose brake reason is 'no route (A*)': tracked obstacles (the planner's own input) vs the true
obstacles in the ego frame within the LiDAR's view (x in [0, 25] m, |y| <= 8 m): counts, phantoms (tracks
with no true obstacle within 0.6 m), and the centre error of matched tracks.
"""
import csv
import json
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402


def one(job):
    name, seed, level = job
    from eval_corridor import make_planner
    from gfsem.sim.closed_loop import run_episode
    from gfsem.sim.geometry import to_frame
    from gfsem.sim.scene import generate_scene
    P = make_planner(name)
    inputs = []
    orig = P.plan

    def plan(obs, scene, tp, st):
        r = orig(obs, scene, tp, st)
        inputs.append(P.p.last_input)
        return r
    P.plan = plan
    sc = generate_scene(seed)
    r = run_episode(sc, P, level, record=True)
    rows = []
    for rec, inp in zip(r.records, inputs):
        if not (rec["brake"] and "no route" in str(rec["info"].get("reason", ""))):
            continue
        T = to_frame(sc.centres, rec["pose"]) if sc.n else np.zeros((0, 2))
        view = (T[:, 0] >= 0) & (T[:, 0] <= 25) & (np.abs(T[:, 1]) <= 8)
        Tv = T[view]
        C = np.asarray(inp.centres).reshape(-1, 2)
        Cv = C[(C[:, 0] >= 0) & (C[:, 0] <= 25) & (np.abs(C[:, 1]) <= 8)] if len(C) else C
        d = np.hypot(Cv[:, None, 0] - Tv[None, :, 0], Cv[:, None, 1] - Tv[None, :, 1]) if len(Cv) and len(Tv) else None
        nn = d.min(1) if d is not None else np.full(len(Cv), np.inf)
        rows.append(dict(t=rec["t"], n_tracks=int(len(Cv)), n_true=int(len(Tv)), phantoms=int((nn > 0.6).sum()),
                         centre_err_matched=float(np.median(nn[nn <= 0.6])) if (nn <= 0.6).any() else None,
                         missed_true=int((d.min(0) > 0.6).sum()) if d is not None else int(len(Tv))))
    return dict(seed=seed, outcome=r.outcome, scans=rows)


def main():
    name, res, level, outp = sys.argv[1:5]
    nw = int(sys.argv[5]) if len(sys.argv) > 5 else 7
    seeds = [int(r["seed"]) for r in csv.DictReader(open(res))
             if r["noise"] == level and r["outcome"] in ("stopped", "timeout") and "no route" in r["fail_cause"]]
    with get_context("spawn").Pool(nw) as pool:
        out = pool.map(one, [(name, s, level) for s in seeds])
    S = [x for e in out for x in e["scans"]]
    summ = {"planner": name, "level": level, "episodes": len(out), "failing_scans": len(S)}
    if S:
        summ.update(scans_with_phantoms=float(np.mean([x["phantoms"] > 0 for x in S])),
                    phantoms_per_scan=float(np.mean([x["phantoms"] for x in S])),
                    tracks_per_scan=float(np.mean([x["n_tracks"] for x in S])),
                    true_per_scan=float(np.mean([x["n_true"] for x in S])),
                    missed_true_per_scan=float(np.mean([x["missed_true"] for x in S])),
                    centre_err_matched_median=float(np.median([x["centre_err_matched"] for x in S
                                                               if x["centre_err_matched"] is not None])))
    json.dump({"summary": summ, "episodes": out}, open(outp, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
