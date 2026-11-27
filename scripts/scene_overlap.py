#!/usr/bin/env python
"""Scene overlap: is any dev scene (seeds 0-499) a near-copy of a training scene (100000+) by obstacle layout?

    python scripts/scene_overlap.py [--out reports/lidar_demo/scene_overlap]

Layout = obstacle centres in track coordinates (s from the field start, d lateral) with their class. Candidates by a
1 m occupancy-grid Jaccard similarity over all 15,000 training scenes, then for the 10 best per dev scene the exact
symmetric chamfer distance (mean nearest-obstacle distance, both ways, in m). Near-copy: same obstacle count, chamfer
<= 0.25 m and every obstacle matched to one of the same class. Sanity check: a training scene against the training set
finds itself at 0.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from gfsem.data.scene_index import SceneIndex  # noqa: E402
from gfsem.sim.config import SceneConfig  # noqa: E402
from gfsem.sim.scene import generate_scene  # noqa: E402

CFG = SceneConfig.from_yaml()


def layout(track, centres, classes, s_goal):
    s0 = s_goal - CFG.field_length - CFG.goal_margin
    s, d = track.frenet(np.asarray(centres, float))
    return np.c_[s - s0, d], np.asarray(classes)


def grid(L, ns, nd=12):
    g = np.zeros((ns, nd), np.float32)
    i = np.clip(L[:, 0].astype(int), 0, ns - 1)
    j = np.clip((L[:, 1] + nd / 2).astype(int), 0, nd - 1)
    g[i, j] = 1
    return g.ravel()


def chamfer(A, B):
    if not len(A) or not len(B):
        return float("inf"), False
    D = np.hypot(A[0][:, None, 0] - B[0][None, :, 0], A[0][:, None, 1] - B[0][None, :, 1])
    ia, ib = D.argmin(1), D.argmin(0)
    ch = 0.5 * (D.min(1).mean() + D.min(0).mean())
    cls_ok = bool((A[1] == B[1][ia]).all() and (B[1] == A[1][ib]).all())
    return float(ch), cls_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/scene_overlap"))
    ap.add_argument("--n", type=int, default=500, help="dev scenes 0..n-1")
    a = ap.parse_args()
    si = SceneIndex([Path("data/sim/scenes_train.npz")])
    z = np.load("data/sim/scenes_train.npz")
    tr_seeds = [int(s) for s in z["seed"]]
    ns = int(np.ceil(CFG.field_length)) + 2
    TR = []
    for s in tr_seeds:
        sc = si.get(s)
        TR.append(layout(sc.track, sc.centres, sc.classes, sc.s_goal))
    Gt = np.stack([grid(L, ns) for L, _ in TR])
    nt = Gt.sum(1)

    def nearest(Lfull, k=10):
        g = grid(Lfull[0], ns)
        inter = Gt @ g
        jac = inter / np.maximum(nt + g.sum() - inter, 1)
        best = []
        for i in np.argsort(-jac)[:k]:
            ch, ok = chamfer(Lfull, TR[i])
            best.append((ch, ok, int(i)))
        return min(best)

    self_ch = nearest(TR[0])[0]                    # sanity: a training scene finds itself (chamfer 0)
    rows = []
    for seed in range(a.n):
        sc = generate_scene(seed)
        Lfull = layout(sc.track, sc.centres, sc.classes, sc.s_goal)
        L = Lfull[0]
        ch, ok, i = nearest(Lfull)
        near_copy = bool(ch <= 0.25 and ok and len(L) == len(TR[i][0]))
        rows.append(dict(seed=seed, n=len(L), nearest_train_seed=tr_seeds[i], n_train=len(TR[i][0]),
                         chamfer=round(ch, 3), classes_match=ok, near_copy=near_copy))
    ch = np.array([r["chamfer"] for r in rows])
    summ = dict(dev_scenes=len(rows), train_scenes=len(TR), sanity_self_chamfer=self_ch,
                near_copies=[r for r in rows if r["near_copy"]],
                chamfer_min=float(ch.min()), chamfer_p5=float(np.percentile(ch, 5)), chamfer_median=float(np.median(ch)),
                closest=sorted(rows, key=lambda r: r["chamfer"])[:5])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "scene_overlap.json").write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
