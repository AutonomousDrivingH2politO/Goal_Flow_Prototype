#!/usr/bin/env python
"""Goal-scorer diagnostics on the whole validation split (frozen seed-0 checkpoint, scorer only).

    python scripts/eval_goal_scorer.py --run runs/l2_full_seed0

Distinct-route rule (fixed before looking at any result, 2026-11-25): walk a frame's teacher routes in
stored order (driven first); a route is *distinct* if its endpoint is >= 1.5 m (x, y) from the endpoint
of every distinct route kept so far. Every metric is reported for all routes and for distinct routes.

Reported:
  * vocabulary coverage: distance (x, y) from every teacher-route endpoint to the nearest vocabulary
    goal: median, p95, share > 0.5 m; driven vs non-driven routes; and the share outside the
    vocabulary's lateral bounds
  * cover@k (k = 1, 2, 3, 5, 10): the top-k goals by final score include a goal within 1.0 m of every
    route endpoint; plain top-k, and with suppression (a goal is skipped if it lies within 1.5 m of a
    goal already taken)
  * rank of the best goal near each route (within 1.0 m) in the plain and suppressed orders
  * l_dis against each route's soft target (T = 0.5 m), next to its floor (the target's entropy), and
    against the mixture target (mean of the distinct routes' soft targets), and (added 2026-11-25 after the
    first run, for later runs) against the mixture over all stored modes, which is what training drew
  Caveat: cover@k uses a 1.0 m radius per route endpoint with 1.5 m distinct-route separation, so one goal
  halfway between two routes can cover both; the selection evaluation adds a one-to-one version.
Validation only.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from gfsem.data.sim_torch import MemmapSplit, SimFrames  # noqa: E402
from gfsem.models.common import pick_device  # noqa: E402
from gfsem.models.goal_scorer import goal_metric  # noqa: E402
from gfsem.train import load  # noqa: E402

DISTINCT_SEP = 1.5      # m between endpoints for two routes to count as distinct
COVER_R = 1.0           # m: a goal "covers" a route endpoint
SUPPRESS_R = 1.5        # m: suppression radius
KS = (1, 2, 3, 5, 10)
T_DIS = 0.5


def distinct_mask(ends: np.ndarray, n: int, sep: float = DISTINCT_SEP) -> np.ndarray:
    keep = np.zeros(len(ends), bool)
    kept = []
    for j in range(n):
        if all(np.hypot(*(ends[j] - ends[i])) >= sep for i in kept):
            keep[j] = True
            kept.append(j)
    return keep


def suppressed_order(score: torch.Tensor, V: torch.Tensor, kmax: int, r: float) -> torch.Tensor:
    """(B, kmax) goal indices: greedy by score, skipping goals within r of any already taken."""
    s = score.clone()
    out = []
    for _ in range(kmax):
        i = s.argmax(-1)  # after exhaustion all -inf: argmax returns 0, harmless for ranking
        out.append(i)
        s = s.masked_fill((V[None, :, :2] - V[i][:, None, :2]).norm(dim=-1) < r, -float("inf"))
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/l2_full_seed0"))
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/l2_selection"))
    a = ap.parse_args()
    dev = pick_device("auto")
    mm = MemmapSplit(Path("data/sim/mm_val"))
    gf = load(a.run / "goalflow_B_best.pt", dev).eval()
    V = gf.scorer.vocab                                                      # (N, 3)
    Vn = V.cpu().numpy()
    ds = SimFrames([mm], driven_only=True)                                   # targets unused; no DAC targets
    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers,
                    multiprocessing_context="spawn" if a.workers else None)
    nb = min(len(dl), a.max_batches or len(dl))
    print(f"val: {len(ds)} frames, {nb} batches, vocab {len(Vn)} y in [{Vn[:, 1].min():.2f}, {Vn[:, 1].max():.2f}] "
          f"x in [{Vn[:, 0].min():.2f}, {Vn[:, 0].max():.2f}]", flush=True)
    # per-route records
    R = {k: [] for k in ("frame", "j", "driven", "distinct", "n_modes", "n_distinct", "nn_xy", "nn_metric",
                         "rank_plain", "rank_supp", "l_dis", "H", "end_y")}
    Fr = {k: [] for k in {"n_modes", "n_distinct", "l_mix", "H_mix", "l_mix_allmodes", "H_mix_allmodes"}
          | {f"cover{s}@{k}_{w}" for s in ("", "_supp") for k in KS for w in ("all", "distinct")}}
    t0 = time.time()
    with torch.no_grad():
        for bi, b in enumerate(dl):
            if bi >= nb:
                break
            bev, ego = b["bev"].to(dev), b["ego"].to(dev)
            dis, dac = gf.scorer(*gf.perceive(bev, ego)[:2])
            score = F.log_softmax(dis, -1) + F.logsigmoid(dac)
            logp = F.log_softmax(dis, -1)
            order = score.argsort(-1, descending=True)
            rank_of = torch.empty_like(order)
            rank_of.scatter_(1, order, torch.arange(order.shape[1], device=dev)[None].expand_as(order))
            supp = suppressed_order(score, V, 64, SUPPRESS_R)                 # (B, 64)
            # full suppressed rank: position of each goal's nearest taken goal is not defined; rank a route
            # by the first suppressed pick within COVER_R of its endpoint (> 64 -> 65)
            modes = b["modes"].numpy()                                          # (B, 8, T, 2)
            # heading at the endpoint from the last segment (stored targets have theta; modes keep x, y)
            nm = b["n_modes"].numpy()
            idx = b["idx"].numpy()
            ends_full = []
            for i in range(len(nm)):
                k0 = int(idx[i])
                kk, jj = ds.index[k0]
                t0_, t1_ = int(mm["targets_off"][jj]), int(mm["targets_off"][jj + 1])
                ends_full.append(np.asarray(mm["targets"][t0_:t1_, -1, :]))     # (m, 3) x, y, theta
            for i in range(len(nm)):
                m = int(nm[i])
                E3 = ends_full[i][:m]
                E = E3[:, :2]
                dm = distinct_mask(E, m)
                Et = torch.as_tensor(E3, dtype=torch.float32, device=dev)
                dxy = torch.cdist(Et[:, :2], V[:, :2])                          # (m, N)
                dmet = torch.cdist(goal_metric(Et), goal_metric(V))
                near = dxy <= COVER_R
                rp = torch.where(near, rank_of[i][None].expand(m, -1), torch.full_like(dxy, 10**6, dtype=torch.long))
                rank_plain = rp.min(1).values + 1
                ds_ = (V[supp[i]][None, :, :2] - Et[:, None, :2]).norm(dim=-1) <= COVER_R   # (m, 64)
                rank_supp = torch.where(ds_.any(1), ds_.float().argmax(1) + 1, torch.full((m,), 65, device=dev))
                tgt = torch.softmax(-dmet / T_DIS, -1)                          # (m, N)
                l_dis = -(tgt * logp[i][None]).sum(-1)
                H = -(tgt * tgt.clamp_min(1e-30).log()).sum(-1)
                mix = tgt[torch.as_tensor(dm, device=dev)].mean(0)
                mix_all = tgt.mean(0)                    # uniform over all stored modes (= training's draw)
                Fr["l_mix_allmodes"].append(float(-(mix_all * logp[i]).sum()))
                Fr["H_mix_allmodes"].append(float(-(mix_all * mix_all.clamp_min(1e-30).log()).sum()))
                R["frame"] += [int(idx[i])] * m
                R["j"] += list(range(m))
                R["driven"] += [j == 0 for j in range(m)]
                R["distinct"] += dm.tolist()
                R["n_modes"] += [m] * m
                R["n_distinct"] += [int(dm.sum())] * m
                R["end_y"] += E[:, 1].tolist()
                R["nn_xy"] += dxy.min(1).values.tolist()
                R["nn_metric"] += dmet.min(1).values.tolist()
                R["rank_plain"] += rank_plain.tolist()
                R["rank_supp"] += rank_supp.tolist()
                R["l_dis"] += l_dis.tolist()
                R["H"] += H.tolist()
                Fr["n_modes"].append(m)
                Fr["n_distinct"].append(int(dm.sum()))
                Fr["l_mix"].append(float(-(mix * logp[i]).sum()))
                Fr["H_mix"].append(float(-(mix * mix.clamp_min(1e-30).log()).sum()))
                for w, sel in (("all", np.ones(m, bool)), ("distinct", dm)):
                    sel_t = torch.as_tensor(sel, device=dev)
                    for k in KS:
                        Fr[f"cover@{k}_{w}"].append(bool((rank_plain[sel_t] <= k).all()))
                        Fr[f"cover_supp@{k}_{w}"].append(bool((rank_supp[sel_t] <= k).all()))
            if bi % 20 == 0 or bi == nb - 1:
                el = time.time() - t0
                print(f"  batch {bi + 1}/{nb}  {el / 60:.1f} min  eta {el / (bi + 1) * (nb - bi - 1) / 60:.1f} min", flush=True)

    R = {k: np.array(v) for k, v in R.items()}
    Fr = {k: np.array(v) for k, v in Fr.items()}
    ylo, yhi = Vn[:, 1].min(), Vn[:, 1].max()
    out: dict = {"frames": int(len(Fr["n_modes"])),
                 "rule": f"distinct routes: endpoints >= {DISTINCT_SEP} m apart (greedy, driven first)",
                 "vocab": {"n": int(len(Vn)), "y_min": float(ylo), "y_max": float(yhi),
                           "x_min": float(Vn[:, 0].min()), "x_max": float(Vn[:, 0].max())}}

    def route_stats(m):
        if not m.any():
            return None
        nn = R["nn_xy"][m]
        rp, rs = R["rank_plain"][m], R["rank_supp"][m]
        return {"routes": int(m.sum()),
                "nn_xy_median": float(np.median(nn)), "nn_xy_p95": float(np.percentile(nn, 95)),
                "nn_xy_share_gt_0.5": float((nn > 0.5).mean()),
                "nn_metric_median": float(np.median(R["nn_metric"][m])),
                "outside_vocab_y_bounds": float(((R["end_y"][m] < ylo) | (R["end_y"][m] > yhi)).mean()),
                "no_goal_within_1m": float((rp >= 10**6).mean()),
                "rank_plain_median": float(np.median(np.minimum(rp, 10**6))),
                "rank_plain_p90": float(np.percentile(np.minimum(rp, 10**6), 90)),
                "rank_plain_share_le_5": float((rp <= 5).mean()), "rank_plain_share_le_10": float((rp <= 10).mean()),
                "rank_supp_median": float(np.median(rs)), "rank_supp_p90": float(np.percentile(rs, 90)),
                "rank_supp_share_le_2": float((rs <= 2).mean()), "rank_supp_share_le_5": float((rs <= 5).mean()),
                "l_dis": float(R["l_dis"][m].mean()), "l_dis_floor_entropy": float(R["H"][m].mean()),
                "l_dis_gap": float((R["l_dis"][m] - R["H"][m]).mean())}

    multi = R["n_modes"] > 1
    dmulti = R["n_distinct"] > 1
    out["routes"] = {
        "all_routes": {"driven": route_stats(R["driven"]), "non_driven": route_stats(~R["driven"]),
                       "non_driven_in_multi_frames": route_stats(~R["driven"] & multi)},
        "distinct_routes": {"driven": route_stats(R["driven"] & R["distinct"]),
                            "non_driven": route_stats(~R["driven"] & R["distinct"]),
                            "driven_in_distinct_multi_frames": route_stats(R["driven"] & dmulti),
                            "non_driven_in_distinct_multi_frames": route_stats(~R["driven"] & R["distinct"] & dmulti)}}
    fm, fd = Fr["n_modes"] > 1, Fr["n_distinct"] > 1
    out["frames_multi_all"] = int(fm.sum())
    out["frames_multi_distinct"] = int(fd.sum())
    cov = {}
    for w, msk in (("all", fm), ("distinct", fd)):
        cov[w] = {f"{'supp_' if s else ''}cover@{k}": float(Fr[f"cover{s}@{k}_{w}"][msk].mean())
                  for s in ("", "_supp") for k in KS}
    out["cover"] = cov
    out["l_mix_all_frames"] = float(Fr["l_mix"].mean())
    out["H_mix_all_frames"] = float(Fr["H_mix"].mean())
    out["l_mix_distinct_multi"] = float(Fr["l_mix"][fd].mean())
    out["H_mix_distinct_multi"] = float(Fr["H_mix"][fd].mean())
    out["l_mix_allmodes_all_frames"] = float(Fr["l_mix_allmodes"].mean())
    out["H_mix_allmodes_all_frames"] = float(Fr["H_mix_allmodes"].mean())
    out["l_mix_allmodes_multi"] = float(Fr["l_mix_allmodes"][fm].mean())
    out["H_mix_allmodes_multi"] = float(Fr["H_mix_allmodes"][fm].mean())
    out["caveat_cover"] = ("cover@k counts a goal within 1.0 m of a route endpoint; with the 1.5 m distinct-route "
                           "rule one goal halfway between two routes can cover both (no one-to-one matching here; "
                           "eval_selection.py adds a one-to-one version)")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "goal_scorer_val.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(a.out / "goal_scorer_val_routes.npz", **R)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
