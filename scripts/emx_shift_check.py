#!/usr/bin/env python
"""Distribution-shift check for GoalFlow at the EMX sensor level (open loop; validation seeds only).

    python scripts/emx_shift_check.py [--scenes 40] [--seed0 250000] [--out reports/lidar_demo/emx_check]

GoalFlow was trained on 0.2 deg rays and the L0-L4 mix; the EMX preset has 0.08 deg rays and 5 cm range noise.
Generates the same validation scenes (seeds seed0.., in the validation range, unused so far) once at EMX and once
at L2 (teacher episodes as in the L1 dataset), then reports per level, on the frozen seed-0 checkpoint:
  * goal scorer: suppressed cover@2/3/5 (greedy, 1.5 m suppression; a goal within 1.0 m of every route endpoint)
    on multi-route frames, balanced DAC accuracy;
  * selection (L3 configuration, NMS-K3 + exact 0.12 m filter against this scan's fitted circles): open-loop
    collision rate of the chosen executed path vs the true obstacles, and the brake rate (no safe candidate).
A clear drop at EMX vs L2 is a sim-to-real warning; retraining with EMX-level data then goes in the queue.
Data go to data/sim/emx_check (git-ignored).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from eval_goal_scorer import suppressed_order  # noqa: E402
from eval_selection import path_clearance  # noqa: E402
from gfsem.data.sim_dataset import DatasetConfig, generate_scene_frames, observation_from_shard, pack_shard  # noqa: E402
from gfsem.data.sim_torch import SimFrames  # noqa: E402
from gfsem.models.common import pick_device  # noqa: E402
from gfsem.planners.corridor import CorridorConfig, cluster_obstacles  # noqa: E402
from gfsem.sim.config import SceneConfig, SimConfig  # noqa: E402
from gfsem.train import load  # noqa: E402


def build(level: str, seeds: list[int], out: Path) -> Path:
    f = out / f"{level}.npz"
    if f.exists():
        return f
    dcfg = replace(DatasetConfig(), noise_mix=((level, 1.0),))
    arr = pack_shard([generate_scene_frames(s, dcfg=dcfg) for s in seeds])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **arr)
    return f


@torch.no_grad()
def evaluate(level: str, shard: Path, gf, dev, veh) -> dict:
    arr = dict(np.load(shard))
    ds = SimFrames([arr], vocab=gf.scorer.vocab.cpu().numpy(), driven_only=True)
    ccfg, scfg = CorridorConfig(), SceneConfig.from_yaml()
    V = gf.scorer.vocab
    cov = {k: [] for k in (2, 3, 5)}
    dac_bal, coll, brake = [], [], []
    for i in range(len(ds)):
        b = ds[i]
        bev, ego = b["bev"][None].to(dev), b["ego"][None].to(dev)
        dis, dac = gf.scorer(*gf.perceive(bev, ego)[:2])
        score = F.log_softmax(dis, -1) + F.logsigmoid(dac)
        m = int(b["n_modes"])
        if m > 1:
            supp = suppressed_order(score, V, 5, 1.5)[0]
            ends = b["modes"][:m, -1].to(dev)
            hit = (V[supp][None, :, :2] - ends[:, None]).norm(dim=-1) <= 1.0          # (m, 5)
            for k in cov:
                cov[k].append(bool(hit[:, :k].any(1).all()))
        tgt = b["dac"].to(dev)
        pred = (dac[0] > 0).float()
        if (tgt == 0).any() and (tgt == 1).any():
            dac_bal.append(float(0.5 * (((pred == 1) & (tgt == 1)).sum() / (tgt == 1).sum()
                                        + ((pred == 0) & (tgt == 0)).sum() / (tgt == 0).sum())))
        obs = observation_from_shard(arr, i)
        C, R, _ = cluster_obstacles(obs, ccfg, scfg)
        circ = np.c_[np.asarray(C).reshape(-1, 2), np.asarray(R)] if len(C) else np.zeros((0, 3))
        torch.manual_seed(i)
        r = gf.plan(bev, ego, M=128, n_steps=1, goal_K=3, safety=True, safety_points=[circ], safety_margin=0.12)
        brake.append(bool(r["no_safe"][0]))
        if not brake[-1]:
            clr = path_clearance(r["traj_main"].cpu(), b["obstacles"][None], b["n_obstacles"][None], veh)
            coll.append(bool(clr[0] < 0))
    return {"level": level, "frames": len(ds), "multi_route_frames": len(cov[2]),
            **{f"suppressed_cover@{k}": float(np.mean(v)) if v else None for k, v in cov.items()},
            "dac_balanced_acc": float(np.mean(dac_bal)) if dac_bal else None,
            "collision_rate_chosen": float(np.mean(coll)) if coll else None,
            "brake_rate": float(np.mean(brake))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=250000)
    ap.add_argument("--out", type=Path, default=Path("reports/lidar_demo/emx_check"))
    a = ap.parse_args()
    assert 200000 <= a.seed0 < 300000, "validation seed range only (300000+ is the spent test split)"
    seeds = list(range(a.seed0, a.seed0 + a.scenes))
    data = Path("data/sim/emx_check")
    dev = pick_device("auto")
    gf = load(Path("runs/l2_full_seed0/goalflow_B_best.pt"), dev).eval()
    veh = SimConfig().vehicle
    rows = []
    for level in ("L2", "EMX"):
        rows.append(evaluate(level, build(level, seeds, data), gf, dev, veh))
        print(json.dumps(rows[-1]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "emx_shift.json").write_text(json.dumps({"seeds": [seeds[0], seeds[-1]], "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
