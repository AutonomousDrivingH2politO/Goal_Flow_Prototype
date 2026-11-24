"""Sample-trajectory plots: BEV background, teacher modes, goal scores, flow samples, selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def plot_samples(model, ds, out: Path, n: int = 6, device="cpu", reg_model=None, M: int = 64, seed: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .sim.raster import RasterConfig

    out.mkdir(parents=True, exist_ok=True)
    rc = RasterConfig()
    ext = [rc.y_max, rc.y_min, rc.x_min, rc.x_max]
    rng = np.random.default_rng(seed)
    # prefer frames with several teacher modes, then fill with random frames
    multi = [i for i in range(len(ds)) if ds.shards[ds.index[i][0]]["targets_off"][ds.index[i][1] + 1]
             - ds.shards[ds.index[i][0]]["targets_off"][ds.index[i][1]] > 1]
    pick = list(rng.choice(multi, min(len(multi), n // 2), replace=False)) if multi else []
    pick += list(rng.choice(len(ds), n - len(pick), replace=False))
    model.eval()
    for k, i in enumerate(pick):
        b = ds[int(i)]
        bev = b["bev"][None].to(device)
        ego = b["ego"][None].to(device)
        r = model.plan(bev, ego, M=M, n_steps=1)
        S = r["samples"][0].cpu().numpy()
        traj = r["traj"][0].cpu().numpy()
        g = r["goal"][0].cpu().numpy()
        score = r["goal_score"][0].cpu().numpy()
        V = model.scorer.vocab.cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        bg = np.clip(0.25 * b["bev"][3].numpy() + b["bev"][0].numpy() + 0.15 * b["bev"][2].numpy(), 0, 1)
        ax.imshow(bg, cmap="Greys", extent=ext, origin="upper", vmin=0, vmax=1.2)
        top = np.argsort(score)[-200:]
        sc = ax.scatter(V[top, 1], V[top, 0], c=score[top], cmap="viridis", s=8, zorder=2, label="goal scores (top 200)")
        for s in S:
            ax.plot(s[:, 1], s[:, 0], color="tab:orange", lw=0.4, alpha=0.5, zorder=3)
        nm = int(b["n_modes"])
        for j in range(nm):
            m = b["modes"][j].numpy()
            ax.plot(m[:, 1], m[:, 0], "c--", lw=1.5, zorder=4, label="teacher modes" if j == 0 else None)
        ax.plot(traj[:, 1], traj[:, 0], "r-", lw=2.5, zorder=5, label="GoalFlow selected")
        ax.plot(g[1], g[0], "r*", ms=14, zorder=6, label="g*")
        if reg_model is not None:
            p = reg_model.norm.inverse(reg_model.reg(*reg_model.perceive(bev, ego)[:2]))[0].cpu().numpy()
            ax.plot(p[:, 1], p[:, 0], "m-", lw=2, zorder=5, label="regression")
        ax.plot(0, 0, "k^", ms=8)
        ax.set_xlim(8, -8)
        ax.set_ylim(-2, 18)
        ax.set_xlabel("y [m] (left is +)")
        ax.set_ylabel("x [m] (forward)")
        ax.set_title(f"frame {int(b['idx'])} seed {int(b['seed'])}: {nm} teacher mode(s), {M} samples")
        ax.legend(loc="lower left", fontsize=7)
        plt.colorbar(sc, ax=ax, fraction=0.04, label="w1 log d_dis + w2 log d_dac")
        fig.tight_layout()
        fig.savefig(out / f"sample_{k:02d}.png", dpi=110)
        plt.close(fig)
