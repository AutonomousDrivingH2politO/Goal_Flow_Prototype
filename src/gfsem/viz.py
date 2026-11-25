"""Sample-trajectory plots: BEV background, teacher modes, goal scores, flow samples, selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def plot_samples(model, ds, out: Path, n: int = 6, device="cpu", reg_model=None, M: int = 128, seed: int = 0,
                 indices: list[int] | None = None, prefix: str = "sample"):
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
    if indices is not None:
        pick = list(indices)
    else:
        pick = list(rng.choice(multi, min(len(multi), n // 2), replace=False)) if multi else []
        pick += list(rng.choice(len(ds), n - len(pick), replace=False))
    model.eval()
    for k, i in enumerate(pick):
        b = ds[int(i)]
        bev = b["bev"][None].to(device)
        ego = b["ego"][None].to(device)
        torch.manual_seed(seed * 100_003 + int(i))   # reproducible samples per frame
        r = model.plan(bev, ego, M=M, n_steps=1)
        S = r["samples"][0].cpu().numpy()
        traj = r["traj"][0].cpu().numpy()
        traj_raw = r["traj_raw"][0].cpu().numpy() if "traj_raw" in r else None
        g = r["goal"][0].cpu().numpy()
        score = r["goal_score"][0].cpu().numpy()
        V = model.scorer.vocab.cpu().numpy()
        fig, ax = plt.subplots(figsize=(6.4, 8.2))
        # background classes as explicit colours (later classes paint over earlier ones)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        cls = [("free / off-track", None, (1.0, 1.0, 1.0)),
               ("drivable (between observed edges)", b["bev"][3].numpy() > 0.5, (0.84, 0.84, 0.84)),
               ("unknown (no ray reached)", b["bev"][2].numpy() > 0.5, (0.80, 0.86, 0.97)),
               ("unknown and drivable", (b["bev"][2].numpy() > 0.5) & (b["bev"][3].numpy() > 0.5), (0.62, 0.71, 0.86)),
               ("LiDAR hits", b["bev"][0].numpy() > 0.5, (0.05, 0.05, 0.05))]
        img = np.ones(b["bev"].shape[1:] + (3,))
        for _, m, c in cls[1:]:
            img[m] = c
        ax.imshow(img, extent=ext, origin="upper", interpolation="nearest")
        bg_handles = [Patch(facecolor=c, edgecolor="0.5", lw=0.5, label=lab) for lab, _, c in cls]
        top = np.argsort(score)[-200:]
        sc = ax.scatter(V[top, 1], V[top, 0], c=score[top], cmap="viridis", s=8, zorder=2, label="goal scores (top 200)")
        for q, s in enumerate(S):                     # all M executed flow samples, faint
            ax.plot(s[:, 1], s[:, 0], color="tab:orange", lw=0.3, alpha=0.18, zorder=3)
        sample_handle = Line2D([], [], color="tab:orange", lw=1.0, alpha=0.7, label=f"flow samples (all {len(S)}, executed, faint)")
        ylo, yhi = float(V[:, 1].min()), float(V[:, 1].max())
        for q, yb in enumerate((ylo, yhi)):          # lateral extent of the goal vocabulary
            ax.axvline(yb, color="tab:green", ls=":", lw=1.0, zorder=2,
                       label=f"vocabulary lateral bounds (y {ylo:.2f} / {yhi:.2f} m)" if q == 0 else None)
        nm = int(b["n_modes"])
        for j in range(nm):
            m = b["modes"][j].numpy()
            ax.plot(m[:, 1], m[:, 0], "c--", lw=1.6, zorder=7, label="teacher modes (drawn on top)" if j == 0 else None)
        ax.plot(traj[:, 1], traj[:, 0], "r-", lw=2.5, zorder=5, label="GoalFlow selected (executed)")
        if traj_raw is not None:
            ax.plot(traj_raw[:, 1], traj_raw[:, 0], "r--", lw=1.0, zorder=5, label="GoalFlow selected (raw)")
        ax.plot(g[1], g[0], "r*", ms=14, zorder=6, label="g*")
        if reg_model is not None:
            from .models.rollout import RolloutConfig, rollout
            pr = reg_model.norm.inverse(reg_model.reg(*reg_model.perceive(bev, ego)[:2]))
            pe = rollout(pr, ego[:, 0], ego[:, 1], RolloutConfig())[0].cpu().numpy()
            pr = pr[0].cpu().numpy()
            ax.plot(pe[:, 1], pe[:, 0], "m-", lw=2, zorder=5, label="regression (executed)")
            ax.plot(pr[:, 1], pr[:, 0], "m--", lw=1.0, zorder=5, label="regression (raw)")
        ax.plot(0, 0, "k^", ms=8)
        ax.set_xlim(8, -8)
        ax.set_ylim(-2, 18)
        ax.set_xlabel("y [m] (left is +)")
        ax.set_ylabel("x [m] (forward)")
        ends = S[:, -1]
        near = [int((np.hypot(*(ends - b["modes"][j].numpy()[-1]).T) <= 1.0).sum()) for j in range(nm)]
        ax.set_title(f"frame {int(b['idx'])} seed {int(b['seed'])}: {nm} teacher mode(s), {len(S)} samples (executed)\n"
                     f"sample endpoint std y = {ends[:, 1].std():.2f} m; samples within 1 m of each mode: "
                     f"{', '.join(map(str, near))}", fontsize=9)
        h, _ = ax.get_legend_handles_labels()
        ax.legend(handles=h[:1] + [sample_handle] + h[1:] + bg_handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=6.5,
                  frameon=False)
        plt.colorbar(sc, ax=ax, fraction=0.04, label="w1 log d_dis + w2 log d_dac")
        fig.tight_layout()
        fig.savefig(out / f"{prefix}_{k:02d}_frame{int(b['idx']):05d}.png", dpi=110)
        plt.close(fig)
