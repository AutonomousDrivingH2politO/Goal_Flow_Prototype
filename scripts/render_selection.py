#!/usr/bin/env python
"""Side-by-side renders of selection variants on validation frames (open loop, frozen seed-0 models).

    python scripts/render_selection.py

Left: M3 (single g*, shadow as trained). Right: NMS-K3 + safety filter, shadow off (main path). Each frame's
evaluation batch is rerun with the same per-batch seed as scripts/eval_selection.py, so the samples drawn are
the evaluated ones and the collision flags match selection_val.json / frames_*.npz. Frames are picked by
fixed rules (first 6 in stored order unless stated), written to RULE.txt before rendering. Validation only.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eval_selection import path_clearance  # noqa: E402
from gfsem.data.scene_index import SceneIndex  # noqa: E402
from gfsem.data.sim_torch import MemmapSplit, SimFrames  # noqa: E402
from gfsem.models.common import pick_device  # noqa: E402
from gfsem.models.rollout import RolloutConfig, rollout  # noqa: E402
from gfsem.sim.config import VehicleConfig  # noqa: E402
from gfsem.train import load  # noqa: E402

SEL = Path("reports/lidar_demo/l2_selection")
OUT = SEL / "renders"
BS, SEED = 32, 0
GOAL_COLORS = ("tab:orange", "tab:purple", "tab:olive")


def rules(idx, C, F):
    pos = {int(f): i for i, f in enumerate(idx)}
    first6 = lambda m: [int(idx[i]) for i in np.flatnonzero(m)[:6]]   # noqa: E731
    return {
        "1_l2full_frames": ("the 12 frames of reports/lidar_demo/l2_full/RULE.txt (val_0000 stored order, >= 2 routes)",
                            [6, 7, 40, 41, 43, 45, 46, 47, 95, 96, 97, 98]),
        "2_regression_collides_k3_not": ("first 6 frames (stored order) where regression's executed path collides and "
                                         "NMS-K3 + safety's chosen path (main, shadow off) does not",
                                         first6(F["regression"]["collision"] & ~F["k3"]["collision_main"])),
        "3_k3_collides": ("first 6 frames where NMS-K3 + safety's chosen path (main, shadow off) collides",
                          first6(F["k3"]["collision_main"])),
        "4_m3_near_gstar": ("first 6 of the 305 frames where M3's shadow switch fires (shadow_dev > 1.5 m; g* is a near "
                            "goal)", first6(F["m3"]["shadow_dev"] > 1.5)),
        "5_k3_covers_homotopy_m3_not": ("first 6 multi-route frames where NMS-K3 + safety's candidate set covers every "
                                        "teacher route by homotopy and M3's does not",
                                        first6((C["n_modes"] > 1) & F["k3"]["cover_all_homotopy"]
                                               & ~F["m3"]["cover_all_homotopy"])),
    }, pos


def draw_panel(ax, b, r, title, reg_ex, veh, chosen_key):
    import matplotlib.pyplot as plt
    from gfsem.sim.raster import RasterConfig
    rc = RasterConfig()
    ext = [rc.y_max, rc.y_min, rc.x_min, rc.x_max]
    bev = b["bev"].cpu().numpy()
    img = np.ones(bev.shape[1:] + (3,))
    img[bev[3] > 0.5] = (0.84, 0.84, 0.84)
    img[bev[2] > 0.5] = (0.80, 0.86, 0.97)
    img[(bev[2] > 0.5) & (bev[3] > 0.5)] = (0.62, 0.71, 0.86)
    img[bev[0] > 0.5] = (0.05, 0.05, 0.05)
    ax.imshow(img, extent=ext, origin="upper", interpolation="nearest")
    S = r["samples"].cpu().numpy()
    go = r["goal_of"].cpu().numpy()
    safe = r["safe"].cpu().numpy() if "safe" in r else np.ones(len(S), bool)
    for s, g, ok in zip(S, go, safe):
        if ok:
            ax.plot(s[:, 1], s[:, 0], color=GOAL_COLORS[g % 3], lw=0.35, alpha=0.35, zorder=3)
        else:
            ax.plot(s[:, 1], s[:, 0], color="0.45", lw=0.5, ls="--", alpha=0.6, zorder=3)
    for k, G in enumerate(r["goals"].cpu().numpy()):
        ax.plot(G[1], G[0], "*", color=GOAL_COLORS[k % 3], mec="k", ms=13, zorder=6)
    O = b["obstacles"].cpu().numpy()[: int(b["n_obstacles"])]
    for x, y, rr in O:
        ax.add_patch(plt.Circle((y, x), rr, fill=False, ec="red", lw=1.1, zorder=5))
    ch = r[chosen_key].cpu().numpy()
    ax.plot(ch[:, 1], ch[:, 0], "-", color="tab:red", lw=2.4, zorder=6)
    ax.plot(reg_ex[:, 1], reg_ex[:, 0], "-", color="m", lw=1.6, zorder=6)
    for j in range(int(b["n_modes"])):
        m = b["modes"][j].cpu().numpy()
        ax.plot(m[:, 1], m[:, 0], "c--", lw=1.5, zorder=7)
    ax.plot(0, 0, "k^", ms=7, zorder=8)
    ax.set_xlim(8, -8)
    ax.set_ylim(-2, 18)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("y [m] (left +)")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    dev = pick_device("auto")
    C = np.load(SEL / "frames_common.npz")
    F = {"m3": np.load(SEL / "frames_M3.npz"), "k3": np.load(SEL / "frames_M3_NMS-K3psafety.npz"),
         "regression": np.load(SEL / "frames_regression.npz")}
    idx = C["idx"]
    R, pos = rules(idx, C, F)
    mm = MemmapSplit(Path("data/sim/mm_val"))
    si = SceneIndex([Path("data/sim/scenes_val.npz")])
    gf = load(Path("runs/l2_full_seed0/goalflow_B_best.pt"), dev).eval()
    rg = load(Path("runs/l2_full_seed0/regression_best.pt"), dev).eval()
    ds = SimFrames([mm], vocab=gf.scorer.vocab.cpu().numpy(), scene_index=si, driven_only=True)
    veh = VehicleConfig.from_yaml()
    mismatches, checked = [], 0
    for name, (text, frames) in R.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "RULE.txt").write_text(f"{text}\nframes (val memmap index): {frames}\n"
                                    "Left: M3 (single g*, shadow as trained). Right: NMS-K3 + safety, shadow off.\n"
                                    "Samples coloured by goal (grey dashed: flagged by the safety filter); stars: goals;"
                                    " red: chosen path; magenta: regression; cyan dashed: teacher routes; red circles:"
                                    " true obstacles. Each frame's evaluation batch is rerun with its seed.\n")
    for name, (text, frames) in R.items():
        for f in frames:
            p = pos[f]
            bi, k = p // BS, p % BS
            batch = [ds[int(i)] for i in idx[bi * BS:(bi + 1) * BS]]
            b = {key: torch.stack([x[key] for x in batch]).to(dev) for key in batch[0] if torch.is_tensor(batch[0][key])}
            with torch.no_grad():
                torch.manual_seed(SEED * 100_003 + bi)
                r1 = gf.plan(b["bev"], b["ego"], M=128, n_steps=1)
                torch.manual_seed(SEED * 100_003 + bi)
                r2 = gf.plan(b["bev"], b["ego"], M=128, n_steps=1, goal_K=3, safety=True, safety_slack=0.0)
                pr = rg.norm.inverse(rg.reg(*rg.perceive(b["bev"], b["ego"])[:2]))
                rx = rollout(pr, b["ego"][:, 0], b["ego"][:, 1], RolloutConfig())
                c1 = path_clearance(r1["traj"], b["obstacles"], b["n_obstacles"], veh)[k] < 0
                c2 = path_clearance(r2["traj_main"], b["obstacles"], b["n_obstacles"], veh)[k] < 0
                cr = path_clearance(rx, b["obstacles"], b["n_obstacles"], veh)[k] < 0
            # the re-run must reproduce the evaluated flags exactly
            ref = (bool(F["m3"]["collision"][p]), bool(F["k3"]["collision_main"][p]), bool(F["regression"]["collision"][p]))
            got = (bool(c1), bool(c2), bool(cr))
            checked += 1
            if got != ref:
                mismatches.append((name, f, ref, got))
            one = lambda r: {kk: (v[k] if torch.is_tensor(v) and v.dim() and v.shape[0] == len(batch) else v)  # noqa: E731
                             for kk, v in r.items()}
            bk = {kk: v[k] for kk, v in b.items()}
            fig, axs = plt.subplots(1, 2, figsize=(11.5, 6.4))
            sh = " (shadow used)" if bool(r1["used_shadow"][k]) else ""
            draw_panel(axs[0], bk, one(r1), f"M3 single g*{sh}: collision {'YES' if c1 else 'no'}",
                       rx[k].cpu().numpy(), veh, "traj")
            draw_panel(axs[1], bk, one(r2), f"NMS-K3 + safety, shadow off: collision {'YES' if c2 else 'no'}",
                       rx[k].cpu().numpy(), veh, "traj_main")
            axs[0].set_ylabel("x [m] (forward)")
            fig.suptitle(f"val frame {f} (seed {int(bk['seed'])}, {int(bk['n_modes'])} teacher route(s)); "
                         f"regression collision {'YES' if cr else 'no'}", fontsize=10)
            hand = [Line2D([], [], color=c, lw=1.5, label=f"samples to goal {i + 1}") for i, c in enumerate(GOAL_COLORS)]
            hand += [Line2D([], [], color="0.45", ls="--", label="flagged by safety filter"),
                     Line2D([], [], color="tab:red", lw=2.4, label="chosen path"),
                     Line2D([], [], color="m", lw=1.6, label="regression (executed)"),
                     Line2D([], [], color="c", ls="--", label="teacher routes"),
                     Line2D([], [], color="red", marker="o", mfc="none", ls="", label="true obstacles")]
            fig.legend(handles=hand, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
            fig.tight_layout(rect=(0, 0.07, 1, 0.97))
            fig.savefig(OUT / name / f"frame{f:05d}.png", dpi=110)
            plt.close(fig)
            print(name, f, "M3", bool(c1), "K3", bool(c2), "reg", bool(cr), "match" if got == ref else "MISMATCH", flush=True)
    (OUT / "CHECK.txt").write_text(f"collision flags re-computed for {checked} frames vs frames_*.npz "
                                   f"(M3 collision, NMS-K3+safety collision_main, regression collision): "
                                   f"{len(mismatches)} mismatches\n" + "".join(f"{m}\n" for m in mismatches))
    print(f"CHECK: {len(mismatches)} mismatches of {checked}", flush=True)
    assert not mismatches, mismatches


if __name__ == "__main__":
    main()
