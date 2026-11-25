"""A6, remaining seeds: is the teacher's driven path feasible in the planner's observed world at the
first failing scan (config b)?

    python scripts/a6_teacher_in_observed.py 108,190,203,399 out.json

Checks on the teacher's driven path from its point nearest the car up to the goal line, in the planner's
own input (with memory) at that scan: plan clearance with the 0.25 m inflation (observed obstacle circles
and the observed-edge test), curvature against the ladder top (0.95 kappa_max), and the observed door gaps
it passes through (door rule). The same path against the true world (true discs, true edges) separates
"observed corridor narrower than the truth" (and why) from lattice incompleteness.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402


def main():
    from gfsem.planners.corridor import CorridorConfig
    from gfsem.planners.corridor_lattice import LatticeCorridorPlanner, l3_config, side_signature_local
    from gfsem.sim.closed_loop import CorridorObserved, CorridorPrivileged, run_episode
    from gfsem.sim.geometry import poses_to_frame, to_frame
    from gfsem.sim.lattice import CollisionWorld
    from gfsem.sim.scene import generate_scene

    class Capture(CorridorObserved):
        def __init__(self, p):
            super().__init__(p, "observed")
            self.inputs = []

        def plan(self, obs, scene, true_pose, state):
            r = super().plan(obs, scene, true_pose, state)
            self.inputs.append(self.p.last_input)
            return r

    out = []
    for seed in map(int, sys.argv[1].split(",")):
        sc = generate_scene(seed)
        teach = run_episode(sc, CorridorPrivileged(LatticeCorridorPlanner(mode="privileged"), "t"), "L0", record=True)
        T = teach.trajectory[:, :3]
        p = LatticeCorridorPlanner(mode="observed", cfg=CorridorConfig(goal_cut_guard=1.0), lcfg=l3_config())
        cap = Capture(p)
        r = run_episode(sc, cap, "L0", record=True)
        br = [i for i, x in enumerate(r.records) if x["brake"]]
        row = {"seed": seed, "outcome_b": r.outcome, "teacher": teach.outcome}
        if not br:
            out.append(row)
            continue
        i = br[0]
        rec = r.records[i]
        pose = rec["pose"]
        G = p._build(cap.inputs[i])
        world = p.world(G)
        P = poses_to_frame(T, pose)                                       # teacher path, ego frame
        j0 = int(np.argmin(np.hypot(P[:, 0], P[:, 1])))
        P = P[j0:]
        P = P[p._goal_side(G, P) < 0]
        row.update(t=rec["t"], reason=str(rec["info"].get("reason", "")),
                   car_offset_from_teacher_path=float(np.hypot(*poses_to_frame(T, pose)[j0, :2])),
                   heading_diff_deg=float(np.degrees(poses_to_frame(T, pose)[j0, 2])), teacher_len=float(
                       np.hypot(*np.diff(P[:, :2], axis=0).T).sum()) if len(P) > 1 else 0.0)
        if len(P) < 2:
            out.append(row)
            continue
        # observed world at the planning margin
        c_obs = world.clearance(P, 0.25)
        # true world: true discs and true edges in the ego frame
        loc = to_frame(sc.centres, pose)
        s0 = sc.track.frenet(pose[:2][None])[0][0]
        ss = np.linspace(s0 - 5, s0 + 30, 141)
        Lw = np.stack([sc.track.pose(s, sc.track.width / 2)[:2] for s in ss])
        Rw = np.stack([sc.track.pose(s, -sc.track.width / 2)[:2] for s in ss])
        tw = CollisionWorld(loc, sc.radii, to_frame(Lw, pose), to_frame(Rw, pose), p.veh)
        c_true = tw.clearance(P, 0.25)
        bad_obs, bad_true = c_obs < 0, c_true < 0
        # what the observed violations are: edge test (-1) or an obstacle circle
        edge_viol = bad_obs & (c_obs == -1.0)
        obst_viol = bad_obs & ~edge_viol
        # curvature of the driven path vs the ladder top
        d = np.diff(P[:, :2], axis=0)
        th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        ds = np.hypot(*d.T)
        k = np.abs(np.diff(th)) / np.maximum(0.5 * (ds[1:] + ds[:-1]), 1e-3)
        kfrac = float(k[ds[1:] > 0.02].max() / p.veh.kappa_max) if len(k) else 0.0
        # observed circles vs truth (fit / occlusion), near the violating poses
        C, R = G["V"][:G["nC"]], G["rad"][:G["nC"]]
        fit = []
        for c, rr in zip(C, R):
            dd = np.hypot(*(loc - c).T)
            jj = int(dd.argmin())
            fit.append({"obs": [round(float(c[0]), 2), round(float(c[1]), 2)], "true_offset": round(float(dd[jj]), 3)})
        missing = [int(jj) for jj in range(sc.n) if (not len(C) or np.hypot(*(C - loc[jj]).T).min() > 0.5)
                   and 0 < loc[jj, 0] < 20 and abs(loc[jj, 1]) < 7]
        # doors passed by the teacher (pairs on opposite sides) and their observed vs true gaps
        sig = dict(side_signature_local(P, C, near=3.0)) if len(C) else {}
        doors = []
        ks = list(sig)
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                u, v = ks[a], ks[b]
                if sig[u] != sig[v]:
                    g_obs = float(np.hypot(*(C[u] - C[v])) - R[u] - R[v])
                    doors.append({"pair": [int(u), int(v)], "gap_obs": round(g_obs, 3),
                                  "passes_door_rule": bool(g_obs >= G["thr"])})
        row.update(observed_violations=int(bad_obs.sum()), observed_edge_violations=int(edge_viol.sum()),
                   observed_obstacle_violations=int(obst_viol.sum()), true_violations=int(bad_true.sum()),
                   first_obs_violation_x=float(P[bad_obs][0, 0]) if bad_obs.any() else None,
                   kappa_frac_max=kfrac, observed_circles=fit, true_obstacles_ahead_not_observed=missing,
                   doors=doors)
        if not bad_obs.any() and kfrac <= 0.95 and all(dd["passes_door_rule"] for dd in doors):
            row["verdict"] = "teacher path feasible in the observed world -> lattice search incompleteness"
        elif bad_obs.any() and not bad_true.any():
            why = []
            if edge_viol.any():
                why.append("observed edge (fill / edge test) narrower than the true edge")
            if obst_viol.any():
                why.append("observed obstacle circle(s) intrude (fit / occlusion)")
            if missing:
                why.append(f"{len(missing)} true obstacle(s) ahead not observed")
            row["verdict"] = "observed corridor narrower than the truth: " + "; ".join(why)
        else:
            row["verdict"] = "teacher path also violates the true world at 0.25 m here (or curvature / door rule)"
        out.append(row)
        print(seed, row["verdict"], flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
