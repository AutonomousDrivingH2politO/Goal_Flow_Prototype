"""A6 tabulation with the ORIGINAL planner (worktree at bb0a9c1, passed as argv[1])."""
import sys, json
SRC = sys.argv[1]; sys.path.insert(0, SRC)
import numpy as np
from multiprocessing import Pool

SEEDS = [17, 27, 64, 71, 82, 86, 102, 106, 108, 146, 155, 189, 190, 200, 202, 203, 225, 273, 287, 308, 333,
         359, 373, 384, 394, 397, 399, 432, 434, 437, 451, 486, 493]


def route_gaps(traj, sc, width):
    """Narrowest gap on the teacher's driven route: obstacle pairs passed on opposite sides
    (|c_i - c_j| - r_i - r_j - width) and obstacle-to-edge (edge distance - r - width)."""
    from gfsem.planners.corridor_lattice import side_signature_local
    P = traj[::4, :2]
    sig = dict(side_signature_local(np.c_[P, np.zeros(len(P))], sc.centres, near=3.0))
    s_path = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    s_of = {j: s_path[int(np.argmin(np.hypot(*(P - sc.centres[j]).T)))] for j in sig}
    best = (np.inf, None)
    ks = list(sig)
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            i, j = ks[a], ks[b]
            if sig[i] != sig[j] and abs(s_of[i] - s_of[j]) < 2.5:
                g = np.hypot(*(sc.centres[i] - sc.centres[j])) - sc.radii[i] - sc.radii[j] - width
                if g < best[0]:
                    best = (g, ("pair", int(i), int(j)))
    _, dc = sc.track.frenet(sc.centres[ks]) if ks else (None, [])
    for i, d in zip(ks, dc):
        # obstacle left of the path (+1): the path passes between it and the right edge
        edge_dist = (sc.track.width / 2 + d) if sig[i] == 1 else (sc.track.width / 2 - d)
        g = edge_dist - sc.radii[i] - width
        if g < best[0]:
            best = (g, ("edge", int(i)))
    return best


def per_seed(seed):
    from gfsem.sim.scene import generate_scene
    from gfsem.sim.closed_loop import run_episode, CorridorObserved, CorridorPrivileged
    from gfsem.planners.corridor_lattice import LatticeCorridorPlanner
    from gfsem.planners.corridor import input_from_observation, cluster_obstacles
    from gfsem.sim.geometry import box_circle_clearance, to_frame, poses_from_frame
    from gfsem.sim.config import SceneConfig
    sc = generate_scene(seed)
    scfg = SceneConfig.from_yaml()
    out = {"seed": seed}
    teach = run_episode(sc, CorridorPrivileged(LatticeCorridorPlanner(mode="privileged"), "teacher"), "L0", record=True)
    pl0 = LatticeCorridorPlanner(mode="observed")
    veh = pl0.veh
    gap, what = route_gaps(teach.trajectory, sc, veh.width)
    out.update(teacher_outcome=teach.outcome, narrowest_gap_minus_width=float(gap), narrowest_gap_kind=what)
    res = {}
    for lvl in ("L0", "L1", "L2"):
        p = LatticeCorridorPlanner(mode="observed")
        r = run_episode(sc, CorridorObserved(p, "observed"), lvl, record=True)
        row = {"outcome": r.outcome, "time": r.time}
        tr = r.trajectory
        brakes = [i for i, x in enumerate(r.records) if x["brake"]]
        if brakes:
            rec = r.records[brakes[0]]
            k = int(round(rec["t"] / 0.05))
            x, y, yaw, v, st = tr[max(k - 1, 0)]
            s_b = float(sc.track.frenet(np.array([[x, y]]))[0][0])
            clr = float(box_circle_clearance(np.array([[x, y, yaw]]), sc.centres, sc.radii, veh.body_x, veh.width).min()) if sc.n else np.inf
            # does the car's own pose fail the 0.12 m check against the observed world?
            inp = input_from_observation(rec["obs"], p.cfg, p.scfg)
            G = p._build(inp)
            own = None
            edge_err = None
            if G is not None:
                w = p.world(G)
                own = bool(w.clearance(np.zeros(3), p.lcfg.validate_inflation)[0] < 0)
                own_edge_exit = bool(w.clearance(np.zeros(3), 0.0)[0] == -1.0)
                pose = np.array([x, y, yaw])
                errs = []
                for e in (G["left"], G["right"]):
                    near = np.hypot(*e.T) < 3.0
                    if near.any():
                        W = poses_from_frame(np.c_[e[near], np.zeros(near.sum())], pose)[:, :2]
                        _, de = sc.track.frenet(W)
                        errs.append(float((sc.track.width / 2 - np.abs(de)).max()))   # >0: edge drawn inside
                edge_err = max(errs) if errs else None
            row.update(first_brake_t=rec["t"], dist_to_goal=float(sc.s_goal - s_b), speed=float(v),
                       steer_deg=float(np.degrees(st)), nearest_obstacle_clearance=clr,
                       own_pose_fails_0p12=own, observed_edge_inward_err_near_car=edge_err,
                       fail_reason=str(rec["info"].get("reason", "")))
        # observed vs true geometry over the run
        rad_err, flips, gap_err, cen_err = [], 0, [], []
        pair = what[1:] if what and what[0] == "pair" else None
        n_assoc = 0
        for rec in r.records:
            C, R, K = cluster_obstacles(rec["obs"], p.cfg, p.scfg)
            if not len(C) or not sc.n:
                continue
            loc = to_frame(sc.centres, rec["pose"])
            d = np.hypot(C[:, None, 0] - loc[None, :, 0], C[:, None, 1] - loc[None, :, 1])
            j = d.argmin(1)
            ok = d[np.arange(len(C)), j] < 0.6
            for ci in np.flatnonzero(ok):
                n_assoc += 1
                rad_err.append(R[ci] - sc.radii[j[ci]])
                cen_err.append(d[ci, j[ci]])
                flips += int(abs(R[ci] - sc.radii[j[ci]]) > 1e-6)
            if pair is not None:
                fi = [ci for ci in np.flatnonzero(ok) if j[ci] == pair[0]]
                fj = [ci for ci in np.flatnonzero(ok) if j[ci] == pair[1]]
                if fi and fj:
                    og = np.hypot(*(C[fi[0]] - C[fj[0]])) - R[fi[0]] - R[fj[0]]
                    tg = np.hypot(*(sc.centres[pair[0]] - sc.centres[pair[1]])) - sc.radii[pair[0]] - sc.radii[pair[1]]
                    gap_err.append(og - tg)
        row.update(n_assoc=n_assoc, radius_err_mean=float(np.mean(rad_err)) if rad_err else None,
                   class_flip_share=flips / max(n_assoc, 1), centre_err_median=float(np.median(cen_err)) if cen_err else None,
                   narrowest_pair_gap_err_median=float(np.median(gap_err)) if gap_err else None,
                   narrowest_pair_gap_err_n=len(gap_err))
        res[lvl] = row
    out["levels"] = res
    return out


if __name__ == "__main__":
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 3) as pool:
        rows = pool.map(per_seed, SEEDS)
    json.dump(rows, open(sys.argv[3], "w"), indent=1, default=float)
    print("done", len(rows))
