"""A6 rerun: the 33 L0-only seeds with a given observed config; table + mechanism per seed.

    python scripts/a6_rerun.py {pre,a,b} out.json [workers]

pre = pre-L3 planner (50e9ad9), a = pre-L3 + goal-line stub fix, b = stub fix + approved L3 set.
At the first brake: the planner's own input (with memory, if any), whether the car's own pose fails the
0.12 m check / is a track exit, the edge stub test, the observed-edge error beside the car, fail causes.
"""
import json
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

SEEDS = [17, 27, 64, 71, 82, 86, 102, 106, 108, 146, 155, 189, 190, 200, 202, 203, 225, 273, 287, 308, 333,
         359, 373, 384, 394, 397, 399, 432, 434, 437, 451, 486, 493]


def make(cfgname):
    from gfsem.planners.corridor import CorridorConfig
    from gfsem.planners.corridor_lattice import LatticeCorridorPlanner, l3_config
    if cfgname == "pre":
        return LatticeCorridorPlanner(mode="observed")
    if cfgname == "a":
        return LatticeCorridorPlanner(mode="observed", cfg=CorridorConfig(goal_cut_guard=1.0))
    return LatticeCorridorPlanner(mode="observed", cfg=CorridorConfig(goal_cut_guard=1.0), lcfg=l3_config())


def per_seed(job):
    seed, cfgname = job
    from gfsem.sim.closed_loop import CorridorObserved, run_episode
    from gfsem.sim.geometry import box_circle_clearance, footprint_corners, poses_from_frame
    from gfsem.sim.scene import generate_scene

    class Capture(CorridorObserved):
        def __init__(self, p):
            super().__init__(p, "observed")
            self.inputs = []

        def plan(self, obs, scene, true_pose, state):
            r = super().plan(obs, scene, true_pose, state)
            self.inputs.append(self.p.last_input)
            return r

    sc = generate_scene(seed)
    out = {"seed": seed, "config": cfgname, "levels": {}}
    for lvl in ("L0", "L1", "L2"):
        p = make(cfgname)
        cap = Capture(p)
        r = run_episode(sc, cap, lvl, record=True)
        row = {"outcome": r.outcome, "time": r.time, "n_brake_plans": r.n_brake_plans,
               "fail_causes": dict(p.fail_causes), "events": dict(getattr(p, "events", {}))}
        br = [i for i, x in enumerate(r.records) if x["brake"]]
        if br:
            i = br[0]
            rec = r.records[i]
            k = int(round(rec["t"] / 0.05))
            x, y, yaw, v, st = r.trajectory[max(k - 1, 0)]
            s_b = float(sc.track.frenet(np.array([[x, y]]))[0][0])
            row.update(first_brake_t=rec["t"], dist_to_goal=float(sc.s_goal - s_b), speed=float(v),
                       steer_deg=float(np.degrees(st)), reason=str(rec["info"].get("reason", "")),
                       nearest_obstacle=float(box_circle_clearance(np.array([[x, y, yaw]]), sc.centres, sc.radii,
                                                                   p.veh.body_x, p.veh.width).min()) if sc.n else None)
            G = p._build(cap.inputs[i])
            if G is not None:
                w = p.world(G)
                cs = footprint_corners(np.zeros((1, 3)), p.veh.body_x, p.veh.width).reshape(-1, 2)
                lens = [float(np.hypot(*np.diff(e, axis=0).T).sum()) if len(e) > 1 else 0.0 for e in (G["left"], G["right"])]
                bad = []
                for (tree, e, nrm), L in zip(w.edges, lens):
                    _, j = tree.query(cs)
                    bad.append(bool((((cs - e[j]) * nrm[j]).sum(1) < 0).any()))
                errs = []
                pose = np.array([x, y, yaw])
                for e in (G["left"], G["right"]):
                    near = np.hypot(*e.T) < 3.0
                    if near.any():
                        Wp = poses_from_frame(np.c_[e[near], np.zeros(near.sum())], pose)[:, :2]
                        _, de = sc.track.frenet(Wp)
                        errs.append(float((sc.track.width / 2 - np.abs(de)).max()))
                row.update(own_fail_0p12=bool(w.clearance(np.zeros(3), 0.12)[0] < 0), own_exit=any(bad),
                           stub=any(b and L < 1.0 for b, L in zip(bad, lens)), edge_lengths=lens,
                           edge_err_near_car=max(errs) if errs else None)
            # does the brake reject future poses because of the edge test? first rejected kept/lattice pose
        out["levels"][lvl] = row
    return out


def mechanism(a):
    if "first_brake_t" not in a:
        return "no brake"
    if a.get("stub"):
        return "goal-line edge stub"
    if a.get("own_fail_0p12") and (a.get("edge_err_near_car") or 0) > 0.12:
        return "fill edge inside true edge"
    if a.get("own_fail_0p12"):
        return "own pose fails 0.12, other"
    return "plan fails, own pose OK"


if __name__ == "__main__":
    cfgname, outp = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    with get_context("spawn").Pool(nw) as pool:
        rows = pool.map(per_seed, [(s, cfgname) for s in SEEDS])
    for r in rows:
        r["mechanism_L0"] = mechanism(r["levels"]["L0"])
    json.dump(rows, open(outp, "w"), indent=1, default=float)
    from collections import Counter
    print(cfgname, "L0 outcomes", dict(Counter(r["levels"]["L0"]["outcome"] for r in rows)),
          "L1", dict(Counter(r["levels"]["L1"]["outcome"] for r in rows)),
          "L2", dict(Counter(r["levels"]["L2"]["outcome"] for r in rows)))
    print("mechanisms", dict(Counter(r["mechanism_L0"] for r in rows)))
