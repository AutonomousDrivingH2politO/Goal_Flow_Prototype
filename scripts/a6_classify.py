import sys, json; sys.path.insert(0, sys.argv[1])
import numpy as np
from multiprocessing import Pool
from gfsem.sim.scene import generate_scene
from gfsem.sim.closed_loop import run_episode, CorridorObserved
from gfsem.planners.corridor_lattice import LatticeCorridorPlanner
from gfsem.planners.corridor import input_from_observation
from gfsem.sim.geometry import footprint_corners

def one(seed):
    sc = generate_scene(seed); p = LatticeCorridorPlanner(mode="observed")
    r = run_episode(sc, CorridorObserved(p, "observed"), "L0", record=True)
    br = [x for x in r.records if x["brake"]]
    if not br: return dict(seed=seed, braked=False, outcome=r.outcome)
    rec = br[0]
    G = p._build(input_from_observation(rec["obs"], p.cfg, p.scfg))
    if G is None: return dict(seed=seed, braked=True, outcome=r.outcome, G=None)
    w = p.world(G)
    cs = footprint_corners(np.zeros((1, 3)), p.veh.body_x, p.veh.width).reshape(-1, 2)
    lens = [float(np.hypot(*np.diff(e, axis=0).T).sum()) if len(e) > 1 else 0.0 for e in (G["left"], G["right"])]
    bad_edge = []
    for (tree, e, n), L in zip(w.edges, lens):
        _, i = tree.query(cs)
        bad_edge.append(bool((((cs - e[i]) * n[i]).sum(1) < 0).any()))
    own_exit = any(bad_edge)
    stub = own_exit and any(b and L < 1.0 for b, L in zip(bad_edge, lens))
    # all brakes in the run: own-pose exit each time?
    return dict(seed=seed, braked=True, outcome=r.outcome, n_brakes=len(br), own_exit=own_exit, stub_edge=stub,
                edge_lengths=lens, own_fail_0p12=bool(w.clearance(np.zeros(3), 0.12)[0] < 0))

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[2].split(",")]
    with Pool(3) as pool: rows = pool.map(one, seeds)
    json.dump(rows, open(sys.argv[3], "w"), indent=1)
