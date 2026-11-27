"""Run observed corridor-lattice episodes with the planner defaults and dump deterministic fields.

    python tests/regression_episodes.py <src root> <out.json>

Used to build tests/data/observed_ref_50e9ad9.json from the 50e9ad9 planner (worktree) and by
tests/test_planner_regression.py, which checks the current defaults reproduce it.
"""
import json
import sys

SEEDS = (0, 27, 64, 106, 170, 288)
LEVELS = ("L0", "L1", "L2")
FIELDS = ("outcome", "time", "n_replans", "n_brake_plans", "flips", "min_clearance")


def run(seeds=SEEDS, levels=LEVELS):
    from gfsem.planners.corridor_lattice import LatticeCorridorPlanner
    from gfsem.sim.closed_loop import CorridorObserved, run_episode
    from gfsem.sim.scene import generate_scene
    out = {}
    for s in seeds:
        sc = generate_scene(s)
        for lv in levels:
            r = run_episode(sc, CorridorObserved(LatticeCorridorPlanner(mode="observed"), "observed"), lv,
                            drift_per_distance=False)          # the 50e9ad9 reference used the time-based drift
            out[f"{s}/{lv}"] = {k: (round(float(getattr(r, k)), 9) if k in ("time", "min_clearance") else getattr(r, k))
                                for k in FIELDS}
    return out


if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1])
    json.dump(run(), open(sys.argv[2], "w"), indent=1)
