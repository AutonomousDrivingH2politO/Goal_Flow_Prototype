"""A6 goal-line stub fix (CorridorConfig.goal_cut_guard, observed mode only)."""
import numpy as np

from gfsem.planners.corridor import CorridorConfig, PlannerInput
from gfsem.planners.corridor_lattice import LatticeCorridorPlanner
from gfsem.sim.geometry import footprint_corners


def _inp(right_x_max=30.0):
    xl = np.arange(-3.0, 30.01, 1.0)
    xr = np.arange(-3.0, right_x_max + 0.01, 1.0)
    left = np.c_[xl, np.full_like(xl, 6.0)]
    right = np.c_[xr, np.full_like(xr, -6.0)]
    # oblique goal line (right end behind the car, as on a curve near the goal): rows = (right, left)
    goal = np.array([[-2.0, -6.0], [10.0, 6.0]])
    return PlannerInput(np.zeros((0, 2)), np.zeros(0), left, right, goal, None, np.zeros(3))


def _planner(guard):
    return LatticeCorridorPlanner(mode="observed", cfg=CorridorConfig(goal_cut_guard=guard))


AHEAD = np.array([[4.0, -3.0, 0.0]])        # a pose 4 m ahead, inside the track


def test_without_guard_the_cut_leaves_a_stub_and_near_poses_count_as_past_the_goal():
    p = _planner(0.0)
    G = p._build(_inp())
    assert np.hypot(*np.diff(G["right"], axis=0).T).sum() <= 1.0 + 1e-9   # the stub behind the car
    assert G["goal"][1][0] < 0                                            # goal endpoint behind the car
    assert p._goal_side(G, AHEAD[:, :2])[0] >= 0      # would skip this pose's track-exit test


def test_guard_keeps_the_edge_beside_the_car_and_the_car_before_the_goal_line():
    p = _planner(1.0)
    G = p._build(_inp())
    x_min = p.veh.body_x[1] + 1.0
    assert min(G["goal"][0][0], G["goal"][1][0]) >= x_min - 1e-9
    corners = footprint_corners(np.zeros((1, 3)), p.veh.body_x, p.veh.width).reshape(-1, 2)
    assert (p._goal_side(G, corners) < 0).all()                         # exit test is not skipped
    assert p._goal_side(G, AHEAD[:, :2])[0] < 0
    assert p.world(G).clearance(np.zeros(3), 0.12)[0] >= 0               # own pose valid


def test_guard_refuses_a_goal_line_behind_the_front_when_no_edge_reaches_past_it():
    p = _planner(1.0)
    assert p._build(_inp(right_x_max=1.0)) is None                       # brake ("no edges"), never skip
