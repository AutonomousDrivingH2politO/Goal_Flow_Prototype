import math

import numpy as np
import pytest

from gfsem.planners.corridor import CorridorPlanner, input_from_privileged, string_pull
from gfsem.sim.closed_loop import CorridorPrivileged, run_episode
from gfsem.sim.config import BOLLARD, CONE, NOISE_LEVELS, LidarConfig, SceneConfig, VehicleConfig
from gfsem.sim.geometry import Track, box_circle_clearance, from_frame, to_frame
from gfsem.sim.lidar import privileged_view, scan
from gfsem.sim.raster import RasterConfig, rasterize, world_to_cell
from gfsem.sim.scene import Scene, generate_scene


def make_scene(centres, classes, kappa=0.0, s_goal=30.0):
    cfg = SceneConfig.from_yaml()
    centres = np.asarray(centres, float).reshape(-1, 2)
    classes = np.asarray(classes, np.int8)
    radii = np.where(classes == BOLLARD, cfg.bollard_radius, cfg.cone_radius)
    heights = np.where(classes == BOLLARD, cfg.bollard_height, cfg.cone_height)
    tr = Track(kappa, 12.0, -10.0, s_goal + 20)
    return Scene(0, tr, centres, radii, heights, classes, (5.0, 25.0), s_goal, True, "test")


def test_vehicle_yaml_is_juno_datasheet():
    v = VehicleConfig.from_yaml()
    assert (v.wheelbase, v.length, v.width) == (1.60, 2.875, 1.165)
    assert v.max_steer_deg == 10.0
    assert 1 / v.kappa_max == pytest.approx(9.07, abs=0.01)
    # door rule: about 2.12 m between bollard centres
    s = SceneConfig.from_yaml()
    assert v.width + 2 * s.door_clearance + 2 * s.bollard_radius == pytest.approx(2.115, abs=1e-3)


@pytest.mark.parametrize("kappa", [0.0, 1 / 30, -1 / 30])
def test_frenet_roundtrip(kappa):
    tr = Track(kappa, 12.0, -10, 50)
    s = np.linspace(-5, 45, 20)
    d = np.linspace(-6, 6, 20)
    s2, d2 = tr.frenet(tr.point(s, d))
    assert np.allclose(s, s2, atol=1e-9) and np.allclose(d, d2, atol=1e-9)


def test_frame_roundtrip():
    pose = np.array([3.0, -2.0, 0.7])
    p = np.random.default_rng(0).normal(size=(10, 2))
    assert np.allclose(from_frame(to_frame(p, pose), pose), p)


def test_box_circle_clearance():
    v = VehicleConfig.from_yaml()
    pose = np.zeros((1, 3))
    front = v.body_x[1]
    c = box_circle_clearance(pose, np.array([[front + 1.0, 0.0]]), np.array([0.25]), v.body_x, v.width)
    assert c[0, 0] == pytest.approx(0.75)
    c = box_circle_clearance(pose, np.array([[0.5, 0.0]]), np.array([0.25]), v.body_x, v.width)
    assert c[0, 0] < 0


def test_lidar_cone_under_beams_and_front_arc():
    lc = LidarConfig()
    # cone 1.5 m in front of the sensor: below the lowest beam; bollard at same range: visible
    near_cone = make_scene([[lc.mount_x + 1.75, 0.0]], [CONE])
    obs = scan(near_cone, np.zeros(3), NOISE_LEVELS["L0"], np.random.default_rng(0), lc)
    assert len(obs.points) == 0
    far_cone = make_scene([[lc.mount_x + 4.0, 0.0]], [CONE])
    obs = scan(far_cone, np.zeros(3), NOISE_LEVELS["L0"], np.random.default_rng(0), lc)
    assert len(obs.points) > 0
    # front arc only: every hit lies on the circle, on the sensor-facing half
    d = obs.points - far_cone.centres[0]
    assert np.allclose(np.hypot(*d.T), 0.25, atol=1e-6)
    assert (d[:, 0] < 0).all()
    # geometric visibility threshold ~2.03 m for 0.40 m cones
    e_min = lc.elev_range[0]
    assert (lc.mount_height - 0.40) / math.tan(-e_min) == pytest.approx(2.03, abs=0.01)


def test_lidar_bollard_occludes():
    lc = LidarConfig()
    sc = make_scene([[8.0, 0.0], [14.0, 0.0]], [BOLLARD, BOLLARD])
    obs = scan(sc, np.zeros(3), NOISE_LEVELS["L0"], np.random.default_rng(0), lc)
    assert (obs.points[:, 0] < 9.0).all()           # nothing from the hidden bollard
    # a cone does not hide a bollard behind it (beams pass over the cone)
    sc = make_scene([[8.0, 0.0], [14.0, 0.0]], [CONE, BOLLARD])
    obs = scan(sc, np.zeros(3), NOISE_LEVELS["L0"], np.random.default_rng(0), lc)
    assert (obs.points[:, 0] > 13.0).any()


def test_raster_channels():
    sc = make_scene([[10.0, 1.0]], [BOLLARD])
    obs = scan(sc, np.zeros(3), NOISE_LEVELS["L1"], np.random.default_rng(1))
    R = rasterize(obs)
    assert R.shape == (6, 128, 128) and R.dtype == np.float32
    assert R.min() >= 0 and R.max() <= 1
    rc = RasterConfig()
    r, c = world_to_cell(np.array([[9.8, 1.0]]), rc)
    assert R[0, r[0] - 1:r[0] + 2, c[0] - 1:c[0] + 2].max() == 1.0     # hit near the bollard front
    assert R[2, r[0] - 8:r[0] - 2, c[0] - 2:c[0] + 2].sum() >= 6     # shadow behind it is unknown
    rf, cf = world_to_cell(np.array([[5.0, 0.0]]), rc)
    assert R[1, rf[0], cf[0]] == 1.0                                   # free in front
    assert np.all(R[1] * R[0] == 0)


def test_string_pull_straight_and_corner():
    s, e = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    p = string_pull(s, [(np.array([5.0, 1.0]), np.array([5.0, -1.0]))], e)
    assert len(p) == 2
    # portal entirely to the left forces a corner
    p = string_pull(s, [(np.array([5.0, 4.0]), np.array([5.0, 2.0]))], e)
    assert len(p) == 3 and np.allclose(p[1], [5.0, 2.0])


def test_scene_deterministic():
    a, b = generate_scene(7), generate_scene(7)
    assert np.array_equal(a.centres, b.centres) and a.feasible == b.feasible


def test_privileged_avoids_single_bollard():
    sc = make_scene([[15.0, 0.0]], [BOLLARD])
    r = run_episode(sc, CorridorPrivileged(CorridorPlanner()), "L0")
    assert r.outcome == "success" and r.min_clearance > 0.2


def test_privileged_brakes_at_wall():
    ys = np.arange(-5.4, 5.5, 1.2)
    sc = make_scene([[15.0, y] for y in ys], [BOLLARD] * len(ys))
    sc.feasible = False
    p = CorridorPlanner()
    assert p.plan(input_from_privileged(privileged_view(sc, np.zeros(3))), 3.0, 0.0).brake
    r = run_episode(sc, CorridorPrivileged(CorridorPlanner()), "L0")
    assert r.outcome == "stopped" and r.correct_brake
