import io
from pathlib import Path

import numpy as np

from gfsem.data.sim_dataset import (DatasetConfig, generate_scene_frames, observation_from_shard,
                                    pack_shard)
from gfsem.planners.corridor import CorridorConfig, input_from_observation
from gfsem.sim.config import SceneConfig
from gfsem.sim.lidar import Observation
from gfsem.sim.raster import RasterConfig, rasterize, world_to_cell


def _obs_left_edge_only():
    x = np.arange(4.0, 22.0, 0.5)
    return Observation(
        points=np.zeros((0, 2), np.float32), labels=np.zeros(0, np.uint8),
        scan_ranges=np.full(601, 40.0, np.float32),
        edge_points=np.stack([x, np.full_like(x, 6.0)], -1).astype(np.float32),
        edge_side=np.ones(len(x), np.int8),
        goal_line=np.array([[40, -6], [40, 6]], np.float32), ego_state=np.zeros(5, np.float32))


def test_raster_drivable_uses_the_planners_edge_fill():
    obs = _obs_left_edge_only()
    R = rasterize(obs)
    rc = RasterConfig()
    cell = lambda x, y: tuple(int(v[0]) for v in world_to_cell(np.array([[x, y]]), rc))
    assert R[3][cell(10.0, -5.5)] == 1.0      # right side filled at 12 m from the left edge
    assert R[3][cell(10.0, -6.5)] == 0.0
    inp = input_from_observation(obs, CorridorConfig(), SceneConfig.from_yaml())
    assert np.allclose(inp.right[:, 1], -6.0, atol=1e-6)   # planners see the same right edge


def _small_cfg():
    return DatasetConfig(K=2, alt_every=50, p_recovery=0.0)


def test_shard_stores_raw_observations_and_is_reproducible():
    a = generate_scene_frames(100_000, dcfg=_small_cfg())
    b = generate_scene_frames(100_000, dcfg=_small_cfg())
    A, B = pack_shard([a]), pack_shard([b])
    assert A.keys() == B.keys()
    for k in A:
        assert np.array_equal(A[k], B[k], equal_nan=A[k].dtype.kind == "f"), k
    buf = io.BytesIO()
    np.savez_compressed(buf, **A)
    buf.seek(0)
    L = dict(np.load(buf))
    assert "points" in L and "scan" in L and not any("raster" in k for k in L)
    for i in (0, len(a["frames"]) // 2):
        o_ref = Observation(**{**{k: v for k, v in zip(
            ("points", "labels", "scan_ranges", "edge_points", "edge_side", "goal_line", "ego_state"),
            (a["frames"][i]["obs"][k] for k in ("points", "labels", "scan", "edge", "edge_side", "goal", "ego")))}})
        o_ref.scan_ranges = o_ref.scan_ranges.astype(np.float32)
        assert np.array_equal(rasterize(observation_from_shard(L, i)), rasterize(o_ref))
    assert (np.diff(L["targets_off"]) >= 1).all()          # every frame has at least the driven mode


def test_edge_fill_rejects_bad_offsets_on_noisy_edges():
    """Noisy edges must not produce fill points inside the track (L4-type noise, seeds 0-29)."""
    from gfsem.sim.config import NOISE_LEVELS
    from gfsem.sim.edges import observed_edges
    from gfsem.sim.geometry import from_frame
    from gfsem.sim.lidar import scan
    from gfsem.sim.scene import generate_scene
    worst = 0.0
    for seed in range(30):
        sc = generate_scene(seed)
        for s in (0.0, sc.s_field[0] - 3.0, sc.s_field[0] + 8.0):
            pose = sc.track.pose(s)
            o = scan(sc, pose, NOISE_LEVELS["L4"], np.random.default_rng([seed, 99]), speed=3.0)
            for side, e in zip((1, -1), observed_edges(o.edge_points, o.edge_side, sc.track.width)):
                if len(e):
                    _, d = sc.track.frenet(from_frame(e, pose))
                    worst = max(worst, float(np.max(np.abs(side * d - sc.track.width / 2))))
    assert worst < 1.6, worst      # 4 sigma of L4 edge noise; the old fill put points ~4 m inside


def test_goal_channel_row0_is_far_edge_and_clipping():
    from gfsem.sim.raster import goal_channel
    rc = RasterConfig()
    drv = np.zeros((128, 128), np.float32)
    drv[:, 40:90] = 1.0
    # goal line across the track at x = 20 m -> rows around (22.4 - 20) / 0.2 = 12
    g = goal_channel(np.array([[20.0, -6.0], [20.0, 6.0]]), drv, rc)
    rows = np.nonzero(g.any(1))[0]
    assert rows.min() >= 10 and rows.max() <= 13
    # goal beyond the raster: far edge = row 0 (max x), 2 px, only inside the drivable corridor
    g = goal_channel(np.array([[40.0, -6.0], [40.0, 6.0]]), drv, rc)
    assert set(np.nonzero(g.any(1))[0]) == {0, 1}
    assert g[0, 40:90].all() and not g[0, :40].any()
    # segment that only partly enters the raster is clipped and drawn
    g = goal_channel(np.array([[10.0, -30.0], [10.0, 0.0]]), drv, rc)
    assert g.sum() > 0


def test_goal_channel_nonempty_in_every_frame_of_a_shard():
    import glob
    import pytest
    from gfsem.sim.raster import goal_channel
    files = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "data/sim/val_*.npz")))
    if not files:
        pytest.skip("no shard on this machine")
    L = dict(np.load(files[0]))
    rc = RasterConfig()
    empty = 0
    for i in range(len(L["seed"])):
        o = observation_from_shard(L, i)
        from gfsem.sim.raster import drivable_mask
        if goal_channel(o.goal_line.astype(np.float64), drivable_mask(o, rc), rc).sum() == 0:
            empty += 1
    assert empty == 0, f"{empty} of {len(L['seed'])} frames have an empty goal channel"
