import io

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
